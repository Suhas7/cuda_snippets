#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Each block computes a (max, sum) pair in one pass using online correction:
// when combining (m_a, d_a) and (m_b, d_b): m = max(m_a, m_b), d = d_a*exp(m_a-m) + d_b*exp(m_b-m)
__global__ void streaming_reduce(const float* input, float* out_max, float* out_sum, int N) {
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load element, seed its running count to 1
    s_max[tid] = idx < N ? input[idx] : -INFINITY;
    s_sum[tid] = idx < N ? 1.0f : 0.0f;
    __syncthreads();

    // Combine pairs via online (max, sum) correction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float ma = s_max[tid],    da = s_sum[tid];
            float mb = s_max[tid + stride], db = s_sum[tid + stride];
            float m  = fmaxf(ma, mb);
            s_max[tid] = m;
            s_sum[tid] = da * expf(ma - m) + db * expf(mb - m);
        }
        __syncthreads();
    }

    // First thread writes the block's (max, sum) pair
    if (tid == 0) {
        out_max[blockIdx.x] = s_max[0];
        out_sum[blockIdx.x] = s_sum[0];
    }
}

__global__ void merge_blocks(const float* in_max, const float* in_sum,
                              float* out_max, float* out_sum, int n) {
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load one block-level (max, sum) pair per thread
    s_max[tid] = idx < n ? in_max[idx] : -INFINITY;
    s_sum[tid] = idx < n ? in_sum[idx] : 0.0f;
    __syncthreads();

    // Combine pairs via online (max, sum) correction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float ma = s_max[tid],    da = s_sum[tid];
            float mb = s_max[tid + stride], db = s_sum[tid + stride];
            float m  = fmaxf(ma, mb);
            s_max[tid] = m;
            s_sum[tid] = da * expf(ma - m) + db * expf(mb - m);
        }
        __syncthreads();
    }

    // First thread writes the merged (max, sum) pair
    if (tid == 0) {
        out_max[blockIdx.x] = s_max[0];
        out_sum[blockIdx.x] = s_sum[0];
    }
}

__global__ void normalize(const float* input, float* output,
                           const float* g_max, const float* g_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = expf(input[idx] - g_max[0]) / g_sum[0];
}

void solve(const float* input, float* output, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_max; float* d_sum;
    cudaMalloc(&d_max, sizeof(float) * numBlocks);
    cudaMalloc(&d_sum, sizeof(float) * numBlocks);

    // First pass: one (max, sum) pair per block
    streaming_reduce<<<numBlocks, BLOCK_SIZE>>>(input, d_max, d_sum, N);

    // Merge block pairs until a single global (max, sum) remains
    int cur = numBlocks;
    while (cur > 1) {
        int next = (cur + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float* tmp_max; float* tmp_sum;
        cudaMalloc(&tmp_max, sizeof(float) * next);
        cudaMalloc(&tmp_sum, sizeof(float) * next);
        merge_blocks<<<next, BLOCK_SIZE>>>(d_max, d_sum, tmp_max, tmp_sum, cur);
        cudaFree(d_max); cudaFree(d_sum);
        d_max = tmp_max; d_sum = tmp_sum;
        cur = next;
    }

    // Subtract the global max, exponentiate, and divide by the global sum
    normalize<<<numBlocks, BLOCK_SIZE>>>(input, output, d_max, d_sum, N);
    cudaDeviceSynchronize();

    cudaFree(d_max);
    cudaFree(d_sum);
}
