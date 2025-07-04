#include "solve.h"
#include <cuda_runtime.h>

#define Bc 32

// One block per query row i; Bc threads per block handle one K/V tile at a time
__global__ void flash_attn(const float* Q, const float* K, const float* V,
                            float* O, int N, int d) {
    int i   = blockIdx.x;
    int tid = threadIdx.x;

    // Carve the shared buffer into K/V tiles, output accumulator, and scratch
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + Bc * d;
    float* o_acc  = smem + 2 * Bc * d;
    float* exp_sc = smem + 2 * Bc * d + d;
    float* tmp    = smem + 2 * Bc * d + d + Bc;

    // Zero the output accumulator
    for (int dk = tid; dk < d; dk += blockDim.x)
        o_acc[dk] = 0.0f;
    __syncthreads();

    float inv_sqrt_d = rsqrtf((float)d);

    // Stream over K/V tiles
    for (int t = 0; t < (N + Bc - 1) / Bc; t++) {
        int j_base = t * Bc, j = j_base + tid;

        // Load this K/V tile into shared memory
        for (int k = tid; k < Bc * d; k += blockDim.x) {
            int jj = j_base + k / d, dk = k % d;
            K_tile[k] = jj < N ? K[jj * d + dk] : 0.0f;
            V_tile[k] = jj < N ? V[jj * d + dk] : 0.0f;
        }
        __syncthreads();

        // Scaled dot product Q[i] . K_tile[tid]
        float score = -INFINITY;
        if (j < N) {
            score = 0.0f;
            for (int dk = 0; dk < d; dk++)
                score += Q[i * d + dk] * K_tile[tid * d + dk];
            score *= inv_sqrt_d;
        }

        // Tile max (block reduction)
        tmp[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) tmp[tid] = fmaxf(tmp[tid], tmp[tid + s]);
            __syncthreads();
        }
        float m_tile = tmp[0];

        // Per-thread exp score and tile sum (block reduction)
        exp_sc[tid] = j < N ? expf(score - m_tile) : 0.0f;
        tmp[tid] = exp_sc[tid];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) tmp[tid] += tmp[tid + s];
            __syncthreads();
        }
        float d_tile = tmp[0];
        (void)d_tile;
    }

    for (int dk = tid; dk < d; dk += blockDim.x)
        O[i * d + dk] = o_acc[dk];
}

void solve(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    size_t smem_sz = (2 * Bc * d + d + 2 * Bc) * sizeof(float);
    flash_attn<<<N, Bc, smem_sz>>>(Q, K, V, O, N, d);
    cudaDeviceSynchronize();
}
