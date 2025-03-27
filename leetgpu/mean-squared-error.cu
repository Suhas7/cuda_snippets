#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_mse_reduce(const float* A, const float* B, float* output, int N) {
    float val = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute squared difference / N
    if (idx < N) {
        float diff = A[idx] - B[idx];
        val = (diff * diff) / N;
    }

    // Warp-level reduction
    val = warp_reduce_sum(val);

    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    if (lane == 0)
        shared[warpId] = val;

    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0)
            output[blockIdx.x] = val;
    }
}

__global__ void reduce(const float* input, float* output, int N) {
    float val = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    if (idx < N)
        val = input[idx];

    // Reduce within warp
    val = warp_reduce_sum(val);

    // Shared memory for warp-level results
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // First thread in each warp writes its result to shared memory
    if (lane == 0)
        shared[warpId] = val;

    __syncthreads();

    // Final reduction by first warp
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0)
            output[blockIdx.x] = val;
    }
}

void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* tmp;
    cudaMalloc(&tmp, sizeof(float) * blocksPerGrid);

    // Fused kernel: computes per-block partial MSEs
    fused_mse_reduce<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, tmp, N);

    // Final reduction(s)
    int remaining = blocksPerGrid;
    while (remaining > 1) {
        int newBlocks = (remaining + threadsPerBlock - 1) / threadsPerBlock;
        reduce<<<newBlocks, threadsPerBlock>>>(tmp, tmp, remaining);
        remaining = newBlocks;
    }

    // Final result in tmp[0]
    cudaMemcpy(mse, tmp, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(tmp);
    cudaDeviceSynchronize();
}
