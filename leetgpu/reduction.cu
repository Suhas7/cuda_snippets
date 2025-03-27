#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void reduce(const float* input, float* output, int N) {
    __shared__ float tmp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    tmp[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tmp[tid] += tmp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = tmp[0];
    }
}

void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate block_sums only once
    float* block_sums;
    cudaMalloc(&block_sums, sizeof(float) * blocksPerGrid);

    // First reduction: input -> block_sums
    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, block_sums, N);
    cudaDeviceSynchronize();

    // Second reduction: block_sums -> output
    if (blocksPerGrid == 1) {
        cudaMemcpy(output, block_sums, sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int finalBlocks = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
        reduce<<<finalBlocks, threadsPerBlock>>>(block_sums, output, blocksPerGrid);
        cudaDeviceSynchronize();

        // If finalBlocks > 1, reduce again
        while (finalBlocks > 1) {
            int temp_n = finalBlocks;
            finalBlocks = (temp_n + threadsPerBlock - 1) / threadsPerBlock;
            float* temp_output;
            cudaMalloc(&temp_output, sizeof(float) * finalBlocks);
            reduce<<<finalBlocks, threadsPerBlock>>>(output, temp_output, temp_n);
            cudaDeviceSynchronize();
            cudaMemcpy(output, temp_output, sizeof(float), cudaMemcpyDeviceToDevice);
            cudaFree(temp_output);
        }
    }

    cudaFree(block_sums);
}