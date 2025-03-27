#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Compute per-block histograms
__global__ void f_histogram(const int* input, int* blockResults, int N, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int val = input[idx];
    if (val < 0 || val >= num_bins) return;
    int base = num_bins * blockIdx.x;
    atomicAdd(&blockResults[base + val], 1);
}

// Reduce per-block histograms into a single histogram
__global__ void reduce_histogram(const int* blockResults, int* histogram, int num_blocks, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bins) return;

    int sum = 0;
    for (int block = 0; block < num_blocks; block++) {
        sum += blockResults[block * num_bins + idx];
    }
    histogram[idx] = sum;
}

void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int size = sizeof(int) * numBlocks * num_bins;

    int* blockResults;
    cudaMalloc(&blockResults, size);
    cudaMemset(blockResults, 0, size);

    // Step 1: Compute per-block histograms
    f_histogram<<<numBlocks, threadsPerBlock>>>(input, blockResults, N, num_bins);
    cudaDeviceSynchronize();

    // Step 2: Reduce per-block histograms into final histogram
    int reduceBlocks = (num_bins + threadsPerBlock - 1) / threadsPerBlock;
    reduce_histogram<<<reduceBlocks, threadsPerBlock>>>(blockResults, histogram, numBlocks, num_bins);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(blockResults);
}