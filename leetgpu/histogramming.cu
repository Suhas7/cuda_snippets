#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void one_hot(const int* input, int* onehot, int N, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int val = input[i];
    if (val >= 0 && val < num_bins)
        onehot[i * num_bins + val] = 1;
}

// Grid is 2D: (rowBlocks, num_bins). Each block reduces a chunk for one bin.
__global__ void reduce_cols(const int* onehot, int* partial, int N, int num_bins) {
    __shared__ int tmp[BLOCK_SIZE];
    int bin = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    tmp[tid] = (row < N) ? onehot[row * num_bins + bin] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) tmp[tid] += tmp[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        partial[bin * gridDim.x + blockIdx.x] = tmp[0];
}

__global__ void accumulate(const int* partial, int* histogram, int row_blocks, int num_bins) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= num_bins) return;
    int sum = 0;
    for (int b = 0; b < row_blocks; b++)
        sum += partial[bin * row_blocks + b];
    histogram[bin] = sum;
}

void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = BLOCK_SIZE;
    int rowBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int* onehot;
    cudaMalloc(&onehot, sizeof(int) * N * num_bins);
    cudaMemset(onehot, 0, sizeof(int) * N * num_bins);

    one_hot<<<rowBlocks, threadsPerBlock>>>(input, onehot, N, num_bins);
    cudaDeviceSynchronize();

    int* partial;
    cudaMalloc(&partial, sizeof(int) * num_bins * rowBlocks);

    dim3 grid(rowBlocks, num_bins);
    reduce_cols<<<grid, threadsPerBlock>>>(onehot, partial, N, num_bins);
    cudaDeviceSynchronize();

    int accumBlocks = (num_bins + threadsPerBlock - 1) / threadsPerBlock;
    accumulate<<<accumBlocks, threadsPerBlock>>>(partial, histogram, rowBlocks, num_bins);
    cudaDeviceSynchronize();

    cudaFree(onehot);
    cudaFree(partial);
}
