#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
typedef void (*ReductionKernel)(const float* input, int N, float* output);

// Generic host function for reduction
void reduction_host(const float* input, int N, float* result, ReductionKernel kernel) {
    float* d_input;
    float* d_output;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int current_N = N;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel recursively
    do {
        numBlocks = (current_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel<<<numBlocks, BLOCK_SIZE>>>(d_input, current_N, d_output);
        cudaMemcpy(d_input, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToDevice);
        current_N = numBlocks;
    } while (current_N > 1 && numBlocks > 1);

    // Copy the final result back to host
    cudaMemcpy(result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Map Kernels
__global__ void div_c(float* input, int N, float sum) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if( idx >= N ) return;
    input[idx] /= sum;
}

__global__ void sub_c_exp(const float* input, float* output, int N, float c) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if( idx >= N ) return;
    output[idx] = expf(input[idx]-c);
}

// Reduce Kernels
__global__ void find_sum(const float* input, int N, float* output) {
    __shared__ float tmp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    tmp[tid] = (idx < N) ? input[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tmp[tid] += tmp[tid + stride];
        }
        __syncthreads();
    }

    if (tid) return;
    output[blockIdx.x] = tmp[0];
}

__global__ void find_max(const float* input, int N, float* output) {
    __shared__ float tmp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    tmp[tid] = (idx < N) ? input[idx] : -INFINITY;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float vA = tmp[tid];
            float vB = tmp[tid + stride];
            tmp[tid] = fmaxf(vA, vB);
        }
        __syncthreads();
    }

    if (tid) return;
    output[blockIdx.x] = tmp[0];
}

void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Find max in log(N) time
    float max;
    reduction_host(input, N, &max, find_max);
    cudaDeviceSynchronize();
    
    // Subtract max val & exponentiate (fused) in C time
    sub_c_exp<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, max);    
    cudaDeviceSynchronize();
    
    // sum in log(N) time
    float total_sum;
    reduction_host(output, N, &total_sum, find_sum);
    cudaDeviceSynchronize();

    // divide in C time
    div_c<<<blocksPerGrid, threadsPerBlock>>>(output, N, total_sum);

    cudaDeviceSynchronize();
}