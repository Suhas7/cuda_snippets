#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx1 >= N/2) return;
    int idx2 = N-1-idx1;
    float val1 = input[idx1];
    float val2 = input[idx2];
    input[idx1] = val2;
    input[idx2] = val1;
}

// [0,1,2,3,4]

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}