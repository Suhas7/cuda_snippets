#include "solve.h"
#include <cuda_runtime.h>

__global__ void conv2d(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= input_rows - kernel_rows + 1) return;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= input_cols - kernel_cols + 1) return;

    float val = 0;
    int row_idx = row * input_cols + col;
    int krow_idx = 0;
    for (int i = 0; i < kernel_rows; i++){
        for (int j = 0; j < kernel_cols; j++){
            val += input[row_idx + j] * kernel[krow_idx + j];
        }
        krow_idx += kernel_cols;
        row_idx += input_cols;
    }
    output[row * (input_cols - kernel_cols + 1) + col] = val;
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(((input_cols - kernel_cols + 1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       ((input_rows - kernel_rows + 1) + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    conv2d<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}