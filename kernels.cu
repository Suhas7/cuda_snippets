#include <torch/extension.h>

#define BS 256

__global__ void k_conv1d(const float* in, const float* ker, float* out, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - K + 1) return;
    float s = 0;
    for (int j = 0; j < K; j++) s += in[i+j] * ker[j];
    out[i] = s;
}

torch::Tensor conv1d(torch::Tensor input, torch::Tensor kernel) {
    int N = input.size(0), K = kernel.size(0);
    auto out = torch::zeros({N-K+1}, input.options());
    k_conv1d<<<(N-K+1+BS-1)/BS, BS>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), out.data_ptr<float>(), N, K);
    return out;
}

__global__ void k_conv2d(const float* in, const float* ker, float* out,
                         int H, int W, int KH, int KW) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outH = H-KH+1, outW = W-KW+1;
    if (row >= outH || col >= outW) return;
    float s = 0;
    for (int i = 0; i < KH; i++)
        for (int j = 0; j < KW; j++)
            s += in[(row+i)*W + col+j] * ker[i*KW+j];
    out[row*outW+col] = s;
}

torch::Tensor conv2d(torch::Tensor input, torch::Tensor kernel) {
    int H = input.size(0), W = input.size(1);
    int KH = kernel.size(0), KW = kernel.size(1);
    auto out = torch::zeros({H-KH+1, W-KW+1}, input.options());
    dim3 threads(16, 16);
    dim3 blocks((W-KW+1+15)/16, (H-KH+1+15)/16);
    k_conv2d<<<blocks, threads>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), out.data_ptr<float>(), H, W, KH, KW);
    return out;
}

__global__ void k_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;
    float s = 0;
    for (int n = 0; n < N; n++) s += A[row*N+n] * B[n*K+col];
    C[row*K+col] = s;
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), N = A.size(1), K = B.size(1);
    auto C = torch::zeros({M, K}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((K+15)/16, (M+15)/16);
    k_matmul<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d", &conv1d);
    m.def("conv2d", &conv2d);
    m.def("matmul", &matmul);
}
