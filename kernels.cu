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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d", &conv1d);
    m.def("conv2d", &conv2d);
}
