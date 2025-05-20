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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d", &conv1d);
}
