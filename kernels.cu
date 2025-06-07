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
    input = input.contiguous(); kernel = kernel.contiguous();
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
    input = input.contiguous(); kernel = kernel.contiguous();
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
    A = A.contiguous(); B = B.contiguous();
    int M = A.size(0), N = A.size(1), K = B.size(1);
    auto C = torch::zeros({M, K}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((K+15)/16, (M+15)/16);
    k_matmul<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

__global__ void k_reduce_max(const float* in, float* out, int N) {
    __shared__ float tmp[1024];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
    tmp[tid] = idx < N ? in[idx] : -INFINITY;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] = fmaxf(tmp[tid], tmp[tid+s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = tmp[0];
}

__global__ void k_reduce_sum(const float* in, float* out, int N) {
    __shared__ float tmp[1024];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
    tmp[tid] = idx < N ? in[idx] : 0.f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] += tmp[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = tmp[0];
}

__global__ void k_sub_exp(const float* in, float* out, int N, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = expf(in[i] - c);
}

__global__ void k_div(float* x, int N, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] /= c;
}

static float device_scalar(const float* d, int N, bool use_max) {
    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    int b = (N + 1023) / 1024;
    auto tmp = torch::empty({b}, opts);
    if (use_max) k_reduce_max<<<b, 1024>>>(d, tmp.data_ptr<float>(), N);
    else         k_reduce_sum<<<b, 1024>>>(d, tmp.data_ptr<float>(), N);
    if (b > 1) {
        auto tmp2 = torch::empty({1}, opts);
        if (use_max) k_reduce_max<<<1, 1024>>>(tmp.data_ptr<float>(), tmp2.data_ptr<float>(), b);
        else         k_reduce_sum<<<1, 1024>>>(tmp.data_ptr<float>(), tmp2.data_ptr<float>(), b);
        float v; cudaMemcpy(&v, tmp2.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost); return v;
    }
    float v; cudaMemcpy(&v, tmp.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost); return v;
}

torch::Tensor softmax(torch::Tensor input) {
    input = input.contiguous();
    int N = input.size(0);
    auto out = torch::empty_like(input);
    int blocks = (N + BS-1) / BS;
    float mx = device_scalar(input.data_ptr<float>(), N, true);
    k_sub_exp<<<blocks, BS>>>(input.data_ptr<float>(), out.data_ptr<float>(), N, mx);
    float sm = device_scalar(out.data_ptr<float>(), N, false);
    k_div<<<blocks, BS>>>(out.data_ptr<float>(), N, sm);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d",  &conv1d);
    m.def("conv2d",  &conv2d);
    m.def("matmul",  &matmul);
    m.def("softmax", &softmax);
}
