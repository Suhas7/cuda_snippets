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


// Streaming softmax: (max, sum) stay on device, no DtoH sync
__global__ void k_stream_reduce(const float* in, float* out_max, float* out_sum, int N) {
    __shared__ float s_max[1024], s_sum[1024];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
    // Load element, seed its running count to 1
    s_max[tid] = idx < N ? in[idx] : -INFINITY;
    s_sum[tid] = idx < N ? 1.0f : 0.0f;
    __syncthreads();
    // Combine pairs via online (max, sum) correction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            float ma = s_max[tid], da = s_sum[tid];
            float mb = s_max[tid+s], db = s_sum[tid+s];
            float m = fmaxf(ma, mb);
            s_max[tid] = m;
            s_sum[tid] = da * expf(ma-m) + db * expf(mb-m);
        }
        __syncthreads();
    }
    // First thread writes the block's (max, sum) pair
    if (tid == 0) { out_max[blockIdx.x] = s_max[0]; out_sum[blockIdx.x] = s_sum[0]; }
}

__global__ void k_stream_merge(const float* in_max, const float* in_sum,
                                float* out_max, float* out_sum, int n) {
    __shared__ float s_max[1024], s_sum[1024];
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
    // Load one block-level (max, sum) pair per thread
    s_max[tid] = idx < n ? in_max[idx] : -INFINITY;
    s_sum[tid] = idx < n ? in_sum[idx] : 0.0f;
    __syncthreads();
    // Combine pairs via online (max, sum) correction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            float ma = s_max[tid], da = s_sum[tid];
            float mb = s_max[tid+s], db = s_sum[tid+s];
            float m = fmaxf(ma, mb);
            s_max[tid] = m;
            s_sum[tid] = da * expf(ma-m) + db * expf(mb-m);
        }
        __syncthreads();
    }
    // First thread writes the merged (max, sum) pair
    if (tid == 0) { out_max[blockIdx.x] = s_max[0]; out_sum[blockIdx.x] = s_sum[0]; }
}

__global__ void k_stream_normalize(const float* in, float* out,
                                    const float* g_max, const float* g_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Subtract the global max, exponentiate, and divide by the global sum
    if (i < N) out[i] = expf(in[i] - g_max[0]) / g_sum[0];
}

torch::Tensor softmax_streaming(torch::Tensor input) {
    input = input.contiguous();
    int N = input.size(0);
    auto out  = torch::empty_like(input);
    auto opts = input.options();
    int cur   = (N + 1023) / 1024;

    // First pass: one (max, sum) pair per block
    auto d_max = torch::empty({cur}, opts);
    auto d_sum = torch::empty({cur}, opts);
    k_stream_reduce<<<cur, 1024>>>(input.data_ptr<float>(),
                                   d_max.data_ptr<float>(), d_sum.data_ptr<float>(), N);

    // Merge block pairs until a single global (max, sum) remains
    while (cur > 1) {
        int next = (cur + 1023) / 1024;
        auto tmp_max = torch::empty({next}, opts);
        auto tmp_sum = torch::empty({next}, opts);
        k_stream_merge<<<next, 1024>>>(d_max.data_ptr<float>(), d_sum.data_ptr<float>(),
                                       tmp_max.data_ptr<float>(), tmp_sum.data_ptr<float>(), cur);
        d_max = tmp_max; d_sum = tmp_sum; cur = next;
    }

    // Subtract the global max, exponentiate, and divide by the global sum
    k_stream_normalize<<<(N+1023)/1024, 1024>>>(input.data_ptr<float>(), out.data_ptr<float>(),
                                                 d_max.data_ptr<float>(), d_sum.data_ptr<float>(), N);
    return out;
}

// ── Attention kernels ────────────────────────────────────────────────────────

#define ATTN_TILE 16
#define ATTN_Bc   32

// Naive attention: materializes the full N×N score matrix
__global__ void k_qkt(const float* Q, const float* K, float* S, int N, int d) {
    __shared__ float tQ[ATTN_TILE][ATTN_TILE], tK[ATTN_TILE][ATTN_TILE];
    int row = blockIdx.y * ATTN_TILE + threadIdx.y;
    int col = blockIdx.x * ATTN_TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (d + ATTN_TILE - 1) / ATTN_TILE; t++) {
        tQ[threadIdx.y][threadIdx.x] = (row < N && t*ATTN_TILE+threadIdx.x < d) ? Q[row*d + t*ATTN_TILE+threadIdx.x] : 0.f;
        tK[threadIdx.x][threadIdx.y] = (col < N && t*ATTN_TILE+threadIdx.y < d) ? K[col*d + t*ATTN_TILE+threadIdx.y] : 0.f;
        __syncthreads();
        for (int k = 0; k < ATTN_TILE; k++) acc += tQ[threadIdx.y][k] * tK[threadIdx.x][k];
        __syncthreads();
    }
    if (row < N && col < N) S[row*N + col] = acc * rsqrtf((float)d);
}

__global__ void k_row_softmax(float* S, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float* s = S + row * N;
    float m = -INFINITY;
    for (int j = 0; j < N; j++) m = fmaxf(m, s[j]);
    float sum = 0.f;
    for (int j = 0; j < N; j++) { s[j] = expf(s[j] - m); sum += s[j]; }
    for (int j = 0; j < N; j++) s[j] /= sum;
}

__global__ void k_sv(const float* S, const float* V, float* O, int N, int d) {
    __shared__ float tS[ATTN_TILE][ATTN_TILE], tV[ATTN_TILE][ATTN_TILE];
    int row = blockIdx.y * ATTN_TILE + threadIdx.y;
    int col = blockIdx.x * ATTN_TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (N + ATTN_TILE - 1) / ATTN_TILE; t++) {
        tS[threadIdx.y][threadIdx.x] = (row < N && t*ATTN_TILE+threadIdx.x < N) ? S[row*N + t*ATTN_TILE+threadIdx.x] : 0.f;
        tV[threadIdx.y][threadIdx.x] = (t*ATTN_TILE+threadIdx.y < N && col < d) ? V[(t*ATTN_TILE+threadIdx.y)*d + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < ATTN_TILE; k++) acc += tS[threadIdx.y][k] * tV[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < d) O[row*d + col] = acc;
}

torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
    int N = Q.size(0), d = Q.size(1);
    auto opts = Q.options();
    auto S = torch::zeros({N, N}, opts);
    auto O = torch::zeros({N, d}, opts);

    // Compute scaled QK^T
    dim3 threads(ATTN_TILE, ATTN_TILE);
    dim3 qkt_blocks((N+ATTN_TILE-1)/ATTN_TILE, (N+ATTN_TILE-1)/ATTN_TILE);
    k_qkt<<<qkt_blocks, threads>>>(Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(), N, d);

    // Row-wise softmax over N×N score matrix
    k_row_softmax<<<(N+255)/256, 256>>>(S.data_ptr<float>(), N);

    // Weighted sum S @ V
    dim3 sv_blocks((d+ATTN_TILE-1)/ATTN_TILE, (N+ATTN_TILE-1)/ATTN_TILE);
    k_sv<<<sv_blocks, threads>>>(S.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), N, d);

    return O;
}


// Flash attention: streaming over K/V tiles, never materializes N×N matrix
__global__ void k_flash_attn(const float* Q, const float* K, const float* V,
                              float* O, int N, int d) {
    int i   = blockIdx.x;
    int tid = threadIdx.x;

    // Carve the shared buffer into K/V tiles, output accumulator, and scratch
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + ATTN_Bc * d;
    float* o_acc  = smem + 2 * ATTN_Bc * d;
    float* exp_sc = smem + 2 * ATTN_Bc * d + d;
    float* tmp    = smem + 2 * ATTN_Bc * d + d + ATTN_Bc;

    // Zero the output accumulator
    for (int dk = tid; dk < d; dk += blockDim.x)
        o_acc[dk] = 0.0f;
    __syncthreads();

    // Running softmax state: row max and denominator
    float m = -INFINITY, denom = 0.0f;
    float inv_sqrt_d = rsqrtf((float)d);

    // Stream over K/V tiles
    for (int t = 0; t < (N + ATTN_Bc - 1) / ATTN_Bc; t++) {
        int j_base = t * ATTN_Bc, j = j_base + tid;

        // Load this K/V tile into shared memory
        for (int k = tid; k < ATTN_Bc * d; k += blockDim.x) {
            int jj = j_base + k / d, dk = k % d;
            K_tile[k] = jj < N ? K[jj * d + dk] : 0.0f;
            V_tile[k] = jj < N ? V[jj * d + dk] : 0.0f;
        }
        __syncthreads();

        // Scaled dot product Q[i] . K_tile[tid]
        float score = -INFINITY;
        if (j < N) {
            score = 0.0f;
            for (int dk = 0; dk < d; dk++)
                score += Q[i * d + dk] * K_tile[tid * d + dk];
            score *= inv_sqrt_d;
        }

        // Tile max (block reduction)
        tmp[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) tmp[tid] = fmaxf(tmp[tid], tmp[tid + s]);
            __syncthreads();
        }
        float m_tile = tmp[0];

        // Per-thread exp score and tile sum (block reduction)
        exp_sc[tid] = j < N ? expf(score - m_tile) : 0.0f;
        tmp[tid] = exp_sc[tid];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) tmp[tid] += tmp[tid + s];
            __syncthreads();
        }
        float d_tile = tmp[0];

        // Online update: merge running (m, denom) with this tile's (m_tile, d_tile)
        float m_new  = fmaxf(m, m_tile);
        float old_sc = expf(m - m_new);
        float new_sc = expf(m_tile - m_new);
        denom = denom * old_sc + d_tile * new_sc;
        m     = m_new;

        // Rescale existing o_acc and accumulate this tile's V contribution.
        // Each thread owns the dk slice {tid, tid+ATTN_Bc, ...} of o_acc,
        // so it loops over all j in the tile — no atomics, no cross-thread conflict.
        for (int dk = tid; dk < d; dk += blockDim.x) {
            o_acc[dk] *= old_sc;
            for (int jj = 0; jj < ATTN_Bc; jj++) {
                if (j_base + jj < N)
                    o_acc[dk] += exp_sc[jj] * new_sc * V_tile[jj * d + dk];
            }
        }
        __syncthreads();  // protect smem before next tile load
    }

    // Normalize by the accumulated denominator
    for (int dk = tid; dk < d; dk += blockDim.x)
        O[i * d + dk] = o_acc[dk] / denom;
}

torch::Tensor attention_flash(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
    int N = Q.size(0), d = Q.size(1);
    auto O = torch::zeros({N, d}, Q.options());
    size_t smem = (2 * ATTN_Bc * d + d + 2 * ATTN_Bc) * sizeof(float);
    k_flash_attn<<<N, ATTN_Bc, smem>>>(Q.data_ptr<float>(), K.data_ptr<float>(),
                                        V.data_ptr<float>(), O.data_ptr<float>(), N, d);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d",            &conv1d);
    m.def("conv2d",            &conv2d);
    m.def("matmul",            &matmul);
    m.def("softmax",           &softmax);
    m.def("softmax_streaming", &softmax_streaming);
    m.def("attention_naive",   &attention_naive);
    m.def("attention_flash",   &attention_flash);
}
