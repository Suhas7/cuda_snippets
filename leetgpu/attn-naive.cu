#include "solve.h"
#include <cuda_runtime.h>

#define TILE 16

__global__ void qkt_kernel(const float* Q, const float* K, float* S, int N, int d) {
    __shared__ float tQ[TILE][TILE], tK[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (d + TILE - 1) / TILE; t++) {
        tQ[threadIdx.y][threadIdx.x] = (row < N && t*TILE+threadIdx.x < d) ? Q[row*d + t*TILE+threadIdx.x] : 0.f;
        tK[threadIdx.x][threadIdx.y] = (col < N && t*TILE+threadIdx.y < d) ? K[col*d + t*TILE+threadIdx.y] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += tQ[threadIdx.y][k] * tK[threadIdx.x][k];
        __syncthreads();
    }
    if (row < N && col < N)
        S[row*N + col] = acc * rsqrtf((float)d);
}

__global__ void row_softmax(float* S, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float* s = S + row * N;
    float m = -INFINITY;
    for (int j = 0; j < N; j++) m = fmaxf(m, s[j]);
    float sum = 0.f;
    for (int j = 0; j < N; j++) { s[j] = expf(s[j] - m); sum += s[j]; }
    for (int j = 0; j < N; j++) s[j] /= sum;
}

__global__ void sv_kernel(const float* S, const float* V, float* O, int N, int d) {
    __shared__ float tS[TILE][TILE], tV[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        tS[threadIdx.y][threadIdx.x] = (row < N && t*TILE+threadIdx.x < N) ? S[row*N + t*TILE+threadIdx.x] : 0.f;
        tV[threadIdx.y][threadIdx.x] = (t*TILE+threadIdx.y < N && col < d) ? V[(t*TILE+threadIdx.y)*d + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += tS[threadIdx.y][k] * tV[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < d) O[row*d + col] = acc;
}

void solve(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    float* S;
    cudaMalloc(&S, sizeof(float) * N * N);

    dim3 threads(TILE, TILE);
    dim3 qkt_blocks((N+TILE-1)/TILE, (N+TILE-1)/TILE);
    qkt_kernel<<<qkt_blocks, threads>>>(Q, K, S, N, d);
    cudaDeviceSynchronize();

    int bs = 256;
    row_softmax<<<(N+bs-1)/bs, bs>>>(S, N);
    cudaDeviceSynchronize();

    dim3 sv_blocks((d+TILE-1)/TILE, (N+TILE-1)/TILE);
    sv_kernel<<<sv_blocks, threads>>>(S, V, O, N, d);
    cudaDeviceSynchronize();

    cudaFree(S);
}
