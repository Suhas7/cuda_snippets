#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void calc_distances(const float* data_x, const float* data_y, int sample_size, float* centroid_x, float* centroid_y, int k, float* distances) {
    // Each block owns blockDim.x = BLOCK_SIZE/k samples (one column of threads
    // per cluster), so block b covers samples [b*blockDim.x, (b+1)*blockDim.x)
    int base = blockIdx.x * blockDim.x;
    int sample_id = threadIdx.x;
    int cluster_id = threadIdx.y;
    if (base + sample_id >= sample_size) return;
    float x_dist = data_x[base + sample_id] - centroid_x[cluster_id];
    float y_dist = data_y[base + sample_id] - centroid_y[cluster_id];
    x_dist *= x_dist;
    y_dist *= y_dist;
    distances[(base + sample_id) * k + cluster_id] = x_dist + y_dist;
}

__global__ void assign_labels(float* distances, int k, int sample_size, int* labels) {
    int base = blockIdx.x * blockDim.x;
    int sample_id = threadIdx.x;
    if (base + sample_id >= sample_size) return;

    float* my_dist = distances + k * (base + sample_id);

    float min_dist = INFINITY;
    int new_label = -1;
    for (int i = 0; i < k; i++) {
        float curr_dist = my_dist[i];
        if (curr_dist < min_dist) {
            new_label = i;
            min_dist = curr_dist;
        }
    }
    labels[base + sample_id] = new_label;
}

__global__ void recalculate_centroids(const float* data_x, const float* data_y, int* labels, int sample_size,
                                      float* centroid_x, float* centroid_y, int* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;
    int c = labels[idx];
    atomicAdd(&centroid_x[c], data_x[idx]);
    atomicAdd(&centroid_y[c], data_y[idx]);
    atomicAdd(&counts[c], 1);
}

__global__ void divide_centroids(float* centroid_x, float* centroid_y, int* counts, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    int n = counts[c];
    if (n > 0) {
        centroid_x[c] /= n;
        centroid_y[c] /= n;
    }
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {

    float* distance_mat;
    cudaMalloc(&distance_mat, sizeof(float) * sample_size * k);

    int* counts;
    cudaMalloc(&counts, sizeof(int) * k);

    cudaMemcpy(final_centroid_x, initial_centroid_x, sizeof(float) * k, cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, sizeof(float) * k, cudaMemcpyDeviceToDevice);

    int samplesPerBlock = BLOCK_SIZE / k;
    dim3 distThreads(BLOCK_SIZE / k, k);

    for (int iter = 0; iter < max_iterations; iter++) {
        int numBlocks = (sample_size + samplesPerBlock - 1) / samplesPerBlock;
        calc_distances<<<numBlocks, distThreads>>>(data_x, data_y, sample_size, final_centroid_x, final_centroid_y, k, distance_mat);

        numBlocks = (sample_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        assign_labels<<<numBlocks, BLOCK_SIZE>>>(distance_mat, k, sample_size, labels);

        cudaMemset(final_centroid_x, 0, sizeof(float) * k);
        cudaMemset(final_centroid_y, 0, sizeof(float) * k);
        cudaMemset(counts, 0, sizeof(int) * k);
        recalculate_centroids<<<numBlocks, BLOCK_SIZE>>>(data_x, data_y, labels, sample_size, final_centroid_x, final_centroid_y, counts);

        int kBlocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        divide_centroids<<<kBlocks, BLOCK_SIZE>>>(final_centroid_x, final_centroid_y, counts, k);
    }

    cudaFree(distance_mat);
    cudaFree(counts);
}
