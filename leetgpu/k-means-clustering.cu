#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void calc_distances(const float* data_x, const float* data_y, int sample_size, float* centroid_x, float* centroid_y, int k, float* distances) {
    int base = blockIdx.x * ((BLOCK_SIZE / k) * k);
    int sample_id = threadIdx.x;
    int cluster_id = threadIdx.y;
    float x_dist = data_x[base + sample_id] - centroid_x[cluster_id];
    float y_dist = data_y[base + sample_id] - centroid_y[cluster_id];
    x_dist *= x_dist;
    y_dist *= y_dist;

    distances[base + sample_id * k + cluster_id] = x_dist + y_dist;
}

__global__ void assign_labels(float* distances, int k, int sample_size, int* labels) {
    int base = blockIdx.x * blockDim.x;
    int sample_id = threadIdx;
    if ( >= sample_size) return;

    float* my_dist = distances + k * (base + sample_id);

    float min_dist = INFINITY;
    // For each sample, manually compute new assignment
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

__global__ void recalculate_centroids(const float* data_x, const float* data_y, int* labels, int sample_size, int k) {
    // Sum reduction
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers 
void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {
    
    // First, calculate distances for each point to each cluster
    float* distance_mat;
    cudaMalloc(&distance_mat, sizeof(float) * sample_size * k);
    int numThreadsPerBlock = ((BLOCK_SIZE / k) * k);
    int numBlocks = (k * sample_size + numThreadsPerBlock - 1) / numThreadsPerBlock;
    dim3 numThreads = {BLOCK_SIZE / k, k};
    calc_distances<<<numBlocks, numThreads>>>(data_x, data_y, sample_size, initial_centroid_x, initial_centroid_y, k, distance_mat);

    // Reassign labels using distances
    numBlocks = (sample_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numThreads = k;
    assign_labels<<<numBlocks, numThreads>>>(distances, k, sample_size, labels);

    // Recalculate centroids

}
