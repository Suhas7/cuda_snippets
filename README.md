# [LeetGPU](https://leetgpu.com) Snippets:

Included the submissions I have saved locally in the `leetgpu` directory.
```
	1d-convolution.cu
	2d-convolution.cu
	attn-naive.cu - basic impl
	attn.cu - flash attn (WIP)
	histogramming_simple.cu - atomicAdd solution
	histogramming.cu - OHE solution
	k-means-clustering.cu
	matrix-multiplication.cu
	mean-squared-error.cu
	reduction.cu
	reverse-array.cu
	softmax-stream.cu - streaming softmax
	softmax.cu - naive
```

Benchmarking scripts focus on latency since I used LeetGPU's online judge to check correctness. `bench-simple.py` captures basic timings, `bench-comprehensive.py` adds NVTX annotations, achieved GB/s and TFLOPS, a matmul size sweep, and a `torch.profiler` kernel breakdown.

Focused on CUDA so I could get conceptual depth - played with Triton, but didn't want to get distracted going too broad on language learning.

## Softmax - Base

### Reduction Host
I used "reduction_host" to remove as much boilerplate as possible when it came to multi-block scaling - there was a lot of repeated logic in find_max, find_sum, etc, so this helped:
1. Break the code into low-bloat testable units (find_sum, find_max)
2. Avoid risky reimplementation of reduction_host for each reduction.

The rolled-up implementation ensures that we collect and reduce results across blocks no matter the amount we're dealing with - specifically, if we have BLOCK_SIZE^n for n ≥ 2, this loop compacts the logic nicely.

### Fusion

The subtract and exponentiate operations are per-element, so it was an obvious choice to fuse these into `sub_c_exp`, short for subtract constant & exponentiate.

### Streaming

There is a streaming variant at `softmax-stream.cu` inspired by flash attention.

## MSE

Since this is one of the simpler kernels, it was a good candidate to test out warp shuffle logic. I used warp-level reductions to improve latency.

I also fused the square/reduction.

## Histogramming

Included 2 approaches I tried. One is the canonical solution relying on atomicAdd that has a lot of contention on the histogram buckets.

I wanted to avoid this, since updating histogram buckets is a high-traffic point of contention. I chose the following approach:

1. Start with the length N vector of samples (range 0-(K-1)): [0,1,2,1,2] N=5, K=3
2. One-hot encode into NxK tensor: [[1,0,0], [0,1,0], [0,0,1], [0,1,0], [0,0,1]]
3. Reduce on dim0 to get a 1xK vector histogram: [1,2,2]

Increases memory pressure by a factor of K, but I wanted to test this approach & figured it'd be a good challenge to implement & work with per-dim reductions.

## K-Means

Focus here was developing a full/complete solution rather than a highly efficient one, since I'd already practiced techniques elsewhere. Hence, some
reliance on atomicAdd.

## Matmul & Conv

Very basic implementations, reviewing Resource(1) here covered enough of the concepts that I chose to focus more on the other kernels.

## Flash Attention (attn.cu)

Basic implementation for instructional purposes. I did it single-head & non-causal so I could just focus on getting correct tiled logic.

Main issue here is that the tiling logic is suboptimal, as the benchmarks show. It needs rework for a single block to handle multiple 
rows, in order for the loaded K/V tiles to be reused in computation (increasing arithmetic intensity and decreasing HBM traffic).

The lack of this prevents it from being a "real" implementation and is the core reason why it lags far behind PyTorch's `scaled_dot_product_attention`.

# Key Resources

(1) [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

This was a great resource to help me:
1. Get my hardware understanding rock-solid
2. Build GPU concurrency intuitions around tiling and streaming
3. See minimal example code to understand concepts like warp-level operations.
4. Contextualize optimizations w/ expected gains & hardware limits (& how to measure).


# AI Involvement

I authored the `leetgpu/*.cu` files from scratch, with AI-assistance (usually Claude) used for debugging & concept discussion.

AI was not used to author this writeup file.
AI was used to bring up the benchmarking scripts and the boilerplate code for the torch bindings (in `kernels.cu`).
