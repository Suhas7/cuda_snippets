import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ops = load(name='cuda_ops', sources=['kernels.cu'], verbose=False)

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def bench(label, fn, *args, warmup=20, iters=200):
    """
    Mean CUDA latency in ms. Wraps the timed region in an NVTX range so
    it shows up by name when profiled with `nsys profile python bench.py`.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(label)
    t0.record()
    for _ in range(iters):
        fn(*args)
    t1.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    return t0.elapsed_time(t1) / iters


def achieved_bw(nbytes, ms):
    """Achieved memory bandwidth in GB/s (for memory-bound kernels)."""
    return nbytes / (ms * 1e6)


def achieved_tflops(flops, ms):
    """Achieved throughput in TFLOPS (for compute-bound kernels)."""
    return flops / (ms * 1e9)


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

dev = 'cuda'
props = torch.cuda.get_device_properties(0)
print(f"device : {props.name}")
print(f"SMs    : {props.multi_processor_count}")
print(f"VRAM   : {props.total_memory / 2**30:.0f} GB")
print()

# ---------------------------------------------------------------------------
# Main latency table
# ---------------------------------------------------------------------------

W = 24
print(f"{'op':<{W}} {'custom ms':>10} {'torch ms':>10} {'ratio':>7}  metric")
print('-' * 65)

# conv1d  (memory-bound: reads input + kernel, writes output)
x1  = torch.randn(65536, device=dev)
k1  = torch.randn(32, device=dev)
mem = (x1.numel() + k1.numel() + x1.numel() - k1.numel() + 1) * x1.element_size()
tc  = bench("custom/conv1d", ops.conv1d, x1, k1)
tt  = bench("torch/conv1d",  lambda a, b: F.conv1d(a.view(1,1,-1), b.view(1,1,-1)).squeeze(), x1, k1)
print(f"{'conv1d (N=65k, K=32)':<{W}} {tc:>10.4f} {tt:>10.4f} {tt/tc:>6.2f}x  {achieved_bw(mem,tc):.1f} GB/s")

# conv2d
x2  = torch.randn(512, 512, device=dev)
k2  = torch.randn(5, 5, device=dev)
oh, ow = x2.shape[0]-k2.shape[0]+1, x2.shape[1]-k2.shape[1]+1
mem = (x2.numel() + k2.numel() + oh*ow) * x2.element_size()
tc  = bench("custom/conv2d", ops.conv2d, x2, k2)
tt  = bench("torch/conv2d",  lambda a, b: F.conv2d(a.view(1,1,512,512), b.view(1,1,5,5)).squeeze(), x2, k2)
print(f"{'conv2d (512x512, k=5)':<{W}} {tc:>10.4f} {tt:>10.4f} {tt/tc:>6.2f}x  {achieved_bw(mem,tc):.1f} GB/s")

# matmul (compute-bound at large N; naive kernel so cuBLAS dominates)
M = N = K = 1024
A   = torch.randn(M, N, device=dev)
B   = torch.randn(N, K, device=dev)
tc  = bench("custom/matmul", ops.matmul, A, B)
tt  = bench("torch/matmul",  torch.mm, A, B)
print(f"{'matmul (1024^3)':<{W}} {tc:>10.4f} {tt:>10.4f} {tt/tc:>6.2f}x  {achieved_tflops(2*M*N*K,tc):.3f} TFLOPS (custom)  {achieved_tflops(2*M*N*K,tt):.3f} TFLOPS (torch/cuBLAS)")

# softmax variants (memory-bound: read + write)
xs  = torch.randn(1 << 20, device=dev)
mem = 2 * xs.numel() * xs.element_size()
tc  = bench("custom/softmax",    ops.softmax, xs)
ts  = bench("stream/softmax",    ops.softmax_streaming, xs)
tt  = bench("torch/softmax",     F.softmax, xs, 0)
print(f"{'softmax/naive (1M)':<{W}} {tc:>10.4f} {tt:>10.4f} {tt/tc:>6.2f}x  {achieved_bw(mem,tc):.1f} GB/s")
print(f"{'softmax/streaming (1M)':<{W}} {ts:>10.4f} {tt:>10.4f} {tt/ts:>6.2f}x  {achieved_bw(mem,ts):.1f} GB/s")

# attention (compute+memory bound; FLOPs = 4*N²*d)
N_attn, d_attn = 1024, 64
Q = torch.randn(N_attn, d_attn, device=dev)
K = torch.randn(N_attn, d_attn, device=dev)
V = torch.randn(N_attn, d_attn, device=dev)
attn_flops = 4 * N_attn**2 * d_attn

def torch_attn(q, k, v):
    return F.scaled_dot_product_attention(q.unsqueeze(0).unsqueeze(0),
                                          k.unsqueeze(0).unsqueeze(0),
                                          v.unsqueeze(0).unsqueeze(0)).squeeze()

tc = bench("naive/attn",  ops.attention_naive, Q, K, V)
tf = bench("flash/attn",  ops.attention_flash, Q, K, V)
tt = bench("torch/attn",  torch_attn, Q, K, V)
print(f"{'attn/naive (N=1024,d=64)':<{W}} {tc:>10.4f} {tt:>10.4f} {tt/tc:>6.2f}x  {achieved_tflops(attn_flops,tc):.3f} TFLOPS")
print(f"{'attn/flash (N=1024,d=64)':<{W}} {tf:>10.4f} {tt:>10.4f} {tt/tf:>6.2f}x  {achieved_tflops(attn_flops,tf):.3f} TFLOPS")

# ---------------------------------------------------------------------------
# Matmul size sweep — shows where naive tiling falls apart vs cuBLAS
# ---------------------------------------------------------------------------

print()
print("matmul size sweep — naive kernel vs cuBLAS (TFLOPS)")
print(f"  {'N':>5}  {'custom':>8}  {'cublas':>8}  {'ratio':>6}")
for n in [128, 256, 512, 1024, 2048]:
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    f = 2 * n**3
    tc = bench(f"custom/matmul/{n}", ops.matmul, a, b, warmup=10, iters=100)
    tt = bench(f"torch/matmul/{n}",  torch.mm, a, b, warmup=10, iters=100)
    print(f"  {n:>5}  {achieved_tflops(f,tc):>8.3f}  {achieved_tflops(f,tt):>8.3f}  {tt/tc:>5.2f}x")

# ---------------------------------------------------------------------------
# Kernel-level breakdown via torch.profiler
# Equivalent to reading NSys kernel rows; run `nsys profile python bench.py`
# for the full trace with NVTX annotations visible per op.
# ---------------------------------------------------------------------------

print()
print("CUDA kernel breakdown — softmax (torch.profiler, 50 iters)")
for label, fn, args in [
    ("custom/naive",     ops.softmax,           (xs,)),
    ("custom/streaming", ops.softmax_streaming,  (xs,)),
    ("torch",            F.softmax,             (xs, 0)),
]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_stack=False,
    ) as prof:
        for _ in range(50):
            fn(*args)
    torch.cuda.synchronize()
    print(f"\n  [{label}]")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6,
                                    max_name_column_width=52))

print()
print("CUDA kernel breakdown — attention (torch.profiler, 20 iters)")
for label, fn, args in [
    ("naive", ops.attention_naive, (Q, K, V)),
    ("flash", ops.attention_flash, (Q, K, V)),
    ("torch", torch_attn,          (Q, K, V)),
]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_stack=False,
    ) as prof:
        for _ in range(20):
            fn(*args)
    torch.cuda.synchronize()
    print(f"\n  [{label}]")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6,
                                    max_name_column_width=52))
