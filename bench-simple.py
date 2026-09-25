import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ops = load(name='cuda_ops', sources=['kernels.cu'], verbose=False)

def bench(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn(*args)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters

dev = 'cuda'
print(f"{'op':<14} {'custom (ms)':>12} {'torch (ms)':>12}")
print('-' * 40)

x1  = torch.randn(65536, device=dev)
k1  = torch.randn(32, device=dev)
tc  = bench(ops.conv1d, x1, k1)
tt  = bench(lambda a, b: F.conv1d(a.view(1,1,-1), b.view(1,1,-1)).squeeze(), x1, k1)
print(f"{'conv1d':<14} {tc:>12.4f} {tt:>12.4f}")

x2  = torch.randn(512, 512, device=dev)
k2  = torch.randn(5, 5, device=dev)
tc  = bench(ops.conv2d, x2, k2)
tt  = bench(lambda a, b: F.conv2d(a.view(1,1,512,512), b.view(1,1,5,5)).squeeze(), x2, k2)
print(f"{'conv2d':<14} {tc:>12.4f} {tt:>12.4f}")

A   = torch.randn(512, 512, device=dev)
B   = torch.randn(512, 512, device=dev)
tc  = bench(ops.matmul, A, B)
tt  = bench(torch.mm, A, B)
print(f"{'matmul':<14} {tc:>12.4f} {tt:>12.4f}")

xs  = torch.randn(65536, device=dev)
tc  = bench(ops.softmax, xs)
ts  = bench(ops.softmax_streaming, xs)
tt  = bench(F.softmax, xs, 0)
print(f"{'softmax':<14} {tc:>12.4f} {tt:>12.4f}")
print(f"{'softmax_stream':<14} {ts:>12.4f} {tt:>12.4f}")

N, d = 1024, 64
Q  = torch.randn(N, d, device=dev)
K  = torch.randn(N, d, device=dev)
V  = torch.randn(N, d, device=dev)
def torch_attn(q, k, v):
    return F.scaled_dot_product_attention(q.unsqueeze(0).unsqueeze(0),
                                          k.unsqueeze(0).unsqueeze(0),
                                          v.unsqueeze(0).unsqueeze(0)).squeeze()
tc  = bench(ops.attention_naive, Q, K, V)
tf  = bench(ops.attention_flash, Q, K, V)
tf2 = bench(ops.attention_flash_v2, Q, K, V)
tt  = bench(torch_attn, Q, K, V)
print(f"{'attn/naive':<14} {tc:>12.4f} {tt:>12.4f}")
print(f"{'attn/flash':<14} {tf:>12.4f} {tt:>12.4f}")
print(f"{'attn/flash_v2':<14} {tf2:>12.4f} {tt:>12.4f}")
