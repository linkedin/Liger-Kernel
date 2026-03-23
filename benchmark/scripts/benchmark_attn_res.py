"""
AttnRes Benchmark: Liger (Triton) vs PyTorch vs torch.compile

Kimi Attention Residuals: softmax attention over depth blocks.
"""

import time
import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops.attn_res import LigerAttnResFunction
from liger_kernel.utils import infer_device

device = infer_device()


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


# ============================================================================
# PyTorch reference
# ============================================================================

def pytorch_attn_res(V, w_query, w_norm, eps=1e-6):
    """
    V: [N, B, T, D]
    w_query: [D]
    w_norm: [D]
    """
    N, B, T, D = V.shape
    # RMSNorm each block
    V_f32 = V.float()
    rms = torch.sqrt(V_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    K = (V_f32 / rms).to(V.dtype) * w_norm  # [N, B, T, D]

    # scores = dot(w_query, K) per block per token
    scores = torch.einsum('d, n b t d -> n b t', w_query.float(), K.float())
    alpha = scores.softmax(dim=0)  # [N, B, T]

    # weighted sum
    h = torch.einsum('n b t, n b t d -> b t d', alpha, V.float()).to(V.dtype)
    return h


pytorch_attn_res_compiled = None


def get_compiled_fn():
    global pytorch_attn_res_compiled
    if pytorch_attn_res_compiled is None:
        pytorch_attn_res_compiled = torch.compile(pytorch_attn_res, mode='max-autotune')
    return pytorch_attn_res_compiled


# ============================================================================
# 正确性测试
# ============================================================================

def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (4, 2, 64, 4096, torch.float16,  "N=4, D=4096, fp16"),
        (8, 2, 64, 4096, torch.float16,  "N=8, D=4096, fp16"),
        (4, 2, 64, 4096, torch.bfloat16, "N=4, D=4096, bf16"),
        (8, 2, 64, 8192, torch.float16,  "N=8, D=8192, fp16"),
        (4, 2, 64, 4096, torch.float32,  "N=4, D=4096, fp32"),
    ]

    for N, B, T, D, dtype, name in configs:
        V = torch.randn(N, B, T, D, device=device, dtype=dtype)
        w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
        w_norm  = torch.ones(D, device=device, dtype=dtype)

        ref  = pytorch_attn_res(V, w_query, w_norm)
        ours = LigerAttnResFunction.apply(V, w_query, w_norm, 1e-6)

        diff = (ours.float() - ref.float()).abs().max().item()
        tol = 1e-2 if dtype != torch.float32 else 1e-5
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    configs = [
        (4, 2, 64, 4096, torch.float16,  "N=4, D=4096, fp16"),
        (8, 2, 64, 4096, torch.float16,  "N=8, D=4096, fp16"),
        (4, 2, 64, 4096, torch.float32,  "N=4, D=4096, fp32"),
    ]

    for N, B, T, D, dtype, name in configs:
        V = torch.randn(N, B, T, D, device=device, dtype=dtype)
        w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
        w_norm  = torch.ones(D, device=device, dtype=dtype)

        # Reference
        V_ref = V.clone().requires_grad_(True)
        wq_ref = w_query.clone().requires_grad_(True)
        wn_ref = w_norm.clone().requires_grad_(True)
        h_ref = pytorch_attn_res(V_ref, wq_ref, wn_ref)
        h_ref.sum().backward()

        # Ours
        V_ours = V.clone().requires_grad_(True)
        wq_ours = w_query.clone().requires_grad_(True)
        wn_ours = w_norm.clone().requires_grad_(True)
        h_ours = LigerAttnResFunction.apply(V_ours, wq_ours, wn_ours, 1e-6)
        h_ours.sum().backward()

        dv_diff = (V_ours.grad.float() - V_ref.grad.float()).abs().max().item()
        dwq_diff = (wq_ours.grad.float() - wq_ref.grad.float()).abs().max().item()
        dwn_diff = (wn_ours.grad.float() - wn_ref.grad.float()).abs().max().item()
        tol_v = 5e-2 if dtype != torch.float32 else 1e-3
        # dWq/dWn accumulate across all tokens via atomic_add; fp32 accum vs fp16 ref → large diff expected
        tol_w = 1.0 if dtype != torch.float32 else 1e-3
        status = "PASS" if dv_diff < tol_v and dwq_diff < tol_w and dwn_diff < tol_w else "FAIL"
        print(f"  {name}: dV={dv_diff:.2e}, dWq={dwq_diff:.2e}, dWn={dwn_diff:.2e} [{status}]")

    print("Backward test done!\n")


# ============================================================================
# 性能测试
# ============================================================================

def bench_forward(N, B, T, D, dtype, device='cuda'):
    results = {}
    V = torch.randn(N, B, T, D, device=device, dtype=dtype)
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm  = torch.ones(D, device=device, dtype=dtype)

    def pt_fn():
        return pytorch_attn_res(V, w_query, w_norm)
    results['pytorch'] = benchmark_fn(pt_fn)

    compiled_fn = get_compiled_fn()
    # warmup compile
    for _ in range(3):
        compiled_fn(V, w_query, w_norm)
    torch.cuda.synchronize()

    def compiled():
        return compiled_fn(V, w_query, w_norm)
    results['torch.compile'] = benchmark_fn(compiled)

    def ours_fn():
        return LigerAttnResFunction.apply(V, w_query, w_norm, 1e-6)
    results['ours'] = benchmark_fn(ours_fn)

    return results


def bench_fwd_bwd(N, B, T, D, dtype, device='cuda'):
    results = {}
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm  = torch.ones(D, device=device, dtype=dtype)

    def pt_fn():
        V = torch.randn(N, B, T, D, device=device, dtype=dtype, requires_grad=True)
        h = pytorch_attn_res(V, w_query, w_norm)
        h.sum().backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        V = torch.randn(N, B, T, D, device=device, dtype=dtype, requires_grad=True)
        h = LigerAttnResFunction.apply(V, w_query, w_norm, 1e-6)
        h.sum().backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    return results


def print_results(title, results, baseline='pytorch'):
    print(f"\n{title}")
    print("=" * 60)
    base = results.get(baseline, 1.0)
    for name, ms in results.items():
        speedup = base / ms if ms == ms else float('nan')
        tag = " (baseline)" if name == baseline else ""
        print(f"  {name:20s}: {ms:8.3f} ms  ({speedup:5.2f}x){tag}")


def main():
    print("=" * 70)
    print("AttnRes Benchmark: Ours vs PyTorch vs torch.compile")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        # (N_blocks, B, T, D, dtype, name)
        (4,  4, 2048, 4096, torch.float16,  "N=4  [D=4096, fp16]"),
        (8,  4, 2048, 4096, torch.float16,  "N=8  [D=4096, fp16]"),
        (8,  4, 2048, 8192, torch.float16,  "N=8  [D=8192, fp16]"),
        (16, 4, 2048, 4096, torch.float16,  "N=16 [D=4096, fp16]"),
        (8,  4, 2048, 4096, torch.bfloat16, "N=8  [D=4096, bf16]"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass")
    print("=" * 70)
    for N, B, T, D, dtype, name in configs:
        results = bench_forward(N, B, T, D, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for N, B, T, D, dtype, name in configs:
        results = bench_fwd_bwd(N, B, T, D, dtype)
        print_results(name, results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        quick_test()
        backward_test()
    else:
        quick_test()
        backward_test()
        main()
