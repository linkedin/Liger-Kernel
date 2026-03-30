"""
Benchmark: LigerFusedMoEFunction vs. reference LigerExperts Python loop.

Measures speed (ms) and memory (MB) for:
  - "liger":     new fused Triton grouped GEMM kernel
  - "reference": original Python loop (LigerExperts with SiLUMulFunction)

Usage:
    python benchmark_fused_moe.py
    python benchmark_fused_moe.py --model llama3-8b --overwrite
    python benchmark_fused_moe.py --sweep_dim num_tokens

Sweep dimensions (--sweep_dim):
    num_tokens  : vary total tokens T (default)
    num_experts : vary number of experts E
    hidden_size : vary hidden dimension H
"""

import argparse
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from liger_kernel.ops.fused_moe import LigerFusedMoEFunction
from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.utils import infer_device

device = infer_device()


# ---------------------------------------------------------------------------
# Reference implementation (matches original LigerExperts loop)
# ---------------------------------------------------------------------------


def _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    T, H = x.shape
    E = gate_up_proj.shape[0]
    final = torch.zeros_like(x)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index.long(), num_classes=E)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for eh in expert_hit:
        eidx = eh[0]
        top_k_pos, token_idx = torch.where(expert_mask[eidx])
        curr = x[token_idx]
        gate, up = nn.functional.linear(curr, gate_up_proj[eidx]).chunk(2, dim=-1)
        curr = LigerSiLUMulFunction.apply(gate, up)
        curr = nn.functional.linear(curr, down_proj[eidx])
        curr = curr * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, curr.to(final.dtype))
    return final


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _make_inputs(T, E, H, I, K, dtype):
    torch.manual_seed(42)
    x = torch.randn(T, H, dtype=dtype, device=device)
    gate_up_proj = torch.randn(E, 2 * I, H, dtype=dtype, device=device) * 0.02
    down_proj = torch.randn(E, H, I, dtype=dtype, device=device) * 0.02
    logits = torch.randn(T, E, device=device)
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)
    top_k_weights = torch.softmax(
        torch.gather(logits, 1, top_k_index.long()), dim=-1
    ).to(dtype)
    return x, gate_up_proj, down_proj, top_k_index, top_k_weights


def _warmup_and_time(fn: Callable, n_warmup: int = 5, n_iters: int = 20) -> float:
    """Return median wall-clock time in ms."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _measure_memory(fn: Callable) -> float:
    """Return peak GPU memory in MB during fn() (forward + backward)."""
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024**2


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class MoEBenchConfig:
    T: int           # total tokens
    E: int           # number of experts
    H: int           # hidden size
    I: int           # expert intermediate size
    K: int           # top-k
    dtype: torch.dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------


def benchmark_config(cfg: MoEBenchConfig, mode: str = "forward") -> Dict:
    """Benchmark one config. mode: 'forward' | 'backward' | 'full'."""
    x, gup, dn, idx, wts = _make_inputs(cfg.T, cfg.E, cfg.H, cfg.I, cfg.K, cfg.dtype)

    def _make_fused_fn(requires_grad: bool):
        x_ = x.clone().requires_grad_(requires_grad)
        gup_ = gup.clone().requires_grad_(requires_grad)
        dn_ = dn.clone().requires_grad_(requires_grad)
        wts_ = wts.clone().requires_grad_(requires_grad)

        def fn():
            out = LigerFusedMoEFunction.apply(x_, gup_, dn_, idx, wts_)
            if requires_grad:
                out.sum().backward()

        return fn

    def _make_ref_fn(requires_grad: bool):
        x_ = x.clone().requires_grad_(requires_grad)
        gup_ = gup.clone().requires_grad_(requires_grad)
        dn_ = dn.clone().requires_grad_(requires_grad)
        wts_ = wts.clone().requires_grad_(requires_grad)

        def fn():
            out = _reference_moe_forward(x_, gup_, dn_, idx, wts_)
            if requires_grad:
                out.sum().backward()

        return fn

    requires_grad = mode in ("backward", "full")

    fused_fn = _make_fused_fn(requires_grad)
    ref_fn = _make_ref_fn(requires_grad)

    # Speed
    fused_ms = _warmup_and_time(fused_fn)
    ref_ms = _warmup_and_time(ref_fn)

    # Memory
    fused_mb = _measure_memory(_make_fused_fn(requires_grad))
    ref_mb = _measure_memory(_make_ref_fn(requires_grad))

    speedup = ref_ms / fused_ms if fused_ms > 0 else float("nan")
    mem_ratio = ref_mb / fused_mb if fused_mb > 0 else float("nan")

    return {
        "T": cfg.T, "E": cfg.E, "H": cfg.H, "I": cfg.I, "K": cfg.K,
        "dtype": str(cfg.dtype).replace("torch.", ""),
        "mode": mode,
        "fused_ms": fused_ms,
        "ref_ms": ref_ms,
        "speedup": speedup,
        "fused_mb": fused_mb,
        "ref_mb": ref_mb,
        "mem_ratio": mem_ratio,
    }


# ---------------------------------------------------------------------------
# Sweep utilities
# ---------------------------------------------------------------------------


def _print_results(results: List[Dict], sweep_key: str):
    """Print a formatted table."""
    header = (
        f"{'':>8} | {'fused_ms':>9} | {'ref_ms':>9} | {'speedup':>8} | "
        f"{'fused_mb':>9} | {'ref_mb':>9} | {'mem_ratio':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{'mode':>6} sweep over {sweep_key}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r[sweep_key]:>8} | {r['fused_ms']:>9.3f} | {r['ref_ms']:>9.3f} | "
            f"{r['speedup']:>7.2f}x | {r['fused_mb']:>9.1f} | {r['ref_mb']:>9.1f} | "
            f"{r['mem_ratio']:>8.2f}x"
        )
    print(sep)


def sweep_num_tokens(base_cfg: MoEBenchConfig, token_values: List[int], mode: str = "full"):
    results = []
    for T in token_values:
        cfg = MoEBenchConfig(T=T, E=base_cfg.E, H=base_cfg.H, I=base_cfg.I, K=base_cfg.K, dtype=base_cfg.dtype)
        r = benchmark_config(cfg, mode=mode)
        print(f"  T={T:5d}: fused={r['fused_ms']:.3f}ms ref={r['ref_ms']:.3f}ms speedup={r['speedup']:.2f}x")
        results.append(r)
    _print_results(results, "T")
    return results


def sweep_num_experts(base_cfg: MoEBenchConfig, expert_values: List[int], mode: str = "full"):
    results = []
    for E in expert_values:
        K = min(base_cfg.K, E)
        cfg = MoEBenchConfig(T=base_cfg.T, E=E, H=base_cfg.H, I=base_cfg.I, K=K, dtype=base_cfg.dtype)
        r = benchmark_config(cfg, mode=mode)
        print(f"  E={E:4d} K={K}: fused={r['fused_ms']:.3f}ms ref={r['ref_ms']:.3f}ms speedup={r['speedup']:.2f}x")
        results.append(r)
    _print_results(results, "E")
    return results


def sweep_hidden_size(base_cfg: MoEBenchConfig, hidden_values: List[int], mode: str = "full"):
    results = []
    for H in hidden_values:
        I = H // 4
        cfg = MoEBenchConfig(T=base_cfg.T, E=base_cfg.E, H=H, I=I, K=base_cfg.K, dtype=base_cfg.dtype)
        r = benchmark_config(cfg, mode=mode)
        print(f"  H={H:5d}: fused={r['fused_ms']:.3f}ms ref={r['ref_ms']:.3f}ms speedup={r['speedup']:.2f}x")
        results.append(r)
    _print_results(results, "H")
    return results


def sweep_granularity(base_cfg: MoEBenchConfig, granularity_values: List[int], mode: str = "full"):
    """Sweep expert granularity G = H/I (finer-grained = smaller I, more experts at same capacity)."""
    results = []
    for G in granularity_values:
        I = max(16, base_cfg.H // G)
        E = G * base_cfg.E  # keep total capacity constant
        K = min(base_cfg.K * G, E)
        cfg = MoEBenchConfig(T=base_cfg.T, E=E, H=base_cfg.H, I=I, K=K, dtype=base_cfg.dtype)
        r = benchmark_config(cfg, mode=mode)
        r["G"] = G
        print(f"  G={G:3d} E={E:4d} I={I:4d} K={K}: fused={r['fused_ms']:.3f}ms ref={r['ref_ms']:.3f}ms speedup={r['speedup']:.2f}x")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# MoE model configs
# ---------------------------------------------------------------------------


MOE_MODEL_CONFIGS = {
    # (H, I, E, K, typical_seq_len)
    "mixtral-8x7b": MoEBenchConfig(T=1024, E=8,  H=4096, I=14336, K=2, dtype=torch.bfloat16),
    "mixtral-8x22b": MoEBenchConfig(T=1024, E=8,  H=6144, I=16384, K=2, dtype=torch.bfloat16),
    "qwen3-moe-30b": MoEBenchConfig(T=1024, E=128, H=2048, I=768,  K=8, dtype=torch.bfloat16),
    "qwen3-moe-235b": MoEBenchConfig(T=512,  E=128, H=7168, I=2560, K=8, dtype=torch.bfloat16),
    "default":       MoEBenchConfig(T=512,  E=16,  H=1024, I=512,  K=4, dtype=torch.bfloat16),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused MoE kernel")
    parser.add_argument("--model", default="default", choices=list(MOE_MODEL_CONFIGS.keys()),
                        help="MoE model config preset")
    parser.add_argument("--sweep_dim", default="num_tokens",
                        choices=["num_tokens", "num_experts", "hidden_size", "granularity"],
                        help="Dimension to sweep over")
    parser.add_argument("--mode", default="full", choices=["forward", "backward", "full"],
                        help="Benchmark mode")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_iters", type=int, default=20)
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    base_cfg = MOE_MODEL_CONFIGS[args.model]
    base_cfg.dtype = dtype_map[args.dtype]

    print(f"\n=== Fused MoE Benchmark ===")
    print(f"Model: {args.model}, sweep: {args.sweep_dim}, mode: {args.mode}, dtype: {args.dtype}")
    print(f"Base config: T={base_cfg.T}, E={base_cfg.E}, H={base_cfg.H}, I={base_cfg.I}, K={base_cfg.K}")
    print(f"Device: {device}")

    if args.sweep_dim == "num_tokens":
        token_values = [2**i for i in range(6, 13)]  # 64 → 4096
        sweep_num_tokens(base_cfg, token_values, mode=args.mode)

    elif args.sweep_dim == "num_experts":
        expert_values = [4, 8, 16, 32, 64, 128]
        sweep_num_experts(base_cfg, expert_values, mode=args.mode)

    elif args.sweep_dim == "hidden_size":
        hidden_values = [256, 512, 1024, 2048, 4096]
        sweep_hidden_size(base_cfg, hidden_values, mode=args.mode)

    elif args.sweep_dim == "granularity":
        # SonicMoE paper benchmark: vary granularity G = H/I
        results = sweep_granularity(base_cfg, [1, 2, 4, 8, 16], mode=args.mode)
        print("\nGranularity sweep (higher G = finer-grained experts):")
        for r in results:
            print(f"  G={r['G']:3d}: speedup={r['speedup']:.2f}x memory_ratio={r['mem_ratio']:.2f}x")

    # Also print a single-config summary
    print(f"\n=== Single-config summary ({args.model}) ===")
    for mode in ["forward", "backward", "full"]:
        r = benchmark_config(base_cfg, mode=mode)
        print(
            f"  {mode:8s}: fused={r['fused_ms']:7.3f}ms  ref={r['ref_ms']:7.3f}ms  "
            f"speedup={r['speedup']:5.2f}x  fused_mem={r['fused_mb']:6.1f}MB  ref_mem={r['ref_mb']:6.1f}MB  "
            f"mem_savings={r['mem_ratio']:5.2f}x"
        )


if __name__ == "__main__":
    main()
