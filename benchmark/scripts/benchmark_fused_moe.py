"""
Benchmark: LigerFusedMoEFunction vs. HuggingFace Python loop.

Integrates with the Liger benchmark framework (SingleBenchmarkRunInput/Output,
run_benchmarks, CSV output to all_benchmark_data.csv).

Usage:
    python benchmark_fused_moe.py                       # T sweep, Qwen3-MoE-30B
    python benchmark_fused_moe.py --sweep-dim num_experts
    python benchmark_fused_moe.py --overwrite
"""

import argparse
import math

import torch
import torch.nn as nn

from benchmark_model_configs import QWEN3_MOE_30B
from benchmark_model_configs import estimate_kernel_peak_memory
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.ops import LigerFusedMoEFunction
from liger_kernel.utils import get_total_gpu_memory
from liger_kernel.utils import infer_device

device = infer_device()


# ---------------------------------------------------------------------------
# HuggingFace reference: Python loop per expert
# Matches transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
# ---------------------------------------------------------------------------


def _huggingface_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
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
        curr = nn.functional.silu(gate) * up
        curr = nn.functional.linear(curr, down_proj[eidx])
        curr = curr * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, curr.to(final.dtype))
    return final


# Expert counts used in the num_experts sweep (independent of model).
EXPERT_SWEEP_VALUES = [8, 16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def _make_moe_inputs(T, E, H, intermediate_dim, K, dtype, requires_grad=True):
    torch.manual_seed(42)
    x = torch.randn(T, H, dtype=dtype, device=device, requires_grad=requires_grad)
    gate_up_proj = (
        torch.randn(E, 2 * intermediate_dim, H, dtype=dtype, device=device, requires_grad=requires_grad) * 0.02
    )
    down_proj = torch.randn(E, H, intermediate_dim, dtype=dtype, device=device, requires_grad=requires_grad) * 0.02
    logits = torch.randn(T, E, device=device)
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)
    top_k_weights = (
        torch.softmax(torch.gather(logits, 1, top_k_index.long()), dim=-1).to(dtype).requires_grad_(requires_grad)
    )
    return x, gate_up_proj, down_proj, top_k_index, top_k_weights


# ---------------------------------------------------------------------------
# Framework-integrated benchmark functions
# ---------------------------------------------------------------------------


def _setup_fused_moe(input: SingleBenchmarkRunInput):
    """Return (fwd_fn, grad_tensors) for the given provider and config.

    extra_benchmark_config keys:
        sweep_dim : "T" or "E" — which dim input.x varies
        T, E      : fixed values for the dimension not being swept (None when swept)
        H, intermediate_dim, K   : model dimensions
        dtype     : torch.dtype
    """
    cfg = input.extra_benchmark_config
    T = int(input.x) if cfg["sweep_dim"] == "T" else cfg["T"]
    E = int(input.x) if cfg["sweep_dim"] == "E" else cfg["E"]
    H, intermediate_dim, K = cfg["H"], cfg["intermediate_dim"], cfg["K"]
    dtype = cfg["dtype"]

    x, gup, dn, idx, wts = _make_moe_inputs(T, E, H, intermediate_dim, K, dtype, requires_grad=True)

    if input.kernel_provider == "liger":

        def fwd_fn():
            return LigerFusedMoEFunction.apply(x, gup, dn, idx, wts)
    elif input.kernel_provider == "huggingface":

        def fwd_fn():
            return _huggingface_moe_forward(x, gup, dn, idx, wts)
    else:
        raise ValueError(f"Unknown provider: {input.kernel_provider}")

    return fwd_fn, [x, gup, dn, wts]


def bench_speed_fused_moe(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    fwd_fn, grad_tensors = _setup_fused_moe(input)
    return run_speed_benchmark(fwd_fn, input.kernel_operation_mode, grad_tensors)


def bench_memory_fused_moe(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    fwd_fn, _ = _setup_fused_moe(input)
    return run_memory_benchmark(fwd_fn, input.kernel_operation_mode)


# ---------------------------------------------------------------------------
# Autotune warmup
# ---------------------------------------------------------------------------


def _warmup_liger(T, E, H, intermediate_dim, K, dtype, sweep_dim):
    """Run one full fwd+bwd to exhaust Triton autotune for (H, intermediate_dim).

    Triton autotune key is (H_dim, I_dim), so a single call is sufficient to
    cache the best config for all subsequent calls with the same H and intermediate_dim.
    For the num_experts sweep we also call this once per E value to warm up
    CUDA caches for each expert count before do_bench starts timing.
    """
    warmup_input = SingleBenchmarkRunInput(
        x=T if sweep_dim == "T" else E,
        kernel_provider="liger",
        extra_benchmark_config={
            "sweep_dim": sweep_dim,
            "T": T,
            "E": E,
            "H": H,
            "intermediate_dim": intermediate_dim,
            "K": K,
            "dtype": dtype,
        },
    )
    warmup_fn, _ = _setup_fused_moe(warmup_input)
    warmup_out = warmup_fn()
    warmup_out.sum().backward()
    del warmup_out
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()
    else:
        torch.cpu.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LigerFusedMoEFunction")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV benchmark data",
    )
    parser.add_argument(
        "--sweep-dim",
        choices=["num_tokens", "num_experts"],
        default="num_tokens",
        help="Dimension to sweep (default: num_tokens)",
    )
    args = parser.parse_args()

    moe_cfg = QWEN3_MOE_30B
    E = moe_cfg.E
    H = moe_cfg.H
    intermediate_dim = moe_cfg.intermediate_dim
    K = moe_cfg.K
    probe_T = moe_cfg.T  # representative token count for probing and warmup
    dtype = torch.bfloat16

    print(
        f"Model: {moe_cfg.name} — E={E}, H={H}, intermediate_dim={intermediate_dim}, K={K}, "
        f"T_base={probe_T}, dtype={dtype}"
    )

    # Memory probe using huggingface (no Triton, higher footprint = safe upper bound).
    def _probe():
        probe_input = SingleBenchmarkRunInput(
            x=probe_T,
            kernel_provider="huggingface",
            extra_benchmark_config={
                "sweep_dim": "T",
                "T": None,
                "E": E,
                "H": H,
                "intermediate_dim": intermediate_dim,
                "K": K,
                "dtype": dtype,
            },
        )
        fwd_fn, _ = _setup_fused_moe(probe_input)
        return fwd_fn()

    peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)
    kernel_bpt = peak_bytes // probe_T

    # Pre-warm Liger's Triton autotune before benchmarks start.
    #
    # Autotune key is (H_dim, I_dim) — one warmup per (H, intermediate_dim) pair is sufficient
    # to cache the best config for the entire sweep.
    #
    # For num_tokens sweep: one pass with the model's base T is enough.
    # For num_experts sweep: one pass per E value in EXPERT_SWEEP_VALUES to also
    #   warm up CUDA caches for each expert count, since weight tensor sizes differ.
    print(f"Pre-warming Liger autotune (H={H}, intermediate_dim={intermediate_dim})...")

    if args.sweep_dim == "num_tokens":
        _warmup_liger(probe_T, E, H, intermediate_dim, K, dtype, sweep_dim="T")
    else:  # num_experts
        for e_val in EXPERT_SWEEP_VALUES:
            print(f"  warmup E={e_val}...")
            _warmup_liger(probe_T, e_val, H, intermediate_dim, K, dtype, sweep_dim="E")

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()
    else:
        torch.cpu.synchronize()

    print("Autotune warmup complete.\n")

    if args.sweep_dim == "num_tokens":
        # Derive a memory-safe upper bound for T from the probe measurement.
        # Target 40% GPU memory utilisation to leave headroom for framework overhead.
        usable_bytes = get_total_gpu_memory() * (1024**3) * 0.4
        max_T = min(32768, max(256, int(usable_bytes / kernel_bpt)))
        # Round down to nearest power-of-two for clean x-axis values.
        max_T = 2 ** int(math.log2(max_T)) if max_T >= 256 else 256
        x_values = [2**i for i in range(7, int(math.log2(max_T)) + 1)]
        extra_configs = [
            {
                "sweep_dim": "T",
                "T": None,  # varied by framework
                "E": E,
                "H": H,
                "intermediate_dim": intermediate_dim,
                "K": K,
                "dtype": dtype,
            }
        ]
        x_name, x_label = "T", "num_tokens"
    else:  # num_experts
        x_values = EXPERT_SWEEP_VALUES
        extra_configs = [
            {
                "sweep_dim": "E",
                "T": probe_T,  # fixed at model's base token count
                "E": None,  # varied by framework
                "H": H,
                "intermediate_dim": intermediate_dim,
                "K": K,
                "dtype": dtype,
            }
        ]
        x_name, x_label = "E", "num_experts"

    common_configs = {
        "kernel_name": "fused_moe",
        "x_name": x_name,
        "x_label": x_label,
        "x_values": x_values,
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": extra_configs,
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_moe,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_moe,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
