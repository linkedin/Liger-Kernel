"""Benchmark Liger's Megatron-LM RMSNorm wrapper.

Compares three providers on the per-token RMSNorm call shape ``[seq, batch, hidden]``:

  - **torch**: vanilla ``torch.nn.RMSNorm`` — the raw PyTorch reference
  - **megatron**: Megatron's ``WrappedTorchNorm`` (the symbol Liger displaces in the
    local-backend path; structurally a factory that returns ``nn.RMSNorm``, so its
    timing should be indistinguishable from ``torch`` — included for explicit parity
    confirmation, since the *point* of the patch is replacing this specific symbol)
  - **liger**: ``LigerMegatronRMSNorm`` — Liger's Triton RMSNorm in the Megatron-shaped
    wrapper (per-layer + final_layernorm slot).

Requires a Liger-supported accelerator (CUDA / ROCm). With megatron-core not
installed, the ``megatron`` provider is silently dropped and the run proceeds with
``liger`` + ``torch``.

Output goes to the shared ``benchmark/data/all_benchmark_data.csv`` — rows are
tagged with ``kernel_name="megatron_rms_norm"`` and the standard visualizer renders
them via:

    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_rms_norm --metric-name speed
    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_rms_norm --metric-name memory
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.megatron import LigerMegatronRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()

try:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    _MEGATRON_AVAILABLE = True
except ImportError:
    WrappedTorchNorm = None
    _MEGATRON_AVAILABLE = False


def _make_config():
    """Duck-typed TransformerConfig accepted by both LigerMegatronRMSNorm and WrappedTorchNorm.

    WrappedTorchNorm asserts a handful of attributes are False; LigerMegatronRMSNorm reads
    ``normalization``, ``sequence_parallel``, and ``layernorm_zero_centered_gamma``.
    """
    return SimpleNamespace(
        normalization="RMSNorm",
        sequence_parallel=False,
        layernorm_zero_centered_gamma=False,
        persist_layer_norm=False,
        memory_efficient_layer_norm=False,
    )


def _make_layer(provider: str, hidden_size: int, eps: float = 1e-6) -> nn.Module:
    config = _make_config()
    if provider == "liger":
        layer = LigerMegatronRMSNorm(config=config, hidden_size=hidden_size, eps=eps)
    elif provider == "torch":
        layer = nn.RMSNorm(normalized_shape=hidden_size, eps=eps)
    elif provider == "megatron":
        if not _MEGATRON_AVAILABLE:
            raise RuntimeError("megatron-core not installed; cannot benchmark 'megatron' provider")
        # WrappedTorchNorm.__new__ returns an nn.RMSNorm instance directly.
        layer = WrappedTorchNorm(config=config, hidden_size=hidden_size, eps=eps)
    else:
        raise ValueError(f"unknown provider: {provider!r}")
    return layer.to(device).to(torch.bfloat16)


def _make_input(s: int, b: int, h: int, requires_grad: bool = True) -> torch.Tensor:
    return torch.randn(s, b, h, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)


def bench_speed_megatron_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    h = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    s = input.extra_benchmark_config["S"]
    b = input.extra_benchmark_config["B"]

    layer = _make_layer(provider, h)
    x = _make_input(s, b, h)

    def fwd():
        return layer(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        # Rerun fwd inside the timed loop so each backward sees a fresh graph (mirrors the
        # megatron CE benchmark's "backward includes forward" convention — subtract the
        # "forward" measurement to derive backward-only timing).
        def _fwd_bwd():
            if x.grad is not None:
                x.grad = None
            out = fwd()
            out.sum().backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(_fwd_bwd, rep=100, quantiles=QUANTILES)
    elif mode == "full":

        def full():
            y = fwd()
            y.sum().backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_megatron_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    h = input.x
    provider = input.kernel_provider
    s = input.extra_benchmark_config["S"]
    b = input.extra_benchmark_config["B"]

    layer = _make_layer(provider, h)
    x = _make_input(s, b, h)

    def full():
        y = layer(x)
        y.sum().backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    providers = ["liger", "torch"]
    if _MEGATRON_AVAILABLE:
        providers.append("megatron")

    common_configs = {
        "kernel_name": "megatron_rms_norm",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 15)],  # 1024 → 16384
        "kernel_providers": providers,
        "extra_benchmark_configs": [{"S": 4096, "B": 1}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_megatron_rms_norm,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_megatron_rms_norm,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
