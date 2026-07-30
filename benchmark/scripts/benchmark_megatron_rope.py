"""Benchmark Liger's Megatron-LM RoPE adapter.

Compares three providers on Megatron's per-tensor RoPE call shape
``[seq, batch, heads, head_dim]`` (SBHD), sweeping the sequence length:

  - **torch**: an eager-PyTorch re-implementation of Megatron's
    ``_apply_rotary_pos_emb_bshd`` (the raw reference).
  - **megatron**: Megatron-Core's own ``_apply_rotary_pos_emb_bshd`` — the
    symbol Liger's ``apply_rotary_pos_emb`` patch routes around. Structurally
    identical to ``torch``; included for explicit parity confirmation.
  - **liger**: ``liger_apply_rotary_pos_emb_bshd`` — Liger's Triton RoPE in the
    Megatron-shaped wrapper.

Requires a Liger-supported accelerator (CUDA / ROCm). With megatron-core not
installed, the ``megatron`` provider is silently dropped and the run proceeds
with ``liger`` + ``torch``.

Output goes to the shared ``benchmark/data/all_benchmark_data.csv`` — rows are
tagged with ``kernel_name="megatron_rope"`` and the standard visualizer renders
them via:

    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_rope --metric-name speed
    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_rope --metric-name memory
"""

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.megatron import liger_apply_rotary_pos_emb_bshd
from liger_kernel.utils import infer_device

device = infer_device()

try:
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

    _MEGATRON_AVAILABLE = True
except ImportError:
    _apply_rotary_pos_emb_bshd = None
    _MEGATRON_AVAILABLE = False

_B = 2
_HEADS = 32
_HEAD_DIM = 128


def _torch_apply_rotary_pos_emb_bshd(t, freqs):
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    x1, x2 = torch.chunk(t_rot, 2, dim=-1)
    rotated = (t_rot * cos_) + (torch.cat((-x2, x1), dim=-1) * sin_)
    return torch.cat((rotated, t_pass), dim=-1)


def _make_inputs(seq_len: int, requires_grad: bool = True):
    t = torch.randn(seq_len, _B, _HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    half = _HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    theta = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
    freqs = torch.cat((theta, theta), dim=-1).reshape(seq_len, 1, 1, _HEAD_DIM)
    return t, freqs


def _fn(provider):
    if provider == "liger":
        return liger_apply_rotary_pos_emb_bshd
    if provider == "torch":
        return _torch_apply_rotary_pos_emb_bshd
    if provider == "megatron":
        if not _MEGATRON_AVAILABLE:
            raise RuntimeError("megatron-core not installed; cannot benchmark 'megatron' provider")
        return _apply_rotary_pos_emb_bshd
    raise ValueError(f"unknown provider: {provider!r}")


def bench_speed_megatron_rope(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    fn = _fn(provider)
    t, freqs = _make_inputs(seq_len)

    def fwd():
        return fn(t, freqs)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
    elif mode == "backward":

        def _fwd_bwd():
            if t.grad is not None:
                t.grad = None
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


def bench_memory_megatron_rope(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider

    fn = _fn(provider)
    t, freqs = _make_inputs(seq_len)

    def full():
        y = fn(t, freqs)
        y.sum().backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    providers = ["liger", "torch"]
    if _MEGATRON_AVAILABLE:
        providers.append("megatron")

    common_configs = {
        "kernel_name": "megatron_rope",
        "x_name": "S",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, 15)],  # 1024 → 16384
        "kernel_providers": providers,
        "extra_benchmark_configs": [{"B": _B, "H": _HEADS, "head_dim": _HEAD_DIM}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_megatron_rope,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_megatron_rope,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
