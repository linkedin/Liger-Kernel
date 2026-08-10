"""Benchmark the Hopper (SM90) CuTe DSL fused *scaled* cross entropy.

The measured operation returns the **per-token NLL** ``[M]`` (never a mean or a
sum), so the backward is driven by a *fixed* per-token gradient vector ``[M]``
-- the same vector for every provider, every repetition and every operation
mode -- instead of the shared harness' fresh ``randn_like`` draw.  That keeps
the three providers numerically comparable and keeps the gradient allocation
out of the timed region.

Sweep parameters (all optional; defaults reproduce the previous behaviour)::

    --tokens M [M ...]              total tokens (the x axis)
    --hidden H                      hidden size
    --vocab V                       vocabulary size
    --providers NAME [NAME ...]     explicit provider list

Providers: ``torch``, ``liger`` (Triton FLCE, ``reduction="none"``),
``cutile``, and ``cutedsl-sm90`` (fixed 1024-token wave-batched backward).

Example::

    python benchmark_fused_scaled_cross_entropy_sm90.py \\
        --tokens 4096 --hidden 4096 --vocab 131072 \\
        --providers torch liger cutile cutedsl-sm90
"""

import argparse
import sys

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import LigerFusedScaledCrossEntropySM90Function
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()

REPRESENTATIVE_CONFIG = {
    "hidden_size": 4096,
    "vocab_size": 131072,
    "dtype": torch.bfloat16,
}

CUTEDSL_PREFIX = "cutedsl-sm90"
CUTILE_PREFIX = "cutile"
# Seed of the fixed per-token upstream gradient, so every provider and every
# repetition sees byte-identical ``d(NLL)/d(loss)`` values.
GRAD_SEED = 1234


class TorchLMHeadCE(torch.nn.Module):
    """Per-token (``reduction="none"``) torch baseline."""

    def __init__(self, hidden_size: int, vocab_size: int, dtype: torch.dtype):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

    def forward(self, x, target):
        logits = self.linear(x).float()
        return torch.nn.functional.cross_entropy(logits, target, reduction="none")


class TritonLMHeadCE(torch.nn.Module):
    """Triton FLCE with ``reduction="none"`` so the semantics match."""

    def __init__(self, hidden_size: int, vocab_size: int, dtype: torch.dtype):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
        self.cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="none", accum_dtype=torch.float32)

    def forward(self, x, target):
        return self.cross_entropy(self.linear.weight, x, target)


class CuteDSLHopperLMHeadScaledCE(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        temperature: float = 1.0,
        m_tiles_per_cluster: int = 1,
    ):
        super().__init__()
        if dtype != torch.bfloat16:
            raise TypeError(f"SM90 fused scaled cross entropy requires bfloat16, got {dtype}")
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype))
        self.temperature = temperature
        self.m_tiles_per_cluster = m_tiles_per_cluster
        torch.nn.init.normal_(self.weight, std=hidden_size**-0.5)

    def forward(self, x, target):
        return LigerFusedScaledCrossEntropySM90Function.apply(
            x,
            self.weight,
            target,
            self.temperature,
            -100,
            self.m_tiles_per_cluster,
        )


class CuTileLMHeadScaledCE(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, dtype: torch.dtype, temperature: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype))
        self.temperature = temperature
        torch.nn.init.normal_(self.weight, std=hidden_size**-0.5)

    def forward(self, x, target):
        from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import (
            LigerFusedLinearScaledCrossEntropyFunction,
        )

        return LigerFusedLinearScaledCrossEntropyFunction.apply(
            x,
            self.weight,
            target,
            self.temperature,
            -100,
            1,
            False,
        )


def fixed_grad_output(tokens: int) -> torch.Tensor:
    """The fixed ``[M]`` upstream gradient of the per-token NLL."""
    generator = torch.Generator(device=device).manual_seed(GRAD_SEED)
    return torch.rand(tokens, generator=generator, device=device, dtype=torch.float32) + 0.25


def setup_fused_scaled_cross_entropy_sm90(input: SingleBenchmarkRunInput):
    """Return ``(x, forward_fn, grad_output)`` for one benchmark point."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        total_tokens = cfg["seq_len"] * cfg["bsz"]
        hidden_size = model_cfg.hidden_size
        vocab_size = model_cfg.vocab_size
        dtype = model_cfg.dtype
    else:
        total_tokens = input.x
        hidden_size = cfg["hidden_size"]
        vocab_size = cfg["vocab_size"]
        dtype = cfg["dtype"]

    x = torch.randn(total_tokens, hidden_size, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(vocab_size, (total_tokens,), dtype=torch.long, device=device)

    provider = input.kernel_provider
    if provider == CUTEDSL_PREFIX:
        layer = CuteDSLHopperLMHeadScaledCE(hidden_size, vocab_size, dtype)
    elif provider == CUTILE_PREFIX:
        layer = CuTileLMHeadScaledCE(hidden_size, vocab_size, dtype)
    elif provider == "liger":
        layer = TritonLMHeadCE(hidden_size, vocab_size, dtype)
    elif provider == "torch":
        layer = TorchLMHeadCE(hidden_size, vocab_size, dtype)
    else:
        raise ValueError(f"Unknown provider {provider!r}")

    layer = layer.to(device)
    weight = layer.weight if hasattr(layer, "weight") else layer.linear.weight
    return x, weight, lambda: layer(x, target), fixed_grad_output(total_tokens)


def probe_forward_fn(x, weight, fwd_fn, grad_output):
    """Adapter for the shared memory-probing helpers (setup returns a 4-tuple)."""
    return fwd_fn()


def bench_speed_fused_scaled_cross_entropy_sm90(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    import triton

    x, weight, fwd_fn, grad_output = setup_fused_scaled_cross_entropy_sm90(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        bench_fn = fwd_fn
    elif mode == "backward":
        y = fwd_fn()

        def bench_fn():
            y.backward(grad_output, retain_graph=True)
    elif mode == "full":

        def bench_fn():
            fwd_fn().backward(grad_output)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    ms_50, ms_20, ms_80 = triton.testing.do_bench(
        bench_fn,
        grad_to_none=[x, weight],
        rep=10,
        quantiles=QUANTILES,
    )
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_fused_scaled_cross_entropy_sm90(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, weight, fwd_fn, grad_output = setup_fused_scaled_cross_entropy_sm90(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        bench_fn = fwd_fn
    elif mode == "backward":
        y = fwd_fn()

        def bench_fn():
            x.grad = None
            weight.grad = None
            y.backward(grad_output, retain_graph=True)
    elif mode == "full":

        def bench_fn():
            x.grad = None
            weight.grad = None
            fwd_fn().backward(grad_output)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    mem_50, mem_20, mem_80 = _test_memory(bench_fn, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


def parse_sweep_args():
    """Script-specific flags, stripped from ``argv`` before the shared parser."""
    parser = argparse.ArgumentParser(add_help=False, description="fused scaled cross entropy sweep options")
    parser.add_argument("--tokens", type=int, nargs="+", default=None, help="Total tokens (M) to sweep.")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden size (H).")
    parser.add_argument("--vocab", type=int, default=None, help="Vocabulary size (V).")
    parser.add_argument("--providers", type=str, nargs="+", default=None, help="Explicit provider list.")
    if any(flag in ("-h", "--help") for flag in sys.argv[1:]):
        parser.print_help()
        print()
    sweep_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return sweep_args


def resolve_providers(sweep_args):
    if sweep_args.providers:
        for provider in sweep_args.providers:
            if provider not in ("torch", "liger", CUTILE_PREFIX, CUTEDSL_PREFIX):
                raise ValueError(f"Unknown provider {provider!r}")
        return list(sweep_args.providers)
    return ["torch", "liger", CUTILE_PREFIX, CUTEDSL_PREFIX]


if __name__ == "__main__":
    sweep_args = parse_sweep_args()
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_scaled_cross_entropy_sm90",
            setup_fn=setup_fused_scaled_cross_entropy_sm90,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            forward_fn=probe_forward_fn,
            probe_provider="torch",
            extra_configs={},
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="fused_scaled_cross_entropy_sm90",
            probe_x=1024,
            model=model,
            setup_fn=setup_fused_scaled_cross_entropy_sm90,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            forward_fn=probe_forward_fn,
            extra_configs={},
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )
        shape = dict(REPRESENTATIVE_CONFIG)
        if sweep_args.hidden is not None:
            shape["hidden_size"] = sweep_args.hidden
        if sweep_args.vocab is not None:
            shape["vocab_size"] = sweep_args.vocab
        if sweep_args.tokens:
            # An explicit M sweep replaces the model-derived token lengths.
            common_configs["x_values"] = sorted(set(sweep_args.tokens))
            common_configs["extra_benchmark_configs"] = [shape]
        else:
            common_configs["x_values"] = sorted(set([*common_configs["x_values"], 4096]))
            common_configs["extra_benchmark_configs"].append(shape)

    common_configs["kernel_providers"] = resolve_providers(sweep_args)

    run_benchmarks(
        bench_test_fn=bench_speed_fused_scaled_cross_entropy_sm90,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_scaled_cross_entropy_sm90,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
