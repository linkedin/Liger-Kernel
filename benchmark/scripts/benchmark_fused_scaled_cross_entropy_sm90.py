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
    --selected-tokens K            selected vocabulary columns (default: 0)
    --selection-mode MODE          native, identity, or disabled
    --return-entropy               include differentiable vocabulary entropy
    --temperature T                positive scalar temperature (default: 1)
    --label-stride N               consume every Nth selected-output row

Providers: ``torch``, ``liger`` (Triton FLCE, ``reduction="none"``),
``fallback``, ``cutile``, and ``cutedsl-sm90`` (fixed 1024-token wave-batched
backward). ``liger`` does not support selected outputs, entropy or temperature.

Example::

    python benchmark_fused_scaled_cross_entropy_sm90.py \\
        --tokens 4096 --hidden 4096 --vocab 131072 \\
        --providers torch liger cutile cutedsl-sm90

    python benchmark_fused_scaled_cross_entropy_sm90.py \\
        --tokens 4096 16384 --hidden 1024 --vocab 151936 \\
        --providers torch fallback cutile cutedsl-sm90 \\
        --selected-tokens 4 --return-entropy --selection-mode native

Repeat with ``--selection-mode identity`` and ``disabled`` for baselines;
``--label-stride 32`` measures a sparse consumer without overcharging its
identity projection. Sparse label targets come from the selected alphabet, so
their anchor logits reuse that projection. Backend rounding is retained.
"""

import argparse
import math
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


class SelectedLMHeadScaledCE(torch.nn.Module):
    """Native selected output or the extra-projection recovery baseline."""

    def __init__(self, provider, hidden, vocab, dtype, config):
        super().__init__()
        self.provider = provider
        self.temperature = config.get("temperature", 1.0)
        self.return_entropy = config.get("return_entropy", False)
        self.selection_mode = config.get("selection_mode", "native")
        self.label_stride = config.get("label_stride", 1)
        generator = torch.Generator(device=device).manual_seed(GRAD_SEED + 1)
        self.weight = torch.nn.Parameter(
            torch.randn(vocab, hidden, device=device, dtype=dtype, generator=generator) / hidden**0.5
        )
        count = config.get("selected_tokens", 0)
        self.register_buffer(
            "selected_token_ids",
            torch.linspace(0, vocab - 1, count, device=device).long() if count else None,
        )

    def fused(self, x, target, ids=None):
        if self.provider == CUTEDSL_PREFIX:
            return LigerFusedScaledCrossEntropySM90Function.apply(
                x, self.weight, target, self.temperature, -100, 1, self.return_entropy, ids
            )
        if self.provider == CUTILE_PREFIX:
            from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import (
                LigerFusedLinearScaledCrossEntropyFunction,
            )

            return LigerFusedLinearScaledCrossEntropyFunction.apply(
                x, self.weight, target, self.temperature, -100, 1, self.return_entropy, ids
            )
        if self.provider == "fallback":
            from liger_kernel.ops.fused_linear_scaled_cross_entropy import _FusedLinearPPOFallbackFunction

            return _FusedLinearPPOFallbackFunction.apply(
                x, self.weight, target, self.temperature, -100, self.return_entropy, ids
            )
        raise ValueError(f"Provider {self.provider!r} does not support scaled CE with selected outputs")

    def forward(self, x, target):
        ids = self.selected_token_ids if self.selection_mode != "disabled" else None
        if self.provider == "torch":
            logits = ((x @ self.weight.t()) / self.temperature).float()
            log_probs = logits.log_softmax(-1)
            valid = target != -100
            nll = -log_probs.gather(1, target.masked_fill(~valid, 0)[:, None]).squeeze(1)
            outputs = (nll.masked_fill(~valid, 0),)
            if self.return_entropy:
                entropy = -(log_probs.exp() * log_probs).sum(-1).to(x.dtype)
                outputs += (entropy.masked_fill(~valid, 0),)
            if ids is not None:
                selected = log_probs[:: self.label_stride].index_select(1, ids)
                outputs += (selected.masked_fill(~valid[:: self.label_stride, None], 0),)
        else:
            native_ids = ids if self.selection_mode == "native" else None
            result = self.fused(x, target, native_ids)
            outputs = result if isinstance(result, tuple) else (result,)
            if ids is not None:
                if self.selection_mode == "native":
                    outputs = outputs[:-1] + (outputs[-1][:: self.label_stride],)
                else:
                    h = x[:: self.label_stride]
                    labels = target[:: self.label_stride]
                    valid = labels != -100
                    if self.provider == CUTEDSL_PREFIX:
                        selected_logits = (h.float() @ self.weight[ids].float().t()).half().float()
                    else:
                        selected_logits = h @ self.weight[ids].t()
                    if self.label_stride > 1:
                        emitted_class = (labels[:, None] == ids).int().argmax(-1)
                        target_logits = selected_logits.gather(1, emitted_class[:, None]).squeeze(1)
                    else:
                        target_weight = self.weight[labels.masked_fill(~valid, 0)]
                        if self.provider == CUTEDSL_PREFIX:
                            target_logits = (h.float() * target_weight.float()).sum(-1).half().float()
                        else:
                            target_logits = torch.bmm(h[:, None, :], target_weight[:, :, None]).flatten()
                    if self.provider != "fallback":
                        selected_logits, target_logits = selected_logits.float(), target_logits.float()
                    selected_logits = (selected_logits / self.temperature).float()
                    target_logits = (target_logits / self.temperature).float()
                    selected = (
                        -outputs[0][:: self.label_stride, None] + selected_logits - target_logits[:, None]
                    ).masked_fill(~valid[:, None], 0)
                    outputs += (selected,)
        return outputs[0] if len(outputs) == 1 else outputs


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
    configured = (
        cfg.get("selected_tokens", 0)
        or cfg.get("return_entropy", False)
        or cfg.get("temperature", 1.0) != 1.0
        or provider == "fallback"
    )
    if configured:
        generator = torch.Generator(device=device).manual_seed(GRAD_SEED)
        x = torch.randn(total_tokens, hidden_size, requires_grad=True, dtype=dtype, device=device, generator=generator)
        target = torch.randint(vocab_size, (total_tokens,), device=device, generator=generator)
        layer = SelectedLMHeadScaledCE(provider, hidden_size, vocab_size, dtype, cfg)
        stride = cfg.get("label_stride", 1)
        if cfg.get("selected_tokens", 0) and stride > 1:
            rows = len(range(0, total_tokens, stride))
            target[::stride] = layer.selected_token_ids[torch.arange(rows, device=device) % cfg["selected_tokens"]]
    elif provider == CUTEDSL_PREFIX:
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
    grad_output = fixed_grad_output(total_tokens)
    if configured:
        grads = (grad_output,)
        generator = torch.Generator(device=device).manual_seed(GRAD_SEED)
        if cfg.get("return_entropy", False):
            grads += (torch.randn(total_tokens, device=device, dtype=dtype, generator=generator),)
        if cfg.get("selected_tokens", 0) and cfg.get("selection_mode") != "disabled":
            rows = len(range(0, total_tokens, cfg.get("label_stride", 1)))
            grads += (torch.randn(rows, cfg["selected_tokens"], device=device, generator=generator),)
        grad_output = grads[0] if len(grads) == 1 else grads
    return x, weight, lambda: layer(x, target), grad_output


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
            torch.autograd.backward(y, grad_output, retain_graph=True)
    elif mode == "full":

        def bench_fn():
            torch.autograd.backward(fwd_fn(), grad_output)
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
            torch.autograd.backward(y, grad_output, retain_graph=True)
    elif mode == "full":

        def bench_fn():
            x.grad = None
            weight.grad = None
            torch.autograd.backward(fwd_fn(), grad_output)
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
    parser.add_argument("--selected-tokens", type=int, default=0, help="Number of selected vocabulary IDs.")
    parser.add_argument("--selection-mode", choices=["disabled", "native", "identity"], default="native")
    parser.add_argument("--return-entropy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--label-stride", type=int, default=1, help="Consume one selected-output row per N tokens.")
    if any(flag in ("-h", "--help") for flag in sys.argv[1:]):
        parser.print_help()
        print()
    sweep_args, remaining = parser.parse_known_args()
    if sweep_args.selected_tokens < 0 or sweep_args.label_stride < 1:
        parser.error("--selected-tokens must be nonnegative and --label-stride must be positive")
    if not math.isfinite(sweep_args.temperature) or sweep_args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    sys.argv = [sys.argv[0], *remaining]
    return sweep_args


def resolve_providers(sweep_args):
    configured = sweep_args.selected_tokens or sweep_args.return_entropy or sweep_args.temperature != 1.0
    if sweep_args.providers:
        for provider in sweep_args.providers:
            if provider not in ("torch", "liger", "fallback", CUTILE_PREFIX, CUTEDSL_PREFIX):
                raise ValueError(f"Unknown provider {provider!r}")
            if configured and provider == "liger":
                raise ValueError("The reduction-oriented Triton FLCE provider does not support this output contract")
        return list(sweep_args.providers)
    if configured:
        return ["torch", "fallback", CUTILE_PREFIX, CUTEDSL_PREFIX]
    return ["torch", "liger", CUTILE_PREFIX, CUTEDSL_PREFIX]


if __name__ == "__main__":
    sweep_args = parse_sweep_args()
    args = parse_benchmark_script_args()
    extra_configs = {
        "selected_tokens": sweep_args.selected_tokens,
        "selection_mode": sweep_args.selection_mode,
        "return_entropy": sweep_args.return_entropy,
        "temperature": sweep_args.temperature,
        "label_stride": sweep_args.label_stride,
    }
    if not (sweep_args.selected_tokens or sweep_args.return_entropy or sweep_args.temperature != 1.0):
        extra_configs = {}
    kernel_name = "fused_linear_selected_log_probs" if sweep_args.selected_tokens else "fused_scaled_cross_entropy_sm90"

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name=kernel_name,
            setup_fn=setup_fused_scaled_cross_entropy_sm90,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            forward_fn=probe_forward_fn,
            probe_provider="torch",
            extra_configs=extra_configs,
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name=kernel_name,
            probe_x=1024,
            model=model,
            setup_fn=setup_fused_scaled_cross_entropy_sm90,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            forward_fn=probe_forward_fn,
            extra_configs=extra_configs,
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )
        shape = dict(REPRESENTATIVE_CONFIG)
        shape.update(extra_configs)
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
