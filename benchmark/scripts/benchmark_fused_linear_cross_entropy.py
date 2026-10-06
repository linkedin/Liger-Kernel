"""Benchmark fused linear cross entropy implementations.

Set ``LIGER_FLCE_BENCH_CHUNK_MEM_CONSTS`` to a comma-separated list (for
example, ``1,2,4,8,16``) to compare Triton FLCE chunk-memory budgets. Provider
names encode the selected value so speed and memory rows remain distinguishable
in the shared benchmark CSV. Chunk sweeps use longer warmup and repetition
windows because a single FLCE iteration can exceed the shared benchmark
default's entire measurement budget. Override the repetition window with
``LIGER_FLCE_BENCH_REP_MS`` for especially long-running shapes.
"""

import os

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()

_CHUNK_SWEEP_ENV = "LIGER_FLCE_BENCH_CHUNK_MEM_CONSTS"
_CHUNK_SWEEP_REP_ENV = "LIGER_FLCE_BENCH_REP_MS"
_CHUNK_PROVIDER_PREFIX = "liger-chunk-c"
_CHUNK_SWEEP_WARMUP_MS = 100
_CHUNK_SWEEP_REP_MS = 400


def _parse_chunk_mem_consts(raw_value: str) -> tuple[int, ...]:
    """Parse an ordered, comma-separated list of positive chunk budgets."""
    if not raw_value.strip():
        return ()

    values = []
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"{_CHUNK_SWEEP_ENV} must be a comma-separated list of positive integers")
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{_CHUNK_SWEEP_ENV} contains a non-integer value: {item!r}") from exc
        if value < 1:
            raise ValueError(f"{_CHUNK_SWEEP_ENV} values must be positive. Got: {value}")
        if value not in values:
            values.append(value)
    return tuple(values)


def _chunk_mem_const_from_provider(provider: str) -> int | None:
    """Decode a chunk budget from a benchmark-only provider name."""
    if not provider.startswith(_CHUNK_PROVIDER_PREFIX):
        return None
    return int(provider.removeprefix(_CHUNK_PROVIDER_PREFIX))


def _parse_positive_int(raw_value: str, env_name: str, default: int) -> int:
    """Parse a positive integer environment override."""
    if not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer. Got: {raw_value!r}") from exc
    if value < 1:
        raise ValueError(f"{env_name} must be positive. Got: {value}")
    return value


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        accum_dtype=None,
        chunk_mem_const: int | None = None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ignore_index = ignore_index
        self.accum_dtype = accum_dtype
        self.chunk_mem_const = chunk_mem_const
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean", accum_dtype=accum_dtype
        )

    def forward(self, x, y):
        if self.chunk_mem_const is not None:
            return liger_fused_linear_cross_entropy(
                input=x,
                weight=self.lin.weight,
                target=y,
                ignore_index=self.ignore_index,
                reduction="mean",
                accum_dtype=self.accum_dtype,
                chunk_mem_const=self.chunk_mem_const,
            )
        return self.ce_loss(self.lin.weight, x, y)


def setup_fused_linear_cross_entropy(input: SingleBenchmarkRunInput):
    """Create input tensor, target, and fused linear CE from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        BT = cfg["seq_len"] * cfg["bsz"]
        V = model_cfg.vocab_size
        H = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        BT = input.x
        V = cfg["vocab_size"]
        H = cfg["hidden_size"]
        dtype = cfg["dtype"]

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    chunk_mem_const = _chunk_mem_const_from_provider(input.kernel_provider)
    if chunk_mem_const is not None:
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype, chunk_mem_const=chunk_mem_const).to(device)
    elif input.kernel_provider == "liger":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif input.kernel_provider == "liger-fp32-accum":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype, accum_dtype=torch.float32).to(device)
    else:
        lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    return _input, lambda _: lm_head_ce(_input, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()
    cutile_backend = os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() == "cutile"
    chunk_mem_consts = _parse_chunk_mem_consts(os.environ.get(_CHUNK_SWEEP_ENV, ""))
    if cutile_backend and chunk_mem_consts:
        raise ValueError(
            f"{_CHUNK_SWEEP_ENV} benchmarks Triton chunk budgets and cannot be combined with LIGER_KERNEL_IMPL=cutile"
        )
    chunk_sweep_rep_ms = _parse_positive_int(
        os.environ.get(_CHUNK_SWEEP_REP_ENV, ""),
        _CHUNK_SWEEP_REP_ENV,
        _CHUNK_SWEEP_REP_MS,
    )

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_linear_cross_entropy",
            setup_fn=setup_fused_linear_cross_entropy,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_provider="torch",
            extra_configs={
                "eps": 1e-6,
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="fused_linear_cross_entropy",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_fused_linear_cross_entropy,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    if chunk_mem_consts:
        common_configs["kernel_providers"] = [
            "torch",
            *[f"{_CHUNK_PROVIDER_PREFIX}{chunk_mem_const}" for chunk_mem_const in chunk_mem_consts],
        ]
    else:
        common_configs["kernel_providers"] = (
            ["torch", "liger"] if cutile_backend else ["torch", "liger", "liger-fp32-accum"]
        )

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(
            setup_fused_linear_cross_entropy,
            warmup=_CHUNK_SWEEP_WARMUP_MS if chunk_mem_consts else 25,
            rep=chunk_sweep_rep_ms if chunk_mem_consts else 10,
        ),
        kernel_operation_modes=["forward", "full"] if cutile_backend else ["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_linear_cross_entropy),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
