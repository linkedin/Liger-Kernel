"""
Standardized benchmark model configurations.

Provides canonical model architecture profiles and device-specific benchmark
parameters.  All benchmark scripts should derive their tensor shapes from these
shared configs rather than defining ad-hoc per-script constants.

Usage::

    from benchmark_model_configs import (
        MODEL_REGISTRY, DEFAULT_MODEL_CONFIG,
        compute_benchmark_shape,
        estimate_kernel_bytes_per_token,
    )

    args = parse_benchmark_script_args()
    model = MODEL_REGISTRY[args.model] if args.model else DEFAULT_MODEL_CONFIG
    total_memory_gb = get_total_gpu_memory()

    # Measure actual memory via a small probe, then compute safe shapes
    bpt = estimate_kernel_bytes_per_token(kernel_fn=..., num_tokens=1024)
    shape = compute_benchmark_shape(total_memory_gb, model, kernel_bytes_per_token=bpt)
"""

import math

from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import Optional

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Canonical model architecture profile.

    Each field corresponds to a standard LLM hyperparameter.  Benchmark scripts
    pick the fields they need (e.g. hidden_size for RMSNorm, vocab_size for
    CrossEntropy) while kernel-specific overrides (e.g. hidden_act for GEGLU)
    are applied locally in the benchmark script.
    """

    name: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class BenchmarkShapeConfig:
    """Computed benchmark shape parameters.

    Returned by :func:`compute_benchmark_shape` — these values are derived
    from the combination of device memory, model dimensions, and kernel-specific
    memory characteristics rather than being hardcoded per device.

    Attributes:
        batch_size: Safe batch size for the given configuration.
        seq_len: Max sequence length for benchmark sweeps.
    """

    batch_size: int
    seq_len: int


# ── Model Profiles ──────────────────────────────────────────────────────────

LLAMA_2_7B = ModelConfig(
    name="llama_2_7b",
    hidden_size=4096,
    intermediate_size=11008,
    vocab_size=32000,
    num_attention_heads=32,
    num_key_value_heads=32,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=4096,
)

LLAMA_3_8B = ModelConfig(
    name="llama_3_8b",
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=128256,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=8192,
)

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "llama_2_7b": LLAMA_2_7B,
    "llama_3_8b": LLAMA_3_8B,
}

DEFAULT_MODEL_CONFIG = LLAMA_3_8B

def estimate_kernel_bytes_per_token(
    kernel_fn: Callable[[], torch.Tensor],
    num_tokens: int,
) -> int:
    """Run a forward + backward probe to measure peak memory per token.

    Call this with the *pure PyTorch* (e.g. huggingface) implementation --
    that typically has the highest memory footprint and therefore gives a
    safe upper-bound estimate.  The returned value is suitable as the
    ``kernel_bytes_per_token`` argument to :func:`compute_benchmark_shape`.

    Example usage with an existing benchmark setup function::

        probe_input = SingleBenchmarkRunInput(
            x=1024, kernel_provider="huggingface",
            extra_benchmark_config={"bsz": 1, ...},
        )
        probe_x, probe_layer = _setup_my_kernel(probe_input)
        bpt = estimate_kernel_bytes_per_token(
            kernel_fn=lambda: probe_layer(probe_x), num_tokens=1024,
        )

    Args:
        kernel_fn: Callable that runs a forward pass and returns an output
            tensor suitable for ``.backward()``.
        num_tokens: Total number of tokens in the probe input
            (``batch_size * seq_len``).
    """
    import gc

    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device_str = "npu"
    else:
        raise RuntimeError(
            "No CUDA or NPU device available for memory measurement"
        )

    torch_device_mod = getattr(torch, device_str)

    gc.collect()
    torch_device_mod.empty_cache()
    torch_device_mod.memory.reset_peak_memory_stats()

    y = kernel_fn()
    y.backward(torch.randn_like(y))

    peak_bytes = torch_device_mod.max_memory_allocated()

    del y
    gc.collect()
    torch_device_mod.empty_cache()

    return max(1, peak_bytes // num_tokens)


def compute_benchmark_shape(
    total_memory_gb: float,
    model_cfg: ModelConfig,
    kernel_bytes_per_token: Optional[int] = None,
    memory_utilization: float = 0.4,
    max_seq_len: Optional[int] = None,
    max_batch_size: int = 32,
) -> BenchmarkShapeConfig:
    """Compute safe ``batch_size`` and ``seq_len`` for a benchmark run.

    Peak memory is estimated as
    ``batch_size * seq_len * kernel_bytes_per_token`` and is capped at
    ``total_memory_gb * memory_utilization``.

    Prefer obtaining *kernel_bytes_per_token* via
    :func:`estimate_kernel_bytes_per_token` (a small runtime probe) rather
    than hardcoding an analytical estimate.

    Args:
        total_memory_gb: Total device memory in gigabytes, typically from
            :func:`~liger_kernel.utils.get_total_gpu_memory`.
        model_cfg: Model architecture config.
        kernel_bytes_per_token: Peak memory **per token** (``batch * seq_len``
            axis).  Best obtained from :func:`estimate_kernel_bytes_per_token`.
            Falls back to a conservative heuristic
            (``hidden_size * dtype_bytes * 16``) when *None*.
        memory_utilization: Fraction of total device memory to target (0\u20131).
            Lower values are safer.  Default ``0.4`` leaves headroom for
            framework overhead and CUDA/NPU context.
        max_seq_len: Hard upper bound for sequence length.  Defaults to
            ``model_cfg.max_position_embeddings`` so the sweep never exceeds
            the model's native context window.
        max_batch_size: Hard upper bound for batch size.
    """
    dtype_bytes = 2 if model_cfg.dtype in (torch.bfloat16, torch.float16) else 4

    if kernel_bytes_per_token is None:
        kernel_bytes_per_token = model_cfg.hidden_size * dtype_bytes * 16

    if max_seq_len is None:
        max_seq_len = model_cfg.max_position_embeddings

    usable_bytes = total_memory_gb * (1024**3) * memory_utilization
    max_tokens = max(1, int(usable_bytes / kernel_bytes_per_token))

    seq_len = min(max_seq_len, max_tokens)
    seq_len = 2 ** int(math.log2(seq_len)) if seq_len >= 1024 else 1024

    batch_size = max(1, min(max_tokens // seq_len, max_batch_size))

    return BenchmarkShapeConfig(batch_size=batch_size, seq_len=seq_len)
