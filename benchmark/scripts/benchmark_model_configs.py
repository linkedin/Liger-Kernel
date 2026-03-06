"""
Standardized benchmark model configurations.

Provides canonical model architecture profiles and device-specific benchmark
parameters.  All benchmark scripts should derive their tensor shapes from these
shared configs rather than defining ad-hoc per-script constants.

Usage::

    from benchmark_model_configs import (
        MODEL_REGISTRY, DEFAULT_MODEL_CONFIG,
        get_device_benchmark_config, compute_benchmark_shape,
        estimate_kernel_bytes_per_token,
    )

    args = parse_benchmark_script_args()
    model = MODEL_REGISTRY[args.model] if args.model else DEFAULT_MODEL_CONFIG
    device_cfg = get_device_benchmark_config(args.device)

    # Measure actual memory via a small probe, then compute safe shapes
    bpt = estimate_kernel_bytes_per_token(kernel_fn=..., num_tokens=1024)
    shape = compute_benchmark_shape(device_cfg, model, kernel_bytes_per_token=bpt)
"""

import math

from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch

from utils import get_gpu_name


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
class DeviceBenchmarkConfig:
    """Device-specific benchmark constraints.

    Attributes:
        device_name: Human-readable device identifier.
        total_memory_gb: Total device memory in gigabytes.  Used together with
            model and kernel information to compute safe ``batch_size`` /
            ``seq_len`` via :func:`compute_benchmark_shape`.
    """

    device_name: str
    total_memory_gb: float


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


# ── Device Configurations ───────────────────────────────────────────────────

_DEVICE_CONFIGS: Dict[str, DeviceBenchmarkConfig] = {
    "h100": DeviceBenchmarkConfig(
        device_name="NVIDIA H100 80GB",
        total_memory_gb=80.0,
    ),
    "atlas_900": DeviceBenchmarkConfig(
        device_name="NPU Atlas 900 A2 64GB",
        total_memory_gb=64.0,
    ),
}

_DEFAULT_MEMORY_GB = 40.0

# Maps device registry keys → substrings to match in PyTorch's GPU name.
_GPU_NAME_PATTERNS: Dict[str, List[str]] = {
    "h100": ["h100"],
    "atlas_900": ["910b"],
}

DEVICE_REGISTRY = _DEVICE_CONFIGS


def _detect_device_memory_gb() -> float:
    """Best-effort auto-detection of device memory in GB."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_mem / (1024**3)
    except Exception:
        pass
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            props = torch.npu.get_device_properties(0)
            return props.total_mem / (1024**3)
    except Exception:
        pass
    return _DEFAULT_MEMORY_GB


def get_device_benchmark_config(
    device_name: Optional[str] = None,
) -> DeviceBenchmarkConfig:
    """Return a device benchmark config by explicit name or auto-detection.

    Args:
        device_name: Registry key (e.g. ``"h100"``).  When *None*, the
            function auto-detects the current GPU by matching its PyTorch
            device name against known patterns.  For unrecognised GPUs the
            device memory is queried at runtime so that
            :func:`compute_benchmark_shape` can still derive safe parameters.
    """
    if device_name is not None:
        if device_name not in _DEVICE_CONFIGS:
            available = ", ".join(sorted(_DEVICE_CONFIGS.keys()))
            raise ValueError(f"Unknown device '{device_name}'. Available: {available}")
        return _DEVICE_CONFIGS[device_name]

    try:
        gpu_name = get_gpu_name().lower()
    except Exception:
        return DeviceBenchmarkConfig(
            device_name="default",
            total_memory_gb=_detect_device_memory_gb(),
        )

    for key, patterns in _GPU_NAME_PATTERNS.items():
        if any(p in gpu_name for p in patterns):
            return _DEVICE_CONFIGS[key]

    return DeviceBenchmarkConfig(
        device_name=gpu_name,
        total_memory_gb=_detect_device_memory_gb(),
    )


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
    device_cfg: DeviceBenchmarkConfig,
    model_cfg: ModelConfig,
    kernel_bytes_per_token: Optional[int] = None,
    memory_utilization: float = 0.4,
    max_seq_len: Optional[int] = None,
    max_batch_size: int = 32,
) -> BenchmarkShapeConfig:
    """Compute safe ``batch_size`` and ``seq_len`` for a benchmark run.

    Peak memory is estimated as
    ``batch_size * seq_len * kernel_bytes_per_token`` and is capped at
    ``device_cfg.total_memory_gb * memory_utilization``.

    Prefer obtaining *kernel_bytes_per_token* via
    :func:`estimate_kernel_bytes_per_token` (a small runtime probe) rather
    than hardcoding an analytical estimate.

    Args:
        device_cfg: Device config with memory capacity.
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

    usable_bytes = device_cfg.total_memory_gb * (1024**3) * memory_utilization
    max_tokens = max(1, int(usable_bytes / kernel_bytes_per_token))

    seq_len = min(max_seq_len, max_tokens)
    seq_len = 2 ** int(math.log2(seq_len)) if seq_len >= 1024 else 1024

    batch_size = max(1, min(max_tokens // seq_len, max_batch_size))

    return BenchmarkShapeConfig(batch_size=batch_size, seq_len=seq_len)
