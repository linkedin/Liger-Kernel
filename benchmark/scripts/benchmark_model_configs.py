"""
Standardized benchmark model configurations.

Provides canonical model architecture profiles and device-specific benchmark
parameters.  All benchmark scripts should derive their tensor shapes from these
shared configs rather than defining ad-hoc per-script constants.

Usage::

    from benchmark_model_configs import (
        MODEL_REGISTRY, DEFAULT_MODEL_CONFIG, get_device_benchmark_config,
    )

    args = parse_benchmark_script_args()
    model = MODEL_REGISTRY[args.model] if args.model else DEFAULT_MODEL_CONFIG
    device_cfg = get_device_benchmark_config(args.device)
"""

from dataclasses import dataclass
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
    rms_norm_eps: float = 1e-5
    dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class DeviceBenchmarkConfig:
    """Device-specific benchmark constraints.

    Attributes:
        device_name: Human-readable device identifier.
        batch_size: Max comfortable batch size for this device.
        seq_len: Max sequence length for benchmark sweeps.  Used as the upper
            bound when generating sweep x_values, or as a fixed sequence length
            parameter when the sweep axis is something else (e.g. vocab_size).
    """

    device_name: str
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
        batch_size=8,
        seq_len=8192,
    ),
    "atlas_900": DeviceBenchmarkConfig(
        device_name="NPU Atlas 900 A2 64GB",
        batch_size=4,
        seq_len=8192,
    ),
}

_DEFAULT_DEVICE_CONFIG = DeviceBenchmarkConfig(
    device_name="default",
    batch_size=8,
    seq_len=8192,
)

# Maps device registry keys → substrings to match in PyTorch's GPU name.
_GPU_NAME_PATTERNS: Dict[str, List[str]] = {
    "h100": ["h100"],
    "atlas_900": ["910b"],
}

DEVICE_REGISTRY = _DEVICE_CONFIGS


def get_device_benchmark_config(
    device_name: Optional[str] = None,
) -> DeviceBenchmarkConfig:
    """Return a device benchmark config by explicit name or auto-detection.

    Args:
        device_name: Registry key (e.g. ``"h100"``).  When *None*, the
            function auto-detects the current GPU by matching its PyTorch
            device name against known patterns.
    """
    if device_name is not None:
        if device_name not in _DEVICE_CONFIGS:
            available = ", ".join(sorted(_DEVICE_CONFIGS.keys()))
            raise ValueError(f"Unknown device '{device_name}'. Available: {available}")
        return _DEVICE_CONFIGS[device_name]

    try:
        gpu_name = get_gpu_name().lower()
    except Exception:
        return _DEFAULT_DEVICE_CONFIG

    for key, patterns in _GPU_NAME_PATTERNS.items():
        if any(p in gpu_name for p in patterns):
            return _DEVICE_CONFIGS[key]

    return _DEFAULT_DEVICE_CONFIG
