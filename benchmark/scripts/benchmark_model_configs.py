"""
Standardized benchmark model configurations.

Provides canonical model architecture profiles and device-specific benchmark
parameters.  All benchmark scripts should derive their tensor shapes from these
shared configs rather than defining ad-hoc per-script constants.

Usage::

    from benchmark_model_configs import (
        get_benchmark_model_config,
        compute_seq_len_sweep_config,
        estimate_kernel_peak_memory,
    )

    args = parse_benchmark_script_args()
    model = get_benchmark_model_config(args.model)

    # Measure actual memory via a small probe, then compute sweep config
    peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)
    bpt = peak_bytes // probe_num_tokens
    config = compute_seq_len_sweep_config(model, kernel_bytes_per_token=bpt)
"""

import gc
import math

from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import Optional

import torch

from liger_kernel.utils import get_total_gpu_memory
from liger_kernel.utils import infer_device


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
class SeqLenSweepConfig:
    """Config for benchmarks that sweep sequence length (e.g. GEGLU, SwiGLU).

    Attributes:
        batch_size: Safe batch size for the sweep.
        seq_len: Max sequence length (upper bound for x_values).
    """

    batch_size: int
    seq_len: int


@dataclass(frozen=True)
class HiddenSizeSweepConfig:
    """Config for benchmarks that sweep hidden_size with fixed BT (e.g. DyT).

    Attributes:
        bt: Fixed batch * seq dimension.
        max_hidden_size: Upper bound for hidden_size sweep.
    """

    bt: int
    max_hidden_size: int


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


def get_benchmark_model_config(model_name: Optional[str] = None) -> ModelConfig:
    """Resolve benchmark model config from name.

    Returns the canonical model architecture profile (hidden_size, vocab_size,
    dtype, etc.) for benchmark runs.  Use this to obtain model attributes
    when building benchmark tensors and shapes.

    Args:
        model_name: Registry key (e.g. ``llama_2_7b``, ``llama_3_8b``).
            If None, returns ``DEFAULT_MODEL_CONFIG``.
    """
    return MODEL_REGISTRY[model_name] if model_name else DEFAULT_MODEL_CONFIG


def estimate_kernel_peak_memory(probe_fn: Callable[[], torch.Tensor]) -> int:
    """Run a forward + backward probe to measure peak memory (bytes).

    Call this with the *pure PyTorch* (e.g. huggingface) implementation --
    that typically has the highest memory footprint and therefore gives a
    safe upper-bound estimate.  Returns the total peak bytes; divide by
    num_tokens if you need bytes-per-token for :func:`compute_seq_len_sweep_config`.

    The probe_fn performs setup and forward pass internally; cleanup is
    automatic, so callers do not need to manage tensor/layer lifecycle.

    Example::

        peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)
        kernel_bpt = peak_bytes // num_tokens  # if needed

    Args:
        probe_fn: Callable that performs setup, runs a forward pass, and
            returns an output tensor suitable for ``.backward()``.
    """
    device_str = infer_device()
    torch_device_mod = getattr(torch, device_str)

    gc.collect()
    torch_device_mod.empty_cache()
    torch_device_mod.memory.reset_peak_memory_stats()

    y = probe_fn()
    y.backward(torch.randn_like(y))

    peak_bytes = torch_device_mod.max_memory_allocated()

    del y
    gc.collect()
    torch_device_mod.empty_cache()

    return max(1, peak_bytes)


def compute_seq_len_sweep_config(
    model_cfg: ModelConfig,
    kernel_bytes_per_token: Optional[int] = None,
    memory_utilization: float = 0.4,
    max_seq_len: Optional[int] = None,
    max_batch_size: int = 32,
) -> SeqLenSweepConfig:
    """Compute safe batch_size and seq_len for sequence-length sweep (e.g. GEGLU).

    Peak memory is estimated as
    ``batch_size * seq_len * kernel_bytes_per_token`` and is capped at
    device memory * memory_utilization.  Device memory is obtained
    internally via :func:`~liger_kernel.utils.get_total_gpu_memory`.

    Prefer obtaining *kernel_bytes_per_token* via
    :func:`estimate_kernel_peak_memory` (divide by num_tokens) rather
    than hardcoding an analytical estimate.

    Args:
        model_cfg: Model architecture config.
        kernel_bytes_per_token: Peak memory **per token** (``batch * seq_len``
            axis).  Best obtained from :func:`estimate_kernel_peak_memory` / num_tokens.
            Falls back to a conservative heuristic
            (``hidden_size * dtype_bytes * 16``) when *None*.
        memory_utilization: Fraction of total device memory to target (0 to 1).
            Lower values are safer.  Default ``0.4`` leaves headroom for
            framework overhead and CUDA/NPU context.
        max_seq_len: Hard upper bound for sequence length.  Defaults to
            ``model_cfg.max_position_embeddings`` so the sweep never exceeds
            the model's native context window.
        max_batch_size: Hard upper bound for batch size.
    """
    total_memory_gb = get_total_gpu_memory()
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

    return SeqLenSweepConfig(batch_size=batch_size, seq_len=seq_len)


def compute_hidden_size_sweep_config(
    model_cfg: ModelConfig,
    kernel_peak_bytes: int,
    bt: int = 4096,
    memory_utilization: float = 0.4,
    max_hidden_size_multiplier: int = 4,
) -> HiddenSizeSweepConfig:
    """Compute safe max_hidden_size for hidden_size sweep (e.g. DyT).

    For kernels with shape (BT, hidden_size) where BT is fixed and we sweep
    hidden_size.  Uses probe peak memory to derive max_hidden_size.
    Device memory is obtained internally via :func:`~liger_kernel.utils.get_total_gpu_memory`.

    Args:
        model_cfg: Model config.
        kernel_peak_bytes: Peak memory from probe (BT, model.hidden_size).
        bt: Fixed BT dimension; must match the probe.
        memory_utilization: Fraction of device memory to use.
        max_hidden_size_multiplier: Cap max_hidden_size at model.hidden_size * this.
    """
    total_memory_gb = get_total_gpu_memory()
    usable_bytes = total_memory_gb * (1024**3) * memory_utilization
    kernel_bpt = max(1, kernel_peak_bytes // bt)
    max_hidden_size = min(
        model_cfg.hidden_size * max_hidden_size_multiplier,
        max(
            model_cfg.hidden_size,
            int(usable_bytes * model_cfg.hidden_size / (bt * kernel_bpt)),
        ),
    )
    max_hidden_size = max(1024, 2 ** int(math.log2(max_hidden_size)))
    return HiddenSizeSweepConfig(bt=bt, max_hidden_size=max_hidden_size)
