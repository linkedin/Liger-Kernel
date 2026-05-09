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
    config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_seq_len)
"""

import gc
import math

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import torch

from utils import SingleBenchmarkRunInput
from utils import default_forward_fn

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

    # ===== MoE-specific (optional) =====
    num_experts: Optional[int] = None
    topk: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None


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
class ModelConfigSweepConfig:
    """Config for benchmarks that sweep across model configs.

    Attributes:
        model_configs: Model configs to benchmark (as tuple for immutability).
        bt: Effective total tokens (batch_size * seq_len).
        batch_size: Safe batch size across all model configs.
        seq_len: Safe sequence length across all model configs.
    """

    model_configs: Tuple[ModelConfig, ...]
    bt: int
    batch_size: int
    seq_len: int


@dataclass(frozen=True)
class MoEModelConfig:
    """MoE model architecture profile for fused MoE benchmarks.

    EP-adjusted values should be baked in: T = total_tokens / ep_size,
    E = total_experts / ep_size.
    """

    name: str
    T: int  # tokens per GPU (EP-adjusted)
    E: int  # experts per GPU (EP-adjusted)
    H: int  # hidden size
    intermediate_dim: int  # expert intermediate size
    K: int  # top-k


# ── MoE Model Profiles ───────────────────────────────────────────────────────

QWEN3_MOE_30B = MoEModelConfig(
    name="qwen3_moe_30b",
    T=8192,
    E=128,
    H=2048,
    intermediate_dim=768,
    K=8,
)


# ── Dense Model Profiles ─────────────────────────────────────────────────────

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

QWEN_2_5_7B = ModelConfig(
    name="qwen2.5_7b",
    hidden_size=3584,
    intermediate_size=18944,
    vocab_size=152064,
    num_attention_heads=28,
    num_key_value_heads=4,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
)

QWEN_2_5_14B = ModelConfig(
    name="qwen2.5_14b",
    hidden_size=5120,
    intermediate_size=13824,
    vocab_size=152064,
    num_attention_heads=40,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
)

QWEN_2_5_72B = ModelConfig(
    name="qwen2.5_72b",
    hidden_size=8192,
    intermediate_size=29568,
    vocab_size=152064,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
)

DEEPSEEK_V2_LITE = ModelConfig(
    name="deepseek_v2_lite",
    hidden_size=2048,
    intermediate_size=10944,
    vocab_size=102400,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=163840,
    moe_intermediate_size=1408,
    num_experts=64,
    topk=6,
)

DEEPSEEK_V3 = ModelConfig(
    name="deepseek_v3",
    hidden_size=7168,
    intermediate_size=18432,
    vocab_size=129280,
    num_attention_heads=128,
    num_key_value_heads=128,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=163840,
    moe_intermediate_size=2048,
    num_experts=256,
    topk=8,
)

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "llama_2_7b": LLAMA_2_7B,
    "llama_3_8b": LLAMA_3_8B,
    "qwen2.5_7b": QWEN_2_5_7B,
    "qwen2.5_14b": QWEN_2_5_14B,
    "qwen2.5_72b": QWEN_2_5_72B,
    "deepseek_v2_lite": DEEPSEEK_V2_LITE,
    "deepseek_v3": DEEPSEEK_V3,
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
    num_tokens if you need bytes-per-token. :func:`compute_seq_len_sweep_config`
    accepts a ``probe_fn`` directly and will run this for you.

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


def _max_seqlen_under_memory(
    *,
    usable_bytes: float,
    peak_bytes: int,
    probe_batch_size: int,
    probe_seq_len: int,
    scaling_method: Literal["linear", "quadratic"],
    target_batch_size: int,
) -> int:
    """Invert peak_bytes(B, L) for L at fixed target_batch_size.

    - ``"linear"``: assumes peak ~ B * L. ``L_max = usable / (B * c_per_BL)``.
    - ``"quadratic"``: assumes peak ~ B * L^2. ``L_max = sqrt(usable / (B * c_per_BL2))``.
    """
    if scaling_method == "linear":
        c = max(1.0, peak_bytes / (probe_batch_size * probe_seq_len))
        return max(1, int(usable_bytes / (target_batch_size * c)))
    if scaling_method == "quadratic":
        c = max(1.0, peak_bytes / (probe_batch_size * probe_seq_len * probe_seq_len))
        return max(1, int(math.sqrt(usable_bytes / (target_batch_size * c))))
    raise ValueError(f"scaling_method must be 'linear' or 'quadratic', got {scaling_method!r}")


def _snap_pow2_seqlen(seq_len: int, max_seq_len: int) -> int:
    """Clamp to *max_seq_len* and snap down to nearest power of 2 (floor at 1024)."""
    seq_len = min(max_seq_len, seq_len)
    return 2 ** int(math.log2(seq_len)) if seq_len >= 1024 else 1024


def compute_seq_len_sweep_config(
    model_cfg: ModelConfig,
    probe_fn: Callable[[], torch.Tensor],
    probe_seq_len: int,
    probe_batch_size: int = 1,
    scaling_method: Literal["linear", "quadratic"] = "linear",
    memory_utilization: float = 0.4,
    max_seq_len: Optional[int] = None,
    max_batch_size: int = 32,
) -> SeqLenSweepConfig:
    """Compute safe batch_size and seq_len for a sequence-length sweep.

    Runs *probe_fn* once via :func:`estimate_kernel_peak_memory` to measure the
    actual peak memory, then inverts the memory model according to
    *scaling_method* to find the largest seq_len that fits in
    ``device_memory * memory_utilization``. Device memory is obtained internally
    via :func:`~liger_kernel.utils.get_total_gpu_memory`.

    For kernels whose memory grows non-linearly with seq_len (e.g. attention
    kernels with O(L^2) scratch), pass ``scaling_method="quadratic"``.

    Args:
        model_cfg: Model architecture config.
        probe_fn: Callable that performs setup, runs a forward pass, and
            returns an output tensor suitable for ``.backward()``. Same
            contract as :func:`estimate_kernel_peak_memory`'s *probe_fn*.
        probe_seq_len: Sequence length used inside *probe_fn*. Required so the
            inversion can isolate the seq-len-dependent term.
        probe_batch_size: Batch size used inside *probe_fn*. Defaults to 1.
        scaling_method: How peak memory scales with seq_len, holding batch
            size fixed. See :func:`_max_seqlen_under_memory`.
        memory_utilization: Fraction of total device memory to target (0 to 1).
            Lower values are safer.  Default ``0.4`` leaves headroom for
            framework overhead and CUDA/NPU context.
        max_seq_len: Hard upper bound for sequence length.  Defaults to
            ``model_cfg.max_position_embeddings`` so the sweep never exceeds
            the model's native context window.
        max_batch_size: Hard upper bound for batch size.
    """
    peak_bytes = estimate_kernel_peak_memory(probe_fn=probe_fn)

    usable_bytes = get_total_gpu_memory() * (1024**3) * memory_utilization
    if max_seq_len is None:
        max_seq_len = model_cfg.max_position_embeddings

    batch_size = max(1, min(max_batch_size, probe_batch_size))
    max_seq_len_from_mem = _max_seqlen_under_memory(
        usable_bytes=usable_bytes,
        peak_bytes=peak_bytes,
        probe_batch_size=probe_batch_size,
        probe_seq_len=probe_seq_len,
        scaling_method=scaling_method,
        target_batch_size=batch_size,
    )
    seq_len = _snap_pow2_seqlen(max_seq_len_from_mem, max_seq_len)

    return SeqLenSweepConfig(batch_size=batch_size, seq_len=seq_len)


def compute_model_config_sweep_config(
    model_configs: List[ModelConfig],
    probe_fn_factory: Callable[[ModelConfig, int], Callable[[], torch.Tensor]],
    bt: int = 1024,
    memory_utilization: float = 0.4,
) -> ModelConfigSweepConfig:
    """Find safe (batch_size, seq_len) that works across all model configs.

    Probes each model config at a small token count to measure peak memory,
    then picks the most conservative parameters that fit within device memory.

    Args:
        model_configs: Model configs to benchmark.
        probe_fn_factory: Factory ``(model_cfg) -> probe_fn``.
            The returned probe_fn should perform setup + forward pass and
            return a tensor suitable for ``.backward()``, same contract as
            :func:`estimate_kernel_peak_memory`'s *probe_fn*.
        bt: Target total tokens (batch_size * seq_len).
        memory_utilization: Fraction of device memory to use.
    """
    total_memory_gb = get_total_gpu_memory()
    usable_bytes = total_memory_gb * (1024**3) * memory_utilization

    probe_seq_len = min(bt, 1024)
    max_bytes_per_token = 0

    for model_cfg in model_configs:
        probe_fn = probe_fn_factory(model_cfg)
        peak_bytes = estimate_kernel_peak_memory(probe_fn)
        bpt = max(1, peak_bytes // probe_seq_len)
        max_bytes_per_token = max(max_bytes_per_token, bpt)

    max_tokens = max(1, int(usable_bytes / max_bytes_per_token))
    safe_bt = min(bt, max_tokens)

    seq_len = min(safe_bt, 8192)
    seq_len = 2 ** int(math.log2(seq_len)) if seq_len >= 1024 else 1024
    batch_size = max(1, safe_bt // seq_len)

    return ModelConfigSweepConfig(
        model_configs=tuple(model_configs),
        bt=batch_size * seq_len,
        batch_size=batch_size,
        seq_len=seq_len,
    )


def build_extra_config(
    model: ModelConfig,
    model_keys: List[str],
    extra_configs: Optional[Dict] = None,
) -> Dict:
    """Construct extra_benchmark_config dict.

    Args:
        model: The model configuration object.
        model_keys: List of attribute names to read from `model`
            (e.g. ["hidden_size", "dtype"]).
        extra_configs: Optional dictionary of additional key/value pairs
            that override or extend the extracted attributes.
    """
    extra_configs = extra_configs or {}
    cfg = {k: getattr(model, k) for k in model_keys}
    cfg.update(extra_configs)
    return cfg


def build_model_config_sweep(
    kernel_name: str,
    all_model_configs: Optional[List[ModelConfig]] = None,
    setup_fn: Callable[[SingleBenchmarkRunInput], Tuple[Any, ...]] = None,
    model_keys: List[str] = None,
    probe_dim: Literal["T", "B", "BT"] = "T",
    forward_fn: Callable[..., torch.Tensor] = default_forward_fn,
    probe_provider: str = "huggingface",
    extra_configs: Optional[Dict] = None,
    bt: int = 2048,
    overwrite: bool = False,
) -> Dict:
    """Build benchmark config dict for model-config sweep.

    Args:
        kernel_name: Name of the kernel being benchmarked.
        all_model_configs: List of model configurations to sweep over.
        setup_fn: Function that prepares inputs and modules given a
            `SingleBenchmarkRunInput`. Returns a tuple of objects consumed
            by `forward_fn`.
        model_keys: List of attributes to extract from each `ModelConfig`
            and include in `extra_benchmark_config`.
        forward_fn: Function that executes the kernel given the outputs of
            `setup_fn`. Defaults to `(x, layer) -> layer(x)`.
        probe_provider: Kernel provider used during memory probing.
        extra_configs: Optional static overrides merged into the benchmark config.
        token_length: Optional token length used for memory probing and sweep config.
        bt: Target total tokens (batch_size * seq_len) used to derive sweep.
        probe_x: Value of x passed to setup_fn during probing. This should be
            specified if the kernel's input.x is not T.
        overwrite: Whether to overwrite existing benchmark results.

    Returns:
        A dictionary consumable by `run_benchmarks`.
    """

    if all_model_configs is None:
        all_model_configs = list(MODEL_REGISTRY.values())

    def probe_fn_factory(model_cfg):
        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=1 if probe_dim == "B" else bt,
                kernel_provider=probe_provider,
                extra_benchmark_config=build_extra_config(
                    model_cfg,
                    model_keys,
                    extra_configs=extra_configs,
                ),
            )
            setup_out = setup_fn(probe_input)

            return forward_fn(*setup_out)

        return _probe

    sweep = compute_model_config_sweep_config(
        all_model_configs,
        probe_fn_factory=probe_fn_factory,
        bt=bt,
    )

    base_config = {"bsz": sweep.batch_size, "seq_len": sweep.seq_len}

    if extra_configs:
        base_config.update(extra_configs)

    return {
        "kernel_name": kernel_name,
        "x_name": "model_config",
        "x_label": "model configuration",
        "x_values": [cfg.name for cfg in sweep.model_configs],
        "extra_benchmark_configs": [base_config],
        "overwrite": overwrite,
    }


def build_token_length_sweep(
    kernel_name: str,
    probe_x: int,
    model: ModelConfig,
    setup_fn: Callable[[SingleBenchmarkRunInput], Tuple[Any, ...]],
    model_keys: List[str],
    extra_configs: Optional[Dict] = None,
    scale_dim: Literal["T", "B", "BT"] = "T",
    forward_fn: Callable[..., torch.Tensor] = default_forward_fn,
    probe_provider: str = "huggingface",
    x_label: str = "sequence length",
    x_values_fn: Optional[Callable[[SeqLenSweepConfig], List[int]]] = None,
    overwrite: bool = False,
) -> Dict:
    """Build benchmark config dict for token-length sweep.

    Args:
        kernel_name: Name of the kernel being benchmarked.
        probe_x: Value of x passed to setup_fn during probing.
        model: Model configuration used for the sweep.
        setup_fn: Function that prepares inputs and modules given a
            `SingleBenchmarkRunInput`. Returns a tuple of objects consumed
            by `forward_fn`.
        model_keys: List of attributes to extract from `model` and include
            in `extra_benchmark_config`.
        extra_configs: Optional static overrides merged into the config.
        forward_fn: Function that executes the kernel given the outputs of
            `setup_fn`. Defaults to `(x, layer) -> layer(x)`.
        probe_provider: Kernel provider used during memory probing.
        scale_dim: Dimension along which to scale the sweep (e.g. "T", "B", or "BT").
        x_label: Label for the x-axis (e.g. "sequence length" or "batch size").
        x_values_fn: Optional function mapping `SeqLenSweepConfig` to a list
            of x values. Defaults to powers of 2 up to max seq_len.
        overwrite: Whether to overwrite existing benchmark results.

    Returns:
        A dictionary consumable by `run_benchmarks`.
    """
    extra_configs = extra_configs or {}

    def probe_fn():
        probe_input = SingleBenchmarkRunInput(
            x=probe_x,
            kernel_provider=probe_provider,
            extra_benchmark_config=build_extra_config(
                model,
                model_keys,
                extra_configs=extra_configs,
            ),
        )
        setup_out = setup_fn(probe_input)
        return forward_fn(*setup_out)

    # ---------------------------------------
    # derive (probe_batch_size, probe_seq_len) based on scale_dim
    # ---------------------------------------
    if scale_dim == "T":
        probe_batch_size = extra_configs.get("B", 1)
        probe_seq_len = probe_x

    elif scale_dim == "B":
        T = extra_configs.get("T")
        if T is None:
            raise ValueError("For B sweep, extra_configs['T'] must be provided")
        probe_batch_size = probe_x
        probe_seq_len = T

    elif scale_dim == "BT":
        probe_batch_size = 1
        probe_seq_len = probe_x

    else:
        raise ValueError(f"Unsupported scale_dim: {scale_dim}")

    config = compute_seq_len_sweep_config(
        model,
        probe_fn=probe_fn,
        probe_seq_len=probe_seq_len,
        probe_batch_size=probe_batch_size,
    )
    if x_values_fn is None:
        if scale_dim == "T":
            x_values_fn = lambda cfg: [2**i for i in range(10, int(math.log2(cfg.seq_len)) + 1)]
        elif scale_dim == "B":
            x_values_fn = lambda cfg: [2**i for i in range(0, int(math.log2(cfg.batch_size)) + 1)]
        elif scale_dim == "BT":
            x_values_fn = lambda cfg: [2**i for i in range(10, int(math.log2(cfg.seq_len * cfg.batch_size)) + 1)]

    return {
        "kernel_name": kernel_name,
        "x_name": scale_dim,
        "x_label": x_label,
        "x_values": x_values_fn(config),
        "extra_benchmark_configs": [
            build_extra_config(
                model,
                model_keys,
                extra_configs=extra_configs,
            )
        ],
        "overwrite": overwrite,
    }
