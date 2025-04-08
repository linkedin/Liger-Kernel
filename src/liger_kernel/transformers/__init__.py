import importlib

# Always-safe imports (independent of 'transformers')
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss  # noqa: F401
from liger_kernel.transformers.dyt import LigerDyT  # noqa: F401
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss  # noqa: F401
from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD  # noqa: F401
from liger_kernel.transformers.geglu import LigerGEGLUMLP  # noqa: F401
from liger_kernel.transformers.jsd import LigerJSD  # noqa: F401
from liger_kernel.transformers.layer_norm import LigerLayerNorm  # noqa: F401
from liger_kernel.transformers.rms_norm import LigerRMSNorm  # noqa: F401
from liger_kernel.transformers.rope import liger_rotary_pos_emb  # noqa: F401
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP  # noqa: F401
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP  # noqa: F401
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP  # noqa: F401
from liger_kernel.transformers.tvd import LigerTVDLoss  # noqa: F401

# Check if 'transformers' is installed
try:
    import transformers  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def is_transformers_available() -> bool:
    """
    Returns True if the 'transformers' package is available.
    Useful for conditional logic in downstream code.
    """
    return _TRANSFORMERS_AVAILABLE


def __getattr__(name: str):
    """
    Handles lazy access to transformer-dependent attributes.
    If 'transformers' is not installed, raises a user-friendly ImportError.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            f"The attribute '{name}' requires the 'transformers' library, which is not installed.\n"
            f"Please install it with `pip install transformers` to use this functionality."
        )

    if name == "AutoLigerKernelForCausalLM":
        module = importlib.import_module("liger_kernel.transformers.auto_model")
        return getattr(module, name)

    monkey_patch_symbols = {
        "_apply_liger_kernel",
        "_apply_liger_kernel_to_instance",
        "apply_liger_kernel_to_gemma",
        "apply_liger_kernel_to_gemma2",
        "apply_liger_kernel_to_gemma3",
        "apply_liger_kernel_to_gemma3_text",
        "apply_liger_kernel_to_granite",
        "apply_liger_kernel_to_llama",
        "apply_liger_kernel_to_llava",
        "apply_liger_kernel_to_mistral",
        "apply_liger_kernel_to_mixtral",
        "apply_liger_kernel_to_mllama",
        "apply_liger_kernel_to_olmo2",
        "apply_liger_kernel_to_paligemma",
        "apply_liger_kernel_to_phi3",
        "apply_liger_kernel_to_qwen2",
        "apply_liger_kernel_to_qwen2_5_vl",
        "apply_liger_kernel_to_qwen2_vl",
    }

    if name in monkey_patch_symbols:
        module = importlib.import_module("liger_kernel.transformers.monkey_patch")
        return getattr(module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")


# Shared symbols in all environments
__all__ = [
    "is_transformers_available",
    "LigerCrossEntropyLoss",
    "LigerDyT",
    "LigerFusedLinearCrossEntropyLoss",
    "LigerFusedLinearJSD",
    "LigerGEGLUMLP",
    "LigerJSD",
    "LigerLayerNorm",
    "LigerRMSNorm",
    "liger_rotary_pos_emb",
    "LigerBlockSparseTop2MLP",
    "LigerPhi3SwiGLUMLP",
    "LigerSwiGLUMLP",
    "LigerTVDLoss",
]

# Add transformer-dependent symbols only if available
if _TRANSFORMERS_AVAILABLE:
    __all__.extend(
        [
            "AutoLigerKernelForCausalLM",
            "_apply_liger_kernel",
            "_apply_liger_kernel_to_instance",
            "apply_liger_kernel_to_gemma",
            "apply_liger_kernel_to_gemma2",
            "apply_liger_kernel_to_gemma3",
            "apply_liger_kernel_to_gemma3_text",
            "apply_liger_kernel_to_granite",
            "apply_liger_kernel_to_llama",
            "apply_liger_kernel_to_llava",
            "apply_liger_kernel_to_mistral",
            "apply_liger_kernel_to_mixtral",
            "apply_liger_kernel_to_mllama",
            "apply_liger_kernel_to_olmo2",
            "apply_liger_kernel_to_paligemma",
            "apply_liger_kernel_to_phi3",
            "apply_liger_kernel_to_qwen2",
            "apply_liger_kernel_to_qwen2_5_vl",
            "apply_liger_kernel_to_qwen2_vl",
        ]
    )
