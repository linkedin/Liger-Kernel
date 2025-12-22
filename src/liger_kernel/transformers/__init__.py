import importlib

from typing import TYPE_CHECKING

# Always-safe imports (independent of 'transformers')
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss  # noqa: F401
from liger_kernel.transformers.dyt import LigerDyT  # noqa: F401
from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm  # noqa: F401
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss  # noqa: F401
from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD  # noqa: F401
from liger_kernel.transformers.geglu import LigerGEGLUMLP  # noqa: F401
from liger_kernel.transformers.jsd import LigerJSD  # noqa: F401
from liger_kernel.transformers.kl_div import LigerKLDIVLoss  # noqa: F401
from liger_kernel.transformers.layer_norm import LigerLayerNorm  # noqa: F401
from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb  # noqa: F401
from liger_kernel.transformers.llama4_rope import liger_llama4_vision_rotary_pos_emb  # noqa: F401
from liger_kernel.transformers.multi_token_attention import LigerMultiTokenAttention  # noqa: F401
from liger_kernel.transformers.poly_norm import LigerPolyNorm  # noqa: F401
from liger_kernel.transformers.rms_norm import LigerRMSNorm  # noqa: F401
from liger_kernel.transformers.rope import liger_rotary_pos_emb  # noqa: F401
from liger_kernel.transformers.softmax import LigerSoftmax  # noqa: F401
from liger_kernel.transformers.sparsemax import LigerSparsemax  # noqa: F401
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP  # noqa: F401
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP  # noqa: F401
from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP  # noqa: F401
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP  # noqa: F401
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP  # noqa: F401
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP  # noqa: F401
from liger_kernel.transformers.tvd import LigerTVDLoss  # noqa: F401

# Static-only imports for IDEs and type checkers
if TYPE_CHECKING:
    from liger_kernel.transformers.auto_model import AutoLigerKernelForCausalLM  # noqa: F401
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel  # noqa: F401
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_exaone4  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_falcon_h1  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma2  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma3  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma3_text  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_glm4  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_glm4v  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_glm4v_moe  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gpt_oss  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_granite  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_hunyuan_v1_dense  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_hunyuan_v1_moe  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_internvl  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama4  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llava  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_mistral  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_mixtral  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_mllama  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_olmo2  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_olmo3  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_paligemma  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_phi3  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen2  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen2_5_vl  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen2_vl  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_moe  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_next  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_vl  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_vl_moe  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_smollm3  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_smolvlm  # noqa: F401


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
        "apply_liger_kernel_to_falcon_h1",
        "apply_liger_kernel_to_gemma",
        "apply_liger_kernel_to_gemma2",
        "apply_liger_kernel_to_gemma3",
        "apply_liger_kernel_to_gemma3_text",
        "apply_liger_kernel_to_glm4",
        "apply_liger_kernel_to_glm4v",
        "apply_liger_kernel_to_glm4v_moe",
        "apply_liger_kernel_to_gpt_oss",
        "apply_liger_kernel_to_granite",
        "apply_liger_kernel_to_internvl",
        "apply_liger_kernel_to_llama",
        "apply_liger_kernel_to_llava",
        "apply_liger_kernel_to_llama4",
        "apply_liger_kernel_to_mistral",
        "apply_liger_kernel_to_mixtral",
        "apply_liger_kernel_to_mllama",
        "apply_liger_kernel_to_olmo2",
        "apply_liger_kernel_to_olmo3",
        "apply_liger_kernel_to_paligemma",
        "apply_liger_kernel_to_phi3",
        "apply_liger_kernel_to_qwen2",
        "apply_liger_kernel_to_qwen2_5_vl",
        "apply_liger_kernel_to_qwen2_vl",
        "apply_liger_kernel_to_qwen3",
        "apply_liger_kernel_to_qwen3_moe",
        "apply_liger_kernel_to_qwen3_next",
        "apply_liger_kernel_to_qwen3_vl",
        "apply_liger_kernel_to_qwen3_vl_moe",
        "apply_liger_kernel_to_smollm3",
        "apply_liger_kernel_to_smolvlm",
        "apply_liger_kernel_to_hunyuan_v1_dense",
        "apply_liger_kernel_to_hunyuan_v1_moe",
        "apply_liger_kernel_to_exaone4",
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
    "LigerFusedAddRMSNorm",
    "LigerPolyNorm",
    "LigerRMSNorm",
    "liger_rotary_pos_emb",
    "liger_llama4_text_rotary_pos_emb",
    "liger_llama4_vision_rotary_pos_emb",
    "LigerBlockSparseTop2MLP",
    "LigerPhi3SwiGLUMLP",
    "LigerQwen3MoeSwiGLUMLP",
    "LigerSwiGLUMLP",
    "LigerTiledGEGLUMLP",
    "LigerTiledSwiGLUMLP",
    "LigerTVDLoss",
    "LigerKLDIVLoss",
    "LigerMultiTokenAttention",
    "LigerSoftmax",
    "LigerSparsemax",
]

# Add transformer-dependent symbols only if available
if _TRANSFORMERS_AVAILABLE:
    __all__.extend(
        [
            "AutoLigerKernelForCausalLM",
            "_apply_liger_kernel",
            "_apply_liger_kernel_to_instance",
            "apply_liger_kernel_to_falcon_h1",
            "apply_liger_kernel_to_gemma",
            "apply_liger_kernel_to_gemma2",
            "apply_liger_kernel_to_gemma3",
            "apply_liger_kernel_to_gemma3_text",
            "apply_liger_kernel_to_glm4",
            "apply_liger_kernel_to_glm4v",
            "apply_liger_kernel_to_glm4v_moe",
            "apply_liger_kernel_to_gpt_oss",
            "apply_liger_kernel_to_granite",
            "apply_liger_kernel_to_internvl",
            "apply_liger_kernel_to_llama",
            "apply_liger_kernel_to_llava",
            "apply_liger_kernel_to_llama4",
            "apply_liger_kernel_to_mistral",
            "apply_liger_kernel_to_mixtral",
            "apply_liger_kernel_to_mllama",
            "apply_liger_kernel_to_olmo2",
            "apply_liger_kernel_to_olmo3",
            "apply_liger_kernel_to_paligemma",
            "apply_liger_kernel_to_phi3",
            "apply_liger_kernel_to_qwen2",
            "apply_liger_kernel_to_qwen2_5_vl",
            "apply_liger_kernel_to_qwen2_vl",
            "apply_liger_kernel_to_qwen3",
            "apply_liger_kernel_to_qwen3_moe",
            "apply_liger_kernel_to_qwen3_next",
            "apply_liger_kernel_to_qwen3_vl",
            "apply_liger_kernel_to_qwen3_vl_moe",
            "apply_liger_kernel_to_smollm3",
            "apply_liger_kernel_to_smolvlm",
            "apply_liger_kernel_to_hunyuan_v1_dense",
            "apply_liger_kernel_to_hunyuan_v1_moe",
            "apply_liger_kernel_to_exaone4",
        ]
    )
