"""
Liger-Kernel operators with automatic vendor-specific replacement.

This module provides two ways to import operators:

1. Import from this package (recommended for Function classes):
       from liger_kernel.ops import LigerGELUMulFunction

   This automatically uses vendor-specific implementation if available.

2. Import from submodules (for kernel functions or specific access):
       from liger_kernel.ops.geglu import geglu_forward, geglu_backward

   This always uses the default implementation (no auto-replacement).

The replacement mechanism:
1. Default implementations are imported from individual modules (e.g., geglu.py)
2. On module load, device is detected via infer_device()
3. If running on a supported vendor device (npu, xpu, etc.), the default
   implementations are replaced with vendor-specific ones
4. All subsequent imports from this package get the replaced versions

Note: Direct imports from submodules (e.g., from liger_kernel.ops.geglu import ...)
      are NOT affected by the replacement mechanism.
"""

# =============================================================================
# Import default implementations
# Both Function classes and kernel functions are imported here.
# All of these can be replaced by vendor-specific implementations.
# =============================================================================

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.cross_entropy import cross_entropy_backward
from liger_kernel.ops.cross_entropy import cross_entropy_forward
from liger_kernel.ops.dyt import LigerDyTFunction
from liger_kernel.ops.experimental.embedding import LigerEmbeddingFunction
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from liger_kernel.ops.fused_add_rms_norm import fused_add_rms_norm_backward
from liger_kernel.ops.fused_add_rms_norm import fused_add_rms_norm_forward
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_backward
from liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_forward
from liger_kernel.ops.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.geglu import geglu_backward
from liger_kernel.ops.geglu import geglu_forward
from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.ops.group_norm import group_norm_backward
from liger_kernel.ops.group_norm import group_norm_forward
from liger_kernel.ops.grpo_loss import GrpoLossFunction
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.jsd import jsd_backward
from liger_kernel.ops.jsd import jsd_forward
from liger_kernel.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.layer_norm import layer_norm_backward
from liger_kernel.ops.layer_norm import layer_norm_forward
from liger_kernel.ops.llama4_rope import LigerLlama4RopeFunction
from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction
from liger_kernel.ops.poly_norm import LigerPolyNormFunction
from liger_kernel.ops.poly_norm import poly_norm_backward
from liger_kernel.ops.poly_norm import poly_norm_forward
from liger_kernel.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.rope import rope_backward
from liger_kernel.ops.rope import rope_forward
from liger_kernel.ops.softmax import LigerSoftmaxFunction
from liger_kernel.ops.sparsemax import LigerSparsemaxFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.swiglu import swiglu_backward
from liger_kernel.ops.swiglu import swiglu_forward
from liger_kernel.ops.tiled_mlp import LigerTiledMLPFunction
from liger_kernel.ops.tiled_mlp import apply_tiled_mlp
from liger_kernel.ops.tvd import LigerTVDLossFunction

# NOTE: __all__ is intentionally NOT defined.
# - Import from this package (liger_kernel.ops) -> subject to vendor replacement
# - Import from submodules (liger_kernel.ops.geglu) -> always use default implementation


# =============================================================================
# Vendor-specific replacement logic
# =============================================================================


def _replace_with_vendor_ops():
    """
    Replace/add vendor-specific operator implementations.

    This function is called automatically on module load. It:
    1. Detects the current device (cuda, npu, xpu, etc.)
    2. Looks up the vendor for that device via VENDOR_REGISTRY
    3. Loads and applies vendor-specific implementations

    Vendor implementations should be placed in:
        liger_kernel/ops/backends/_<vendor>/ops/

    If the vendor module defines __all__, only those symbols are exported.
    Otherwise, all public symbols (not starting with _) are auto-discovered.

    Note: Vendor can both override existing ops AND add new vendor-specific ops.
    """
    from liger_kernel.ops.backends import get_vendor_for_device
    from liger_kernel.utils import infer_device

    device = infer_device()

    # Look up vendor info for this device
    vendor_info = get_vendor_for_device(device)
    if vendor_info is None:
        return

    try:
        import importlib

        vendor_ops = importlib.import_module(vendor_info.module_path)

        # Get names to export: use __all__ if defined, otherwise auto-discover
        names_to_export = getattr(vendor_ops, "__all__", None)

        if names_to_export is None:
            # Auto-discover: find all public symbols (classes and functions)
            names_to_export = [name for name in dir(vendor_ops) if not name.startswith("_")]

        # Replace or add to this module's globals
        for name in names_to_export:
            globals()[name] = getattr(vendor_ops, name)

    except ImportError:
        # Vendor module not available, use default implementations
        pass


_replace_with_vendor_ops()
