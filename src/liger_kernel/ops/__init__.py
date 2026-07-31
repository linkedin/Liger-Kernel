"""
Liger-Kernel operators with automatic implementation-specific replacement.

This module provides two ways to import operators:

1. Import from this package (recommended for Function classes):
       from liger_kernel.ops import LigerGELUMulFunction

   This automatically uses the active implementation if any is selected.

2. Import from submodules (for kernel functions or specific access):
       from liger_kernel.ops.geglu import geglu_forward, geglu_backward

   This always uses the default implementation (no auto-replacement).

The replacement mechanism:
1. Default implementations are imported from individual modules (e.g., geglu.py)
2. On module load, device is detected via infer_device() and the env var
   LIGER_KERNEL_IMPL is read
3. select_impl() picks an active implementation (auto-applied for the device,
   or explicitly requested via env var)
4. If one is selected, its operators replace/extend the symbols here
5. All subsequent imports from this package get the replaced versions

Note: Direct imports from submodules (e.g., from liger_kernel.ops.geglu import ...)
      are NOT affected by the replacement mechanism.
"""

# =============================================================================
# Import default implementations
# Both Function classes and kernel functions are imported here.
# All of these can be replaced by backend-specific implementations.
# =============================================================================

from liger_kernel.ops.attn_res import LigerAttnResFunction  # noqa: F401
from liger_kernel.ops.attn_res import attn_res_backward  # noqa: F401
from liger_kernel.ops.attn_res import attn_res_forward  # noqa: F401
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction  # noqa: F401
from liger_kernel.ops.cross_entropy import cross_entropy_backward  # noqa: F401
from liger_kernel.ops.cross_entropy import cross_entropy_forward  # noqa: F401
from liger_kernel.ops.dyt import LigerDyTFunction  # noqa: F401
from liger_kernel.ops.experimental.embedding import LigerEmbeddingFunction  # noqa: F401
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction  # noqa: F401
from liger_kernel.ops.fused_add_rms_norm import fused_add_rms_norm_backward  # noqa: F401
from liger_kernel.ops.fused_add_rms_norm import fused_add_rms_norm_forward  # noqa: F401
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction  # noqa: F401
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward  # noqa: F401
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward  # noqa: F401
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction  # noqa: F401
from liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_backward  # noqa: F401
from liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_forward  # noqa: F401
from liger_kernel.ops.fused_moe import LigerFusedMoEFunction  # noqa: F401
from liger_kernel.ops.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionFunction  # noqa: F401
from liger_kernel.ops.geglu import LigerGELUMulFunction  # noqa: F401
from liger_kernel.ops.geglu import geglu_backward  # noqa: F401
from liger_kernel.ops.geglu import geglu_forward  # noqa: F401
from liger_kernel.ops.group_norm import LigerGroupNormFunction  # noqa: F401
from liger_kernel.ops.group_norm import group_norm_backward  # noqa: F401
from liger_kernel.ops.group_norm import group_norm_forward  # noqa: F401
from liger_kernel.ops.grpo_loss import GrpoLossFunction  # noqa: F401
from liger_kernel.ops.jsd import LigerJSDFunction  # noqa: F401
from liger_kernel.ops.jsd import jsd_backward  # noqa: F401
from liger_kernel.ops.jsd import jsd_forward  # noqa: F401
from liger_kernel.ops.kl_div import LigerKLDivLossFunction  # noqa: F401
from liger_kernel.ops.layer_norm import LigerLayerNormFunction  # noqa: F401
from liger_kernel.ops.layer_norm import layer_norm_backward  # noqa: F401
from liger_kernel.ops.layer_norm import layer_norm_forward  # noqa: F401
from liger_kernel.ops.lfm2_moe_router import LigerLfm2MoeRouterFunction  # noqa: F401
from liger_kernel.ops.lfm2_short_conv import LigerLfm2ShortConvFunction  # noqa: F401
from liger_kernel.ops.llama4_rope import LigerLlama4RopeFunction  # noqa: F401
from liger_kernel.ops.mhc import LigerMHCCoeffsFunction  # noqa: F401
from liger_kernel.ops.mhc import LigerMHCPostResFunction  # noqa: F401
from liger_kernel.ops.mhc import LigerMHCPreFunction  # noqa: F401
from liger_kernel.ops.modulated_rms_norm import LigerModulatedRMSNormFunction  # noqa: F401
from liger_kernel.ops.modulated_rms_norm import modulated_rms_norm_backward  # noqa: F401
from liger_kernel.ops.modulated_rms_norm import modulated_rms_norm_forward  # noqa: F401
from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction  # noqa: F401
from liger_kernel.ops.poly_norm import LigerPolyNormFunction  # noqa: F401
from liger_kernel.ops.poly_norm import poly_norm_backward  # noqa: F401
from liger_kernel.ops.poly_norm import poly_norm_forward  # noqa: F401
from liger_kernel.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction  # noqa: F401
from liger_kernel.ops.relu_squared import LigerReLUSquaredFunction  # noqa: F401
from liger_kernel.ops.relu_squared import relu_squared_backward  # noqa: F401
from liger_kernel.ops.relu_squared import relu_squared_forward  # noqa: F401
from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # noqa: F401
from liger_kernel.ops.rms_norm import rms_norm_backward  # noqa: F401
from liger_kernel.ops.rms_norm import rms_norm_forward  # noqa: F401
from liger_kernel.ops.rope import LigerRopeFunction  # noqa: F401
from liger_kernel.ops.rope import rope_backward  # noqa: F401
from liger_kernel.ops.rope import rope_forward  # noqa: F401
from liger_kernel.ops.softmax import LigerSoftmaxFunction  # noqa: F401
from liger_kernel.ops.sparsemax import LigerSparsemaxFunction  # noqa: F401
from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401
from liger_kernel.ops.swiglu import swiglu_backward  # noqa: F401
from liger_kernel.ops.swiglu import swiglu_forward  # noqa: F401
from liger_kernel.ops.tiled_mlp import LigerTiledMLPFunction  # noqa: F401
from liger_kernel.ops.tiled_mlp import apply_tiled_mlp  # noqa: F401
from liger_kernel.ops.tvd import LigerTVDLossFunction  # noqa: F401
from liger_kernel.ops.vocab_parallel_cross_entropy import LigerVocabParallelCEFunction  # noqa: F401

# NOTE: __all__ is intentionally NOT defined.
# - Import from this package (liger_kernel.ops) -> subject to backend replacement
# - Import from submodules (liger_kernel.ops.geglu) -> always use default implementation


# =============================================================================
# Implementation discovery + dispatch
# =============================================================================


def _discover_impls():
    """
    Trigger self-registration of all implementations.

    Two sources of implementations:
      - Hardware backends in ``backends/_<name>/`` (loaded by
        ``backends/__init__.py``'s own auto-import loop).
      - DSL alternatives at the top level of ``ops/`` (e.g., ``cutile/``).
        Each DSL subpackage's ``__init__.py`` calls ``register_impl()``
        when imported.
    """
    import importlib
    import pkgutil

    # Hardware backends self-register when `backends` is imported.
    importlib.import_module("liger_kernel.ops.backends")

    # DSL alternatives — non-private subpackages of `ops/`, minus reserved
    # directories that aren't implementation containers.
    reserved = {"backends", "experimental"}
    for _, modname, ispkg in pkgutil.iter_modules(__path__):
        if ispkg and not modname.startswith("_") and modname not in reserved:
            importlib.import_module(f"{__name__}.{modname}")


def _replace_with_impl_ops():
    """
    Replace/add implementation-specific operators on top of the defaults.

    This function is called automatically on module load. It:
    1. Detects the current device (cuda, npu, xpu, etc.).
    2. Selects the active implementation via ``select_impl()``, honoring any
       explicit ``LIGER_KERNEL_IMPL`` override.
    3. Loads and applies the implementation's operators.

    Implementations live either at:
        liger_kernel/ops/<name>/ops/                     (DSL alternatives)
        liger_kernel/ops/backends/_<name>/ops/           (hardware backends)

    If the implementation module defines ``__all__``, only those symbols are
    exported. Otherwise, all public symbols (not starting with ``_``) are
    auto-discovered.

    Note: Implementations can both override existing ops AND add new ones.
    """
    import os

    from liger_kernel.ops.backends import LIGER_KERNEL_IMPL_ENV
    from liger_kernel.ops.backends import select_impl
    from liger_kernel.utils import infer_device

    device = infer_device()
    explicit = os.environ.get(LIGER_KERNEL_IMPL_ENV, "").strip().lower() or None
    impl_info = select_impl(device, explicit=explicit)
    if impl_info is None:
        return

    try:
        import importlib

        impl_ops = importlib.import_module(impl_info.module_path)

        # Get names to export: use __all__ if defined, otherwise auto-discover.
        names_to_export = getattr(impl_ops, "__all__", None)
        if names_to_export is None:
            names_to_export = [name for name in dir(impl_ops) if not name.startswith("_")]

        # Replace or add to this module's globals.
        for name in names_to_export:
            globals()[name] = getattr(impl_ops, name)

    except ImportError:
        # An auto-selected implementation that fails to import (e.g., missing
        # optional vendor SDK) silently falls back to defaults. An explicitly
        # requested implementation, however, must succeed — re-raise so the
        # user sees the underlying error.
        if explicit:
            raise


_discover_impls()
_replace_with_impl_ops()
