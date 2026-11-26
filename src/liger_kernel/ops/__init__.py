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
1. Default implementations are imported on demand from individual modules (e.g., geglu.py)
2. On the first public operator request, device is detected via infer_device() and
   the env var LIGER_KERNEL_IMPL is read
3. select_impl() picks an active implementation (auto-applied for the device,
   or explicitly requested via env var)
4. If one is selected, its operators replace/extend the symbols here
5. All subsequent imports from this package get the replaced versions

Note: Direct imports from submodules (e.g., from liger_kernel.ops.geglu import ...)
      are NOT affected by the replacement mechanism.
      TileLang does not fall back; unsupported operators raise when called.
"""

import importlib
import importlib.util
import os
import threading as _threading

from types import ModuleType as _ModuleType

# Preserve the public names without importing their default Triton modules.
_DEFAULT_EXPORTS = {
    "LigerAttnResFunction": "attn_res",
    "attn_res_backward": "attn_res",
    "attn_res_forward": "attn_res",
    "LigerCrossEntropyFunction": "cross_entropy",
    "cross_entropy_backward": "cross_entropy",
    "cross_entropy_forward": "cross_entropy",
    "LigerDyTFunction": "dyt",
    "LigerEmbeddingFunction": "experimental.embedding",
    "LigerFusedAddRMSNormFunction": "fused_add_rms_norm",
    "fused_add_rms_norm_backward": "fused_add_rms_norm",
    "fused_add_rms_norm_forward": "fused_add_rms_norm",
    "LigerFusedCETVDFunction": "fused_ce_tvd",
    "fused_ce_tvd_backward": "fused_ce_tvd",
    "fused_ce_tvd_forward": "fused_ce_tvd",
    "LigerFusedLinearCrossEntropyFunction": "fused_linear_cross_entropy",
    "fused_linear_cross_entropy_backward": "fused_linear_cross_entropy",
    "fused_linear_cross_entropy_forward": "fused_linear_cross_entropy",
    "LigerFusedLinearJSDFunction": "fused_linear_jsd",
    "fused_linear_jsd_backward": "fused_linear_jsd",
    "fused_linear_jsd_forward": "fused_linear_jsd",
    "LigerFusedLinearKLDivFunction": "fused_linear_kl_div",
    "fused_linear_kl_div_backward": "fused_linear_kl_div",
    "fused_linear_kl_div_forward": "fused_linear_kl_div",
    "LigerFusedLinearScaledCrossEntropyFunction": "fused_linear_scaled_cross_entropy",
    "LigerFusedLinearScaledCrossEntropyTPFunction": "fused_linear_scaled_cross_entropy",
    "ISCorrection": "fused_linear_score_centering",
    "LigerFusedLinearScoreCenteringFunction": "fused_linear_score_centering",
    "fused_linear_score_centering": "fused_linear_score_centering",
    "LigerFusedMoEFunction": "fused_moe",
    "LigerFusedNeighborhoodAttentionFunction": "fused_neighborhood_attention",
    "LigerGELUMulFunction": "geglu",
    "geglu_backward": "geglu",
    "geglu_forward": "geglu",
    "LigerGroupNormFunction": "group_norm",
    "group_norm_backward": "group_norm",
    "group_norm_forward": "group_norm",
    "GrpoLossFunction": "grpo_loss",
    "LigerJSDFunction": "jsd",
    "jsd_backward": "jsd",
    "jsd_forward": "jsd",
    "LigerKLDivLossFunction": "kl_div",
    "LigerLayerNormFunction": "layer_norm",
    "layer_norm_backward": "layer_norm",
    "layer_norm_forward": "layer_norm",
    "LigerLlama4RopeFunction": "llama4_rope",
    "LigerMHCCoeffsFunction": "mhc",
    "LigerMHCPostResFunction": "mhc",
    "LigerMHCPreFunction": "mhc",
    "LigerMLPFunction": "mlp",
    "LigerModulatedRMSNormFunction": "modulated_rms_norm",
    "modulated_rms_norm_backward": "modulated_rms_norm",
    "modulated_rms_norm_forward": "modulated_rms_norm",
    "LigerMultiTokenAttentionFunction": "multi_token_attention",
    "LigerPolyNormFunction": "poly_norm",
    "poly_norm_backward": "poly_norm",
    "poly_norm_forward": "poly_norm",
    "LigerQwen2VLMRopeFunction": "qwen2vl_mrope",
    "LigerReLUSquaredFunction": "relu_squared",
    "relu_squared_backward": "relu_squared",
    "relu_squared_forward": "relu_squared",
    "LigerRMSNormFunction": "rms_norm",
    "rms_norm_backward": "rms_norm",
    "rms_norm_forward": "rms_norm",
    "LigerRopeFunction": "rope",
    "rope_backward": "rope",
    "rope_forward": "rope",
    "LigerSoftmaxFunction": "softmax",
    "LigerSparsemaxFunction": "sparsemax",
    "LigerFusedGateUpSiLUMulFunction": "swiglu",
    "LigerSiLUMulFunction": "swiglu",
    "swiglu_backward": "swiglu",
    "swiglu_forward": "swiglu",
    "LigerTiledMLPFunction": "tiled_mlp",
    "apply_tiled_mlp": "tiled_mlp",
    "LigerTVDLossFunction": "tvd",
    "LigerVocabParallelCEFunction": "vocab_parallel_cross_entropy",
}

# Auxiliary Python types are independent of the selected kernel implementation.
_AUXILIARY_EXPORTS = {"ISCorrection"}

_impl_loaded = False
_impl_lock = _threading.RLock()


def _unavailable_export(name, impl_name):
    from liger_kernel.backends import ImplNotAvailableError

    def unavailable(*args, **kwargs):
        raise ImplNotAvailableError(f"Implementation '{impl_name}' does not provide operator '{name}'.")

    unavailable.__name__ = name
    if name.endswith("Function"):
        import torch

        return type(name, (torch.autograd.Function,), {"__module__": __name__, "forward": staticmethod(unavailable)})
    return unavailable


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

    This function is called when a legacy public export is first requested. It:
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
    from liger_kernel.ops.backends import LIGER_KERNEL_IMPL_ENV
    from liger_kernel.ops.backends import select_impl
    from liger_kernel.utils import infer_device

    device = infer_device()
    explicit = os.environ.get(LIGER_KERNEL_IMPL_ENV, "").strip().lower() or None
    impl_info = select_impl(device, explicit=explicit)
    if impl_info is None:
        return

    try:
        impl_ops = importlib.import_module(impl_info.module_path)

        # Get names to export: use __all__ if defined, otherwise auto-discover.
        names_to_export = getattr(impl_ops, "__all__", None)
        if names_to_export is None:
            names_to_export = [name for name in dir(impl_ops) if not name.startswith("_")]

        # Replace or add to this module's globals.
        for name in names_to_export:
            globals()[name] = getattr(impl_ops, name)

        if impl_info.module_path.startswith(f"{__name__}.tilelang."):
            for name in _DEFAULT_EXPORTS:
                if name not in names_to_export and name not in _AUXILIARY_EXPORTS:
                    globals()[name] = _unavailable_export(name, impl_info.name)

    except ImportError:
        # An auto-selected implementation that fails to import (e.g., missing
        # optional vendor SDK) silently falls back to defaults. An explicitly
        # requested implementation, however, must succeed — re-raise so the
        # user sees the underlying error.
        if explicit:
            raise


def _load_legacy_exports():
    global _impl_loaded
    with _impl_lock:
        if _impl_loaded:
            return

        # Mark before importing implementations, which may themselves use public ops.
        _impl_loaded = True
        try:
            _discover_impls()
            _replace_with_impl_ops()
        except Exception:
            _impl_loaded = False
            raise


def __getattr__(name):
    if name == "__all__":
        _load_legacy_exports()
        return sorted(
            set(_DEFAULT_EXPORTS)
            | {key for key in globals() if not key.startswith("_") and key not in ("importlib", "os")}
        )
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _DEFAULT_EXPORTS and importlib.util.find_spec(f"{__name__}.{name}") is not None:
        # Let Python import a submodule without activating legacy operator exports.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _load_legacy_exports()
    if name in globals():
        return globals()[name]
    if name not in _DEFAULT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _DEFAULT_EXPORTS[name]
    saved_exports = {
        export: globals()[export]
        for export, source in _DEFAULT_EXPORTS.items()
        if source == module_name and export in globals() and not isinstance(globals()[export], _ModuleType)
    }
    module = importlib.import_module(f"{__name__}.{module_name}")
    # Bind sibling exports too: importing a submodule installs the module itself
    # on its parent, which can otherwise shadow a public function of the same name.
    for export, source in _DEFAULT_EXPORTS.items():
        if source == module_name:
            globals()[export] = saved_exports[export] if export in saved_exports else getattr(module, export)
    return globals()[name]


def __dir__():
    return sorted(set(globals()) | set(_DEFAULT_EXPORTS))
