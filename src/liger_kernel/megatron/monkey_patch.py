"""Monkey-patch entry point for applying Liger kernels to Megatron-Core."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_MARKER = "__liger_patched__"


def apply_liger_kernel_to_megatron(rms_norm: bool = True, force: bool = False) -> None:
    """Patch Megatron-Core to use Liger Triton kernels.

    Idempotent. A no-op when Megatron-Core has native Liger support (the
    upstream ``LigerSpecProvider``) unless ``force=True``. Targets Megatron's
    ``BackendSpecProvider`` classes so every model that routes through the
    standard spec system benefits without per-model code.

    Args:
        rms_norm: Replace ``LocalSpecProvider.layer_norm`` so it returns
            ``LigerMegatronRMSNorm`` when ``rms_norm=True``. Default ``True``.
        force: Apply the monkey-patch even if native Liger support is
            detected. Useful for testing or to override Megatron's wired
            integration. Default ``False``.

    Notes:
        Call this BEFORE building your model. Patching after instantiation
        will not retroactively swap modules already created.

        This only affects the local (non-TE) backend. Mixing Liger norms with
        ``TESpecProvider`` requires a custom ``BackendSpecProvider`` subclass
        because TE's ``TELayerNormColumnParallelLinear`` folds the norm into
        the QKV linear; naive substitution would either double-norm or skip
        the norm. See the project README for the mixing recipe.
    """
    if not force and _native_liger_support_available():
        logger.info(
            "Megatron-Core has native LigerSpecProvider; skipping monkey-patch. "
            "Pass force=True to override."
        )
        return

    if rms_norm:
        _patch_local_spec_provider_layer_norm()


def _native_liger_support_available() -> bool:
    try:
        from megatron.core.extensions.liger_kernel_spec_provider import (  # noqa: F401
            LigerSpecProvider,
        )
        return True
    except ImportError:
        return False


def _patch_local_spec_provider_layer_norm() -> None:
    from megatron.core.models import backends

    from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm

    if getattr(backends.LocalSpecProvider.layer_norm, _PATCH_MARKER, False):
        return  # already patched

    original_layer_norm = backends.LocalSpecProvider.layer_norm

    def patched_layer_norm(
        self,
        rms_norm: bool = False,
        for_qk: bool = False,
        has_residual: bool = False,
    ):
        if rms_norm:
            return LigerMegatronRMSNorm
        return original_layer_norm(
            self, rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual
        )

    setattr(patched_layer_norm, _PATCH_MARKER, True)
    setattr(patched_layer_norm, "__wrapped__", original_layer_norm)
    backends.LocalSpecProvider.layer_norm = patched_layer_norm

    logger.info(
        "Patched megatron.core.models.backends.LocalSpecProvider.layer_norm "
        "to return LigerMegatronRMSNorm for rms_norm=True."
    )
