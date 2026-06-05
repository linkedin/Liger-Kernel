"""Monkey-patch entry point for applying Liger kernels to Megatron-Core."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_MARKER = "__liger_patched__"


def apply_liger_kernel_to_megatron(rms_norm: bool = True) -> None:
    """Patch Megatron-Core to use Liger Triton kernels.

    Idempotent. Targets Megatron's ``BackendSpecProvider`` and
    ``transformer_block.LayerNormImpl`` so every model that routes through
    the standard spec system benefits without per-model code.

    Args:
        rms_norm: When ``True`` (default) replace both
            ``LocalSpecProvider.layer_norm`` (per-layer norm slots) and
            ``transformer_block.LayerNormImpl`` (the block-level
            ``final_layernorm`` slot) so all RMSNorm modules in the model
            become ``LigerMegatronRMSNorm``.

    Notes:
        Call this BEFORE building your model. Patching after instantiation
        will not retroactively swap modules already created.

        This only affects the local (non-TE) backend. Mixing Liger norms with
        ``TESpecProvider`` requires a custom ``BackendSpecProvider`` subclass
        because TE's ``TELayerNormColumnParallelLinear`` folds the norm into
        the QKV linear; naive substitution would either double-norm or skip
        the norm. See the project README for the mixing recipe.
    """
    if rms_norm:
        _patch_local_spec_provider_layer_norm()
        _patch_transformer_block_layernorm_impl()


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
        return original_layer_norm(self, rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual)

    setattr(patched_layer_norm, _PATCH_MARKER, True)
    setattr(patched_layer_norm, "__wrapped__", original_layer_norm)
    backends.LocalSpecProvider.layer_norm = patched_layer_norm

    logger.info(
        "Patched megatron.core.models.backends.LocalSpecProvider.layer_norm "
        "to return LigerMegatronRMSNorm for rms_norm=True."
    )


def _patch_transformer_block_layernorm_impl() -> None:
    """Cover the block-level ``final_layernorm`` slot.

    ``TransformerBlock`` uses a module-level global ``LayerNormImpl`` (chosen
    at import time from TE / Apex / WrappedTorchNorm) to fill the
    ``final_layernorm`` slot when the caller passes a per-layer spec rather
    than a ``TransformerBlockSubmodules``. Our spec-provider patch only
    catches the per-layer slots, so without this second patch the trailing
    norm stays as PyTorch's ``nn.RMSNorm``.

    We only displace the ``WrappedTorchNorm`` fallback. Users on TE or Apex
    chose those deliberately; replacing them would undo TE's LN+Linear
    fusion or surprise Apex users.
    """
    from megatron.core.transformer import transformer_block
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm

    if getattr(transformer_block.LayerNormImpl, _PATCH_MARKER, False):
        return

    original = transformer_block.LayerNormImpl
    if original is not WrappedTorchNorm:
        logger.info(
            "transformer_block.LayerNormImpl is %s; skipping block-level patch "
            "(Liger only displaces the pure-torch fallback).",
            getattr(original, "__name__", str(original)),
        )
        return

    class _LigerOrTorchNorm:
        """Routes to ``LigerMegatronRMSNorm`` when ``config.normalization``
        is ``"RMSNorm"``; otherwise instantiates the original implementation.

        Keeps LayerNorm users on PyTorch's ``nn.LayerNorm`` while routing
        RMSNorm users through Liger for the final norm slot.
        """

        def __new__(cls, config, hidden_size, eps=1e-5, **kwargs):
            if getattr(config, "normalization", None) == "RMSNorm":
                return LigerMegatronRMSNorm(config=config, hidden_size=hidden_size, eps=eps, **kwargs)
            return original(config=config, hidden_size=hidden_size, eps=eps, **kwargs)

    setattr(_LigerOrTorchNorm, _PATCH_MARKER, True)
    setattr(_LigerOrTorchNorm, "__wrapped__", original)
    transformer_block.LayerNormImpl = _LigerOrTorchNorm

    logger.info(
        "Patched megatron.core.transformer.transformer_block.LayerNormImpl "
        "to route RMSNorm configs through LigerMegatronRMSNorm."
    )
