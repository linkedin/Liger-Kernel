"""Monkey-patch entry point for applying Liger kernels to Megatron-Core."""

from __future__ import annotations

import logging

from liger_kernel.megatron.cross_entropy import _patch_fused_vocab_parallel_cross_entropy

logger = logging.getLogger(__name__)

_PATCH_MARKER = "__liger_patched__"


def apply_liger_kernel_to_megatron(
    rms_norm: bool = True,
    cross_entropy: bool = False,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "none",
) -> None:
    """Patch Megatron-Core to use Liger Triton kernels.

    Idempotent. Targets Megatron's ``BackendSpecProvider``,
    ``transformer_block.LayerNormImpl``, and (optionally)
    ``fused_vocab_parallel_cross_entropy`` so models that route through the
    standard spec system pick up Liger without per-model code.

    Args:
        rms_norm: When ``True`` (default) replace both
            ``LocalSpecProvider.layer_norm`` (per-layer norm slots) and
            ``transformer_block.LayerNormImpl`` (the block-level
            ``final_layernorm`` slot) so all RMSNorm modules in the model
            become ``LigerMegatronRMSNorm``.
        cross_entropy: When ``True`` replace
            ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``
            with Liger's Triton cross-entropy (online softmax, in-place
            gradients, no full-softmax materialization). Default ``False``
            because this path currently supports
            ``tensor_model_parallel_size=1`` only.
        ignore_index: Cross-entropy ignore index. Only used when
            ``cross_entropy=True``.
        label_smoothing: Cross-entropy label smoothing factor. Liger does not
            auto-detect this — pass ``cfg.label_smoothing_factor`` (or
            equivalent) from your Megatron ``TransformerConfig`` if label
            smoothing is enabled, to preserve native behavior. Only used when
            ``cross_entropy=True``.
        reduction: Must be ``"none"`` when ``cross_entropy=True``; Megatron's
            fused-CE contract returns per-token loss shaped ``[seq, batch]``
            and handles reduction itself downstream.

    Notes:
        Call this BEFORE building your model. Patching after instantiation
        will not retroactively swap modules already created.

        The RMSNorm patches only affect the local (non-TE) backend. Mixing
        Liger norms with ``TESpecProvider`` requires a custom
        ``BackendSpecProvider`` subclass because TE's
        ``TELayerNormColumnParallelLinear`` folds the norm into the QKV
        linear; naive substitution would either double-norm or skip the norm.

    Raises:
        RuntimeError: When ``cross_entropy=True`` and Megatron's parallel
            state already reports ``tensor_model_parallel_size > 1``.
    """
    if rms_norm:
        _patch_local_spec_provider_layer_norm()
        _patch_transformer_block_layernorm_impl()
    if cross_entropy:
        _check_tensor_parallel_size_at_patch_time()
        _patch_fused_vocab_parallel_cross_entropy(
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )


def _check_tensor_parallel_size_at_patch_time() -> None:
    """Raise RuntimeError if Megatron's parallel state already reports TP>1.

    If Megatron is importable but the parallel state is not yet initialized
    (for example, ``apply_liger_kernel_to_megatron`` is called before
    ``initialize_megatron``), silently defer; per-kernel wrappers check again
    at call time against the ``tp_group`` argument Megatron supplies.
    """
    try:
        from megatron.core import parallel_state
    except ImportError:
        return
    try:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    except (AssertionError, RuntimeError):
        return
    if tp_size > 1:
        raise RuntimeError(
            f"apply_liger_kernel_to_megatron(cross_entropy=True) currently requires "
            f"tensor_model_parallel_size=1, got {tp_size}. Vocab-parallel cross-entropy "
            f"support is planned as follow-up work."
        )


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
