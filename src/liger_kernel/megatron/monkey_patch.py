"""Monkey-patch entry point for applying Liger kernels to Megatron-Core."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_PATCH_MARKER = "__liger_patched__"


def apply_liger_kernel_to_megatron(
    rms_norm: bool = True,
    cross_entropy: bool = False,
    rope: bool = False,
) -> None:
    """Patch Megatron-Core to use Liger Triton kernels.

    Idempotent. Targets Megatron's ``BackendSpecProvider``,
    ``transformer_block.LayerNormImpl``, and (optionally) both of Megatron's
    vocab-parallel cross-entropy entry points so models that route through
    the standard spec system pick up Liger without per-model code.

    Args:
        rms_norm: When ``True`` (default) replace both
            ``LocalSpecProvider.layer_norm`` (per-layer norm slots) and
            ``transformer_block.LayerNormImpl`` (the block-level
            ``final_layernorm`` slot) so all RMSNorm modules in the model
            become ``LigerMegatronRMSNorm``.
        cross_entropy: When ``True`` replace both
            ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``
            (fused path) and
            ``megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy``
            (unfused path) with Liger's Triton cross-entropy. Default
            ``False`` so adopters opt in explicitly. All tensor-parallel sizes
            are supported (TP=1 and TP>1 share the same vocab-parallel
            kernel). The fused wrapper matches native's
            ``(logits, target, tp_group)`` signature exactly; the unfused
            wrapper additionally honors a runtime ``label_smoothing``
            argument, matching native's
            ``(logits, target, label_smoothing=0.0, tp_group=None)``.
        rope: When ``True`` replace
            ``megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb``
            (and every module that imported the symbol, e.g.
            ``megatron.core.transformer.attention``) with Liger's Triton RoPE.
            Default ``False`` so adopters opt in explicitly. Only the standard
            unfused ``bshd`` path is routed through Liger; fused (TE/Apex) RoPE,
            packed ``thd`` sequences, interleaved rotation, multi-latent
            attention, ``mscale != 1`` and per-batch ``freqs`` transparently
            fall back to Megatron's native implementation.

    Notes:
        Call this BEFORE building your model. Patching after instantiation
        will not retroactively swap modules already created.

        The RMSNorm patches only affect the local (non-TE) backend. Mixing
        Liger norms with ``TESpecProvider`` requires a custom
        ``BackendSpecProvider`` subclass because TE's
        ``TELayerNormColumnParallelLinear`` folds the norm into the QKV
        linear; naive substitution would either double-norm or skip the norm.

        For explicit kernel configuration (custom ``ignore_index``,
        ``label_smoothing``, etc.) instantiate ``LigerMegatronCrossEntropy``
        directly and wire it into your model (Mode 2). The monkey-patch path
        is intentionally a transparent drop-in: it matches Megatron's native
        defaults so callers can flip Liger on without touching loss config.
    """
    if rms_norm:
        _patch_local_spec_provider_layer_norm()
        _patch_transformer_block_layernorm_impl()
    if cross_entropy:
        _patch_fused_vocab_parallel_cross_entropy()
        _patch_vocab_parallel_cross_entropy()
    if rope:
        _patch_apply_rotary_pos_emb()


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


# Sentinel for "caller did not pass this kwarg". A plain ``0.0`` default would be
# observationally indistinguishable from "user explicitly asked for 0.0" — and Megatron's
# native vocab_parallel_cross_entropy accepts call-time ``label_smoothing=0.0`` as a real
# request for that value. We must not silently override.
_LABEL_SMOOTHING_UNSET = object()


def _patch_fused_vocab_parallel_cross_entropy() -> None:
    """Replace ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``.

    Wraps a single ``LigerMegatronCrossEntropy`` instance (constructed with class defaults
    that match Megatron's native fused-CE behavior) in a closure matching Megatron's fused-CE
    signature ``(logits, target, tp_group)``. Idempotent: a sentinel attribute on the
    replacement prevents wrappers from stacking.
    """
    try:
        import megatron.core.fusions.fused_cross_entropy as fused_ce
    except ImportError as exc:
        raise ImportError(
            "apply_liger_kernel_to_megatron(cross_entropy=True) requires megatron-core to be "
            "installed. Expected symbol path: "
            "megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy."
        ) from exc

    if not hasattr(fused_ce, "fused_vocab_parallel_cross_entropy"):
        raise ImportError(
            "megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy not "
            "found. The symbol path may have changed in your Megatron-LM version. Please file "
            "an issue on https://github.com/linkedin/Liger-Kernel with your megatron-core version."
        )

    if getattr(fused_ce.fused_vocab_parallel_cross_entropy, _PATCH_MARKER, False):
        return  # already patched

    original = fused_ce.fused_vocab_parallel_cross_entropy

    from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy

    ce = LigerMegatronCrossEntropy()

    def liger_fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=None):
        return ce(vocab_parallel_logits, target, tp_group=tp_group)

    setattr(liger_fused_vocab_parallel_cross_entropy, _PATCH_MARKER, True)
    setattr(liger_fused_vocab_parallel_cross_entropy, "__wrapped__", original)
    fused_ce.fused_vocab_parallel_cross_entropy = liger_fused_vocab_parallel_cross_entropy

    logger.info(
        "Patched megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy with Liger cross-entropy."
    )


def _patch_vocab_parallel_cross_entropy() -> None:
    """Replace ``megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy``.

    This is Megatron's *unfused* eager-Python vocab-parallel CE path, dispatched to when
    ``config.cross_entropy_loss_fusion=False``. Its signature accepts ``label_smoothing``
    at call time, so the wrapper honors a runtime value when the caller actually passed
    one. A sentinel disambiguates "caller passed 0.0" (use 0.0) from "caller didn't pass"
    (use class default).
    """
    try:
        import megatron.core.tensor_parallel.cross_entropy as unfused_ce
    except ImportError as exc:
        raise ImportError(
            "apply_liger_kernel_to_megatron(cross_entropy=True) requires megatron-core to be "
            "installed. Expected symbol path: "
            "megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy."
        ) from exc

    if not hasattr(unfused_ce, "vocab_parallel_cross_entropy"):
        raise ImportError(
            "megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy not "
            "found. The symbol path may have changed in your Megatron-LM version. Please file "
            "an issue on https://github.com/linkedin/Liger-Kernel with your megatron-core version."
        )

    if getattr(unfused_ce.vocab_parallel_cross_entropy, _PATCH_MARKER, False):
        return  # already patched

    original = unfused_ce.vocab_parallel_cross_entropy

    from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy

    # Class-default instance; reused for every call where the caller doesn't pass
    # label_smoothing. Avoids allocating a fresh module per CE call in the common case
    # (Megatron's own LanguageModule.compute_language_model_loss dispatch does not pass
    # label_smoothing — it always lands here).
    default_ce = LigerMegatronCrossEntropy()

    def liger_vocab_parallel_cross_entropy(
        vocab_parallel_logits,
        target,
        label_smoothing=_LABEL_SMOOTHING_UNSET,
        tp_group=None,
    ):
        # Sentinel-based "did the caller pass this?" check so that an explicit
        # label_smoothing=0.0 from the caller is honored verbatim (matching Megatron's
        # native vocab_parallel_cross_entropy contract). Construct a fresh
        # LigerMegatronCrossEntropy only on the runtime-override path; nn.Module
        # construction is microseconds vs. CE-kernel milliseconds.
        if label_smoothing is _LABEL_SMOOTHING_UNSET:
            return default_ce(vocab_parallel_logits, target, tp_group=tp_group)
        ce = LigerMegatronCrossEntropy(label_smoothing=label_smoothing)
        return ce(vocab_parallel_logits, target, tp_group=tp_group)

    setattr(liger_vocab_parallel_cross_entropy, _PATCH_MARKER, True)
    setattr(liger_vocab_parallel_cross_entropy, "__wrapped__", original)
    unfused_ce.vocab_parallel_cross_entropy = liger_vocab_parallel_cross_entropy

    logger.info(
        "Patched megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy with Liger cross-entropy."
    )


def _patch_apply_rotary_pos_emb() -> None:
    """Replace ``rope_utils.apply_rotary_pos_emb`` with Liger's Triton RoPE.

    Megatron's RoPE dispatcher lives in
    ``megatron.core.models.common.embeddings.rope_utils`` but is imported by
    value into consumer modules (``megatron.core.transformer.attention`` does
    ``from ...rope_utils import apply_rotary_pos_emb`` at import time). Rebinding
    only the definition module would therefore miss the copy attention already
    holds. We rebind the symbol on *every* module whose ``apply_rotary_pos_emb``
    still points at the original, so all live call sites pick up Liger.

    Only the standard unfused ``bshd`` path is routed through Liger. Fused RoPE,
    packed ``thd`` sequences, interleaved rotation, multi-latent attention,
    ``mscale != 1`` and per-batch ``freqs`` delegate to the captured original so
    numerics never silently change.
    """
    try:
        import megatron.core.models.common.embeddings.rope_utils as rope_utils
    except ImportError as exc:
        raise ImportError(
            "apply_liger_kernel_to_megatron(rope=True) requires megatron-core to be "
            "installed. Expected symbol path: "
            "megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb."
        ) from exc

    if not hasattr(rope_utils, "apply_rotary_pos_emb"):
        raise ImportError(
            "megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb not "
            "found. The symbol path may have changed in your Megatron-LM version. Please file "
            "an issue on https://github.com/linkedin/Liger-Kernel with your megatron-core version."
        )

    if getattr(rope_utils.apply_rotary_pos_emb, _PATCH_MARKER, False):
        return  # already patched

    original = rope_utils.apply_rotary_pos_emb

    from liger_kernel.megatron.rope import liger_apply_rotary_pos_emb_bshd

    def liger_apply_rotary_pos_emb(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        # Route only the standard unfused bshd path to Liger; everything else
        # (fused TE/Apex RoPE, packed thd, interleaved, multi-latent attention,
        # mscale scaling, per-batch/mRoPE freqs) falls back to native so we
        # never change numerics for configs Liger's kernel does not cover.
        unsupported = (
            cu_seqlens is not None
            or mscale != 1.0
            or getattr(config, "apply_rope_fusion", False)
            or getattr(config, "rotary_interleaved", False)
            or getattr(config, "multi_latent_attention", False)
            or freqs.shape[1] != 1
            or freqs.shape[2] != 1
        )
        if unsupported:
            return original(t, freqs, config, cu_seqlens=cu_seqlens, mscale=mscale, cp_group=cp_group)
        return liger_apply_rotary_pos_emb_bshd(t, freqs)

    setattr(liger_apply_rotary_pos_emb, _PATCH_MARKER, True)
    setattr(liger_apply_rotary_pos_emb, "__wrapped__", original)

    # Rebind on every module that imported the original symbol by value.
    patched_modules = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if getattr(module, "apply_rotary_pos_emb", None) is original:
            module.apply_rotary_pos_emb = liger_apply_rotary_pos_emb
            patched_modules.append(name)

    # Guarantee the definition module is patched even if the scan missed it.
    if rope_utils.apply_rotary_pos_emb is original:
        rope_utils.apply_rotary_pos_emb = liger_apply_rotary_pos_emb
        patched_modules.append(rope_utils.__name__)

    logger.info(
        "Patched apply_rotary_pos_emb with Liger RoPE on modules: %s.",
        ", ".join(sorted(set(patched_modules))),
    )
