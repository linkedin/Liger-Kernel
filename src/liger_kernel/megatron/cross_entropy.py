"""Megatron-Core compatible cross-entropy backed by Liger's Triton kernel."""

from __future__ import annotations

import torch

from liger_kernel.transformers.functional import liger_cross_entropy


class LigerMegatronCrossEntropy(torch.nn.Module):
    """``nn.Module`` drop-in for Megatron's vocab-parallel cross-entropy.

    Conforms to ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``'s
    signature, ``(vocab_parallel_logits, target, tp_group=None)``. Public Mode-2 (hand-built)
    API: instantiate once with the per-training-run config, then call from your overridden
    ``LanguageModule.compute_language_model_loss`` (or wherever Megatron's CE would live in your
    custom model).

    Mirrors the ``LigerMegatronRMSNorm`` pattern shipped in PR #1254: config-time kwargs on
    ``__init__``, data-only ``forward``. Single source of truth for the underlying Liger call;
    the monkey-patch wrappers in ``monkey_patch.py`` instantiate this class.

    Args:
        ignore_index: Target index to ignore.
        label_smoothing: Cross-entropy label smoothing factor.
        reduction: Must be ``"none"`` — Megatron's vocab-parallel CE contract returns per-token
            loss shaped ``[seq, batch]`` and handles reduction itself downstream.

    Scope:
        TP=1 only. Vocab-parallel cross-entropy (TP>1) requires cross-rank reductions
        that Liger's kernel does not perform; tracked as Phase 1.5 follow-up. Raises
        ``RuntimeError`` at call time if a multi-rank ``tp_group`` is supplied.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
    ):
        super().__init__()
        if reduction != "none":
            raise ValueError(
                f"Megatron's vocab-parallel CE contract requires per-token loss; "
                f"reduction must be 'none', got {reduction!r}."
            )
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        tp_group=None,
    ) -> torch.Tensor:
        if tp_group is not None and hasattr(tp_group, "size") and tp_group.size() > 1:
            raise RuntimeError(
                f"LigerMegatronCrossEntropy requires tensor_model_parallel_size=1, "
                f"got tp_group.size()={tp_group.size()}. Vocab-parallel support is "
                f"tracked as follow-up work."
            )
        if vocab_parallel_logits.dim() != 3:
            raise ValueError(
                f"vocab_parallel_logits must be 3-D ([seq, batch, vocab]); "
                f"got shape {tuple(vocab_parallel_logits.shape)}. (HuggingFace's "
                f"[batch, seq, vocab] callers must transpose before calling.)"
            )
        s, b, v = vocab_parallel_logits.shape
        loss = liger_cross_entropy(
            vocab_parallel_logits.reshape(-1, v),
            target.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        return loss.reshape(s, b)

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}"
        )
