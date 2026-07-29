"""Megatron-Core compatible cross-entropy.

Thin subclass of :class:`liger_kernel.transformers.LigerVocabParallelCrossEntropy`
that overrides the defaults to match Megatron-LM's ``vocab_parallel_cross_entropy``
bit-for-bit (``formula="megatron"``, ``mode="partition"``). The actual kernel +
autograd op live in :mod:`liger_kernel.ops.vocab_parallel_cross_entropy`; the
general-purpose Module wrapper lives in :mod:`liger_kernel.transformers.vocab_parallel_cross_entropy`.

With ``label_smoothing=0`` (the common case) this class is indistinguishable from
``LigerVocabParallelCrossEntropy``. With ``label_smoothing > 0`` it reproduces
Megatron's NeMo smoothing formula and Megatron's V_local averaging choice — so users
monkey-patching from Megatron see zero numerical change at any TP size.

Conforms to ``megatron.core.fusions.fused_cross_entropy.fused_vocab_parallel_cross_entropy``'s
signature, ``(vocab_parallel_logits, target, tp_group=None)``.
"""

from __future__ import annotations

from liger_kernel.transformers.vocab_parallel_cross_entropy import LigerVocabParallelCrossEntropy


class LigerMegatronCrossEntropy(LigerVocabParallelCrossEntropy):
    """``nn.Module`` drop-in for Megatron-LM's vocab-parallel cross-entropy.

    Args mirror :class:`LigerVocabParallelCrossEntropy`; only the defaults differ.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        label_smoothing_formula: str = "megatron",
        label_smoothing_mode: str = "partition",
    ):
        super().__init__(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
            label_smoothing_formula=label_smoothing_formula,
            label_smoothing_mode=label_smoothing_mode,
        )
