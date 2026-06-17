"""Megatron-Core compatible fused linear + vocab-parallel cross-entropy.

Thin subclass of
:class:`liger_kernel.transformers.LigerFusedLinearVocabParallelCrossEntropy`
that overrides the defaults to match Megatron-LM's
``vocab_parallel_cross_entropy`` bit-for-bit (``formula="megatron"``,
``mode="partition"``). With ``label_smoothing=0`` (the common case) this class
is indistinguishable from the transformers/ wrapper.

Pairs with :class:`liger_kernel.megatron.LigerMegatronCrossEntropy` (which
takes pre-materialized logits): same defaults, but consumes ``hidden_states`` +
``output_weight`` directly so the per-rank ``[BT, V_local]`` logits tensor is
never materialized — the dominant activation in long-context LLM training.

Replaces the ``ColumnParallelLinear(hidden) -> vocab_parallel_cross_entropy(logits)``
two-step in Megatron's LM-head with one fused op.
"""

from __future__ import annotations

from liger_kernel.transformers.fused_linear_vocab_parallel_cross_entropy import (
    LigerFusedLinearVocabParallelCrossEntropy,
)


class LigerMegatronFusedLinearCrossEntropy(LigerFusedLinearVocabParallelCrossEntropy):
    """``nn.Module`` drop-in for Megatron-LM's LM-head + vocab-parallel CE.

    Args mirror :class:`LigerFusedLinearVocabParallelCrossEntropy`; only the
    defaults differ (Megatron formula + partition mode + fp32 accum for
    ``gradient_accumulation_fusion`` interop).
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        label_smoothing_formula: str = "megatron",
        label_smoothing_mode: str = "partition",
        accum_dtype=None,
    ):
        super().__init__(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
            label_smoothing_formula=label_smoothing_formula,
            label_smoothing_mode=label_smoothing_mode,
            accum_dtype=accum_dtype,
        )
