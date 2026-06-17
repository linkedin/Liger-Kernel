"""General-purpose fused linear + vocab-parallel cross-entropy nn.Module.

Wraps :class:`liger_kernel.ops.fused_linear_vocab_parallel_cross_entropy.LigerFusedLinearVPCEFunction`
with the standard Liger Module-in-transformers/ pattern. Megatron's drop-in subclass
lives in :mod:`liger_kernel.megatron.fused_linear_cross_entropy` and only overrides
the defaults.

Pairs with :class:`liger_kernel.transformers.LigerVocabParallelCrossEntropy` (which
takes pre-materialized logits): same label-smoothing surface, but consumes
``hidden_states`` + ``weight`` directly so the per-rank ``[BT, V_local]`` logits
tensor is never materialized.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from liger_kernel.ops.fused_linear_vocab_parallel_cross_entropy import LigerFusedLinearVPCEFunction
from liger_kernel.ops.fused_linear_vocab_parallel_cross_entropy import validate_formula_and_mode


class LigerFusedLinearVocabParallelCrossEntropy(nn.Module):
    """Fused linear + vocab-parallel cross-entropy. PyTorch / F.cross_entropy defaults.

    Args:
        ignore_index: Target value to ignore (no loss, no gradient).
        label_smoothing: Smoothing strength in [0, 1).
        reduction: Must be ``"none"`` — vocab-parallel CE returns per-token loss
            shaped ``[seq, batch]``; reduction happens downstream.
        label_smoothing_formula: ``"pytorch"`` (default) or ``"megatron"``.
        label_smoothing_mode: ``"global"`` (default) or ``"partition"``.
        accum_dtype: Accumulator dtype for ``grad_W`` / ``grad_b``. ``None``
            (default) means ``weight.dtype``. Pass ``torch.float32`` for
            Megatron's ``gradient_accumulation_fusion=True`` interop or whenever
            fp32 wgrad accumulation is desired.

    Shape:
        Input: ``hidden_states`` shape ``[S, B, H]``; ``weight`` shape
            ``[V_local, H]``; optional ``bias`` shape ``[V_local]``; ``target``
            shape ``[S, B]`` with GLOBAL vocab indices in ``[0, V_global)``.
        Output: per-token loss shape ``[S, B]``.

    Forward:
        ``ce(hidden_states, weight, target, bias=None, tp_group=None)``. Pass
        ``tp_group`` when running with TP>1; ``None`` for TP=1 / single-GPU.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        label_smoothing_formula: str = "pytorch",
        label_smoothing_mode: str = "global",
        accum_dtype=None,
    ):
        super().__init__()
        if reduction != "none":
            raise ValueError(f"Vocab-parallel CE returns per-token loss; reduction must be 'none', got {reduction!r}.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")
        validate_formula_and_mode(label_smoothing_formula, label_smoothing_mode)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.label_smoothing_formula = label_smoothing_formula
        self.label_smoothing_mode = label_smoothing_mode
        self.accum_dtype = accum_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor = None,
        tp_group=None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be 3-D ([seq, batch, hidden]); got shape {tuple(hidden_states.shape)}."
            )
        return LigerFusedLinearVPCEFunction.apply(
            hidden_states,
            weight,
            target,
            bias,
            tp_group,
            self.ignore_index,
            self.label_smoothing,
            self.label_smoothing_formula,
            self.label_smoothing_mode,
            self.accum_dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}, label_smoothing_formula={self.label_smoothing_formula!r}, "
            f"label_smoothing_mode={self.label_smoothing_mode!r}, accum_dtype={self.accum_dtype}"
        )
