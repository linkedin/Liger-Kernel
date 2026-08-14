"""General-purpose vocab-parallel cross-entropy nn.Module.

Wraps :class:`liger_kernel.ops.vocab_parallel_cross_entropy.LigerVocabParallelCEFunction`
with the standard Liger Module-in-transformers/ pattern. Megatron's drop-in subclass
lives in :mod:`liger_kernel.megatron.cross_entropy` and only overrides the defaults.

Label smoothing — two formulas:

  - ``"pytorch"`` (default): ``q' = (1 - alpha) * q + (alpha / K) * uniform``. Same as
    ``torch.nn.functional.cross_entropy(..., label_smoothing=alpha)``.
  - ``"megatron"``: ``q' = (1 - alpha) * q + (alpha / (K - 1)) * uniform_excluding_gt``.
    Same as Megatron-LM's NeMo-derived formula.

The structural form is identical (``loss = (1 - alpha_eff) * H(q, p) + alpha_eff * H(u, p)``)
so the kernel only needs ``alpha_eff = alpha`` (PyTorch) or ``alpha_eff = alpha * K / (K - 1)``
(Megatron). The Python wrapper rescales and forwards.

Averaging scope at TP>1 (irrelevant at TP=1 since V_local == V_global):

  - ``"global"`` (default): ``H(u, p)`` averaged over V_global. Adds one AllReduce on
    ``sum(log_softmax_local, dim=-1)``. Exact result — matches a single-rank reference
    computed on the full logits.
  - ``"partition"``: ``H(u, p)`` averaged over V_local on each rank. No extra
    AllReduce. Loss tensor differs per rank under this mode — this mirrors Megatron's
    existing behavior verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from liger_kernel.ops.vocab_parallel_cross_entropy import LigerVocabParallelCEFunction
from liger_kernel.ops.vocab_parallel_cross_entropy import validate_formula_and_mode


class LigerVocabParallelCrossEntropy(nn.Module):
    """General-purpose vocab-parallel cross-entropy. PyTorch / F.cross_entropy defaults.

    Args:
        ignore_index: Target value to ignore (no loss, no gradient).
        label_smoothing: Smoothing strength in [0, 1).
        reduction: Must be ``"none"`` — vocab-parallel CE returns per-token loss
            shaped ``[seq, batch]``; reduction happens downstream.
        label_smoothing_formula: ``"pytorch"`` (default) or ``"megatron"``. Selects
            which label-smoothing definition the math follows. With
            ``label_smoothing=0`` both are identical.
        label_smoothing_mode: ``"global"`` (default) or ``"partition"``. Selects
            whether the ``H(u, p)`` averaging covers V_global (one extra
            AllReduce, exact at TP>1) or V_local on each rank (no extra
            AllReduce, per-rank loss may differ). Irrelevant at TP=1.

    Shape:
        Input: ``vocab_parallel_logits`` shape ``[S, B, V_local]``; ``target`` shape
            ``[S, B]``. ``target`` values are GLOBAL vocab indices in ``[0, V_global)``.
        Output: per-token loss shape ``[S, B]``.

    Forward:
        ``ce(logits, target, tp_group=None)``. Pass ``tp_group`` (a
        ``torch.distributed.ProcessGroup``) when running with TP>1; pass ``None``
        for TP=1 / single-GPU. Passing a single-rank group is also valid.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        label_smoothing_formula: str = "pytorch",
        label_smoothing_mode: str = "global",
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

    def forward(
        self,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        tp_group=None,
    ) -> torch.Tensor:
        if vocab_parallel_logits.dim() != 3:
            raise ValueError(
                f"vocab_parallel_logits must be 3-D ([seq, batch, vocab]); "
                f"got shape {tuple(vocab_parallel_logits.shape)}. (HuggingFace's "
                f"[batch, seq, vocab] callers must transpose before calling.)"
            )
        return LigerVocabParallelCEFunction.apply(
            vocab_parallel_logits,
            target,
            tp_group,
            self.ignore_index,
            self.label_smoothing,
            self.label_smoothing_formula,
            self.label_smoothing_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}, label_smoothing_formula={self.label_smoothing_formula!r}, "
            f"label_smoothing_mode={self.label_smoothing_mode!r}"
        )
