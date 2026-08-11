"""Megatron-facing module for tensor-parallel fused linear cross entropy."""

from __future__ import annotations

import torch
import torch.nn as nn

from liger_kernel.ops import liger_megatron_fused_linear_cross_entropy


class LigerMegatronFusedLinearCrossEntropy(nn.Module):
    """Fuse a vocab-sharded output projection with per-token cross entropy.

    ``hidden`` is replicated across TP ranks and ``weight`` contains the local
    contiguous vocabulary shard. The output shape matches ``target``.
    """

    def __init__(
        self,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_group=None,
    ) -> torch.Tensor:
        return liger_megatron_fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            tp_group=tp_group,
            ignore_index=self.ignore_index,
        )

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}"
