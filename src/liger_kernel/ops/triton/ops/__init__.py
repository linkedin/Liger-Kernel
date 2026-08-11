"""Operators replaced by the opt-in all-Triton implementation."""

from liger_kernel.ops.triton.ops.megatron_fused_linear_cross_entropy import LigerMegatronFusedLinearCrossEntropyFunction
from liger_kernel.ops.triton.ops.megatron_fused_linear_cross_entropy import liger_megatron_fused_linear_cross_entropy

__all__ = [
    "LigerMegatronFusedLinearCrossEntropyFunction",
    "liger_megatron_fused_linear_cross_entropy",
]
