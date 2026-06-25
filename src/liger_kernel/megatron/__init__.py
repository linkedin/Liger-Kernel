"""Megatron-Core adapter for Liger-Kernel.

Public API:
    LigerMegatronRMSNorm — RMSNorm module conforming to Megatron-Core's
        LayerNormBuilder protocol.
    LigerMegatronCrossEntropy — drop-in for Megatron-LM's vocab-parallel
        cross-entropy (Megatron defaults). Supports all TP sizes.
    apply_liger_kernel_to_megatron — patches Megatron-Core so existing training
        scripts pick up Liger kernels with one line. Currently supports
        RMSNorm (via BackendSpecProvider) plus both the fused and unfused
        vocab-parallel cross-entropy paths.

The general-purpose ``LigerVocabParallelCrossEntropy`` Module lives under
``liger_kernel.transformers`` alongside the other nn.Module wrappers; the
``LigerVocabParallelCEFunction`` autograd op lives under ``liger_kernel.ops``
alongside the Triton kernels. Import them from there directly when you need
the general-purpose (non-Megatron-default) variant.
"""

from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy
from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron
from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm

__all__ = [
    "LigerMegatronCrossEntropy",
    "LigerMegatronRMSNorm",
    "apply_liger_kernel_to_megatron",
]
