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
alongside the Triton kernels. Both are re-exported from this package for
discoverability since this is where Megatron users land first.
"""

from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy
from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron
from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm
from liger_kernel.transformers.vocab_parallel_cross_entropy import LigerVocabParallelCrossEntropy

__all__ = [
    "LigerMegatronCrossEntropy",
    "LigerMegatronRMSNorm",
    "LigerVocabParallelCrossEntropy",
    "apply_liger_kernel_to_megatron",
]
