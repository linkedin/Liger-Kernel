"""Megatron-Core adapter for Liger-Kernel.

Public API:
    LigerMegatronRMSNorm — RMSNorm module conforming to Megatron-Core's
        LayerNormBuilder protocol.
    LigerVocabParallelCrossEntropy — general-purpose vocab-parallel CE
        (PyTorch / F.cross_entropy semantics by default).
    LigerMegatronCrossEntropy — drop-in for Megatron-LM's
        vocab-parallel cross-entropy (Megatron defaults). Supports all TP sizes.
    apply_liger_kernel_to_megatron — patches Megatron-Core so existing training
        scripts pick up Liger kernels with one line. Currently supports
        RMSNorm (via BackendSpecProvider) plus both the fused and unfused
        vocab-parallel cross-entropy paths.
"""

from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy
from liger_kernel.megatron.cross_entropy import LigerVocabParallelCrossEntropy
from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron
from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm

__all__ = [
    "LigerMegatronCrossEntropy",
    "LigerMegatronRMSNorm",
    "LigerVocabParallelCrossEntropy",
    "apply_liger_kernel_to_megatron",
]
