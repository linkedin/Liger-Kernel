"""Megatron-Core adapter for Liger-Kernel.

Public API:
    LigerMegatronRMSNorm — RMSNorm module conforming to Megatron-Core's
        LayerNormBuilder protocol.
    LigerMegatronCrossEntropy — drop-in for Megatron-LM's vocab-parallel
        cross-entropy (Megatron defaults). Supports all TP sizes.
    LigerMegatronSwiGLU — MLP subclass for Mode 2, replacing the fused
        gated-SiLU activation used by the dense MLP and the MoE shared
        experts. Mode 1 patches ``fused_bias_swiglu.SwiGLUFunction`` instead;
        both fall back to Megatron for FP8 input store and CPU activation
        offload, and neither touches the bias or MoE-routed variants.
    apply_liger_kernel_to_megatron — patches Megatron-Core so existing training
        scripts pick up Liger kernels with one line. Currently supports
        RMSNorm (via BackendSpecProvider), both the fused and unfused
        vocab-parallel cross-entropy paths, and SwiGLU.

The general-purpose ``LigerVocabParallelCrossEntropy`` Module lives under
``liger_kernel.transformers`` alongside the other nn.Module wrappers; the
``LigerVocabParallelCEFunction`` autograd op lives under ``liger_kernel.ops``
alongside the Triton kernels. Import them from there directly when you need
the general-purpose (non-Megatron-default) variant.
"""

from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy
from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron
from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm
from liger_kernel.megatron.swiglu import LigerMegatronSwiGLU

__all__ = [
    "LigerMegatronCrossEntropy",
    "LigerMegatronRMSNorm",
    "LigerMegatronSwiGLU",
    "apply_liger_kernel_to_megatron",
]
