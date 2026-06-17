"""Megatron-Core adapter for Liger-Kernel.

Public API:
    LigerMegatronRMSNorm — RMSNorm module conforming to Megatron-Core's
        LayerNormBuilder protocol.
    LigerMegatronCrossEntropy — drop-in for Megatron-LM's vocab-parallel
        cross-entropy (Megatron defaults, takes pre-materialized logits).
    LigerMegatronFusedLinearCrossEntropy — fused LM-head + vocab-parallel
        cross-entropy. Consumes ``hidden_states`` + ``output_weight`` directly,
        chunked over BT so the per-rank ``[BT, V_local]`` logits tensor is
        never materialized. The right choice for long-context / large-vocab
        training; saves ~11 GiB/rank at Llama3-70B TP=8 seqlen=128K.
    apply_liger_kernel_to_megatron — patches Megatron-Core so existing training
        scripts pick up Liger kernels with one line.

The general-purpose ``LigerVocabParallelCrossEntropy`` Module lives under
``liger_kernel.transformers`` alongside the other nn.Module wrappers; the
``LigerVocabParallelCEFunction`` autograd op lives under ``liger_kernel.ops``
alongside the Triton kernels. Import them from there directly when you need
the general-purpose (non-Megatron-default) variant.
"""

from liger_kernel.megatron.cross_entropy import LigerMegatronCrossEntropy
from liger_kernel.megatron.fused_linear_cross_entropy import LigerMegatronFusedLinearCrossEntropy
from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron
from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm

__all__ = [
    "LigerMegatronCrossEntropy",
    "LigerMegatronFusedLinearCrossEntropy",
    "LigerMegatronRMSNorm",
    "apply_liger_kernel_to_megatron",
]
