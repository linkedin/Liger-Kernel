"""Megatron-Core adapter for Liger-Kernel.

Public API:
    LigerMegatronRMSNorm — RMSNorm module conforming to Megatron-Core's
        LayerNormBuilder protocol.
    apply_liger_kernel_to_megatron — patches Megatron-Core's BackendSpecProvider
        so existing scripts pick up Liger kernels with one line.
"""

from liger_kernel.megatron.monkey_patch import apply_liger_kernel_to_megatron
from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm

__all__ = ["LigerMegatronRMSNorm", "apply_liger_kernel_to_megatron"]
