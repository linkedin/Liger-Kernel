"""
cute-specific operator implementations — fused MoE over NVSHMEM.

Imported only when the ``cute`` implementation is actively selected
(``LIGER_KERNEL_IMPL=cute``). The native kernels live in the separate
``liger_cute_kernels`` lck wheel; importing this module loads that compiled
extension and raises a clear ImportError if it is not installed (see
``liger_kernel.ops.cute._load_tvm_ffi``).
"""

from liger_kernel.ops.cute.ops.moe import LigerExpertParallelFusedMoEFunction
from liger_kernel.ops.cute.ops.moe import moe_fused

__all__ = [
    "LigerExpertParallelFusedMoEFunction",
    "moe_fused",
]
