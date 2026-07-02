"""
RMSNorm implementation for the cuTeDSL backend.

This module reuses the cuTile RMSNorm kernels to ensure behavior parity while
the dedicated cuTeDSL kernel implementation is being developed.
"""

from liger_kernel.ops.cutile.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.cutile.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.cutile.ops.rms_norm import rms_norm_forward

__all__ = [
    "LigerRMSNormFunction",
    "rms_norm_forward",
    "rms_norm_backward",
]
