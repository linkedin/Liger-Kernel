"""
FlyDSL-specific operator implementations.
"""

try:
    import flydsl.compiler as flyc  # noqa: F401
    import flydsl.expr as fx  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "flydsl backend requires the ROCm FlyDSL package. "
        "Install it with `pip install flydsl`, or when installing "
        "Liger-Kernel use `pip install 'liger-kernel[flydsl]'`."
    ) from exc

from liger_kernel.ops.flydsl.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.flydsl.ops.cross_entropy import cross_entropy_backward
from liger_kernel.ops.flydsl.ops.cross_entropy import cross_entropy_forward
from liger_kernel.ops.flydsl.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.flydsl.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
from liger_kernel.ops.flydsl.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

__all__ = [
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
    "LigerFusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy_backward",
    "fused_linear_cross_entropy_forward",
]
