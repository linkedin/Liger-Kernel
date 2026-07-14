"""
CuTe DSL-specific operator implementations.
"""

try:
    import cutlass.cute as cute  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "cutedsl backend requires the NVIDIA CUTLASS Python DSL (CuTe DSL). "
        "Install it with `pip install nvidia-cutlass-dsl`, or when installing "
        "Liger-Kernel use `pip install 'liger-kernel[cutedsl]'`."
    ) from exc

from liger_kernel.ops.cutedsl.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.cutedsl.ops.cross_entropy import cross_entropy_backward
from liger_kernel.ops.cutedsl.ops.cross_entropy import cross_entropy_forward

__all__ = [
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
]
