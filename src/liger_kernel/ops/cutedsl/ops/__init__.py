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
from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.cutedsl.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.cutedsl.ops.swiglu import LigerSiLUMulCuteDSLFunction as LigerSiLUMulFunction
from liger_kernel.ops.cutedsl.ops.swiglu import fused_swiglu
from liger_kernel.ops.cutedsl.ops.swiglu import pack_swiglu_weights
from liger_kernel.ops.cutedsl.ops.swiglu import swiglu_backward
from liger_kernel.ops.cutedsl.ops.swiglu import swiglu_forward

__all__ = [
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
    "LigerFusedLinearCrossEntropyFunction",
    "LigerRMSNormFunction",
    "rms_norm_backward",
    "rms_norm_forward",
    "LigerSiLUMulFunction",
    "fused_swiglu",
    "pack_swiglu_weights",
    "swiglu_backward",
    "swiglu_forward",
]
