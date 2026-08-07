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
from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import LigerFusedScaledCrossEntropySM90Function
from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import fused_scaled_cross_entropy_backward
from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import fused_scaled_cross_entropy_forward
from liger_kernel.ops.cutedsl.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_forward

# The SM90 fused scaled cross entropy implementation is selected by the
# root-level ``LigerFusedLinearScaledCrossEntropyFunction`` frontend. It
# deliberately does not replace or alias
# ``LigerFusedLinearCrossEntropyFunction``, which keeps its reduction and
# legacy-option surface.
__all__ = [
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
    "LigerFusedLinearCrossEntropyFunction",
    "LigerFusedScaledCrossEntropySM90Function",
    "fused_scaled_cross_entropy_backward",
    "fused_scaled_cross_entropy_forward",
    "LigerRMSNormFunction",
    "rms_norm_backward",
    "rms_norm_forward",
]
