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
from liger_kernel.ops.cutedsl.ops.megatron_fused_linear_cross_entropy import (
    LigerMegatronFusedLinearCrossEntropyFunction,
)
from liger_kernel.ops.cutedsl.ops.megatron_fused_linear_cross_entropy import liger_megatron_fused_linear_cross_entropy
from liger_kernel.ops.cutedsl.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.cutedsl.ops.rope import LigerRopeFunction
from liger_kernel.ops.cutedsl.ops.rope import rope_backward
from liger_kernel.ops.cutedsl.ops.rope import rope_forward
from liger_kernel.ops.cutedsl.ops.swiglu import LigerSiLUMulCuteDSLFunction as LigerSiLUMulFunction
from liger_kernel.ops.cutedsl.ops.swiglu import swiglu_backward
from liger_kernel.ops.cutedsl.ops.swiglu import swiglu_forward

# ``LigerFusedScaledCrossEntropySM90Function`` is an *additional* CuTe DSL
# operator (per-token NLL only, Hopper BF16); it deliberately does not replace
# or alias the Triton ``LigerFusedLinearCrossEntropyFunction``, which keeps its
# reduction and legacy-option surface.
# NOTE: rope and swiglu are fork-only CuTe DSL kernels (not present upstream).
# The OSS sync must keep exporting them so ``LIGER_KERNEL_IMPL=cutedsl`` routes
# RoPE/SwiGLU to these kernels instead of silently falling back to Triton.
__all__ = [
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
    "LigerFusedLinearCrossEntropyFunction",
    "LigerFusedScaledCrossEntropySM90Function",
    "fused_scaled_cross_entropy_backward",
    "fused_scaled_cross_entropy_forward",
    "LigerMegatronFusedLinearCrossEntropyFunction",
    "liger_megatron_fused_linear_cross_entropy",
    "LigerRMSNormFunction",
    "rms_norm_backward",
    "rms_norm_forward",
    "LigerRopeFunction",
    "rope_backward",
    "rope_forward",
    "LigerSiLUMulFunction",
    "swiglu_backward",
    "swiglu_forward",
]
