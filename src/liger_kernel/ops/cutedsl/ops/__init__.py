"""
cuTeDSL-specific operator implementations.
"""

try:
    # Current parity implementation shares the same CUDA DSL runtime.
    import cuda.tile as ct  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "cuTeDSL backend currently requires the CUDA tile runtime. Install with `pip install cuda-tile` "
        "or `pip install 'cuda-tile[tileiras]'`."
    ) from exc

from liger_kernel.ops.cutedsl.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.cutedsl.ops.rms_norm import rms_norm_forward

__all__ = [
    "LigerRMSNormFunction",
    "rms_norm_backward",
    "rms_norm_forward",
]
