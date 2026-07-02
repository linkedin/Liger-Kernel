"""
cuTile-specific operator implementations.
"""

try:
    import cuda.tile as ct  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "cuTile backend requires cuda-tile. Install it with `pip install cuda-tile` "
        "or `pip install 'cuda-tile[tileiras]'` to include the optional tileiras compiler. "
        "When installing Liger-Kernel, use `pip install 'liger-kernel[cutile]'` "
        "or `pip install 'liger-kernel[cutile-tileiras]'`."
    ) from exc

from liger_kernel.ops.cutile.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.cutile.ops.cross_entropy import cross_entropy_backward
from liger_kernel.ops.cutile.ops.cross_entropy import cross_entropy_forward
from liger_kernel.ops.cutile.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.cutile.ops.fused_linear_jsd import fused_linear_jsd_backward
from liger_kernel.ops.cutile.ops.fused_linear_jsd import fused_linear_jsd_forward
from liger_kernel.ops.cutile.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.cutile.ops.geglu import geglu_backward
from liger_kernel.ops.cutile.ops.geglu import geglu_forward
from liger_kernel.ops.cutile.ops.jsd import LigerJSDFunction
from liger_kernel.ops.cutile.ops.jsd import jsd_backward
from liger_kernel.ops.cutile.ops.jsd import jsd_forward
from liger_kernel.ops.cutile.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.cutile.ops.layer_norm import layer_norm_backward
from liger_kernel.ops.cutile.ops.layer_norm import layer_norm_forward
from liger_kernel.ops.cutile.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.cutile.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.cutile.ops.rms_norm import rms_norm_forward

__all__ = [
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
    "LigerFusedLinearJSDFunction",
    "fused_linear_jsd_backward",
    "fused_linear_jsd_forward",
    "LigerGELUMulFunction",
    "geglu_backward",
    "geglu_forward",
    "LigerJSDFunction",
    "jsd_backward",
    "jsd_forward",
    "LigerLayerNormFunction",
    "layer_norm_backward",
    "layer_norm_forward",
    "LigerRMSNormFunction",
    "rms_norm_backward",
    "rms_norm_forward",
]
