"""
Ascend NPU operator implementations.

This module exports Ascend NPU-optimized implementations that will automatically
replace the default implementations when running on NPU devices.

Both Function classes and kernel functions can be exported here.

To add a new operator:
1. Create the implementation file (e.g., rms_norm.py)
2. Import the Function class and/or kernel functions here
3. Optionally add to __all__ for explicit control

If __all__ is not defined, all public symbols will be auto-discovered.
"""

from liger_kernel.ops.backends._ascend.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.backends._ascend.ops.geglu import geglu_backward
from liger_kernel.ops.backends._ascend.ops.geglu import geglu_forward
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import qwen2vl_mrope_backward
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import qwen2vl_mrope_forward
from liger_kernel.ops.backends._ascend.ops.rope import LigerRopeFunction
from liger_kernel.ops.backends._ascend.ops.rope import rope_backward
from liger_kernel.ops.backends._ascend.ops.rope import rope_forward
from liger_kernel.ops.backends._ascend.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.backends._ascend.ops.swiglu import swiglu_backward
from liger_kernel.ops.backends._ascend.ops.swiglu import swiglu_forward

__all__ = [
    "LigerGELUMulFunction",
    "geglu_forward",
    "geglu_backward",
    "LigerQwen2VLMRopeFunction",
    "qwen2vl_mrope_forward",
    "qwen2vl_mrope_backward",
    "LigerRopeFunction",
    "rope_forward",
    "rope_backward",
    "LigerSiLUMulFunction",
    "swiglu_forward",
    "swiglu_backward",
]
