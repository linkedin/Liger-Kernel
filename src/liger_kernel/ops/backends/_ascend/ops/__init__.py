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

from liger_kernel.ops.backends._ascend.ops.embedding import LigerEmbeddingFunction
from liger_kernel.ops.backends._ascend.ops.embedding import embedding_backward
from liger_kernel.ops.backends._ascend.ops.embedding import embedding_forward
from liger_kernel.ops.backends._ascend.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.backends._ascend.ops.geglu import geglu_backward
from liger_kernel.ops.backends._ascend.ops.geglu import geglu_forward
from liger_kernel.ops.backends._ascend.ops.llama4_rope import LigerLlama4RopeFunction
from liger_kernel.ops.backends._ascend.ops.llama4_rope import llama4_rope_backward
from liger_kernel.ops.backends._ascend.ops.llama4_rope import llama4_rope_forward
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import qwen2vl_mrope_backward
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import qwen2vl_mrope_forward
from liger_kernel.ops.backends._ascend.ops.rope import LigerRopeFunction
from liger_kernel.ops.backends._ascend.ops.rope import rope_backward
from liger_kernel.ops.backends._ascend.ops.rope import rope_forward
from liger_kernel.ops.backends._ascend.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.backends._ascend.ops.swiglu import swiglu_backward
from liger_kernel.ops.backends._ascend.ops.swiglu import swiglu_forward
from liger_kernel.ops.backends._ascend.ops.tvd import LigerTVDLossFunction
from liger_kernel.ops.backends._ascend.ops.tvd import tv_distance_forward_triton
from liger_kernel.ops.backends._ascend.ops.tvd import tvd_backward_triton

__all__ = [
    "LigerEmbeddingFunction",
    "embedding_forward",
    "embedding_backward",
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
    "LigerTVDLossFunction",
    "tv_distance_forward_triton",
    "tvd_backward_triton",
    "LigerLlama4RopeFunction",
    "llama4_rope_forward",
    "llama4_rope_backward",
]
