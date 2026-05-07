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

from liger_kernel.ops.backends._ascend.ops.attn_res import LigerAttnResFunction
from liger_kernel.ops.backends._ascend.ops.attn_res import attn_res_backward
from liger_kernel.ops.backends._ascend.ops.attn_res import attn_res_forward
from liger_kernel.ops.backends._ascend.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.backends._ascend.ops.cross_entropy import cross_entropy_backward
from liger_kernel.ops.backends._ascend.ops.cross_entropy import cross_entropy_forward
from liger_kernel.ops.backends._ascend.ops.dyt import LigerDyTFunction
from liger_kernel.ops.backends._ascend.ops.dyt import liger_dyt_bwd
from liger_kernel.ops.backends._ascend.ops.dyt import liger_dyt_fwd
from liger_kernel.ops.backends._ascend.ops.embedding import LigerEmbeddingFunction
from liger_kernel.ops.backends._ascend.ops.embedding import embedding_backward
from liger_kernel.ops.backends._ascend.ops.embedding import embedding_forward
from liger_kernel.ops.backends._ascend.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from liger_kernel.ops.backends._ascend.ops.fused_add_rms_norm import fused_add_rms_norm_backward
from liger_kernel.ops.backends._ascend.ops.fused_add_rms_norm import fused_add_rms_norm_forward
from liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
from liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
from liger_kernel.ops.backends._ascend.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.backends._ascend.ops.fused_linear_jsd import fused_linear_jsd_backward
from liger_kernel.ops.backends._ascend.ops.fused_linear_jsd import fused_linear_jsd_forward
from liger_kernel.ops.backends._ascend.ops.fused_moe import LigerFusedMoEFunction
from liger_kernel.ops.backends._ascend.ops.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionFunction
from liger_kernel.ops.backends._ascend.ops.fused_neighborhood_attention import fused_neighborhood_attention_forward
from liger_kernel.ops.backends._ascend.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.backends._ascend.ops.geglu import geglu_backward
from liger_kernel.ops.backends._ascend.ops.geglu import geglu_forward
from liger_kernel.ops.backends._ascend.ops.group_norm import LigerGroupNormFunction
from liger_kernel.ops.backends._ascend.ops.group_norm import group_norm_backward
from liger_kernel.ops.backends._ascend.ops.group_norm import group_norm_forward
from liger_kernel.ops.backends._ascend.ops.grpo_loss import GrpoLossFunction
from liger_kernel.ops.backends._ascend.ops.grpo_loss import grpo_loss_backward_triton
from liger_kernel.ops.backends._ascend.ops.grpo_loss import grpo_loss_forward_triton
from liger_kernel.ops.backends._ascend.ops.jsd import LigerJSDFunction
from liger_kernel.ops.backends._ascend.ops.jsd import jsd_backward
from liger_kernel.ops.backends._ascend.ops.jsd import jsd_forward
from liger_kernel.ops.backends._ascend.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.backends._ascend.ops.kl_div import kldiv_backward_triton
from liger_kernel.ops.backends._ascend.ops.kl_div import kldiv_forward_triton
from liger_kernel.ops.backends._ascend.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.backends._ascend.ops.layer_norm import layer_norm_backward
from liger_kernel.ops.backends._ascend.ops.layer_norm import layer_norm_forward
from liger_kernel.ops.backends._ascend.ops.llama4_rope import LigerLlama4RopeFunction
from liger_kernel.ops.backends._ascend.ops.llama4_rope import llama4_rope_backward
from liger_kernel.ops.backends._ascend.ops.llama4_rope import llama4_rope_forward
from liger_kernel.ops.backends._ascend.ops.mhc import LigerMHCCoeffsFunction
from liger_kernel.ops.backends._ascend.ops.mhc import LigerMHCPostResFunction
from liger_kernel.ops.backends._ascend.ops.mhc import LigerMHCPreFunction
from liger_kernel.ops.backends._ascend.ops.poly_norm import LigerPolyNormFunction
from liger_kernel.ops.backends._ascend.ops.poly_norm import poly_norm_backward
from liger_kernel.ops.backends._ascend.ops.poly_norm import poly_norm_forward
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import qwen2vl_mrope_backward
from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import qwen2vl_mrope_forward
from liger_kernel.ops.backends._ascend.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.backends._ascend.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.backends._ascend.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.backends._ascend.ops.rope import LigerRopeFunction
from liger_kernel.ops.backends._ascend.ops.rope import rope_backward
from liger_kernel.ops.backends._ascend.ops.rope import rope_forward
from liger_kernel.ops.backends._ascend.ops.softmax import LigerSoftmaxFunction
from liger_kernel.ops.backends._ascend.ops.softmax import _softmax_backward
from liger_kernel.ops.backends._ascend.ops.softmax import _softmax_forward
from liger_kernel.ops.backends._ascend.ops.sparsemax import LigerSparsemaxFunction
from liger_kernel.ops.backends._ascend.ops.sparsemax import sparsemax_backward
from liger_kernel.ops.backends._ascend.ops.sparsemax import sparsemax_forward
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
    "LigerFusedAddRMSNormFunction",
    "fused_add_rms_norm_forward",
    "fused_add_rms_norm_backward",
    "LigerGELUMulFunction",
    "geglu_forward",
    "geglu_backward",
    "LigerQwen2VLMRopeFunction",
    "qwen2vl_mrope_forward",
    "qwen2vl_mrope_backward",
    "LigerRMSNormFunction",
    "rms_norm_forward",
    "rms_norm_backward",
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
    "LigerPolyNormFunction",
    "poly_norm_forward",
    "poly_norm_backward",
    "LigerDyTFunction",
    "liger_dyt_fwd",
    "liger_dyt_bwd",
    "LigerKLDivLossFunction",
    "kldiv_forward_triton",
    "kldiv_backward_triton",
    "LigerLayerNormFunction",
    "layer_norm_backward",
    "layer_norm_forward",
    "LigerSoftmaxFunction",
    "_softmax_forward",
    "_softmax_backward",
    "LigerJSDFunction",
    "jsd_forward",
    "jsd_backward",
    "LigerCrossEntropyFunction",
    "cross_entropy_backward",
    "cross_entropy_forward",
    "GrpoLossFunction",
    "grpo_loss_forward_triton",
    "grpo_loss_backward_triton",
    "LigerFusedLinearJSDFunction",
    "fused_linear_jsd_forward",
    "fused_linear_jsd_backward",
    "LigerGroupNormFunction",
    "group_norm_forward",
    "group_norm_backward",
    "LigerSparsemaxFunction",
    "sparsemax_forward",
    "sparsemax_backward",
    "LigerFusedNeighborhoodAttentionFunction",
    "fused_neighborhood_attention_forward",
    "LigerFusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy_forward",
    "fused_linear_cross_entropy_backward",
    "LigerFusedMoEFunction",
    "LigerMHCCoeffsFunction",
    "LigerMHCPreFunction",
    "LigerMHCPostResFunction",
    "LigerAttnResFunction",
    "attn_res_forward",
    "attn_res_backward",
]
