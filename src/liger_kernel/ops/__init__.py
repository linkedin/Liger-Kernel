import importlib
import inspect

from liger_kernel.ops.backends import VENDOR_REGISTRY
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.dyt import LigerDyTFunction
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.ops.grpo_loss import GrpoLossFunction
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.llama4_rope import LigerLlama4RopeFunction
from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction
from liger_kernel.ops.poly_norm import LigerPolyNormFunction
from liger_kernel.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.softmax import LigerSoftmaxFunction
from liger_kernel.ops.sparsemax import LigerSparsemaxFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.tiled_mlp import LigerTiledMLPFunction
from liger_kernel.ops.tvd import LigerTVDLossFunction
from liger_kernel.utils import infer_device

device = infer_device()

__all__ = [
    "LigerCrossEntropyFunction",
    "LigerDyTFunction",
    "LigerFusedAddRMSNormFunction",
    "LigerFusedLinearCrossEntropyFunction",
    "LigerFusedLinearJSDFunction",
    "LigerFusedNeighborhoodAttentionFunction",
    "LigerGELUMulFunction",
    "LigerGroupNormFunction",
    "GrpoLossFunction",
    "LigerJSDFunction",
    "LigerKLDivLossFunction",
    "LigerLayerNormFunction",
    "LigerLlama4RopeFunction",
    "LigerMultiTokenAttentionFunction",
    "LigerPolyNormFunction",
    "LigerQwen2VLMRopeFunction",
    "LigerRMSNormFunction",
    "LigerRopeFunction",
    "LigerSoftmaxFunction",
    "LigerSparsemaxFunction",
    "LigerSiLUMulFunction",
    "LigerTiledMLPFunction",
    "LigerTVDLossFunction",
]
_globals = globals()

if device in VENDOR_REGISTRY:
    backend_module_path = VENDOR_REGISTRY[device].module_path
    module = importlib.import_module(backend_module_path)
    ops = []
    if module is not None:
        ops += inspect.getmembers(module, inspect.isclass)
    for fn_name, fn in ops:
        if fn_name not in __all__:
            __all__.append(fn_name)
        _globals[fn_name] = fn
