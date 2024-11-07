from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

liger_swiglu = LigerSiLUMulFunction.apply
liger_cross_entropy = LigerCrossEntropyFunction.apply
liger_fused_linear_cross_entropy = LigerFusedLinearCrossEntropyFunction.apply
liger_geglu = LigerGELUMulFunction.apply
liger_rms_norm = LigerRMSNormFunction.apply
liger_rope = LigerRopeFunction.apply
liger_layer_norm = LigerLayerNormFunction.apply
liger_kl_div = LigerKLDivLossFunction.apply
liger_jsd = LigerJSDFunction.apply
liger_fused_linear_jsd = LigerFusedLinearJSDFunction.apply
