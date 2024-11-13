from typing import Optional

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

liger_swiglu = LigerSiLUMulFunction.apply
liger_fused_linear_cross_entropy = LigerFusedLinearCrossEntropyFunction.apply
liger_geglu = LigerGELUMulFunction.apply
liger_rms_norm = LigerRMSNormFunction.apply
liger_rope = LigerRopeFunction.apply
liger_layer_norm = LigerLayerNormFunction.apply
liger_kl_div = LigerKLDivLossFunction.apply
liger_jsd = LigerJSDFunction.apply
liger_fused_linear_jsd = LigerFusedLinearJSDFunction.apply
liger_group_norm = LigerGroupNormFunction.apply


# conform to the function signature in https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
# `weight` and `size_average` are placeholders and not implemented yet
def liger_cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index: int = -100,
    reduce=None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
):
    loss, z_loss = LigerCrossEntropyFunction.apply(
        input,
        target,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
    )
    if not return_z_loss:
        return loss
    return loss, z_loss
