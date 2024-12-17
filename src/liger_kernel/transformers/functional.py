from typing import Optional

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


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


def liger_fused_linear_cross_entropy(
    input,
    weight,
    target,
    bias=None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
):
    return LigerFusedLinearCrossEntropyFunction.apply(
        input,
        weight,
        target,
        bias,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
    )


def liger_fused_linear_jsd(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels=None,
    jsd_beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
):
    return LigerFusedLinearJSDFunction.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        jsd_beta,
        ignore_index,
        temperature,
    )


def liger_geglu(a, b):
    return LigerGELUMulFunction.apply(a, b)


def liger_group_norm(
    X,
    affine_scaling_weight,
    affine_shifting_bias,
    num_channels,
    num_groups,
    eps,
):
    return LigerGroupNormFunction.apply(
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    )


def liger_jsd(
    input,
    target,
    shift_labels=None,
    beta: float = 0.5,
    ignore_index: int = -100,
):
    return LigerJSDFunction.apply(
        input,
        target,
        shift_labels,
        beta,
        ignore_index,
    )


# conform to the function signature in https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html#torch.nn.functional.kl_div
# `size_average` and `mean` are being deprecated in torch API and are placeholders here
def liger_kl_div(
    input,
    target,
    size_average: bool = True,
    reduce: bool = True,
    reduction: str = "mean",
    log_target: bool = False,
    eps: float = 1e-10,
):
    # Note: the default reduction in torch is `mean`, but being `batchmean` in Liger
    return LigerKLDivLossFunction.apply(
        input,
        target,
        reduction,
        log_target,
        eps,
    )


def liger_layer_norm(X, W, B, eps):
    return LigerLayerNormFunction.apply(X, W, B, eps)


def liger_qwen2vl_mrope(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    return LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section, unsqueeze_dim)


def liger_rms_norm(X, W, eps, offset: float = 0.0, casting_mode: str = "llama", in_place: bool = True):
    return LigerRMSNormFunction.apply(X, W, eps, offset, casting_mode, in_place)


def liger_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def liger_swiglu(a, b):
    return LigerSiLUMulFunction.apply(a, b)
