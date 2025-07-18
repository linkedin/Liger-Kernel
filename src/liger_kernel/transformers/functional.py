from typing import Optional

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.dyt import LigerDyTFunction
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from liger_kernel.ops.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.ops.jsd import LigerJSDFunction
from liger_kernel.ops.kl_div import LigerKLDivLossFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction
from liger_kernel.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.softmax import LigerSoftmaxFunction
from liger_kernel.ops.sparsemax import LigerSparsemaxFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.tvd import LigerTVDLossFunction


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
        weight,
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
    ce_weight=None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
):
    loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
        input,
        weight,
        target,
        bias,
        ce_weight,
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


def liger_sparsemax(
    input,
    dim: int = -1,
):
    return LigerSparsemaxFunction.apply(input, dim)


def liger_multi_token_attention(
    scores,
    weight,
    bias=None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparse: bool = False,
):
    """
    Functional interface for multi-token attention.

    Args:
        scores: Input tensor of shape (B, C_in, L, L)
        weight: Convolution weight tensor of shape (C_out, C_in // groups, K, K)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for the convolution (default: 1)
        padding: Padding for the convolution (default: 0)
        dilation: Dilation factor for the convolution (default: 1)
        groups: Number of groups for the convolution (default: 1)
        sparse: Specifies if input tensors are expected to be sparse (default: False)
    Returns:
        Output tensor after applying multi-token attention.
    """
    return LigerMultiTokenAttentionFunction.apply(scores, weight, bias, stride, padding, dilation, groups, sparse)


def liger_fused_neighborhood_attention(
    query,
    key,
    value,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
):
    """
    Liger fused neighborhood attention.

    paper: https://arxiv.org/pdf/2504.16922

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        kernel_size: Size of the neighborhood window (default: 7)
        dilation: Dilation factor for the neighborhood (default: 1)
        scale: Scaling factor for attention scores (default: rsqrt(head_dim))

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    return LigerFusedNeighborhoodAttentionFunction.apply(query, key, value, kernel_size, dilation, scale)


def liger_tvd(
    input,
    target,
    shift_labels=None,
    reduction: str = "mean",
    ignore_index: int = -100,
):
    return LigerTVDLossFunction.apply(
        input,
        target,
        shift_labels,
        reduction,
        ignore_index,
    )


def liger_layer_norm(X, W, B, eps):
    return LigerLayerNormFunction.apply(X, W, B, eps)


def liger_qwen2vl_mrope(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    return LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section, unsqueeze_dim)


def liger_rms_norm(X, W, eps, offset: float = 0.0, casting_mode: str = "llama", in_place: bool = True):
    return LigerRMSNormFunction.apply(X, W, eps, offset, casting_mode, in_place)


def liger_fused_add_rms_norm(X, R, W, eps, offset: float = 0.0, casting_mode: str = "llama", in_place: bool = True):
    return LigerFusedAddRMSNormFunction.apply(X, R, W, eps, offset, casting_mode, in_place)


def liger_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def liger_swiglu(a, b):
    return LigerSiLUMulFunction.apply(a, b)


def liger_softmax(x):
    return LigerSoftmaxFunction.apply(x)


def liger_dyt(x, alpha, gamma, beta):
    return LigerDyTFunction.apply(x, alpha, gamma, beta)
