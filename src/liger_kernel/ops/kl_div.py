from typing import Literal

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


MAX_FUSED_SIZE = 65536 // 4  # 65536 // 4 or 8 works the best

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

_REDUCTION_MODE_NONE = tl.constexpr(0)
_REDUCTION_MODE_SUM = tl.constexpr(1)
_REDUCTION_MODE_MEAN = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}


@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(y_true) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.sum(loss, axis=0)
            tl.store(loss_ptr, loss)
            loss_ptr += 1  # in case of reduction, the output tensor has dimensions [B,], therefore stride is always 1


@triton.jit
def _kldiv_kernel_backward(
    input_ptr,
    input_stride,
    target_ptr,
    target_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)

    input_ptr += pid * input_stride
    target_ptr += pid * target_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)

        tl.store(input_ptr + offsets, res, mask=mask)


def kldiv_forward_triton(y_pred, y_true, log_target, reduction):  # [B, S]  # [B, S]
    B, S = y_pred.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(S))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (B,)
    reduction = _str_to_reduction_mode[reduction]

    out_size = (B, S) if reduction == _REDUCTION_MODE_NONE.value else (B,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)

    _kldiv_kernel_forward[grid](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction=reduction,
    )

    # calculated according to the reduction mode same as in Pytorch. In the later versions, `mean` will be changed to the same behavior as `batchmean`
    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # https://github.com/pytorch/pytorch/blob/d7b57c4d63edb42e1deeeba9497fcb5f1f748ff2/torch/nn/functional.py#L3372
    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / B
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.mean(dim=0)
    else:
        return output_tensor


def kldiv_backward_triton(input, target, grad_output, log_target):
    B, S = input.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(S))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (B,)

    # We store the gradients in-place in the input tensor
    _kldiv_kernel_backward[grid](
        input,
        input.stride(0),
        target,
        target.stride(0),
        S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul then.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return input

    return input * grad_output


class LigerKLDivLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the KL Divergence Loss using Triton, as defined by the following formula:
    ```python
    if log_target:
        loss = target * (target.log() - input)
    else:
        loss = target.exp() * (target - input)
    ```,
    then the loss is reduced according to the `reduction` parameter.
    as defined in the PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: REDUCTION_LITERAL = "batchmean",
        log_target: bool = False,
    ) -> torch.Tensor:
        """A forward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            y_pred (torch.Tensor): A tensor of shape (BT, V) containing the predicted values, expected to be log-probabilities.
            y_true (torch.Tensor): A tensor of shape (BT, V) containing the target values, expected to be either probabilities or log-probabilities, depending on the value of `log_target`.
            reduction (REDUCTION_LITERAL, optional): Reduction to be used. Defaults to "batchmean".
            log_target (bool, optional): If set to true, expects the ground truth to already be log-probabilities. Defaults to False.

        Returns:
            torch.Tensor: The computed KL Divergence Loss, with shape (BT, V) if `reduction` is "none", else a scalar.
        """
        ctx.save_for_backward(y_pred, y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return kldiv_forward_triton(
            y_pred, y_true, log_target=log_target, reduction=reduction
        )

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """A backward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None, None, None]: The gradient of the loss with respect to the inputs and None for the other arguments of the forward method.
        """
        y_pred, y_true = ctx.saved_tensors

        derivative = kldiv_backward_triton(y_pred, y_true, grad_output, ctx.log_target)

        if ctx.reduction == "batchmean":
            derivative = derivative / y_pred.shape[0]
        elif ctx.reduction == "sum" or ctx.reduction == "none":
            pass
        elif ctx.reduction == "mean":
            derivative = derivative / (y_pred.shape[0] * y_pred.shape[1])

        return (
            derivative,
            None,
            None,
            None,
        )
