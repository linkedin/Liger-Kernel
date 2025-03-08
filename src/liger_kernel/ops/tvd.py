from typing import Literal
from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_SIZE = 65536 // 4

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


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


@triton.jit
def _tv_distance_kernel(
    p_ptr,
    p_stride,
    q_ptr,
    q_stride,
    loss_ptr,
    loss_stride,
    grads_ptr,
    grads_stride,
    label_ptr,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    p_ptr += pid * p_stride
    q_ptr += pid * q_stride
    loss_ptr += pid * loss_stride
    grads_ptr += pid * grads_stride
    label_ptr += pid

    base_offsets = tl.arange(0, BLOCK_SIZE)

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols
                tl.store(grads_ptr + offsets, 0.0, mask=mask)
                if reduction == _REDUCTION_MODE_NONE:
                    tl.store(loss_ptr + offsets, 0.0, mask=mask)
            return

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)

        # TVD(P || Q) = 0.5 * |P - Q|
        tv_loss = 0.5 * tl.abs(p - q)

        grad_res = tl.where(p > q, 0.5, -0.5)

        tl.store(grads_ptr + offsets, grad_res, mask=mask)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, tv_loss, mask=mask)
        else:
            loss_sum += tl.sum(tv_loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)


def tv_distance_forward_triton(p, q, shift_labels, reduction, ignore_index, has_label):
    BT, V = p.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (BT,)

    reduction = _str_to_reduction_mode[reduction]

    out_size = (BT, V) if reduction == _REDUCTION_MODE_NONE.value else (BT,)
    output_tensor = torch.zeros(out_size, device=p.device, dtype=torch.float32)
    grads = torch.empty_like(p)

    n_non_ignore = (shift_labels != ignore_index).sum().item() if has_label else BT

    _tv_distance_kernel[grid](
        p,
        p.stride(0),
        q,
        q.stride(0),
        output_tensor,
        output_tensor.stride(0),
        grads,
        grads.stride(0),
        shift_labels if has_label else torch.empty(1, device=p.device),
        ignore_index,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_LABEL=has_label,
        num_warps=num_warps,
        reduction=reduction,
    )

    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / n_non_ignore, grads / n_non_ignore
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0), grads
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.sum() / (n_non_ignore * V), grads / (n_non_ignore * V)
    else:
        return output_tensor, grads


def tvd_backward_triton(grad_output, grads):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul then.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return grads

    return grads * grad_output


class LigerTVDLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the Total Variation Distance Loss using Triton.
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        p: torch.Tensor,
        q: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
        reduction: REDUCTION_LITERAL = "batchmean",
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """A forward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            p (torch.Tensor): A tensor of shape (BT, V) containing the first distribution.
            q (torch.Tensor): A tensor of shape (BT, V) containing the second distribution.
            shift_labels (Optional[torch.Tensor]): A tensor of shape (BT,) containing the labels.
            reduction (REDUCTION_LITERAL, optional): The reduction method to be applied. Defaults to "batchmean".
            ignore_index (int, optional): The index to ignore during loss calculation. Defaults to -100.

        Returns:
            torch.Tensor: The computed Total Variation Distance Loss.
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (p.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grads = tv_distance_forward_triton(p, q, shift_labels, reduction, ignore_index, has_label)
        ctx.save_for_backward(grads)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """A backward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None, None, None, None]: The gradient of the loss with respect to the inputs.
        """
        (grads,) = ctx.saved_tensors
        grads = tvd_backward_triton(grad_output, grads)

        return grads, None, None, None, None
