"""
Extended cross entropy kernel that computes token accuracy without materializing logits
"""

import operator

from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def liger_cross_entropy_kernel_with_accuracy(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    accuracy_ptr,  # NEW: pointer to store accuracy
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    RETURN_ACCURACY: tl.constexpr,  # NEW: flag to compute accuracy
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    """
    Extended kernel that computes cross entropy loss AND token accuracy without materializing logits.

    Key optimization: We track argmax during the same pass where we find max for softmax.
    This adds negligible overhead since we're already loading the data.
    """
    program_id = tl.program_id(0).to(tl.int64)

    # Load target
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # Locate input
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # Set gradients to 0 for ignored indices
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        # Store 0 for accuracy (ignored tokens don't count)
        if RETURN_ACCURACY:
            tl.store(accuracy_ptr + program_id * loss_stride, 0.0)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride
    if RETURN_ACCURACY:
        accuracy_ptr += program_id * loss_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: Find max + sum
    # NEW: Also track argmax for accuracy
    m = float("-inf")
    d = 0.0
    argmax_idx = 0  # NEW: track which index has the max value

    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    # First pass: find max, sum, AND argmax
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)

        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)

        block_max = tl.max(X_block)

        # NEW: Track argmax - find which element in this block is the max
        if RETURN_ACCURACY:
            # Check if this block contains the global maximum
            if block_max > m:
                # Create a mask for elements equal to block_max
                is_max_mask = X_block == block_max
                # Use tl.where to get the first matching index
                # We multiply by a large number and take min to get the first true index
                masked_offsets = tl.where(is_max_mask, X_offsets, n_cols)
                argmax_idx = tl.min(masked_offsets)

        if label_smoothing > 0:
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))

        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    # Second pass: compute gradients (same as original)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)

        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate

        if not HAS_WEIGHT:
            X_block = tl.exp(X_block - m) / d
            X_block += 2 * lse_square_scale * lse * X_block
            X_block += -eps
            X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
            if reduction == "mean":
                X_block = X_block / n_non_ignore
        else:
            weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
            softmax_X = tl.exp(X_block - m) / d
            dloss_ori = (1 - label_smoothing) * softmax_X
            dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
            dloss_ori = dloss_ori * weight_y
            dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
            dz_loss = 2 * lse_square_scale * lse * softmax_X
            if reduction == "mean":
                dloss_ori = dloss_ori / sum_non_ignore_weight
                dloss_smooth = dloss_smooth / sum_non_ignore_weight
                dz_loss = dz_loss / n_non_ignore
            X_block = dloss_ori + dloss_smooth + dz_loss

        if HAS_SOFTCAPPING:
            X_block = X_block * (1 - intermediate * intermediate)

        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    # Calculate loss (same as original)
    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    z_loss = lse_square_scale * lse * lse
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)

    # NEW: Store accuracy (1.0 if prediction matches target, 0.0 otherwise)
    if RETURN_ACCURACY:
        is_correct = 1.0 if argmax_idx == y else 0.0
        tl.store(accuracy_ptr, is_correct)


MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2


def cross_entropy_forward_with_accuracy(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    return_token_accuracy=True,  # NEW: option to compute accuracy
):
    """
    Forward pass that computes loss and optionally accuracy without materializing logits.

    Args:
        return_token_accuracy: If True, returns per-token accuracy (1.0 if correct, 0.0 if wrong)

    Returns:
        loss: scalar or per-token loss
        z_loss: scalar or per-token z_loss (or None)
        accuracy: per-token accuracy (or None if return_token_accuracy=False)
        _input: modified input with gradients
    """
    assert isinstance(return_z_loss, bool)
    assert isinstance(return_token_accuracy, bool)

    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    accuracy_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_token_accuracy else None

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()

    assert (target * target_mask).max() < V, f"Target {target.max()} out of bounds"
    assert (target * target_mask).min() >= 0, f"Target {target.min()} out of bounds"

    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0

    if weight is not None:
        assert weight.shape[0] == V
        assert torch.is_floating_point(weight)
        sum_non_ignore_weight = torch.gather(weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        weight_sum = weight.sum().item()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # Launch kernel with accuracy computation
    liger_cross_entropy_kernel_with_accuracy[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        weight_ptr=weight,
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        accuracy_ptr=accuracy_1d,  # NEW
        loss_stride=loss_1d.stride(-1),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=sum_non_ignore_weight,
        ignore_index=ignore_index,
        weight_sum=weight_sum,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=return_z_loss,
        RETURN_ACCURACY=return_token_accuracy,  # NEW
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        num_warps=32 if not is_hip() else 16,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        accuracy = accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        # For accuracy, compute mean of non-ignored tokens
        if return_token_accuracy:
            # Only count non-ignored tokens for accuracy
            if n_non_ignore > 0:
                accuracy = torch.sum(accuracy_1d) / n_non_ignore
            else:
                accuracy = torch.tensor(0.0, device=_input.device, dtype=_input.dtype)
        else:
            accuracy = None

    return loss, z_loss, accuracy, _input


class LigerCrossEntropyFunctionWithAccuracy(torch.autograd.Function):
    """
    Cross entropy with optional accuracy computation, without materializing logits.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.FloatTensor],
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,  # NEW
    ):
        """
        Forward pass that optionally returns accuracy.

        Returns:
            loss: computed loss
            z_loss: z_loss if return_z_loss=True, else None
            accuracy: mean token accuracy if return_token_accuracy=True, else None
        """
        loss, z_loss, accuracy, _input = cross_entropy_forward_with_accuracy(
            _input,
            target,
            weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            return_token_accuracy,
        )

        ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy

        return loss, z_loss, accuracy

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3):
        """Backward pass"""
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3  # accuracy is not differentiable

        (_input,) = ctx.saved_tensors

        # Reuse the backward from original implementation
        from liger_kernel.ops.cross_entropy import cross_entropy_backward

        _input = cross_entropy_backward(_input, grad_output)

        return (
            _input,
            None,  # target
            None,  # weight
            None,  # ignore_index
            None,  # lse_square_scale
            None,  # label_smoothing
            None,  # reduction
            None,  # softcap
            None,  # return_z_loss
            None,  # return_token_accuracy
        )
