from typing import Literal

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

_REDUCTION_MODE_NONE: tl.constexpr = tl.constexpr(0)
_REDUCTION_MODE_SUM: tl.constexpr = tl.constexpr(1)
_REDUCTION_MODE_MEAN: tl.constexpr = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN: tl.constexpr = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}

# -----------------------------------------------------------------------------
# Kernels (2D Tiling + Persistent Programs)
# -----------------------------------------------------------------------------


@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    gt_ptr,  # [B, S], ground truth ptr
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    n_rows,  # int, number of rows in the input tensor
    n_cols,  # int, number of columns in the input tensor
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    grid_n = tl.cdiv(n_cols, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    # Persistent-program loop over logical 2D blocks.
    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        offset_m = tl.arange(0, BLOCK_SIZE_M) + block_m * BLOCK_SIZE_M
        offset_n = tl.arange(0, BLOCK_SIZE_N) + block_n * BLOCK_SIZE_N

        mask_m = offset_m < n_rows
        mask_n = offset_n < n_cols

        offset = offset_m[:, None] * n_cols + offset_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        y = tl.load(y_ptr + offset, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offset, mask=mask, other=0.0)

        # KL(y_true || y_pred) with y_pred provided in log-space.
        # - log_target=False: y_true is probability space; clamp with eps before log.
        # - log_target=True : y_true is log-probability space.
        if log_target:
            loss = tl.exp(y_true) * (y_true - y)
        else:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offset, loss, mask=mask)
        else:
            # Multiple block_n tiles may update the same row, so atomic_add is required.
            loss_sum = tl.sum(loss, axis=1)
            tl.atomic_add(loss_ptr + offset_m, loss_sum, mask=mask_m)


@triton.jit
def _kldiv_kernel_backward(
    target_ptr,
    new_grads_ptr,
    grad_output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    grid_n = tl.cdiv(n_cols, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    # For reduced losses, grad_output is a scalar. Load it once per program.
    if reduction != _REDUCTION_MODE_NONE:
        grad_output_scalar = tl.load(grad_output_ptr)

    # Persistent-program loop over logical 2D blocks.
    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        offset_m = tl.arange(0, BLOCK_SIZE_M) + block_m * BLOCK_SIZE_M
        offset_n = tl.arange(0, BLOCK_SIZE_N) + block_n * BLOCK_SIZE_N

        mask_m = offset_m < n_rows
        mask_n = offset_n < n_cols

        offset = offset_m[:, None] * n_cols + offset_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        y_true = tl.load(target_ptr + offset, mask=mask, other=0.0)

        if log_target:
            res = -tl.exp(y_true)
        else:
            res = y_true * -1

        if reduction != _REDUCTION_MODE_NONE:
            res = res * grad_output_scalar
        else:
            grad_output = tl.load(grad_output_ptr + offset, mask=mask, other=0.0)
            res = res * grad_output

        if reduction == _REDUCTION_MODE_BATCHMEAN:
            res = res / n_rows
        elif reduction == _REDUCTION_MODE_MEAN:
            res = res / (n_rows * n_cols)

        tl.store(new_grads_ptr + offset, res, mask=mask)


# -----------------------------------------------------------------------------
# Helper: Call compute_default_tiling_strategy
# -----------------------------------------------------------------------------


def get_optimal_block_size(
    n_rows,
    dtype_size,
    BLOCK_SIZE_N: tl.constexpr,
    log_target: bool = False,
    is_backward: bool = False,
    is_scalar_grad_output: bool = True,
):
    """
    Calculate optimal BLOCK_SIZE_M using compute_default_tiling_strategy.
    """
    # 1) Set memory multiplier
    # Backward is lighter than forward in this op, so we typically use a smaller multiplier.
    # If backward also needs to stream a full grad_output tile (i.e., grad_output is not a scalar),
    # its memory footprint becomes closer to forward, so we bump the multiplier.
    if is_backward:
        multiplier = 2.5 if is_scalar_grad_output else 3.0
    else:
        multiplier = 3.0 if log_target else 6.0

    # For bf16/fp16 (dtype_size < 4), compile-time UB overflow was observed on some shapes.
    # Clamp to fp32 size for a conservative tiling estimate; this can be refined later.
    dtype_size = max(dtype_size, 4)

    # 2) Call tiling strategy (tile only dim 0 / rows)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=dtype_size,
        memory_multiplier=multiplier,
        shapes=((n_rows, BLOCK_SIZE_N),),
        tiling_dims=(0,),
    )

    # 3) Parse result
    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return block_size
    else:
        return triton.next_power_of_2(min(128, n_rows))


def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):  # [BT, V]
    BT, V = y_pred.shape
    reduction = _str_to_reduction_mode[reduction]

    out_size = (BT, V) if reduction == _REDUCTION_MODE_NONE.value else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)

    BLOCK_SIZE_N = triton.next_power_of_2(min(128, V))
    BLOCK_SIZE_M = get_optimal_block_size(BT, y_pred.element_size(), BLOCK_SIZE_N, log_target=log_target)
    num_cores = get_npu_core_count()
    total_blocks = triton.cdiv(BT, BLOCK_SIZE_M) * triton.cdiv(V, BLOCK_SIZE_N)
    grid = min(num_cores, total_blocks)

    _kldiv_kernel_forward[(grid,)](
        y_pred,
        y_true,
        output_tensor,
        BT,
        V,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        log_target=log_target,
        reduction=reduction,
    )

    # Final reduction follows PyTorch KLDivLoss semantics.
    # Note: In newer PyTorch versions, `mean` is planned to match `batchmean`.
    # See: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / BT
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.sum() / (BT * V)
    else:
        return output_tensor


def kldiv_backward_triton(target, grad_output, new_grads, log_target, reduction):
    BT, V = target.shape
    reduction = _str_to_reduction_mode[reduction]

    BLOCK_SIZE_N = triton.next_power_of_2(min(128, V))
    # grad_output handling:
    # - numel() == 1: use scalar grad_output path in kernel.
    # - numel() != 1: stream per-element grad_output tile in kernel.
    is_scalar_grad_output = grad_output.numel() == 1
    BLOCK_SIZE_M = get_optimal_block_size(
        BT,
        target.element_size(),
        BLOCK_SIZE_N,
        log_target=log_target,
        is_backward=True,
        is_scalar_grad_output=is_scalar_grad_output,
    )
    num_cores = get_npu_core_count()
    total_blocks = triton.cdiv(BT, BLOCK_SIZE_M) * triton.cdiv(V, BLOCK_SIZE_N)
    grid = min(num_cores, total_blocks)

    _kldiv_kernel_backward[(grid,)](
        target,
        new_grads,
        grad_output,
        BT,
        V,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        log_target=log_target,
        reduction=reduction,
    )

    return new_grads


class LigerKLDivLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the KL Divergence Loss using Triton, as defined by the following formula:
    ```python
    if log_target:
        loss = target.exp() * (target - input)
    else:
        loss = target * (target.log() - input)
    ```,
    then the loss is reduced according to the `reduction` parameter.
    as defined in the PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: REDUCTION_LITERAL = "batchmean",
        log_target: bool = False,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """A forward pass for the KL Divergence Loss.

        Args:
            y_pred (torch.Tensor): A tensor of shape (BT, V) containing the predicted values, expected to be log-probabilities.
            y_true (torch.Tensor): A tensor of shape (BT, V) containing the target values, expected to be either probabilities or log-probabilities, depending on the value of `log_target`.
            reduction (REDUCTION_LITERAL, optional): Reduction to be used. Defaults to "batchmean".
            log_target (bool, optional): If set to true, expects the ground truth to already be log-probabilities. Defaults to False.
            eps: (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

        Returns:
            torch.Tensor: The computed KL Divergence Loss, with shape (BT, V) if `reduction` is "none", else a scalar.
        """
        return kldiv_forward_triton(y_pred, y_true, log_target=log_target, reduction=reduction, eps=eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        y_true = inputs[1]
        reduction = inputs[2] if len(inputs) > 2 else "batchmean"
        log_target = inputs[3] if len(inputs) > 3 else False
        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """A backward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None, None, None, None]: The gradient of the loss with respect to the inputs and None for the other arguments of the forward method.
        """
        (y_true,) = ctx.saved_tensors

        new_grads = torch.empty_like(y_true)

        derivative = kldiv_backward_triton(y_true, grad_output, new_grads, ctx.log_target, ctx.reduction)

        return (
            derivative,
            None,
            None,
            None,
            None,
        )
