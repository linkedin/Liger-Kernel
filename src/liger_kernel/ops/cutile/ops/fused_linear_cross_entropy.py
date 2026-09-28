# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Hopper-optimized cuTile fused linear cross entropy.

The projection and input-gradient GEMMs use PyTorch's Hopper-tuned BLAS path.
cuTile computes partitioned CE statistics, overwrites logits with dZ, and
computes dW with an FP32-accumulating 128x256x64 MMA schedule.
"""

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]

LOG2E = 1.4426950408889634
DW_TILE_M = 128
DW_TILE_N = 128
DW_TILE_K = 64
CE_BLOCK_SIZE = 2048
LOGITS_STATS_BLOCK_SIZE = 4096
MAX_STATS_BLOCK_SIZE = 1024


@ct.kernel(num_worker_warps=ct.ByTarget(sm_90=8), opt_level=3)
def _matmul_tn_kernel(
    a,
    b,
    output,
    grad_output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    """Compute ``output = (a.T @ b) * grad_output``."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    a_view = a.tiled_view((TILE_K, TILE_M), padding_mode=ct.PaddingMode.ZERO)
    b_view = b.tiled_view((TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    for pid_k in range(a_view.num_tiles(0)):
        a_tile = a_view.load((pid_k, pid_m), latency=10)
        b_tile = b_view.load((pid_k, pid_n), latency=10)
        acc = ct.mma(ct.transpose(a_tile), b_tile, acc)

    scale = ct.astype(ct.gather(grad_output, (), check_bounds=False), ct.float32)
    ct.store(output, (pid_m, pid_n), ct.astype(acc * scale, output.dtype), latency=1)


@ct.kernel(num_worker_warps=ct.ByTarget(sm_90=8), opt_level=3)
def _matmul_tn_n4_kernel(
    a,
    b,
    output,
    grad_output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    LOAD_LATENCY: ConstInt,
):
    """Compute four adjacent dW column tiles while sharing each dZ load."""
    pid_n = ct.bid(0) * 4
    pid_m = ct.bid(1)
    a_view = a.tiled_view((TILE_K, TILE_M), padding_mode=ct.PaddingMode.ZERO)
    b_view = b.tiled_view((TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    output_view = output.tiled_view((TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    acc0 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    acc1 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    acc2 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    acc3 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    for pid_k in range(a_view.num_tiles(0)):
        a_tile = ct.transpose(a_view.load((pid_k, pid_m), latency=LOAD_LATENCY))
        acc0 = ct.mma(a_tile, b_view.load((pid_k, pid_n), latency=LOAD_LATENCY), acc0)
        acc1 = ct.mma(a_tile, b_view.load((pid_k, pid_n + 1), latency=LOAD_LATENCY), acc1)
        acc2 = ct.mma(a_tile, b_view.load((pid_k, pid_n + 2), latency=LOAD_LATENCY), acc2)
        acc3 = ct.mma(a_tile, b_view.load((pid_k, pid_n + 3), latency=LOAD_LATENCY), acc3)

    scale = ct.astype(ct.gather(grad_output, (), check_bounds=False), ct.float32)
    output_view.store((pid_m, pid_n), ct.astype(acc0 * scale, output.dtype), latency=1)
    output_view.store((pid_m, pid_n + 1), ct.astype(acc1 * scale, output.dtype), latency=1)
    output_view.store((pid_m, pid_n + 2), ct.astype(acc2 * scale, output.dtype), latency=1)
    output_view.store((pid_m, pid_n + 3), ct.astype(acc3 * scale, output.dtype), latency=1)


@ct.kernel(occupancy=8)
def _fused_cross_entropy_dz_kernel(
    logits,
    target,
    loss,
    loss_scale,
    partial_max,
    partial_sum,
    completion_count,
    vocab_size,
    num_partitions,
    ignore_index,
    STATS_BLOCK_SIZE: ConstInt,
    CE_BLOCK_SIZE: ConstInt,
    PARTIALS_BLOCK_SIZE: ConstInt,
    HAS_GRADIENTS: ConstBool,
    REDUCTION_MEAN: ConstBool,
):
    program = ct.bid(0)
    row = program // num_partitions
    partition = program % num_partitions
    label = ct.load(target, row, shape=())
    stat_cols = ct.arange(STATS_BLOCK_SIZE, dtype=ct.int32) + partition * STATS_BLOCK_SIZE
    values = ct.astype(
        ct.gather(logits, (row, stat_cols), check_bounds=True, padding_value=-float("inf"), latency=4),
        ct.float32,
    )
    block_max = ct.max(values, 0, keepdims=False)
    block_sum = ct.sum(ct.exp2((values - block_max) * LOG2E, flush_to_zero=True), 0, keepdims=False)
    ct.scatter(partial_max, (row, partition), block_max, check_bounds=False)
    ct.scatter(partial_sum, (row, partition), block_sum, check_bounds=False)

    completed = ct.atomic_add(
        completion_count,
        row,
        1,
        check_bounds=False,
        memory_order=ct.MemoryOrder.ACQ_REL,
        memory_scope=ct.MemoryScope.DEVICE,
    )
    if completed != num_partitions - 1:
        return

    if label == ignore_index:
        if HAS_GRADIENTS:
            for chunk in range((vocab_size + CE_BLOCK_SIZE - 1) // CE_BLOCK_SIZE):
                cols = ct.arange(CE_BLOCK_SIZE, dtype=ct.int32) + chunk * CE_BLOCK_SIZE
                ct.scatter(logits, (row, cols), ct.zeros((CE_BLOCK_SIZE,), dtype=logits.dtype), check_bounds=True)
        ct.scatter(loss, row, ct.astype(0.0, loss.dtype))
        return

    partial_cols = ct.arange(PARTIALS_BLOCK_SIZE, dtype=ct.int32)
    tile_max = ct.astype(
        ct.gather(partial_max, (row, partial_cols), check_bounds=True, padding_value=-float("inf"), latency=2),
        ct.float32,
    )
    tile_sum = ct.astype(
        ct.gather(partial_sum, (row, partial_cols), check_bounds=True, padding_value=0.0, latency=2),
        ct.float32,
    )
    valid_stats = ct.less(partial_cols, num_partitions)
    tile_max = ct.where(valid_stats, tile_max, -float("inf"))
    tile_sum = ct.where(valid_stats, tile_sum, 0.0)

    max_value = ct.max(tile_max, 0, keepdims=False)
    exp_sum = ct.sum(tile_sum * ct.exp2((tile_max - max_value) * LOG2E, flush_to_zero=True), 0, keepdims=False)
    target_logit = ct.astype(ct.gather(logits, (row, label), check_bounds=False), ct.float32)
    row_scale = 1.0
    if REDUCTION_MEAN:
        row_scale = ct.astype(ct.gather(loss_scale, (), check_bounds=False), ct.float32)
    ct.scatter(loss, row, ct.astype((max_value + ct.log(exp_sum) - target_logit) * row_scale, loss.dtype))

    if HAS_GRADIENTS:
        inv_sum = 1.0 / exp_sum
        for chunk in range((vocab_size + CE_BLOCK_SIZE - 1) // CE_BLOCK_SIZE):
            cols = ct.arange(CE_BLOCK_SIZE, dtype=ct.int32) + chunk * CE_BLOCK_SIZE
            logits_tile = ct.astype(
                ct.gather(logits, (row, cols), check_bounds=True, padding_value=-float("inf"), latency=4),
                ct.float32,
            )
            gradient = ct.exp2((logits_tile - max_value) * LOG2E, flush_to_zero=True) * inv_sum
            gradient = ct.where(ct.equal(cols, label), gradient - 1.0, gradient)
            ct.scatter(logits, (row, cols), ct.astype(gradient, logits.dtype), check_bounds=True)


def _launch_matmul_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
) -> None:
    if output.shape[1] % 256 == 0:
        _launch_cutile(
            output.device,
            (
                output.shape[1] // 256,
                (output.shape[0] + DW_TILE_M - 1) // DW_TILE_M,
                1,
            ),
            _matmul_tn_n4_kernel,
            (a, b, output, grad_output, DW_TILE_M, 64, DW_TILE_K, 5),
        )
        return

    tile_n = min(256, _next_power_of_2(output.shape[1]))
    _launch_cutile(
        output.device,
        (
            (output.shape[0] + DW_TILE_M - 1) // DW_TILE_M,
            (output.shape[1] + tile_n - 1) // tile_n,
            1,
        ),
        _matmul_tn_kernel,
        (a, b, output, grad_output, DW_TILE_M, tile_n, DW_TILE_K),
    )


def _launch_cutile(device: torch.device, grid, kernel, args) -> None:
    with torch.cuda.device(device):
        ct.launch(torch.cuda.current_stream(device), grid, kernel, args)


def _reject_unsupported(
    bias,
    ce_weight,
    lse_square_scale,
    label_smoothing,
    softcap,
    return_z_loss,
    accum_dtype,
    use_token_scaling,
    return_token_accuracy,
    return_predicted_tokens,
) -> None:
    unsupported = {
        "bias": bias is not None,
        "class weights": ce_weight is not None,
        "z-loss": bool(lse_square_scale),
        "label smoothing": bool(label_smoothing),
        "softcap": softcap is not None,
        "return_z_loss": return_z_loss,
        "accum_dtype": accum_dtype is not None,
        "token scaling": use_token_scaling,
        "return_token_accuracy": return_token_accuracy,
        "return_predicted_tokens": return_predicted_tokens,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise NotImplementedError(f"cuTile SM90 FLCE does not support: {', '.join(enabled)}")


def _validate_inputs(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(_input.device) != (9, 0):
        raise RuntimeError("cuTile FLCE requires a Hopper (compute capability 9.0) GPU")
    if _input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("cuTile FLCE supports BF16 input and weight only")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M, H], weight[V, H], and target[M]")
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"incompatible input {tuple(_input.shape)}, weight {tuple(weight.shape)}, and target {tuple(target.shape)}"
        )
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    if not _input.is_contiguous() or not weight.is_contiguous() or not target.is_contiguous():
        raise ValueError("cuTile FLCE requires contiguous input, weight, and target tensors")
    num_stats_partitions = (weight.shape[0] + LOGITS_STATS_BLOCK_SIZE - 1) // LOGITS_STATS_BLOCK_SIZE
    if num_stats_partitions > MAX_STATS_BLOCK_SIZE:
        raise NotImplementedError(
            f"cuTile FLCE supports at most {MAX_STATS_BLOCK_SIZE * LOGITS_STATS_BLOCK_SIZE} vocabulary entries"
        )


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    _reject_unsupported(
        bias,
        ce_weight,
        lse_square_scale,
        label_smoothing,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    _validate_inputs(_input, weight, target, reduction)

    tokens, hidden_size = _input.shape
    vocab_size = weight.shape[0]
    num_vocab_tiles = (vocab_size + LOGITS_STATS_BLOCK_SIZE - 1) // LOGITS_STATS_BLOCK_SIZE
    stats_block_size = _next_power_of_2(num_vocab_tiles)
    needs_dz = _input.requires_grad or weight.requires_grad
    logits = torch.empty((tokens, vocab_size), dtype=torch.bfloat16, device=_input.device)
    loss_1d = torch.empty(tokens, dtype=torch.float32, device=_input.device)
    partial_max = torch.empty((tokens, num_vocab_tiles), dtype=torch.float32, device=_input.device)
    partial_sum = torch.empty_like(partial_max)
    completion_count = torch.zeros(tokens, dtype=torch.int32, device=_input.device)
    target_mask = target != ignore_index
    valid_target = ~target_mask | ((target >= 0) & (target < vocab_size))
    torch._assert_async(valid_target.all(), f"target values must be in [0, {vocab_size}) or equal ignore_index")
    loss_scale = (
        target_mask.sum().clamp_min(1).to(torch.float32).reciprocal()
        if reduction == "mean"
        else torch.empty((), dtype=torch.float32, device=_input.device)
    )

    torch.mm(_input, weight.t(), out=logits)
    _launch_cutile(
        logits.device,
        (num_vocab_tiles * tokens, 1, 1),
        _fused_cross_entropy_dz_kernel,
        (
            logits,
            target,
            loss_1d,
            loss_scale,
            partial_max,
            partial_sum,
            completion_count,
            vocab_size,
            num_vocab_tiles,
            ignore_index,
            LOGITS_STATS_BLOCK_SIZE,
            CE_BLOCK_SIZE,
            stats_block_size,
            needs_dz,
            reduction == "mean",
        ),
    )

    grad_input = None
    if _input.requires_grad:
        grad_input = torch.empty((tokens, hidden_size), dtype=_input.dtype, device=_input.device)
        torch.mm(logits, weight, out=grad_input)

    loss = loss_1d.sum()
    gradient_scale = loss_scale if reduction == "mean" else torch.ones((), dtype=torch.float32, device=_input.device)
    return loss, grad_input, logits if weight.requires_grad else None, gradient_scale


def fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_logits: Optional[torch.Tensor],
    _input: Optional[torch.Tensor],
    gradient_scale: torch.Tensor,
    weight_shape: torch.Size,
    weight_dtype: torch.dtype,
):
    total_scale = grad_output * gradient_scale
    if grad_input is not None:
        grad_input = grad_input * total_scale

    grad_weight = None
    if grad_logits is not None:
        grad_weight = torch.empty(weight_shape, dtype=weight_dtype, device=grad_logits.device)
        _launch_matmul_tn(grad_logits, _input, grad_weight, total_scale)

    return grad_input, grad_weight


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
        accum_dtype=None,
        use_token_scaling=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    ):
        loss, grad_input, grad_logits, gradient_scale = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            ce_weight=ce_weight,
            bias=bias,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
        )

        ctx.save_for_backward(
            grad_input,
            grad_logits,
            _input.detach() if weight.requires_grad else None,
            gradient_scale,
        )
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return loss, None, None, None

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        del grad_output2, grad_output3, grad_output4
        grad_input, grad_logits, _input, gradient_scale = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_cross_entropy_backward(
            grad_output,
            grad_input,
            grad_logits,
            _input,
            gradient_scale,
            ctx.weight_shape,
            ctx.weight_dtype,
        )
        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


__all__ = [
    "LigerFusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy_backward",
    "fused_linear_cross_entropy_forward",
]
