"""CuTile tensor-parallel fused linear cross entropy for Megatron."""

from __future__ import annotations

import math

import cuda.tile as ct
import torch
import torch.distributed as dist

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.megatron_fused_linear_cross_entropy import _tp_rank_and_world

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634
MAX_ROW_BLOCK_SIZE = 4096


@ct.function
def _matmul_body(
    a,
    b,
    bias,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    HAS_BIAS: ConstBool,
    SWIZZLE: ConstBool,
):
    num_m_tiles = ct.num_tiles(a, axis=0, shape=(TILE_M, TILE_K))
    num_n_tiles = ct.num_tiles(b, axis=1, shape=(TILE_K, TILE_N))
    num_k_tiles = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))
    block = ct.bid(0)
    if SWIZZLE:
        group_size_m = 8
        blocks_per_group = group_size_m * num_n_tiles
        group = block // blocks_per_group
        first_tile_m = group * group_size_m
        active_group_size_m = min(num_m_tiles - first_tile_m, group_size_m)
        tile_m = first_tile_m + (block % active_group_size_m)
        tile_n = (block % blocks_per_group) // active_group_size_m
    else:
        tile_m = block // num_n_tiles
        tile_n = block % num_n_tiles

    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    for tile_k in range(num_k_tiles):
        a_tile = ct.load(
            a,
            index=(tile_m, tile_k),
            shape=(TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b,
            index=(tile_k, tile_n),
            shape=(TILE_K, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    if HAS_BIAS:
        bias_tile = ct.load(
            bias,
            index=(tile_n,),
            shape=(TILE_N,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        accumulator = accumulator + ct.astype(bias_tile, ct.float32)

    ct.store(output, index=(tile_m, tile_n), tile=ct.astype(accumulator, output.dtype))


@ct.kernel(num_ctas=1)
def _matmul_1cta_kernel(
    a,
    b,
    bias,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    HAS_BIAS: ConstBool,
    SWIZZLE: ConstBool,
):
    _matmul_body(a, b, bias, output, TILE_M, TILE_N, TILE_K, HAS_BIAS, SWIZZLE)


@ct.kernel(num_ctas=2)
def _matmul_2cta_kernel(
    a,
    b,
    bias,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    HAS_BIAS: ConstBool,
    SWIZZLE: ConstBool,
):
    _matmul_body(a, b, bias, output, TILE_M, TILE_N, TILE_K, HAS_BIAS, SWIZZLE)


@ct.kernel(occupancy=4)
def _row_max_kernel(
    input,
    output,
    n_cols,
    BLOCK_SIZE: ConstInt,
):
    row = ct.bid(0)
    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_max_tile = ct.full((1,), -math.inf, dtype=ct.float32)

    for chunk in range(num_chunks):
        columns = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk * BLOCK_SIZE
        values = ct.astype(
            ct.gather(
                input,
                (row, columns),
                check_bounds=True,
                padding_value=-math.inf,
                latency=3,
            ),
            ct.float32,
        )
        row_max = ct.maximum(
            ct.sum(row_max_tile, 0, keepdims=False),
            ct.max(values, 0, keepdims=False),
        )
        row_max_tile = ct.full((1,), row_max, dtype=ct.float32)

    ct.scatter(output, row, ct.sum(row_max_tile, 0, keepdims=False))


@ct.kernel(occupancy=4)
def _vocab_parallel_ce_forward_kernel(
    logits,
    logits_max,
    target,
    predicted_logit,
    sum_exp,
    vocab_start,
    n_cols,
    ignore_index,
    BLOCK_SIZE: ConstInt,
):
    row = ct.bid(0)
    y_global = ct.load(target, row, shape=())
    maximum = ct.astype(ct.load(logits_max, row, shape=()), ct.float32)
    is_ignored = y_global == ignore_index
    target_off_rank = (y_global < vocab_start) or (y_global >= vocab_start + n_cols)
    y_local = ct.astype(y_global - vocab_start, ct.int32)

    if is_ignored or target_off_rank:
        predicted = 0.0
    else:
        target_index = ct.add(ct.arange(1, dtype=ct.int32), y_local)
        target_tile = ct.gather(logits, (row, target_index), check_bounds=False)
        predicted = ct.sum(ct.astype(target_tile, ct.float32), 0, keepdims=False) - maximum

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_exp_tile = ct.full((1,), 0.0, dtype=ct.float32)
    for chunk in range(num_chunks):
        columns = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk * BLOCK_SIZE
        in_bounds = columns < n_cols
        values = ct.astype(
            ct.gather(
                logits,
                (row, columns),
                check_bounds=True,
                padding_value=-math.inf,
                latency=3,
            ),
            ct.float32,
        )
        exponentials = ct.exp2((values - maximum) * LOG2E, flush_to_zero=True)
        exponentials = ct.where(in_bounds, exponentials, 0.0)
        running_sum = ct.sum(sum_exp_tile, 0, keepdims=False)
        sum_exp_tile = ct.full(
            (1,),
            running_sum + ct.sum(exponentials, 0, keepdims=False),
            dtype=ct.float32,
        )
        ct.scatter(
            logits,
            (row, columns),
            ct.astype(exponentials, logits.dtype),
            check_bounds=True,
        )

    ct.scatter(predicted_logit, row, predicted)
    ct.scatter(sum_exp, row, ct.sum(sum_exp_tile, 0, keepdims=False))


@ct.kernel(occupancy=4)
def _vocab_parallel_ce_backward_kernel(
    exp_buffer,
    sum_exp,
    target,
    grad_output,
    vocab_start,
    n_cols,
    ignore_index,
    BLOCK_SIZE: ConstInt,
):
    row = ct.bid(0)
    y_global = ct.load(target, row, shape=())
    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    if y_global == ignore_index:
        for chunk in range(num_chunks):
            columns = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk * BLOCK_SIZE
            zeros = ct.full((BLOCK_SIZE,), 0.0, dtype=exp_buffer.dtype)
            ct.scatter(exp_buffer, (row, columns), zeros, check_bounds=True)
        return

    target_off_rank = (y_global < vocab_start) or (y_global >= vocab_start + n_cols)
    y_local = ct.astype(y_global - vocab_start, ct.int32)
    global_sum = ct.astype(ct.load(sum_exp, row, shape=()), ct.float32)
    upstream = ct.astype(ct.load(grad_output, row, shape=()), ct.float32)

    for chunk in range(num_chunks):
        columns = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk * BLOCK_SIZE
        exponentials = ct.astype(
            ct.gather(exp_buffer, (row, columns), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        gradient = exponentials / global_sum
        if not target_off_rank:
            gradient = ct.where(columns == y_local, gradient - 1.0, gradient)
        gradient = gradient * upstream
        ct.scatter(
            exp_buffer,
            (row, columns),
            ct.astype(gradient, exp_buffer.dtype),
            check_bounds=True,
        )


@ct.kernel(occupancy=4)
def _loss_kernel(
    sum_exp,
    predicted_logit,
    target,
    output,
    ignore_index,
):
    row = ct.bid(0)
    y_global = ct.load(target, row, shape=())
    if y_global == ignore_index:
        loss = 0.0
    else:
        denominator = ct.astype(ct.load(sum_exp, row, shape=()), ct.float32)
        predicted = ct.astype(ct.load(predicted_logit, row, shape=()), ct.float32)
        loss = ct.log(denominator) - predicted
    ct.scatter(output, row, loss)


@ct.kernel(occupancy=4)
def _column_sum_kernel(
    input,
    output,
    n_rows,
    BLOCK_SIZE: ConstInt,
):
    column = ct.bid(0)
    num_chunks = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_tile = ct.full((1,), 0.0, dtype=ct.float32)

    for chunk in range(num_chunks):
        rows = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk * BLOCK_SIZE
        values = ct.astype(
            ct.gather(input, (rows, column), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        running_total = ct.sum(total_tile, 0, keepdims=False)
        total_tile = ct.full(
            (1,),
            running_total + ct.sum(values, 0, keepdims=False),
            dtype=ct.float32,
        )

    ct.scatter(output, column, ct.astype(ct.sum(total_tile, 0, keepdims=False), output.dtype))


def _select_row_block_size(size: int) -> int:
    return min(MAX_ROW_BLOCK_SIZE, _next_power_of_2(size))


def _cutile_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    operation: str,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul expects [M, K] @ [K, N], got {tuple(a.shape)} and {tuple(b.shape)}.")

    output = torch.empty(
        (a.shape[0], b.shape[1]),
        device=a.device,
        dtype=output_dtype or a.dtype,
    )
    if operation == "projection":
        kernel, tile = (
            (_matmul_2cta_kernel, (256, 256, 128)) if a.shape[0] <= 1024 else (_matmul_2cta_kernel, (512, 256, 64))
        )
    elif operation == "dx":
        if a.shape[0] <= 1024:
            kernel, tile = _matmul_1cta_kernel, (128, 128, 64)
        elif a.shape[1] > 16000 or a.shape[0] > 16384:
            kernel, tile = _matmul_2cta_kernel, (512, 256, 64)
        else:
            kernel, tile = _matmul_1cta_kernel, (256, 256, 64)
    elif operation == "dw":
        if a.shape[1] >= 16384 and (a.shape[0] > 16000 or a.shape[1] == 16384):
            kernel, tile = _matmul_2cta_kernel, (512, 256, 64)
        else:
            use_single_cta = a.shape[1] <= 1024 or a.shape[0] > 16000 or a.shape[1] > 16384
            kernel = _matmul_1cta_kernel if use_single_cta else _matmul_2cta_kernel
            tile = (256, 256, 64)
    else:
        raise ValueError(f"unknown FLCE GEMM operation: {operation!r}.")

    tile_m, tile_n, tile_k = tile
    swizzle = operation == "projection" and b.shape[1] > 16000
    grid = (
        ct.cdiv(a.shape[0], tile_m) * ct.cdiv(b.shape[1], tile_n),
        1,
        1,
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            a,
            b,
            bias if bias is not None else output,
            output,
            tile_m,
            tile_n,
            tile_k,
            bias is not None,
            swizzle,
        ),
    )
    return output


def _cutile_row_max(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty(input.shape[0], device=input.device, dtype=torch.float32)
    block_size = 16384 if input.shape[1] > 16384 else _select_row_block_size(input.shape[1])
    ct.launch(
        torch.cuda.current_stream(),
        (input.shape[0], 1, 1),
        _row_max_kernel,
        (input, output, int(input.shape[1]), int(block_size)),
    )
    return output


def _cutile_ce_forward(
    logits: torch.Tensor,
    logits_max: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab_local = logits.shape
    stats = torch.empty((2, rows), device=logits.device, dtype=torch.float32)
    predicted_logit = stats[0]
    sum_exp = stats[1]
    block_size = 16384 if vocab_local > 16384 else _select_row_block_size(vocab_local)
    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        _vocab_parallel_ce_forward_kernel,
        (
            logits,
            logits_max,
            target,
            predicted_logit,
            sum_exp,
            int(vocab_start),
            int(vocab_local),
            int(ignore_index),
            int(block_size),
        ),
    )
    return logits, stats


def _cutile_ce_backward(
    exp_buffer: torch.Tensor,
    sum_exp: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
) -> None:
    block_size = min(2048, _select_row_block_size(exp_buffer.shape[1]))
    ct.launch(
        torch.cuda.current_stream(),
        (exp_buffer.shape[0], 1, 1),
        _vocab_parallel_ce_backward_kernel,
        (
            exp_buffer,
            sum_exp,
            target,
            grad_output,
            int(vocab_start),
            int(exp_buffer.shape[1]),
            int(ignore_index),
            int(block_size),
        ),
    )


def _cutile_loss(
    sum_exp: torch.Tensor,
    predicted_logit: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    output = torch.empty_like(sum_exp)
    ct.launch(
        torch.cuda.current_stream(),
        (target.numel(), 1, 1),
        _loss_kernel,
        (sum_exp, predicted_logit, target, output, int(ignore_index)),
    )
    return output


def _cutile_column_sum(input: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    output = torch.empty(input.shape[1], device=input.device, dtype=output_dtype)
    block_size = _select_row_block_size(input.shape[0])
    ct.launch(
        torch.cuda.current_stream(),
        (input.shape[1], 1, 1),
        _column_sum_kernel,
        (input, output, int(input.shape[0]), int(block_size)),
    )
    return output


def _materialized_backward(ctx, grad_output: torch.Tensor):
    hidden, weight, exp_buffer, sum_exp, target = ctx.saved_tensors
    grad_output_1d = grad_output.contiguous().reshape(-1).float()
    _cutile_ce_backward(
        exp_buffer,
        sum_exp,
        target,
        grad_output_1d,
        ctx.vocab_start,
        ctx.ignore_index,
    )

    grad_hidden = _cutile_matmul(
        exp_buffer,
        weight,
        operation="dx",
        output_dtype=torch.float32 if exp_buffer.shape[0] <= 1024 else None,
    )
    reduce_work = (
        dist.all_reduce(
            grad_hidden,
            op=dist.ReduceOp.SUM,
            group=ctx.tp_group,
            async_op=True,
        )
        if ctx.tp_world > 1
        else None
    )
    grad_weight = _cutile_matmul(exp_buffer.t(), hidden, operation="dw")
    grad_bias = _cutile_column_sum(exp_buffer, ctx.bias_dtype) if ctx.has_bias else None

    if reduce_work is not None:
        reduce_work.wait()
    grad_hidden = grad_hidden.to(ctx.hidden_dtype).reshape(ctx.original_hidden_shape)
    return grad_hidden, grad_weight, grad_bias


class LigerMegatronFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Hidden-to-loss tensor-parallel FLCE using CuTile local kernels."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None,
        tp_group,
        ignore_index: int,
    ) -> torch.Tensor:
        if hidden.ndim < 2:
            raise ValueError(f"hidden must have at least 2 dimensions, got shape {tuple(hidden.shape)}.")
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D [V_local, H], got shape {tuple(weight.shape)}.")
        if tuple(target.shape) != tuple(hidden.shape[:-1]):
            raise ValueError(
                f"target shape must equal hidden.shape[:-1]; got target={tuple(target.shape)}, "
                f"hidden={tuple(hidden.shape)}."
            )
        if hidden.shape[-1] != weight.shape[1]:
            raise ValueError(f"hidden size mismatch: hidden has H={hidden.shape[-1]}, weight has H={weight.shape[1]}.")
        if hidden.dtype != weight.dtype:
            raise TypeError(f"hidden and weight must have the same dtype, got {hidden.dtype} and {weight.dtype}.")
        if hidden.device != weight.device or hidden.device != target.device:
            raise ValueError("hidden, weight, and target must be on the same device.")
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
                raise ValueError(f"bias must have shape ({weight.shape[0]},), got {tuple(bias.shape)}.")
            if bias.device != hidden.device or bias.dtype != hidden.dtype:
                raise TypeError("bias must have the same device and dtype as hidden.")
        if hidden.device.type != "cuda" or hidden.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("CuTile Megatron FLCE requires a CUDA GPU and float16 or bfloat16 inputs.")

        tp_rank, tp_world = _tp_rank_and_world(tp_group)
        vocab_local = weight.shape[0]
        vocab_global = vocab_local * tp_world
        vocab_start = tp_rank * vocab_local

        flat_target = target.reshape(-1).to(torch.int64).contiguous()
        valid = flat_target != ignore_index
        invalid = valid & ((flat_target < 0) | (flat_target >= vocab_global))
        valid_targets = ~torch.any(invalid)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(valid_targets, f"non-ignored targets must be in [0, {vocab_global}).")
        elif not valid_targets.item():
            raise ValueError(f"non-ignored targets must be in [0, {vocab_global}).")

        original_hidden_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        weight_2d = weight.contiguous()
        bias_1d = bias.contiguous() if bias is not None else None

        logits = _cutile_matmul(hidden_2d, weight_2d.t(), operation="projection", bias=bias_1d)
        logits_max = _cutile_row_max(logits)
        if tp_world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        exp_buffer, stats = _cutile_ce_forward(
            logits,
            logits_max,
            flat_target,
            vocab_start,
            ignore_index,
        )
        if tp_world > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)
        predicted_logit = stats[0]
        sum_exp = stats[1]

        loss = _cutile_loss(sum_exp, predicted_logit, flat_target, ignore_index)

        ctx.save_for_backward(hidden_2d, weight_2d, exp_buffer, sum_exp, flat_target)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.tp_group = tp_group
        ctx.tp_world = tp_world
        ctx.vocab_start = vocab_start
        ctx.ignore_index = ignore_index
        ctx.original_hidden_shape = original_hidden_shape
        ctx.hidden_dtype = hidden.dtype
        return loss.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_hidden, grad_weight, grad_bias = _materialized_backward(ctx, grad_output)
        return grad_hidden, grad_weight, None, grad_bias, None, None


def liger_megatron_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    tp_group=None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute Megatron FLCE with CuTile local kernels and NCCL TP collectives."""
    return LigerMegatronFusedLinearCrossEntropyFunction.apply(
        hidden,
        weight,
        target,
        bias,
        tp_group,
        ignore_index,
    )
