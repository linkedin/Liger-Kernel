import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

MAX_FUSED_SIZE = 16384
MAX_FUSED_FORWARD_SIZE = 4096
GRID_OVERSUB_FACTOR = 8
REDUCE_BATCH_CHUNK = 32
MAX_GRID_SIZE = 65535


def _select_reduce_block_size(hidden_size: int) -> int:
    if hidden_size >= 2048:
        return 1024
    if hidden_size >= 1024:
        return 512
    if hidden_size >= 512:
        return 256
    return 128


def _group_norm_forward_stats_block_size(hidden_size: int, element_size: int) -> int:
    vv_alignment = 32
    required = vv_alignment // element_size
    block_h = _select_reduce_block_size(hidden_size)
    if hidden_size <= MAX_FUSED_SIZE:
        block_h = max(block_h, triton.next_power_of_2(hidden_size))
    block_h = min(block_h, 1024)
    block_h = max(block_h, required)
    return block_h


def _group_norm_forward_stats_batch_block_size(hidden_size: int) -> int:
    block_h = min(1024, triton.next_power_of_2(hidden_size))
    return max(1, min(8, 4096 // block_h))


def _group_norm_forward_affine_spatial_block_size(hidden_size_per_channel: int) -> int:
    return min(256, max(32, triton.next_power_of_2(hidden_size_per_channel)))


def _group_norm_forward_affine_channel_block_size(channels_per_group: int, block_h: int) -> int:
    max_block_ch = min(8, triton.next_power_of_2(channels_per_group))
    target_block_ch = max(1, 1024 // block_h)
    block_ch = 1
    while block_ch * 2 <= max_block_ch and block_ch * 2 <= target_block_ch:
        block_ch *= 2
    return block_ch


def _group_norm_forward_affine_batch_block_size(block_h: int, block_ch: int) -> int:
    tile_elems = block_h * block_ch
    return max(1, min(8, 2048 // tile_elems))


def _group_norm_forward_launch_config(num_tiles: int, batch_size: int, block_batch: int) -> tuple[int, int]:
    batch_blocks = triton.cdiv(batch_size, block_batch)
    num_cores = get_npu_core_count()

    if num_tiles * batch_blocks >= num_cores * GRID_OVERSUB_FACTOR:
        return block_batch, batch_blocks

    target_batch_blocks = min(batch_size, max(batch_blocks, triton.cdiv(num_cores * GRID_OVERSUB_FACTOR, num_tiles)))
    block_batch = max(1, triton.cdiv(batch_size, target_batch_blocks))
    return block_batch, triton.cdiv(batch_size, block_batch)


def _group_norm_backward_spatial_block_size(hidden_size_per_channel: int) -> int:
    return min(1024, triton.next_power_of_2(hidden_size_per_channel))


def _group_norm_backward_batch_block_size(hidden_size_per_channel: int) -> int:
    block_h = _group_norm_backward_spatial_block_size(hidden_size_per_channel)
    return max(1, min(8, 4096 // block_h))


# tl.arange(0, BLOCK_BATCH) and register pressure — stay within a typical Triton tile bound.
_BACKWARD_DX_DWDB_MAX_BLOCK_BATCH = 1024


def _group_norm_backward_dx_dwdb_launch_config(
    batch_size: int,
    num_groups: int,
    hidden_size_per_channel: int,
) -> tuple[int, int]:
    """
    Ascend Triton requires the launch grid to stay below 65536 blocks (product of grid dims).
    This kernel uses grid (num_groups, num_partial_rows) with num_partial_rows = cdiv(batch_size, block_batch).
    When batch is large and block_batch is tiny (default ≤ 8), the product overflows; raise block_batch as needed.
    """
    if num_groups > MAX_GRID_SIZE:
        raise RuntimeError(f"group_norm backward: num_groups={num_groups} exceeds Ascend grid limit {MAX_GRID_SIZE}")

    block_batch = _group_norm_backward_batch_block_size(hidden_size_per_channel)
    max_partial_rows = max(1, MAX_GRID_SIZE // num_groups)
    needed_block_batch = triton.cdiv(batch_size, max_partial_rows)
    block_batch = max(block_batch, needed_block_batch)
    block_batch = min(block_batch, batch_size, _BACKWARD_DX_DWDB_MAX_BLOCK_BATCH)

    num_partial_rows = triton.cdiv(batch_size, block_batch)
    while num_groups * num_partial_rows > MAX_GRID_SIZE and block_batch < batch_size:
        block_batch = min(batch_size, block_batch * 2)
        num_partial_rows = triton.cdiv(batch_size, block_batch)

    if num_groups * num_partial_rows > MAX_GRID_SIZE:
        block_batch = batch_size
        num_partial_rows = 1

    return block_batch, num_partial_rows


@triton.jit
def _group_norm_forward_single_task_kernel(
    Y_ptr,
    X_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    num_groups,
    total_tasks,
    hidden_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    BLOCK_CH: tl.constexpr,
):
    hidden_size_per_channel: tl.constexpr = hidden_size // channels_per_group
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    grid_stride = num_progs

    ch_offsets = tl.arange(0, BLOCK_CH)
    h_offsets = tl.arange(0, BLOCK_H)
    ch_mask = ch_offsets < channels_per_group
    h_mask = h_offsets < hidden_size_per_channel
    mask = ch_mask[:, None] & h_mask[None, :]

    for task_iter in range(0, total_tasks, grid_stride):
        pid_task = task_iter + pid

        pid_group = pid_task % num_groups
        task_base = pid_task * hidden_size
        ptrs = task_base + ch_offsets[:, None] * hidden_size_per_channel + h_offsets[None, :]

        X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_acc = tl.sum(tl.sum(X_vals, axis=1), axis=0)
        sum_sq_acc = tl.sum(tl.sum(X_vals * X_vals, axis=1), axis=0)

        mean = sum_acc / hidden_size
        variance = tl.maximum(sum_sq_acc / hidden_size - mean * mean, 0.0)
        rstd = rsqrt(variance + eps)

        group_ch_start = pid_group * channels_per_group
        W_vals = tl.load(W_ptr + group_ch_start + ch_offsets, mask=ch_mask, other=1.0).to(tl.float32)
        B_vals = tl.load(B_ptr + group_ch_start + ch_offsets, mask=ch_mask, other=0.0).to(tl.float32)
        Y_vals = (X_vals - mean) * rstd
        Y_vals = Y_vals * W_vals[:, None] + B_vals[:, None]
        tl.store(Y_ptr + ptrs, Y_vals.to(Y_ptr.dtype.element_ty), mask=mask)

        tl.store(Mean_ptr + pid_task, mean)
        tl.store(RSTD_ptr + pid_task, rstd)


@triton.jit
def _group_norm_forward_stats_kernel(
    X_ptr,
    Mean_ptr,
    RSTD_ptr,
    batch_size,
    num_groups,
    hidden_size: tl.constexpr,
    eps,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_group = tl.program_id(0)
    pid_batch_block = tl.program_id(1)
    batch_offsets = pid_batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    task_offsets = batch_offsets * num_groups + pid_group
    inv_hidden_size = 1.0 / hidden_size

    sum_acc = tl.zeros([BLOCK_BATCH], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_BATCH], dtype=tl.float32)
    h_offsets = tl.arange(0, BLOCK_SIZE_H)

    for start in tl.range(0, hidden_size, BLOCK_SIZE_H):
        idx = start + h_offsets
        h_mask = idx < hidden_size
        mask = batch_mask[:, None] & h_mask[None, :]
        ptrs = task_offsets[:, None] * hidden_size + idx[None, :]
        X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_acc += tl.sum(X_vals, axis=1)
        sum_sq_acc += tl.sum(X_vals * X_vals, axis=1)

    mean = sum_acc * inv_hidden_size
    variance = tl.maximum(sum_sq_acc * inv_hidden_size - mean * mean, 0.0)
    rstd = rsqrt(variance + eps)
    tl.store(Mean_ptr + task_offsets, mean, mask=batch_mask)
    tl.store(RSTD_ptr + task_offsets, rstd, mask=batch_mask)


@triton.jit
def _group_norm_forward_affine_kernel(
    Y_ptr,
    X_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    batch_size,
    num_groups,
    hidden_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    channel_blocks,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    hidden_size_per_channel: tl.constexpr = hidden_size // channels_per_group
    pid_tile = tl.program_id(0)
    pid_batch_block = tl.program_id(1)
    pid_group = pid_tile // channel_blocks
    pid_channel_block = pid_tile - pid_group * channel_blocks

    batch_offsets = pid_batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    task_offsets = batch_offsets * num_groups + pid_group
    mean = tl.load(Mean_ptr + task_offsets, mask=batch_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + task_offsets, mask=batch_mask, other=0.0).to(tl.float32)

    ch_offsets = pid_channel_block * BLOCK_SIZE_CH + tl.arange(0, BLOCK_SIZE_CH)
    ch_mask = ch_offsets < channels_per_group
    group_ch_offsets = pid_group * channels_per_group + ch_offsets
    W_vals = tl.load(W_ptr + group_ch_offsets, mask=ch_mask, other=1.0).to(tl.float32)
    B_vals = tl.load(B_ptr + group_ch_offsets, mask=ch_mask, other=0.0).to(tl.float32)
    h_offsets = tl.arange(0, BLOCK_SIZE_H)

    for start in tl.range(0, hidden_size_per_channel, BLOCK_SIZE_H):
        idx = start + h_offsets
        h_mask = idx < hidden_size_per_channel
        mask = batch_mask[:, None, None] & ch_mask[None, :, None] & h_mask[None, None, :]
        ptrs = (
            task_offsets[:, None, None] * hidden_size
            + ch_offsets[None, :, None] * hidden_size_per_channel
            + idx[None, None, :]
        )
        X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        Y_vals = (X_vals - mean[:, None, None]) * rstd[:, None, None]
        Y_vals = Y_vals * W_vals[None, :, None] + B_vals[None, :, None]
        tl.store(Y_ptr + ptrs, Y_vals.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _group_norm_backward_dx_dwdb_kernel(
    X_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    DX_ptr,
    DW_partial_ptr,
    DB_partial_ptr,
    DY_ptr,
    batch_size,
    num_groups,
    hidden_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    hidden_size_per_channel: tl.constexpr = hidden_size // channels_per_group
    pid_group = tl.program_id(0)
    pid_batch_block = tl.program_id(1)

    batch_offsets = pid_batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    task_offsets = batch_offsets * num_groups + pid_group
    h_offsets = tl.arange(0, BLOCK_H)

    mean = tl.load(Mean_ptr + task_offsets, mask=batch_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + task_offsets, mask=batch_mask, other=0.0).to(tl.float32)

    inv_hidden_size = 1.0 / hidden_size
    group_ch_start = pid_group * channels_per_group

    c1 = tl.zeros([BLOCK_BATCH], dtype=tl.float32)
    c2 = tl.zeros([BLOCK_BATCH], dtype=tl.float32)

    for local_ch in range(0, channels_per_group):
        channel_base = local_ch * hidden_size_per_channel
        W = tl.load(W_ptr + group_ch_start + local_ch).to(tl.float32)
        dW_acc = 0.0
        dB_acc = 0.0

        for start in range(0, hidden_size_per_channel, BLOCK_H):
            idx = start + h_offsets
            h_mask = idx < hidden_size_per_channel
            mask = batch_mask[:, None] & h_mask[None, :]
            ptrs = task_offsets[:, None] * hidden_size + channel_base + idx[None, :]

            X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            DY_vals = tl.load(DY_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

            x_hat = (X_vals - mean[:, None]) * rstd[:, None]
            wdy = W * DY_vals

            dW_acc += tl.sum(tl.sum(tl.where(mask, DY_vals * x_hat, 0.0), axis=1), axis=0)
            dB_acc += tl.sum(tl.sum(tl.where(mask, DY_vals, 0.0), axis=1), axis=0)
            c1 += tl.sum(tl.where(mask, x_hat * wdy, 0.0), axis=1)
            c2 += tl.sum(tl.where(mask, wdy, 0.0), axis=1)

        partial_offset = pid_batch_block * (num_groups * channels_per_group) + group_ch_start + local_ch
        tl.store(DW_partial_ptr + partial_offset, dW_acc)
        tl.store(DB_partial_ptr + partial_offset, dB_acc)

    c1 *= inv_hidden_size
    c2 *= inv_hidden_size

    for local_ch in range(0, channels_per_group):
        channel_base = local_ch * hidden_size_per_channel
        W = tl.load(W_ptr + group_ch_start + local_ch).to(tl.float32)

        for start in range(0, hidden_size_per_channel, BLOCK_H):
            idx = start + h_offsets
            h_mask = idx < hidden_size_per_channel
            mask = batch_mask[:, None] & h_mask[None, :]
            ptrs = task_offsets[:, None] * hidden_size + channel_base + idx[None, :]

            X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            DY_vals = tl.load(DY_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            x_hat = (X_vals - mean[:, None]) * rstd[:, None]
            DX_vals = (W * DY_vals - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX_ptr + ptrs, DX_vals.to(DX_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _group_norm_reduce_param_grads_kernel(
    DW_partial_ptr,
    DB_partial_ptr,
    DW_ptr,
    DB_ptr,
    num_partial_rows,
    num_channels,
    BLOCK_SIZE_CH: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    ch_offsets = pid * BLOCK_SIZE_CH + tl.arange(0, BLOCK_SIZE_CH)
    ch_mask = ch_offsets < num_channels
    row_offsets = tl.arange(0, BLOCK_SIZE_ROWS)

    dW_acc = tl.zeros([BLOCK_SIZE_CH], dtype=tl.float32)
    dB_acc = tl.zeros([BLOCK_SIZE_CH], dtype=tl.float32)

    for row_start in range(0, num_partial_rows, BLOCK_SIZE_ROWS):
        rows = row_start + row_offsets
        row_mask = rows < num_partial_rows
        mask = row_mask[:, None] & ch_mask[None, :]
        ptrs = rows[:, None] * num_channels + ch_offsets[None, :]
        dW_vals = tl.load(DW_partial_ptr + ptrs, mask=mask, other=0.0)
        dB_vals = tl.load(DB_partial_ptr + ptrs, mask=mask, other=0.0)
        dW_acc += tl.sum(dW_vals, axis=0)
        dB_acc += tl.sum(dB_vals, axis=0)

    tl.store(DW_ptr + ch_offsets, dW_acc, mask=ch_mask)
    tl.store(DB_ptr + ch_offsets, dB_acc, mask=ch_mask)


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups

    X = X.view(batch_size, num_groups, -1)
    hidden_size = X.shape[-1]
    hidden_size_per_channel = hidden_size // channels_per_group
    Y = torch.empty_like(X)
    total_tasks = batch_size * num_groups
    Mean = torch.empty(total_tasks, dtype=torch.float32, device=X.device)
    RSTD = torch.empty(total_tasks, dtype=torch.float32, device=X.device)

    if hidden_size <= MAX_FUSED_FORWARD_SIZE:
        block_h = _group_norm_backward_spatial_block_size(hidden_size_per_channel)
        block_ch = triton.next_power_of_2(channels_per_group)
        num_cores = get_npu_core_count()
        grid_size = min(total_tasks, num_cores * 2, MAX_GRID_SIZE)
        _group_norm_forward_single_task_kernel[(grid_size,)](
            Y,
            X,
            Mean,
            RSTD,
            W,
            B,
            num_groups,
            total_tasks,
            hidden_size,
            channels_per_group,
            eps,
            BLOCK_H=block_h,
            BLOCK_CH=block_ch,
        )
        return Y.view(*shape), X.view(*shape), Mean.view(batch_size, num_groups), RSTD.view(batch_size, num_groups)

    stats_block_h = _group_norm_forward_stats_block_size(hidden_size, X.element_size())
    stats_block_batch = _group_norm_forward_stats_batch_block_size(hidden_size)
    stats_block_batch, stats_grid_batch = _group_norm_forward_launch_config(num_groups, batch_size, stats_block_batch)
    _group_norm_forward_stats_kernel[(num_groups, stats_grid_batch)](
        X,
        Mean,
        RSTD,
        batch_size,
        num_groups,
        hidden_size,
        eps,
        BLOCK_SIZE_H=stats_block_h,
        BLOCK_BATCH=stats_block_batch,
    )

    affine_block_h = _group_norm_forward_affine_spatial_block_size(hidden_size_per_channel)
    affine_block_ch = _group_norm_forward_affine_channel_block_size(channels_per_group, affine_block_h)
    channel_blocks = triton.cdiv(channels_per_group, affine_block_ch)
    affine_block_batch = _group_norm_forward_affine_batch_block_size(affine_block_h, affine_block_ch)
    affine_block_batch, affine_grid_batch = _group_norm_forward_launch_config(
        num_groups * channel_blocks,
        batch_size,
        affine_block_batch,
    )
    _group_norm_forward_affine_kernel[(num_groups * channel_blocks, affine_grid_batch)](
        Y,
        X,
        Mean,
        RSTD,
        W,
        B,
        batch_size,
        num_groups,
        hidden_size,
        channels_per_group,
        channel_blocks,
        BLOCK_SIZE_H=affine_block_h,
        BLOCK_SIZE_CH=affine_block_ch,
        BLOCK_BATCH=affine_block_batch,
    )

    return Y.view(*shape), X.view(*shape), Mean.view(batch_size, num_groups), RSTD.view(batch_size, num_groups)


def group_norm_backward(dY, X, W, Mean, RSTD, num_channels, num_groups, bias_dtype):
    shape = dY.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups

    dY = dY.view(batch_size, num_groups, -1)
    X = X.view(batch_size, num_groups, -1)
    hidden_size = dY.shape[-1]
    hidden_size_per_channel = hidden_size // channels_per_group

    DX = torch.empty_like(dY)
    block_h = _group_norm_backward_spatial_block_size(hidden_size_per_channel)
    block_batch, num_partial_rows = _group_norm_backward_dx_dwdb_launch_config(
        batch_size,
        num_groups,
        hidden_size_per_channel,
    )

    DW_partial = torch.empty((num_partial_rows, num_channels), dtype=torch.float32, device=W.device)
    DB_partial = torch.empty((num_partial_rows, num_channels), dtype=torch.float32, device=W.device)
    DW_accum = torch.empty(num_channels, dtype=torch.float32, device=W.device)
    DB_accum = torch.empty(num_channels, dtype=torch.float32, device=W.device)

    _group_norm_backward_dx_dwdb_kernel[(num_groups, num_partial_rows)](
        X,
        W,
        Mean,
        RSTD,
        DX,
        DW_partial,
        DB_partial,
        dY,
        batch_size,
        num_groups,
        hidden_size,
        channels_per_group,
        BLOCK_H=block_h,
        BLOCK_BATCH=block_batch,
    )

    reduce_block_ch = min(256, triton.next_power_of_2(num_channels))
    _group_norm_reduce_param_grads_kernel[(triton.cdiv(num_channels, reduce_block_ch),)](
        DW_partial,
        DB_partial,
        DW_accum,
        DB_accum,
        num_partial_rows,
        num_channels,
        BLOCK_SIZE_CH=reduce_block_ch,
        BLOCK_SIZE_ROWS=REDUCE_BATCH_CHUNK,
    )

    return DX.view(*shape), DW_accum.to(W.dtype), DB_accum.to(bias_dtype)


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    ):
        Y, X, Mean, RSTD = group_norm_forward(
            X,
            num_channels,
            num_groups,
            affine_scaling_weight,
            affine_shifting_bias,
            eps,
        )
        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.bias_dtype = affine_shifting_bias.dtype
        ctx.save_for_backward(X, affine_scaling_weight, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = group_norm_backward(
            dY,
            X,
            W,
            Mean,
            RSTD,
            ctx.num_channels,
            ctx.num_groups,
            ctx.bias_dtype,
        )
        return DX, DW, DB, None, None, None
