import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# -----------------------------------------------------------------------------
# Kernels (2D row/col tiling + persistent programs)
# -----------------------------------------------------------------------------


@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (B, G, hidden_size)
    Y_row_stride,  # stride of each batch row in Y
    Y_col_stride,  # stride of each group row in Y
    X_ptr,  # pointer to input, shape (B, G, hidden_size)
    X_row_stride,  # stride of each batch row in X
    X_col_stride,  # stride of each group row in X
    Mean_ptr,  # pointer to mean output, shape (B, G)
    Mean_row_stride,  # stride of each batch row in Mean
    Mean_col_stride,  # stride of each group row in Mean
    RSTD_ptr,  # pointer to rstd output, shape (B, G)
    RSTD_row_stride,  # stride of each batch row in RSTD
    RSTD_col_stride,  # stride of each group row in RSTD
    W_ptr,  # pointer to affine scale weights, shape (C)
    B_ptr,  # pointer to affine bias weights, shape (C)
    n_rows,  # total logical rows = B * G
    hidden_size,
    channels_per_group,
    num_groups,
    SINGLE_CHANNEL_TILE: tl.constexpr,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_col_blocks = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    hidden_size_per_channel = hidden_size // channels_per_group
    hidden_size_inv = 1.0 / hidden_size
    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    col_offsets_base = tl.arange(0, BLOCK_SIZE_N)

    # Persistent-program loop over row tiles.
    for block_m in tl.range(pid, grid_m, num_progs):
        row_idx = block_m * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        batch_idx = row_idx // num_groups
        group_idx = row_idx % num_groups

        row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        row_square_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        # Pass 1: accumulate E[x] and E[x^2] for each row tile.
        for cb in range(num_col_blocks):
            col_offsets = cb * BLOCK_SIZE_N + col_offsets_base
            col_mask = col_offsets < hidden_size
            mask = row_mask[:, None] & col_mask[None, :]
            X_ptrs = (
                X_ptr + batch_idx[:, None] * X_row_stride + group_idx[:, None] * X_col_stride + col_offsets[None, :]
            )
            X_block = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
            row_sum += tl.sum(X_block, axis=1)
            row_square_sum += tl.sum(X_block * X_block, axis=1)

        mean = row_sum * hidden_size_inv
        var = row_square_sum * hidden_size_inv - mean * mean
        rstd = rsqrt(tl.maximum(var, 0.0) + eps)

        mean_ptrs = Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride
        rstd_ptrs = RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride
        tl.store(mean_ptrs, mean, mask=row_mask)
        tl.store(rstd_ptrs, rstd, mask=row_mask)

        # Pass 2: normalize + affine transform.
        # SINGLE_CHANNEL_TILE indicates the current column tile maps to one channel,
        # so W/B can be loaded once per row and broadcast to the tile.
        for cb in range(num_col_blocks):
            col_offsets = cb * BLOCK_SIZE_N + col_offsets_base
            col_mask = col_offsets < hidden_size
            mask = row_mask[:, None] & col_mask[None, :]
            X_ptrs = (
                X_ptr + batch_idx[:, None] * X_row_stride + group_idx[:, None] * X_col_stride + col_offsets[None, :]
            )
            X_block = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
            if SINGLE_CHANNEL_TILE:
                local_channel = (cb * BLOCK_SIZE_N) // hidden_size_per_channel
                global_channel = group_idx * channels_per_group + local_channel
                W_block = tl.load(W_ptr + global_channel, mask=row_mask, other=0.0).to(tl.float32)[:, None]
                B_block = tl.load(B_ptr + global_channel, mask=row_mask, other=0.0).to(tl.float32)[:, None]
            else:
                local_channel = col_offsets // hidden_size_per_channel
                global_channel = group_idx[:, None] * channels_per_group + local_channel[None, :]
                W_block = tl.load(W_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)
                B_block = tl.load(B_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)
            Y_block = (X_block - mean[:, None]) * rstd[:, None] * W_block + B_block
            Y_ptrs = (
                Y_ptr + batch_idx[:, None] * Y_row_stride + group_idx[:, None] * Y_col_stride + col_offsets[None, :]
            )
            tl.store(Y_ptrs, Y_block, mask=mask)


@triton.jit
def _group_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (B, G, hidden_size)
    X_row_stride,  # stride of each batch row in X
    X_col_stride,  # stride of each group row in X
    W_ptr,  # pointer to affine scale weights, shape (C)
    Mean_ptr,  # pointer to saved group mean, shape (B, G)
    Mean_row_stride,  # stride of each batch row in Mean
    Mean_col_stride,  # stride of each group row in Mean
    RSTD_ptr,  # pointer to saved reciprocal std, shape (B, G)
    DX_ptr,  # pointer to input gradients, shape (B, G, hidden_size)
    DW_scratch_ptr,  # pointer to scratch buffer for dW partial sums, shape (grid, C)
    DW_scratch_stride,  # row stride for DW_scratch
    DB_scratch_ptr,  # pointer to scratch buffer for dB partial sums, shape (grid, C)
    DB_scratch_stride,  # row stride for DB_scratch
    DY_ptr,  # pointer to upstream gradients, shape (B, G, hidden_size)
    DY_row_stride,  # stride of each batch row in DY
    DY_col_stride,  # stride of each group row in DY
    n_rows,  # total logical rows = B * G
    hidden_size,
    channels_per_group,
    num_groups,
    SINGLE_CHANNEL_TILE: tl.constexpr,
    COMPUTE_PARAM_GRAD: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_col_blocks = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    hidden_size_per_channel = hidden_size // channels_per_group
    N_inv = 1.0 / hidden_size
    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    col_offsets_base = tl.arange(0, BLOCK_SIZE_N)
    DW_scratch_base = DW_scratch_ptr + pid * DW_scratch_stride
    DB_scratch_base = DB_scratch_ptr + pid * DB_scratch_stride

    # Persistent-program loop over row tiles.
    for block_m in tl.range(pid, grid_m, num_progs):
        row_idx = block_m * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        batch_idx = row_idx // num_groups
        group_idx = row_idx % num_groups

        mean = tl.load(
            Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        rstd = tl.load(
            RSTD_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        sum_x_hat_wdy = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        sum_wdy = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        # Pass 1: compute row-wise reduction terms (c1, c2).
        for cb in range(num_col_blocks):
            col_offsets = cb * BLOCK_SIZE_N + col_offsets_base
            col_mask = col_offsets < hidden_size
            mask = row_mask[:, None] & col_mask[None, :]

            X_ptrs = (
                X_ptr + batch_idx[:, None] * X_row_stride + group_idx[:, None] * X_col_stride + col_offsets[None, :]
            )
            DY_ptrs = (
                DY_ptr + batch_idx[:, None] * DY_row_stride + group_idx[:, None] * DY_col_stride + col_offsets[None, :]
            )
            X_block = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
            DY_block = tl.load(DY_ptrs, mask=mask, other=0.0).to(tl.float32)

            if SINGLE_CHANNEL_TILE:
                local_channel = (cb * BLOCK_SIZE_N) // hidden_size_per_channel
                global_channel = group_idx * channels_per_group + local_channel
                W_block = tl.load(W_ptr + global_channel, mask=row_mask, other=0.0).to(tl.float32)[:, None]
            else:
                local_channel = col_offsets // hidden_size_per_channel
                global_channel = group_idx[:, None] * channels_per_group + local_channel[None, :]
                W_block = tl.load(W_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)

            x_hat = (X_block - mean[:, None]) * rstd[:, None]
            wdy = W_block * DY_block
            sum_x_hat_wdy += tl.sum(tl.where(mask, x_hat * wdy, 0.0), axis=1)
            sum_wdy += tl.sum(tl.where(mask, wdy, 0.0), axis=1)

        c1 = sum_x_hat_wdy * N_inv
        c2 = sum_wdy * N_inv

        # Pass 2: compute DX and optionally accumulate DW/DB.
        # COMPUTE_PARAM_GRAD=False is used to skip expensive atomics in cases
        # where host-side dense reduction is faster/more stable.
        for cb in range(num_col_blocks):
            col_offsets = cb * BLOCK_SIZE_N + col_offsets_base
            col_mask = col_offsets < hidden_size
            mask = row_mask[:, None] & col_mask[None, :]

            X_ptrs = (
                X_ptr + batch_idx[:, None] * X_row_stride + group_idx[:, None] * X_col_stride + col_offsets[None, :]
            )
            DY_ptrs = (
                DY_ptr + batch_idx[:, None] * DY_row_stride + group_idx[:, None] * DY_col_stride + col_offsets[None, :]
            )
            X_block = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
            DY_block = tl.load(DY_ptrs, mask=mask, other=0.0).to(tl.float32)

            if SINGLE_CHANNEL_TILE:
                local_channel = (cb * BLOCK_SIZE_N) // hidden_size_per_channel
                global_channel = group_idx * channels_per_group + local_channel
                W_block = tl.load(W_ptr + global_channel, mask=row_mask, other=0.0).to(tl.float32)[:, None]
            else:
                local_channel = col_offsets // hidden_size_per_channel
                global_channel = group_idx[:, None] * channels_per_group + local_channel[None, :]
                W_block = tl.load(W_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)

            x_hat = (X_block - mean[:, None]) * rstd[:, None]
            wdy = W_block * DY_block
            DX_block = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

            DX_ptrs = (
                DX_ptr + batch_idx[:, None] * X_row_stride + group_idx[:, None] * X_col_stride + col_offsets[None, :]
            )
            tl.store(DX_ptrs, DX_block.to(X_ptr.dtype.element_ty), mask=mask)

            if COMPUTE_PARAM_GRAD:
                if SINGLE_CHANNEL_TILE:
                    dW_partial = tl.sum(tl.where(mask, DY_block * x_hat, 0.0), axis=1)
                    dB_partial = tl.sum(tl.where(mask, DY_block, 0.0), axis=1)
                    tl.atomic_add(DW_scratch_base + global_channel, dW_partial, mask=row_mask)
                    tl.atomic_add(DB_scratch_base + global_channel, dB_partial, mask=row_mask)
                else:
                    dW_block = tl.where(mask, DY_block * x_hat, 0.0)
                    dB_block = tl.where(mask, DY_block, 0.0)
                    tl.atomic_add(DW_scratch_base + global_channel, dW_block, mask=mask)
                    tl.atomic_add(DB_scratch_base + global_channel, dB_block, mask=mask)


# -----------------------------------------------------------------------------
# Helper: call compute_default_tiling_strategy
# -----------------------------------------------------------------------------


def get_optimal_block_size(n_rows, dtype_size, BLOCK_SIZE_N, is_backward: bool = False):
    # Backward keeps larger live-state than forward in this kernel.
    multiplier = 7.0 if is_backward else 6.0

    # Use fp32-size as conservative UB estimate for tiling.
    dtype_size = max(dtype_size, 4)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=dtype_size,
        memory_multiplier=multiplier,
        shapes=((n_rows, BLOCK_SIZE_N),),
        tiling_dims=(0,),
    )
    if tile_shapes and len(tile_shapes) > 0:
        return tile_shapes[0][0]
    return triton.next_power_of_2(min(128, n_rows))


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    # Reshape X so that the mean / std are computed across each group
    X = X.view(batch_size, num_groups, -1).contiguous()

    hidden_size = X.shape[-1]
    hidden_size_per_channel = hidden_size // channels_per_group
    n_rows = batch_size * num_groups

    BLOCK_SIZE_N = min(128, triton.next_power_of_2(hidden_size))
    BLOCK_SIZE_M = get_optimal_block_size(n_rows, X.element_size(), BLOCK_SIZE_N)

    # Fast path condition: each column tile must lie entirely inside one channel
    # segment of length `hidden_size_per_channel`.
    #
    # Layout of a row:
    #   | channel0 | channel1 | channel2 | ...
    #   |----Hc----|----Hc----|
    #   Hc = hidden_size_per_channel
    #
    # The kernel processes tiles of shape (BLOCK_SIZE_M, BLOCK_SIZE_N).
    # Channel boundaries exist only along the column dimension, because
    # each row corresponds to a different (batch, group).
    #
    # Therefore only BLOCK_SIZE_N matters for whether a tile crosses
    # channel boundaries; BLOCK_SIZE_M does not affect channel mapping.
    #
    # If BLOCK_SIZE_N divides Hc and is <= Hc, each column tile belongs
    # to exactly one channel. In that case W/B can be loaded once and
    # broadcast across the tile (fast path).
    #
    # Otherwise a tile may span multiple channels, requiring per-element
    # channel index computation and parameter loads (slow path).
    single_channel_tile = BLOCK_SIZE_N <= hidden_size_per_channel and hidden_size_per_channel % BLOCK_SIZE_N == 0

    num_cores = get_npu_core_count()
    grid = min(num_cores, triton.cdiv(n_rows, BLOCK_SIZE_M))

    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.empty((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.empty((batch_size, num_groups), dtype=X.dtype, device=X.device)

    _group_norm_forward_kernel[(grid,)](
        Y,
        Y.stride(0),
        Y.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        W,
        B,
        n_rows,
        hidden_size,
        channels_per_group,
        num_groups,
        SINGLE_CHANNEL_TILE=single_channel_tile,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return Y.view(*shape), X.view(*shape), Mean, RSTD


def group_norm_backward(dY, X, W, B, Mean, RSTD, num_channels, num_groups):
    shape = dY.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    X_grouped = X.view(batch_size, num_groups, -1)
    dY_grouped = dY.view(batch_size, num_groups, -1)
    hidden_size = dY_grouped.shape[-1]
    hidden_size_per_channel = hidden_size // channels_per_group
    n_rows = batch_size * num_groups

    BLOCK_SIZE_N = min(128, triton.next_power_of_2(hidden_size))
    BLOCK_SIZE_M = get_optimal_block_size(
        n_rows,
        X.element_size(),
        BLOCK_SIZE_N,
        is_backward=True,
    )

    # Same condition as forward:
    # if true, each BLOCK_SIZE_N tile maps cleanly to one channel segment.
    single_channel_tile = BLOCK_SIZE_N <= hidden_size_per_channel and hidden_size_per_channel % BLOCK_SIZE_N == 0

    num_cores = get_npu_core_count()
    grid = min(num_cores, triton.cdiv(n_rows, BLOCK_SIZE_M))
    # For non-single-channel tiles, per-element atomic updates are costly.
    # In that case, kernel computes DX only and DW/DB are reduced on host side.
    compute_param_grad = single_channel_tile

    DX = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    if compute_param_grad:
        DW_scratch = torch.zeros((grid, num_channels), dtype=torch.float32, device=W.device)
        DB_scratch = torch.zeros((grid, num_channels), dtype=torch.float32, device=W.device)
    else:
        # Placeholder buffers (unused in kernel when COMPUTE_PARAM_GRAD=False)
        DW_scratch = torch.empty((1, 1), dtype=torch.float32, device=W.device)
        DB_scratch = torch.empty((1, 1), dtype=torch.float32, device=W.device)

    _group_norm_backward_kernel[(grid,)](
        X_grouped,
        X_grouped.stride(0),
        X_grouped.stride(1),
        W,
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        DX,
        DW_scratch,
        DW_scratch.stride(0),
        DB_scratch,
        DB_scratch.stride(0),
        dY_grouped,
        dY_grouped.stride(0),
        dY_grouped.stride(1),
        n_rows,
        hidden_size,
        channels_per_group,
        num_groups,
        SINGLE_CHANNEL_TILE=single_channel_tile,
        COMPUTE_PARAM_GRAD=compute_param_grad,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    # Precision note:
    # - In-kernel atomic_add on floating-point values is order-dependent under parallel
    #   scheduling (non-associative summation), which can introduce run-to-run numerical
    #   differences in DW/DB for contention-heavy shapes.
    # - Host-side dense reduction provides a more stable accumulation pattern for these
    #   difficult layouts.
    if compute_param_grad:
        DW = DW_scratch.sum(dim=0).to(W.dtype)
        DB = DB_scratch.sum(dim=0).to(W.dtype)
    else:
        # Fallback path to avoid severe atomic contention when SINGLE_CHANNEL_TILE=False.
        # Layout: [B, G, hidden_size] -> [B, G, C_per_G, hidden_per_channel]
        X4 = X_grouped.reshape(batch_size, num_groups, channels_per_group, hidden_size_per_channel).to(torch.float32)
        dY4 = dY_grouped.reshape(batch_size, num_groups, channels_per_group, hidden_size_per_channel).to(torch.float32)
        mean4 = Mean.reshape(batch_size, num_groups, 1, 1).to(torch.float32)
        rstd4 = RSTD.reshape(batch_size, num_groups, 1, 1).to(torch.float32)

        x_hat4 = (X4 - mean4) * rstd4
        DW = (dY4 * x_hat4).sum(dim=(0, 3)).reshape(-1).to(W.dtype)
        DB = dY4.sum(dim=(0, 3)).reshape(-1).to(W.dtype)

    return DX.view(*shape), DW, DB


class LigerGroupNormFunction(torch.autograd.Function):
    """
    Group Normalization autograd function for Ascend NPU.

    Forward computes, for each sample/group:
        y = (x - mean) * rstd * weight + bias
    where:
        mean = E[x], rstd = 1 / sqrt(Var[x] + eps)

    The kernel uses row/column tiling with persistent programs. Backward computes
    input gradients in Triton and computes parameter gradients either via Triton
    atomics (fast path) or host-side dense reduction (fallback path), depending
    on the tile/channel layout.
    """

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
        ctx.save_for_backward(X, affine_scaling_weight, affine_shifting_bias, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = group_norm_backward(dY, X, W, B, Mean, RSTD, ctx.num_channels, ctx.num_groups)
        return DX, DW, DB, None, None, None
