"""Ascend LayerNorm: UB-aware fused 2D, row, or column-tiled Triton kernels."""

import functools

import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# Peak live fp32 2D tiles in fused forward (single X load, x_hat, Y).
_FUSED_FWD_MEM_MULT = 6.0
# Tighter estimate for full-width (4096) 2-row fused tiles: X_f32 + Y_f32 only.
_FUSED_FWD_MEM_MULT_WIDE = 4.0
# Peak live fp32 vectors in per-program backward reduction.
_FUSED_BWD_MEM_MULT = 8.0
_UB_SAFETY_MARGIN = 0.85


@triton.jit
def _layer_norm_forward_row(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Row kernel when BLOCK_SIZE fully covers n_cols (no column mask)."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols

    W_row = tl.load(W_ptr + col_offsets, eviction_policy="evict_last")
    B_row = tl.load(B_ptr + col_offsets, eviction_policy="evict_last")

    for row_idx in tl.range(pid, n_rows, num_progs):
        row_X_ptr = X_ptr + row_idx * X_row_stride
        row_Y_ptr = Y_ptr + row_idx * Y_row_stride
        row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
        row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        X_row = tl.load(row_X_ptr + col_offsets, eviction_policy="evict_first")
        X_f32 = X_row.to(tl.float32)
        mean = tl.sum(X_f32, axis=0) * n_cols_inv
        mean_sq = tl.sum(X_f32 * X_f32, axis=0) * n_cols_inv
        rstd = rsqrt(mean_sq - mean * mean + eps)

        tl.store(row_Mean_ptr, mean)
        tl.store(row_RSTD_ptr, rstd)
        x_hat = ((X_f32 - mean) * rstd).to(X_row.dtype)
        Y_row = x_hat * W_row + B_row
        tl.store(row_Y_ptr + col_offsets, Y_row)


@triton.jit
def _layer_norm_forward_fused_2d(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """Full-width fused tile without column mask: E[x^2]-mean^2, fp32 affine."""
    row_block_start = tl.program_id(0) * ROWS_PER_BLOCK
    row_block_step = tl.num_programs(0) * ROWS_PER_BLOCK

    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    n_cols_inv = 1.0 / n_cols

    W_f32 = tl.load(W_ptr + col_offsets, eviction_policy="evict_last").to(tl.float32)
    B_f32 = tl.load(B_ptr + col_offsets, eviction_policy="evict_last").to(tl.float32)

    for row_block_idx in tl.range(row_block_start, n_rows, row_block_step):
        row_idx = row_block_idx + row_offsets
        row_mask = row_idx < n_rows

        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        X_f32 = X_rows.to(tl.float32)

        mean_rows = tl.sum(X_f32, axis=1) * n_cols_inv
        mean_sq_rows = tl.sum(X_f32 * X_f32, axis=1) * n_cols_inv
        rstd_rows = rsqrt(mean_sq_rows - mean_rows * mean_rows + eps)

        tl.store(Mean_ptr + row_idx * Mean_row_stride, mean_rows, mask=row_mask)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd_rows, mask=row_mask)

        Y_f32 = (X_f32 - mean_rows[:, None]) * rstd_rows[:, None] * W_f32[None, :] + B_f32[None, :]
        tl.store(
            Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
            Y_f32.to(X_rows.dtype),
            mask=row_mask[:, None],
        )


@triton.jit
def _layer_norm_forward_tiled(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fallback forward kernel that tiles columns when a full row does not fit UB."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols

    for row_idx in tl.range(pid, n_rows, num_progs):
        X_row_ptr = X_ptr + row_idx * X_row_stride
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        Mean_row_ptr = Mean_ptr + row_idx * Mean_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Three passes (mean, variance, affine): each needs a full column sweep.
        row_sum = 0.0
        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            row_sum += tl.sum(X_block)
        mean = row_sum * n_cols_inv

        var_sum = 0.0
        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            x_hat = X_block - mean
            var_sum += tl.sum(tl.where(mask, x_hat * x_hat, 0.0))

        rstd = rsqrt(var_sum * n_cols_inv + eps)
        tl.store(Mean_row_ptr, mean)
        tl.store(RSTD_row_ptr, rstd)

        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            B_block = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            Y_f32 = (X_block - mean) * rstd * W_block + B_block
            tl.store(Y_row_ptr + col_offsets, Y_f32.to(X_block.dtype), mask=mask)


@triton.jit
def _layer_norm_backward_rows(
    X_ptr,
    X_row_stride,
    W_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    DX_ptr,
    DX_row_stride,
    DW_ptr,
    DB_ptr,
    DY_ptr,
    DY_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Grid-stride backward: each program reduces dW/dB via atomic_add."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    n_cols_inv = 1.0 / n_cols

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dB_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for row_idx in tl.range(pid, n_rows, num_progs):
        row_X_ptr = X_ptr + row_idx * X_row_stride
        row_DX_ptr = DX_ptr + row_idx * DX_row_stride
        row_DY_ptr = DY_ptr + row_idx * DY_row_stride
        row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
        row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        x = tl.load(row_X_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first")
        dy = tl.load(row_DY_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first")
        mean = tl.load(row_Mean_ptr).to(tl.float32)
        rstd = tl.load(row_RSTD_ptr).to(tl.float32)

        x_f32 = x.to(tl.float32)
        dy_f32 = dy.to(tl.float32)
        x_hat = (x_f32 - mean) * rstd
        wdy = w_f32 * dy_f32
        c1 = tl.sum(x_hat * wdy, axis=0) * n_cols_inv
        c2 = tl.sum(wdy, axis=0) * n_cols_inv
        dx = (wdy - (x_hat * c1 + c2)) * rstd

        tl.store(row_DX_ptr + cols, dx.to(x.dtype), mask=mask)
        dW_row += dy_f32 * x_hat
        dB_row += dy_f32

    tl.atomic_add(DW_ptr + cols, dW_row, mask=mask)
    tl.atomic_add(DB_ptr + cols, dB_row, mask=mask)


@triton.jit
def _layer_norm_backward_tiled(
    X_ptr,
    X_row_stride,
    W_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    DX_ptr,
    DX_row_stride,
    DW_ptr,
    DB_ptr,
    DY_ptr,
    DY_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fallback backward kernel using column tiles and per-tile atomics."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols

    for row_idx in tl.range(pid, n_rows, num_progs):
        X_row_ptr = X_ptr + row_idx * X_row_stride
        DY_row_ptr = DY_ptr + row_idx * DY_row_stride
        DX_row_ptr = DX_ptr + row_idx * DX_row_stride
        Mean_row_ptr = Mean_ptr + row_idx * Mean_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        mean = tl.load(Mean_row_ptr).to(tl.float32)
        rstd = tl.load(RSTD_row_ptr).to(tl.float32)

        sum_x_hat_wdy = 0.0
        sum_wdy = 0.0
        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            DY_block = tl.load(DY_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            x_hat = (X_block - mean) * rstd
            wdy = W_block * DY_block
            sum_x_hat_wdy += tl.sum(tl.where(mask, x_hat * wdy, 0.0))
            sum_wdy += tl.sum(tl.where(mask, wdy, 0.0))

        c1 = sum_x_hat_wdy * n_cols_inv
        c2 = sum_wdy * n_cols_inv

        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            DY_block = tl.load(DY_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(
                tl.float32
            )
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            x_hat = (X_block - mean) * rstd
            wdy = W_block * DY_block
            DX_block = (wdy - (x_hat * c1 + c2)) * rstd
            tl.store(DX_row_ptr + col_offsets, DX_block.to(X_ptr.dtype.element_ty), mask=mask)

            tl.atomic_add(DW_ptr + col_offsets, DY_block * x_hat, mask=mask)
            tl.atomic_add(DB_ptr + col_offsets, DY_block, mask=mask)


def _safe_column_block(n_cols: int, memory_multiplier: float) -> int:
    """Largest power-of-2 column tile that fits in UB for 1D vector kernels."""
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((n_cols,),),
        tiling_dims=(0,),
    )
    if tile_shapes:
        return max(128, tile_shapes[0][0])
    return 1024


def _safe_fused_rows_per_block(n_cols: int, col_block: int) -> int:
    """Max row tile count for fused 2D kernel at a fixed column width."""
    desired_rows = 8
    mem_mult = _FUSED_FWD_MEM_MULT_WIDE if col_block >= n_cols and n_cols >= 2048 else _FUSED_FWD_MEM_MULT
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=4,
        memory_multiplier=mem_mult,
        shapes=((desired_rows, col_block),),
        tiling_dims=(0,),
    )
    if tile_shapes:
        return max(1, tile_shapes[0][0])
    return 1


@functools.lru_cache(maxsize=32)
def _fused_forward_tile(n_cols: int) -> tuple[int, int, bool]:
    """
    Plan fused-2D vs column-tiled forward.

    Returns:
        (col_block, rows_per_block, use_fused_2d)
    """
    safe_col = _safe_column_block(n_cols, _FUSED_FWD_MEM_MULT)
    col_pow2 = triton.next_power_of_2(n_cols)
    col_block = min(col_pow2, safe_col)
    # Mask-free kernels assume BLOCK_SIZE == n_cols; padded pow2 tiles need column masks.
    if col_block < n_cols or col_pow2 > n_cols:
        return _safe_column_block(n_cols, 8.0), 1, False

    rows_per_block = _safe_fused_rows_per_block(n_cols, col_block)
    return col_block, rows_per_block, True


def _forward_grid_size(n_rows: int, num_cores: int) -> int:
    # Ascend910 row LN: 4096 -> cores*2; 8192+ -> cores*4.
    if n_rows <= 1024:
        return min(num_cores * 2, n_rows)
    if n_rows >= 8192:
        return min(num_cores * 4, n_rows)
    if n_rows >= 4096:
        return min(num_cores * 2, n_rows)
    return min(num_cores, n_rows)


def _backward_grid_size(n_rows: int, num_cores: int) -> int:
    """Grid size for row-wise backward kernel."""
    return min(num_cores, n_rows)


def layer_norm_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward LayerNorm on Ascend.

    Dispatches mask-free row / fused-2D kernels when UB allows full-width tiles;
    otherwise uses a three-pass column-tiled kernel with 2x grid oversubscription.
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    num_cores = get_npu_core_count()
    col_block, rows_per_block, use_fused = _fused_forward_tile(n_cols)

    if use_fused:
        if rows_per_block <= 1:
            num_programs = _forward_grid_size(n_rows, num_cores)
            _layer_norm_forward_row[(num_programs,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                B,
                Mean,
                Mean.stride(0),
                RSTD,
                RSTD.stride(0),
                n_rows,
                n_cols,
                eps,
                BLOCK_SIZE=col_block,
            )
        else:
            num_row_blocks = triton.cdiv(n_rows, rows_per_block)
            num_programs = min(num_cores, num_row_blocks)
            _layer_norm_forward_fused_2d[(num_programs,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                B,
                Mean,
                Mean.stride(0),
                RSTD,
                RSTD.stride(0),
                n_rows,
                n_cols,
                eps,
                BLOCK_SIZE=col_block,
                ROWS_PER_BLOCK=rows_per_block,
            )
    else:
        num_programs = min(num_cores * 2, n_rows)
        _layer_norm_forward_tiled[(num_programs,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            B,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=col_block,
        )

    return Y.view(*shape), X, Mean, RSTD


def layer_norm_backward(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    _B: torch.Tensor,
    Mean: torch.Tensor,
    RSTD: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward LayerNorm matching tensors saved by layer_norm_forward (_B unused)."""
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    num_cores = get_npu_core_count()
    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    safe_col = _safe_column_block(n_cols, _FUSED_BWD_MEM_MULT)
    col_block = min(triton.next_power_of_2(n_cols), safe_col)

    DW = torch.zeros(n_cols, dtype=torch.float32, device=W.device)
    DB = torch.zeros(n_cols, dtype=torch.float32, device=W.device)

    if col_block >= n_cols:
        grid_size = _backward_grid_size(n_rows, num_cores)

        _layer_norm_backward_rows[(grid_size,)](
            X,
            X.stride(0),
            W,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            DX,
            DX.stride(0),
            DW,
            DB,
            dY,
            dY.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=col_block,
        )
    else:
        col_block = _safe_column_block(n_cols, 10.0)
        num_programs = min(num_cores * 2, n_rows)

        _layer_norm_backward_tiled[(num_programs,)](
            X,
            X.stride(0),
            W,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            DX,
            DX.stride(0),
            DW,
            DB,
            dY,
            dY.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=col_block,
        )

    return DX.view(*shape), DW.to(W.dtype), DB.to(_B.dtype)


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None
