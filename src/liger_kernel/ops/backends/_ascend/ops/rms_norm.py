"""Ascend RMSNorm: UB-aware fused row / 2D / column-tiled Triton kernels.

Profiling on 910B (BT=8192, H=4096, bf16) showed the previous two-pass tiled
forward spent 350us at aiv_mte2_ratio=0.69 because it reloaded X after the
rstd reduction. torch_npu.npu_rms_norm finishes in 107us at vec_ratio=0.81.
Full-width single-pass kernels (same pattern as Ascend LayerNorm) keep X in UB
and cut that extra HBM round-trip. Column tiling remains only when a full row
does not fit UB.
"""

import functools

import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import torch_to_triton_dtype

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

# Peak live fp32 tiles. Wide rows keep fewer copies so a full 4096/8192 vector fits.
_FUSED_FWD_MEM_MULT = 5.0
_FUSED_FWD_MEM_MULT_WIDE = 4.0
_FUSED_BWD_MEM_MULT = 7.0
_TILED_MEM_MULT = 8.0
_UB_SAFETY_MARGIN = 0.85


# -----------------------------------------------------------------------------
# Forward kernels
# -----------------------------------------------------------------------------


@triton.jit
def _rms_norm_forward_row(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Grid-stride full-width row forward. One vector load of X per row."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    n_cols_inv = 1.0 / n_cols
    # Python float scalars can specialize to fp64 (#1358). Pin so rsqrt / W+offset stay fp32.
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)
    else:
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)

    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)

    for row_idx in tl.range(row_start, row_end):
        row_X_ptr = X_ptr + row_idx * X_row_stride
        row_Y_ptr = Y_ptr + row_idx * Y_row_stride

        X_row = tl.load(row_X_ptr + col_offsets, mask=col_mask, other=0.0)
        if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
            X_f32 = X_row.to(tl.float32)
        else:
            X_f32 = X_row

        sum_sq = tl.sum(tl.where(col_mask, X_f32 * X_f32, 0.0), axis=0)
        rstd = rsqrt(sum_sq * n_cols_inv + eps)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)

        X_hat = X_f32 * rstd
        if casting_mode == _CASTING_MODE_LLAMA:
            X_hat = X_hat.to(X_DTYPE)
        if elementwise_affine:
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_row = X_hat * (offset + W_row.to(tl.float32))
            else:
                Y_row = X_hat * (offset + W_row)
        else:
            Y_row = X_hat
        if casting_mode == _CASTING_MODE_GEMMA:
            Y_row = Y_row.to(X_DTYPE)
        tl.store(row_Y_ptr + col_offsets, Y_row, mask=col_mask)


@triton.jit
def _rms_norm_forward_fused_2d(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """2D fused forward: ROWS_PER_BLOCK rows x full n_cols in one load."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    col_mask = col_offsets < n_cols
    n_cols_inv = 1.0 / n_cols
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)
    else:
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)

    n_row_blocks = tl.cdiv(n_rows, ROWS_PER_BLOCK)
    blocks_per_prog = (n_row_blocks + num_progs - 1) // num_progs
    block_start = pid * blocks_per_prog
    block_end = tl.minimum(block_start + blocks_per_prog, n_row_blocks)

    for block_i in tl.range(block_start, block_end):
        row_idx = block_i * ROWS_PER_BLOCK + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
            X_f32 = X_rows.to(tl.float32)
        else:
            X_f32 = X_rows

        sum_sq = tl.sum(tl.where(block_mask, X_f32 * X_f32, 0.0), axis=1)
        rstd = rsqrt(sum_sq * n_cols_inv + eps)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, mask=row_mask)

        X_hat = X_f32 * rstd[:, None]
        if casting_mode == _CASTING_MODE_LLAMA:
            X_hat = X_hat.to(X_DTYPE)
        if elementwise_affine:
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_rows = X_hat * (offset + W_row.to(tl.float32))[None, :]
            else:
                Y_rows = X_hat * (offset + W_row)[None, :]
        else:
            Y_rows = X_hat
        if casting_mode == _CASTING_MODE_GEMMA:
            Y_rows = Y_rows.to(X_DTYPE)
        tl.store(
            Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
            Y_rows,
            mask=block_mask,
        )


@triton.jit
def _rms_norm_forward_tiled(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Two-pass column-tiled forward when a full row does not fit UB."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)
    else:
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)

    for row_idx in tl.range(row_start, row_end):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride

        sum_square = 0.0
        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                X_block = X_block.to(tl.float32)
            sum_square += tl.sum(tl.where(mask, X_block * X_block, 0.0))

        rstd = rsqrt(sum_square * n_cols_inv + eps)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)

        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
            if casting_mode == _CASTING_MODE_GEMMA:
                X_block = X_block.to(tl.float32)
                if elementwise_affine:
                    W_block = W_block.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                X_block = X_block.to(tl.float32)

            X_block = X_block * rstd
            if casting_mode == _CASTING_MODE_LLAMA:
                X_block = X_block.to(X_DTYPE)
            if elementwise_affine:
                Y_block = X_block * (offset + W_block)
            else:
                Y_block = X_block
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_block = Y_block.to(X_DTYPE)
            tl.store(Y_row_ptr + col_offsets, Y_block, mask=mask)


# -----------------------------------------------------------------------------
# Backward kernels
# -----------------------------------------------------------------------------


@triton.jit
def _rms_norm_backward_row(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Full-width row backward. Per-program dW is reduced on the host."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    n_cols_inv = 1.0 / n_cols
    offset = offset.to(tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_offset = W_row + offset
        dW_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)

    for row_idx in tl.range(row_start, row_end):
        dY_row = tl.load(dY_ptr + row_idx * dY_row_stride + col_offsets, mask=col_mask, other=0.0)
        X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=col_mask, other=0.0)
        rstd = tl.load(RSTD_ptr + row_idx * RSTD_row_stride)
        X_f32 = X_row.to(tl.float32)
        X_hat = X_f32 * rstd

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                m = (dY_row * W_offset).to(tl.float32)
                X_hat_dw = X_hat.to(X_dtype)
            elif casting_mode == _CASTING_MODE_GEMMA:
                m = dY_row.to(tl.float32) * W_offset
                X_hat_dw = X_hat
            else:
                m = dY_row * W_offset
                X_hat_dw = X_hat
        else:
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                m = dY_row.to(tl.float32)
            else:
                m = dY_row
            X_hat_dw = X_hat

        sum_m_X = tl.sum(tl.where(col_mask, m * X_f32, 0.0), axis=0)
        correction = -n_cols_inv * rstd * rstd * sum_m_X
        dX = rstd * m + rstd * correction * X_f32
        tl.store(dX_ptr + row_idx * dX_row_stride + col_offsets, dX.to(X_dtype), mask=col_mask)

        if elementwise_affine:
            dW_acc += tl.where(col_mask, (dY_row * X_hat_dw).to(tl.float32), 0.0)

    if elementwise_affine:
        tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_acc, mask=col_mask)


@triton.jit
def _rms_norm_backward_fused_2d(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """2D fused backward: several rows x full n_cols."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROWS_PER_BLOCK)
    col_mask = col_offsets < n_cols
    n_cols_inv = 1.0 / n_cols
    offset = offset.to(tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_offset = W_row + offset
        dW_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    n_row_blocks = tl.cdiv(n_rows, ROWS_PER_BLOCK)
    blocks_per_prog = (n_row_blocks + num_progs - 1) // num_progs
    block_start = pid * blocks_per_prog
    block_end = tl.minimum(block_start + blocks_per_prog, n_row_blocks)

    for block_i in tl.range(block_start, block_end):
        row_idx = block_i * ROWS_PER_BLOCK + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        dY_rows = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        rstd = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, mask=row_mask, other=0.0)
        X_f32 = X_rows.to(tl.float32)
        X_hat = X_f32 * rstd[:, None]

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                m = (dY_rows * W_offset[None, :]).to(tl.float32)
                X_hat_dw = X_hat.to(X_dtype)
            elif casting_mode == _CASTING_MODE_GEMMA:
                m = dY_rows.to(tl.float32) * W_offset[None, :]
                X_hat_dw = X_hat
            else:
                m = dY_rows * W_offset[None, :]
                X_hat_dw = X_hat
        else:
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                m = dY_rows.to(tl.float32)
            else:
                m = dY_rows
            X_hat_dw = X_hat

        sum_m_X = tl.sum(tl.where(block_mask, m * X_f32, 0.0), axis=1)
        correction = -n_cols_inv * rstd * rstd * sum_m_X
        dX = rstd[:, None] * m + rstd[:, None] * correction[:, None] * X_f32
        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX.to(X_dtype),
            mask=block_mask,
        )
        if elementwise_affine:
            dW_acc += tl.sum(tl.where(block_mask, (dY_rows * X_hat_dw).to(tl.float32), 0.0), axis=0)

    if elementwise_affine:
        tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_acc, mask=col_mask)


@triton.jit
def _rms_norm_backward_tiled(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Two-pass column-tiled backward when a full row does not fit UB."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols
    offset = offset.to(tl.float32)

    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)

    for row_idx in tl.range(row_start, row_end):
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        rstd = tl.load(RSTD_ptr + row_idx * RSTD_row_stride)

        sum_m_X = 0.0
        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
                W_offset = W_block + offset
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dY_block * W_offset).to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32) * W_offset
                else:
                    m = dY_block * W_offset
            else:
                if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32)
                else:
                    m = dY_block
            sum_m_X += tl.sum(tl.where(mask, m * X_block, 0.0))

        correction = -n_cols_inv * rstd * rstd * sum_m_X

        for col_block_idx in range(num_col_blocks):
            col_offsets = col_block_idx * BLOCK_SIZE + offsets
            mask = col_offsets < n_cols
            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
                W_offset = W_block + offset
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dY_block * W_offset).to(tl.float32)
                    dW_block = dY_block * (X_block * rstd).to(X_dtype)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32) * W_offset
                    dW_block = dY_block * (X_block * rstd)
                else:
                    m = dY_block * W_offset
                    dW_block = dY_block * (X_block * rstd)
            else:
                if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32)
                else:
                    m = dY_block
                dW_block = None

            dX_block = rstd * m + rstd * correction * X_block
            tl.store(dX_row_ptr + col_offsets, dX_block.to(X_dtype), mask=mask)

            if elementwise_affine:
                dW_row_ptr = dW_ptr + pid * dW_row_stride
                existing = tl.load(dW_row_ptr + col_offsets, mask=mask, other=0.0)
                tl.store(dW_row_ptr + col_offsets, existing + dW_block.to(tl.float32), mask=mask)


# -----------------------------------------------------------------------------
# Tiling helpers
# -----------------------------------------------------------------------------


def _safe_column_block(n_cols: int, memory_multiplier: float) -> int:
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


def _safe_fused_rows_per_block(n_cols: int, col_block: int, memory_multiplier: float) -> int:
    desired_rows = 8
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((desired_rows, col_block),),
        tiling_dims=(0,),
    )
    if tile_shapes:
        return max(1, tile_shapes[0][0])
    return 1


@functools.lru_cache(maxsize=64)
def _fused_tile(n_cols: int, is_forward: bool) -> tuple[int, int, bool]:
    """Return (col_block, rows_per_block, use_fused_full_width)."""
    mem_mult = _FUSED_FWD_MEM_MULT if is_forward else _FUSED_BWD_MEM_MULT
    if is_forward and n_cols >= 2048:
        mem_mult = _FUSED_FWD_MEM_MULT_WIDE
    safe_col = _safe_column_block(n_cols, mem_mult)
    col_pow2 = triton.next_power_of_2(n_cols)
    col_block = min(col_pow2, safe_col)
    if col_block < n_cols:
        return _safe_column_block(n_cols, _TILED_MEM_MULT), 1, False
    wide_mult = _FUSED_FWD_MEM_MULT_WIDE if is_forward else _FUSED_BWD_MEM_MULT
    rows_per_block = _safe_fused_rows_per_block(n_cols, col_block, wide_mult)
    return col_block, rows_per_block, True


def _forward_grid_size(n_rows: int, num_cores: int) -> int:
    if n_rows <= 1024:
        return min(num_cores * 2, n_rows)
    if n_rows >= 8192:
        return min(num_cores * 4, n_rows)
    if n_rows >= 4096:
        return min(num_cores * 2, n_rows)
    return min(num_cores, n_rows)


def _backward_grid_size(n_rows: int, num_cores: int) -> int:
    if n_rows >= 8192:
        return min(num_cores * 2, n_rows)
    return min(num_cores, n_rows)


# -----------------------------------------------------------------------------
# Launchers
# -----------------------------------------------------------------------------


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def rms_norm_forward(X, W, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    X_DTYPE = torch_to_triton_dtype[X.dtype]

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    if W is not None:
        assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension"
        elementwise_affine = True
        w_arg = W
    else:
        elementwise_affine = False
        w_arg = X

    num_cores = get_npu_core_count()
    col_block, rows_per_block, use_fused = _fused_tile(n_cols, True)

    common = dict(
        Y_ptr=Y,
        Y_row_stride=Y.stride(0),
        X_ptr=X,
        X_row_stride=X.stride(0),
        W_ptr=w_arg,
        RSTD_ptr=RSTD,
        RSTD_row_stride=RSTD.stride(0),
        n_rows=n_rows,
        n_cols=n_cols,
        eps=eps,
        offset=offset,
        casting_mode=casting_mode,
        elementwise_affine=elementwise_affine,
        X_DTYPE=X_DTYPE,
        BLOCK_SIZE=col_block,
    )

    if use_fused:
        if rows_per_block <= 1:
            grid_size = _forward_grid_size(n_rows, num_cores)
            _rms_norm_forward_row[(grid_size,)](**common)
        else:
            num_row_blocks = triton.cdiv(n_rows, rows_per_block)
            grid_size = min(num_cores, num_row_blocks)
            _rms_norm_forward_fused_2d[(grid_size,)](**common, ROWS_PER_BLOCK=rows_per_block)
    else:
        grid_size = min(num_cores * 2, n_rows)
        _rms_norm_forward_tiled[(grid_size,)](**common)

    return Y.view(*shape), X, RSTD, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, in_place):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    num_cores = get_npu_core_count()
    col_block, rows_per_block, use_fused = _fused_tile(n_cols, False)

    if use_fused:
        grid_size = _backward_grid_size(n_rows, num_cores)
        if rows_per_block > 1:
            num_row_blocks = triton.cdiv(n_rows, rows_per_block)
            grid_size = min(num_cores, num_row_blocks)
    else:
        grid_size = min(num_cores * 2, n_rows)

    if W is not None:
        _dW = torch.zeros((grid_size, n_cols), dtype=torch.float32, device=W.device)
        elementwise_affine = True
        w_arg = W
        dw_stride = _dW.stride(0)
    else:
        _dW = X
        elementwise_affine = False
        w_arg = X
        dw_stride = 0

    dX = dY if in_place else torch.empty_like(dY)

    common = dict(
        dY_ptr=dY,
        dY_row_stride=dY.stride(0),
        dX_ptr=dX,
        dX_row_stride=dX.stride(0),
        X_ptr=X,
        X_row_stride=X.stride(0),
        X_dtype=torch_to_triton_dtype[X.dtype],
        W_ptr=w_arg,
        RSTD_ptr=RSTD,
        RSTD_row_stride=RSTD.stride(0),
        dW_ptr=_dW,
        dW_row_stride=dw_stride,
        n_rows=n_rows,
        n_cols=n_cols,
        offset=offset,
        casting_mode=casting_mode,
        elementwise_affine=elementwise_affine,
        BLOCK_SIZE=col_block,
    )

    if use_fused:
        if rows_per_block <= 1:
            _rms_norm_backward_row[(grid_size,)](**common)
        else:
            _rms_norm_backward_fused_2d[(grid_size,)](**common, ROWS_PER_BLOCK=rows_per_block)
    else:
        _rms_norm_backward_tiled[(grid_size,)](**common)

    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype) if elementwise_affine else None
    return dX, dW


class LigerRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if isinstance(X, torch.distributed.tensor.DTensor):
            X = X.full_tensor()

        Y, X, RSTD, casting_mode = rms_norm_forward(X, W, eps, offset, casting_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None
        if isinstance(dY, torch.distributed.tensor.DTensor):
            dY = dY.full_tensor()

        dX, dW = rms_norm_backward(dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.in_place)
        return dX, dW, None, None, None, None, None
