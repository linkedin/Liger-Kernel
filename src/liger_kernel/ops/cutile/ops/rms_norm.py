"""Register-resident RMSNorm forward and persistent fused backward for cuTile."""

import math
import os

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous

ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]
ConstInt = ct.Constant[int]

_CASTING_MODE_NONE = -1
_CASTING_MODE_LLAMA = 0
_CASTING_MODE_GEMMA = 1

_STR_TO_CASTING_MODE = {
    "llama": _CASTING_MODE_LLAMA,
    "gemma": _CASTING_MODE_GEMMA,
    "none": _CASTING_MODE_NONE,
}

_DW_REDUCTION_POLICY = os.environ.get("LIGER_RMS_DW_REDUCTION", "auto")
_DW_REDUCTION_TILE_SIZE = int(os.environ.get("LIGER_RMS_DW_REDUCTION_TILE_SIZE") or 32)
_DW_REDUCTION_ROW_TILE_SIZE = int(os.environ.get("LIGER_RMS_DW_REDUCTION_ROW_TILE_SIZE") or 32)


def _calculate_settings(n_cols):
    block_size = _next_power_of_2(n_cols)
    if block_size > 65536:
        raise RuntimeError(f"Hidden dimension {n_cols} exceeds maximum supported size of 65536.")
    return block_size


@ct.kernel(occupancy=1)
def _rms_norm_fwd_kernel_ct(
    x,
    weight,
    y,
    rstd_out,
    n_cols: ConstInt,
    eps: ConstFloat,
    offset: ConstFloat,
    casting_mode: ConstInt,
    elementwise_affine: ConstBool,
    block_size: ConstInt,
    aligned: ConstBool,
):
    """One block per row; retain X across its reduction and output phases."""
    row = ct.bid(0)
    columns = ct.arange(block_size, dtype=ct.int32)
    check_bounds = not aligned

    x_input = ct.gather(x, (row, columns), check_bounds=check_bounds, padding_value=0.0, latency=4)
    x_float = ct.astype(x_input, ct.float32)
    sum_square = ct.sum(x_float * x_float, 0, keepdims=False)
    reciprocal_rms = ct.rsqrt(sum_square / n_cols + eps)
    ct.scatter(rstd_out, row, reciprocal_rms)

    normalized = x_float * reciprocal_rms
    if casting_mode == _CASTING_MODE_LLAMA:
        normalized = ct.astype(ct.astype(normalized, x.dtype), ct.float32)
    if elementwise_affine:
        weight_tile = ct.astype(
            ct.gather(weight, columns, check_bounds=check_bounds, padding_value=0.0, latency=3),
            ct.float32,
        )
        normalized = normalized * (weight_tile + offset)
    ct.scatter(y, (row, columns), ct.astype(normalized, y.dtype), check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _rms_norm_bwd_dx_kernel_ct(
    x,
    dy,
    weight,
    rstd,
    dx,
    n_cols: ConstInt,
    offset: ConstFloat,
    casting_mode: ConstInt,
    elementwise_affine: ConstBool,
    block_size: ConstInt,
    aligned: ConstBool,
):
    """One block per row for the non-affine or dX-only backward path."""
    row = ct.bid(0)
    columns = ct.arange(block_size, dtype=ct.int32)
    check_bounds = not aligned

    x_tile = ct.astype(
        ct.gather(x, (row, columns), check_bounds=check_bounds, padding_value=0.0, latency=4),
        ct.float32,
    )
    dy_tile = ct.astype(
        ct.gather(dy, (row, columns), check_bounds=check_bounds, padding_value=0.0, latency=4),
        ct.float32,
    )
    reciprocal_rms = ct.astype(ct.load(rstd, row, shape=(), latency=3), ct.float32)
    m = dy_tile
    if elementwise_affine:
        weight_tile = ct.astype(
            ct.gather(weight, columns, check_bounds=check_bounds, padding_value=0.0, latency=3),
            ct.float32,
        )
        m = dy_tile * (weight_tile + offset)

    dot = ct.sum(m * x_tile, 0, keepdims=False)
    coefficient = -(reciprocal_rms * reciprocal_rms * dot) / n_cols
    dx_tile = reciprocal_rms * (m + coefficient * x_tile)
    ct.scatter(dx, (row, columns), ct.astype(dx_tile, dx.dtype), check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _rms_norm_bwd_combined_kernel_ct(
    x,
    dy,
    weight,
    rstd,
    dx,
    dw_partial,
    n_rows: ConstInt,
    n_cols: ConstInt,
    rows_per_program: ConstInt,
    offset: ConstFloat,
    casting_mode: ConstInt,
    block_size: ConstInt,
    aligned: ConstBool,
):
    """Persistent fused dX+dW; retain W and FP32 dW accumulators across rows."""
    program = ct.bid(0)
    columns = ct.arange(block_size, dtype=ct.int32)
    check_bounds = not aligned

    weight_tile = ct.astype(
        ct.gather(weight, columns, check_bounds=check_bounds, padding_value=0.0, latency=3),
        ct.float32,
    )
    dw_accumulator = ct.full((block_size,), 0.0, dtype=ct.float32)

    for row_offset in range(rows_per_program):
        row = program * rows_per_program + row_offset
        if row < n_rows:
            x_input = ct.gather(x, (row, columns), check_bounds=check_bounds, padding_value=0.0, latency=4)
            dy_input = ct.gather(dy, (row, columns), check_bounds=check_bounds, padding_value=0.0, latency=4)
            x_tile = ct.astype(x_input, ct.float32)
            dy_tile = ct.astype(dy_input, ct.float32)
            reciprocal_rms = ct.astype(ct.load(rstd, row, shape=(), latency=3), ct.float32)

            m = dy_tile * (weight_tile + offset)
            dot = ct.sum(m * x_tile, 0, keepdims=False)
            coefficient = -(reciprocal_rms * reciprocal_rms * dot) / n_cols
            dx_tile = reciprocal_rms * (m + coefficient * x_tile)
            ct.scatter(dx, (row, columns), ct.astype(dx_tile, dx.dtype), check_bounds=check_bounds)

            normalized = x_tile * reciprocal_rms
            if casting_mode == _CASTING_MODE_LLAMA:
                normalized = ct.astype(ct.astype(normalized, x.dtype), ct.float32)
            dw_accumulator = dw_accumulator + dy_tile * normalized

    ct.scatter(dw_partial, (program, columns), dw_accumulator, check_bounds=check_bounds)


@ct.kernel(occupancy=1)
def _rms_norm_dw_reduce_kernel_ct(
    dw_partial,
    dw,
    num_programs: ConstInt,
    n_cols: ConstInt,
    tile_size: ConstInt,
    row_tile_size: ConstInt,
    aligned: ConstBool,
):
    """Reduce FP32 dW partials and cast once to the output dtype."""
    column_tile = ct.bid(0)
    columns = column_tile * tile_size + ct.arange(tile_size, dtype=ct.int32)
    check_bounds = not aligned
    accumulator = ct.full((tile_size,), 0.0, dtype=ct.float32)

    num_row_tiles = (num_programs + row_tile_size - 1) // row_tile_size
    for row_tile in range(num_row_tiles):
        partials = ct.load(
            dw_partial,
            index=(row_tile, column_tile),
            shape=(row_tile_size, tile_size),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        )
        accumulator += ct.sum(partials, 0, keepdims=False)

    ct.scatter(dw, columns, ct.astype(accumulator, dw.dtype), check_bounds=check_bounds)


def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
    with device_context(X.device):
        del row_mode
        if not isinstance(casting_mode, int):
            if casting_mode not in _STR_TO_CASTING_MODE:
                raise ValueError(f"Invalid casting mode: {casting_mode}")
            casting_mode = _STR_TO_CASTING_MODE[casting_mode]
        elif casting_mode not in _STR_TO_CASTING_MODE.values():
            raise ValueError(f"Invalid casting mode: {casting_mode}")

        shape = X.shape
        hidden_size = shape[-1]
        x_2d = X.contiguous().view(-1, hidden_size)
        n_rows, n_cols = x_2d.shape
        block_size = _calculate_settings(n_cols)
        aligned = n_cols == block_size
        elementwise_affine = W is not None
        if elementwise_affine:
            if W.shape != (n_cols,):
                raise ValueError(f"expected weight shape {(n_cols,)}, got {tuple(W.shape)}")
            weight = W.contiguous()

        y = torch.empty_like(x_2d)
        reciprocal_rms = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        weight_arg = weight if elementwise_affine else reciprocal_rms
        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            _rms_norm_fwd_kernel_ct,
            (
                x_2d,
                weight_arg,
                y,
                reciprocal_rms,
                int(n_cols),
                float(eps),
                float(offset),
                int(casting_mode),
                bool(elementwise_affine),
                int(block_size),
                bool(aligned),
            ),
        )
        return y.view(*shape), x_2d, reciprocal_rms, block_size, None, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode):
    with device_context(X.device):
        del num_warps, row_mode
        shape = dY.shape
        hidden_size = shape[-1]
        dy_2d = dY.contiguous().view(-1, hidden_size)
        x_2d = X.contiguous().view(-1, hidden_size)
        n_rows, n_cols = dy_2d.shape
        block_size = _calculate_settings(n_cols) if BLOCK_SIZE is None else BLOCK_SIZE
        aligned = n_cols == block_size
        elementwise_affine = W is not None
        dx = dy_2d if in_place else torch.empty_like(dy_2d)

        if not elementwise_affine:
            ct.launch(
                torch.cuda.current_stream(),
                (n_rows, 1, 1),
                _rms_norm_bwd_dx_kernel_ct,
                (
                    x_2d,
                    dy_2d,
                    RSTD.contiguous(),
                    RSTD.contiguous(),
                    dx,
                    int(n_cols),
                    float(offset),
                    int(casting_mode),
                    False,
                    int(block_size),
                    bool(aligned),
                ),
            )
            return dx.view(*shape), None

        weight = W.contiguous()
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
        strip_multiplier = 2 if n_rows >= 16 * sm_count else 1
        num_programs = max(1, min(n_rows, strip_multiplier * sm_count))
        rows_per_program = math.ceil(n_rows / num_programs)
        dw_partial = torch.empty((num_programs, n_cols), dtype=torch.float32, device=W.device)

        ct.launch(
            torch.cuda.current_stream(),
            (num_programs, 1, 1),
            _rms_norm_bwd_combined_kernel_ct,
            (
                x_2d,
                dy_2d,
                weight,
                RSTD.contiguous(),
                dx,
                dw_partial,
                int(n_rows),
                int(n_cols),
                int(rows_per_program),
                float(offset),
                int(casting_mode),
                int(block_size),
                bool(aligned),
            ),
        )
        if _DW_REDUCTION_POLICY in ("auto", "custom"):
            tile_size = _DW_REDUCTION_TILE_SIZE
            if tile_size <= 0 or tile_size & (tile_size - 1):
                raise ValueError(f"LIGER_RMS_DW_REDUCTION_TILE_SIZE must be a positive power of two, got {tile_size}")
            row_tile_size = _DW_REDUCTION_ROW_TILE_SIZE
            if row_tile_size <= 0 or row_tile_size & (row_tile_size - 1):
                raise ValueError(
                    f"LIGER_RMS_DW_REDUCTION_ROW_TILE_SIZE must be a positive power of two, got {row_tile_size}"
                )
            dw = torch.empty_like(weight)
            ct.launch(
                torch.cuda.current_stream(),
                (math.ceil(n_cols / tile_size), 1, 1),
                _rms_norm_dw_reduce_kernel_ct,
                (
                    dw_partial,
                    dw,
                    int(num_programs),
                    int(n_cols),
                    int(tile_size),
                    int(row_tile_size),
                    bool(n_cols % tile_size == 0),
                ),
            )
        elif _DW_REDUCTION_POLICY == "torch":
            dw = dw_partial.sum(dim=0).to(W.dtype)
        else:
            raise ValueError(f"Invalid LIGER_RMS_DW_REDUCTION policy: {_DW_REDUCTION_POLICY}")
        return dx.view(*shape), dw


class LigerRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(
            X,
            W,
            eps,
            offset,
            casting_mode,
            row_mode,
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        if W is None:
            ctx.save_for_backward(X, RSTD)
        else:
            ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None
        dX, dW = rms_norm_backward(
            dY,
            X,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.in_place,
            ctx.row_mode,
        )
        return dX, dW, None, None, None, None, None
