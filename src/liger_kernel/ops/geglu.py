import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import infer_device_arch
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


# Wide one-row blocks use 128 registers/thread on SM103, limiting occupancy to
# 25%. Column tiles keep the same elementwise math while increasing occupancy.
_GEGLU_SM103_TILE_SIZE = 1024
_GEGLU_SM103_TILE_MIN_BLOCK = 16384


def _should_use_sm103_tiling(n_cols):
    return infer_device_arch() == "blackwell_ultra" and triton.next_power_of_2(n_cols) >= _GEGLU_SM103_TILE_MIN_BLOCK


def _geglu_sm103_tile_settings(n_cols):
    block_size = min(_GEGLU_SM103_TILE_SIZE, triton.next_power_of_2(n_cols))
    return block_size, 4


@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a.cast(b_row.dtype) * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    geglu_a = geglu_a.to(dc_row.dtype).to(tl.float32)

    db_row = dc_row.cast(tl.float32) * geglu_a

    # Gradient w.r.t. a can be computed with:
    # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row.to(dc_row.dtype), mask=mask)


@triton.jit
def _geglu_tanh_forward_kernel_tiled(a, b, c, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The second grid axis selects a fixed-width column tile within each row.
    row = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1)

    a += row * stride
    b += row * stride
    c += row * stride

    col_offsets = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a.cast(b_row.dtype) * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel_tiled(dc, a, b, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The second grid axis selects a fixed-width column tile within each row.
    row = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1)

    dc += row * stride
    a += row * stride
    b += row * stride

    col_offsets = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    geglu_a = geglu_a.to(dc_row.dtype).to(tl.float32)
    db_row = dc_row.cast(tl.float32) * geglu_a

    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row.to(dc_row.dtype), mask=mask)


def geglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    if _should_use_sm103_tiling(n_cols):
        block_size, num_warps = _geglu_sm103_tile_settings(n_cols)
        grid = (n_rows, triton.cdiv(n_cols, block_size))
        with device_context(a.device):
            _geglu_tanh_forward_kernel_tiled[grid](
                a,
                b,
                c,
                c.stride(-2),
                n_cols,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        return a, b, c.view(*ori_shape)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with device_context(a.device):
        _geglu_tanh_forward_kernel[(n_rows,)](
            a,
            b,
            c,
            c.stride(-2),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return a, b, c.view(*ori_shape)


def geglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    if _should_use_sm103_tiling(n_cols):
        block_size, num_warps = _geglu_sm103_tile_settings(n_cols)
        grid = (n_rows, triton.cdiv(n_cols, block_size))
        with device_context(a.device):
            _geglu_tanh_backward_kernel_tiled[grid](
                dc,
                a,
                b,
                dc.stride(-2),
                n_cols,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        return a.view(*ori_shape), b.view(*ori_shape)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with device_context(a.device):
        _geglu_tanh_backward_kernel[(n_rows,)](
            dc,
            a,
            b,
            dc.stride(-2),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return a.view(*ori_shape), b.view(*ori_shape)


class LigerGELUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = geglu_backward(a, b, dc)
        return a, b
