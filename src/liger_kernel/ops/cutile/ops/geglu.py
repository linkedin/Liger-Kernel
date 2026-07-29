# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]

MAX_FUSED_SIZE_FWD = 4096
MAX_FUSED_SIZE_BWD = 512
SQRT_2_OVER_PI = 0.7978845608028654


def _calculate_block_size(n_cols, max_fused_size):
    BLOCK_SIZE = _next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, max_fused_size)
    return BLOCK_SIZE


@ct.kernel(occupancy=1)
def _geglu_fwd_kernel_ct(
    a,  # (n_rows, n_cols) input a
    b,  # (n_rows, n_cols) input b
    c,  # (n_rows, n_cols) output c
    n_cols: ConstInt,
    BLOCK_SIZE: ConstInt,
    CHECK_BOUNDS: ConstBool,
):
    """GEGLU forward. CHECK_BOUNDS=False (aligned path) is ~17-20% faster on B200."""
    row_idx = ct.bid(0)

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        a_tile = ct.astype(ct.gather(a, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)
        b_tile = ct.gather(b, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)

        a_sq = a_tile * a_tile
        tanh_arg = SQRT_2_OVER_PI * (a_tile + 0.044715 * a_sq * a_tile)
        tanh_result = ct.tanh(tanh_arg)
        geglu_a = 0.5 * a_tile * (1.0 + tanh_result)

        c_tile = ct.astype(geglu_a, b_tile.dtype) * b_tile
        ct.scatter(c, (row_idx, col_idx), c_tile, check_bounds=CHECK_BOUNDS)


@ct.kernel
def _geglu_bwd_kernel_ct(
    dc,  # (n_rows, n_cols) upstream gradient
    a,  # (n_rows, n_cols) saved input a; da written in-place
    b,  # (n_rows, n_cols) saved input b; db written in-place
    n_cols: ConstInt,
    BLOCK_SIZE: ConstInt,
    CHECK_BOUNDS: ConstBool,
):
    """GEGLU backward. CHECK_BOUNDS=False (aligned path) is faster on B200."""
    row_idx = ct.bid(0)

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        dc_tile = ct.gather(dc, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)
        a_tile = ct.astype(ct.gather(a, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)
        b_tile = ct.gather(b, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)

        a_sq = a_tile * a_tile
        tanh_arg = SQRT_2_OVER_PI * (a_tile + 0.044715 * a_sq * a_tile)
        tanh_result = ct.tanh(tanh_arg)
        geglu_a = 0.5 * a_tile * (1.0 + tanh_result)
        geglu_a = ct.astype(ct.astype(geglu_a, dc_tile.dtype), ct.float32)

        db = ct.astype(dc_tile, ct.float32) * geglu_a

        # da = dc * b * (term1 + 0.5*a*(1-tanh^2)*sqrt(2/pi)*(1+3*0.044715*a^2))
        term1 = 0.5 * (1.0 + tanh_result)
        sech2 = (1 - tanh_result) * (1 + tanh_result)
        term2 = 0.5 * a_tile * sech2 * (SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * a_sq))
        da = dc_tile * b_tile * (term1 + term2)

        ct.scatter(a, (row_idx, col_idx), ct.astype(da, a.dtype), check_bounds=CHECK_BOUNDS)
        ct.scatter(b, (row_idx, col_idx), ct.astype(db, dc_tile.dtype), check_bounds=CHECK_BOUNDS)


def geglu_forward(a, b):
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols).contiguous()
    b = b.view(-1, n_cols).contiguous()
    n_rows = a.shape[0]
    c = torch.empty_like(a)
    BLOCK_SIZE = _calculate_block_size(n_cols, MAX_FUSED_SIZE_FWD)
    aligned = n_cols % BLOCK_SIZE == 0
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        _geglu_fwd_kernel_ct,
        (a, b, c, int(n_cols), int(BLOCK_SIZE), not aligned),
    )
    return a, b, c.view(*ori_shape)


def geglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols).contiguous()
    n_rows = dc.shape[0]
    BLOCK_SIZE = _calculate_block_size(n_cols, MAX_FUSED_SIZE_BWD)
    aligned = n_cols % BLOCK_SIZE == 0
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        _geglu_bwd_kernel_ct,
        (dc, a, b, int(n_cols), int(BLOCK_SIZE), not aligned),
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
