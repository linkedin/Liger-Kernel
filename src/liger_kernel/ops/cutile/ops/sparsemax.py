# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous

# 20 bisections give fp32-scale tau precision (~1e-6 relative interval).
_BSEARCH_ITER = 20


def _select_block_size(n_cols: int) -> int:
    return min(_next_power_of_2(n_cols), 4096)


@ct.kernel(occupancy=4)
def _sparsemax_bsearch_kernel_ct(
    y_output,
    x_input,
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    BSEARCH_ITER: ct.Constant[int],
):
    row_idx = ct.bid(0)
    n_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE

    x_max = ct.full((1,), -1e38, dtype=ct.float32)

    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        x_tile = ct.astype(
            ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=-1e38),
            ct.float32,
        )
        x_max = ct.maximum(x_max, ct.max(x_tile, 0, keepdims=True))

    # tau_lo = x_max - 1 is the tightest universally-valid lower bound:
    # f(x_max - 1) = sum_{x > x_max - 1}(x - (x_max - 1)) >= (x_max - (x_max - 1)) = 1.
    # Using (sum_x - 1)/n_cols breaks when some entries are large-negative mask sentinels
    # (e.g. -1e9): the lower bound falls below the true tau on rows with few finite entries,
    # and the bisection range becomes too wide for 20 iterations to converge precisely.
    tau_lo = x_max - ct.full((1,), 1.0, ct.float32)
    tau_hi = x_max

    one = ct.full((1,), 1.0, ct.float32)
    half = ct.full((1,), 0.5, ct.float32)

    for _ in range(BSEARCH_ITER):
        tau_mid = half * (tau_lo + tau_hi)
        f = ct.full((1,), 0.0, ct.float32)

        for ci in range(n_chunks):
            col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
            x_tile = ct.astype(
                ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=-1e38),
                ct.float32,
            )
            valid_mask = ct.astype(col_idx < N_COLS, ct.float32)
            in_supp = ct.astype(x_tile > tau_mid, ct.float32) * valid_mask
            f = f + ct.sum(in_supp * (x_tile - tau_mid), 0, keepdims=True)

        tau_lo = ct.where(f >= one, tau_mid, tau_lo)
        tau_hi = ct.where(f < one, tau_mid, tau_hi)

    tau = half * (tau_lo + tau_hi)

    zero = ct.full((BLOCK_SIZE,), 0.0, ct.float32)
    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        x_tile = ct.astype(
            ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        y_tile = ct.maximum(x_tile - tau, zero)
        ct.scatter(y_output, (row_idx, col_idx), ct.astype(y_tile, y_output.dtype), check_bounds=True)


# Low-occupancy variant for large N (>16384): at high occupancy 7×128KB/SM thrashes L2;
# occ=2 keeps each row resident in L2 across bisection passes. Generated via replace_hints
# from the same kernel definition to avoid duplicating ~60 lines of bisection code.
_sparsemax_bsearch_kernel_large_ct = _sparsemax_bsearch_kernel_ct.replace_hints(occupancy=2)


@ct.kernel
def _sparsemax_bwd_kernel_ct(
    grad_input,
    output,
    grad_output,
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    row_idx = ct.bid(0)
    n_chunks = (N_COLS + BLOCK_SIZE - 1) // BLOCK_SIZE

    go_sum_tile = ct.full((BLOCK_SIZE,), 0.0, ct.float32)
    supp_cnt_tile = ct.full((BLOCK_SIZE,), 0.0, ct.float32)

    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        o_tile = ct.astype(ct.gather(output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        go_tile = ct.astype(
            ct.gather(grad_output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32
        )
        supp_f = ct.astype(o_tile > ct.full((BLOCK_SIZE,), 0.0, ct.float32), ct.float32)
        go_sum_tile = go_sum_tile + supp_f * go_tile
        supp_cnt_tile = supp_cnt_tile + supp_f

    go_sum = ct.sum(go_sum_tile, 0, keepdims=False)
    supp_cnt = ct.sum(supp_cnt_tile, 0, keepdims=False)
    mean_go = go_sum / (supp_cnt + 1e-6)

    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
        o_tile = ct.astype(ct.gather(output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        go_tile = ct.astype(
            ct.gather(grad_output, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32
        )
        supp_f = ct.astype(o_tile > ct.full((BLOCK_SIZE,), 0.0, ct.float32), ct.float32)
        gi_tile = supp_f * (go_tile - mean_go)
        ct.scatter(grad_input, (row_idx, col_idx), ct.astype(gi_tile, grad_input.dtype), check_bounds=True)


def _sparsemax_forward(x: torch.Tensor, dim: int):
    if dim < 0:
        dim += x.dim()
    x_sw = x.transpose(dim, -1).contiguous()
    n_cols = x_sw.size(-1)
    n_rows = x_sw.numel() // n_cols
    x_flat = x_sw.view(n_rows, n_cols)

    BLOCK_SIZE = _select_block_size(n_cols)
    out_flat = torch.empty_like(x_flat)
    kernel = _sparsemax_bsearch_kernel_large_ct if n_cols > 16384 else _sparsemax_bsearch_kernel_ct
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        kernel,
        (out_flat, x_flat, int(n_cols), int(BLOCK_SIZE), int(_BSEARCH_ITER)),
    )

    return out_flat.view_as(x_sw).transpose(dim, -1).contiguous(), out_flat


def _sparsemax_backward(grad_out: torch.Tensor, out_flat: torch.Tensor, dim: int):
    grad_sw = grad_out.transpose(dim, -1).contiguous()
    n_cols = grad_sw.size(-1)
    n_rows = grad_sw.numel() // n_cols
    go_flat = grad_sw.view(n_rows, n_cols).contiguous()

    BLOCK_SIZE = _select_block_size(n_cols)
    dx_flat = torch.empty_like(go_flat)
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        _sparsemax_bwd_kernel_ct,
        (dx_flat, out_flat, go_flat, int(n_cols), int(BLOCK_SIZE)),
    )

    return dx_flat.view_as(grad_sw).transpose(dim, -1)


class LigerSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, dim: int):
        y, out_flat = _sparsemax_forward(x, dim)
        ctx.save_for_backward(out_flat)
        ctx.dim = dim
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor):
        (out_flat,) = ctx.saved_tensors
        dx = _sparsemax_backward(grad_out, out_flat, ctx.dim)
        return dx, None
