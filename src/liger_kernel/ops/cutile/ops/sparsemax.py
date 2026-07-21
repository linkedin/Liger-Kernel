# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous


# Exact sparsemax threshold (Martins & Astudillo 2016, Alg. 1):
# on the descending-sorted row z, find support size k = max{ j : z_(j) > (cssv_j - 1)/j },
# then tau = (sum_{i<=k} z_(i) - 1)/k. The whole row lives in one BLOCK_SIZE tile so ct.cumsum
# gives the running prefix sum.
@ct.kernel(occupancy=4)
def _sparsemax_fwd_kernel_ct(
    y_output,
    x_input,
    x_sorted,  # row-wise descending sort of x_input (fp32), produced by torch.sort
    N_COLS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],  # = next_pow2(N_COLS): whole row in one tile for the cumsum
):
    row_idx = ct.bid(0)
    col_idx = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    one_b = ct.full((BLOCK_SIZE,), 1.0, ct.float32)
    zero_b = ct.full((BLOCK_SIZE,), 0.0, ct.float32)
    valid_mask = col_idx < N_COLS
    valid_f = ct.astype(valid_mask, ct.float32)

    z_sorted = ct.astype(
        ct.gather(x_sorted, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
        ct.float32,
    )
    # Masked entries (col >= N_COLS) contribute 0 to the prefix sum / are excluded from support.
    z_valid = z_sorted * valid_f
    cssv = ct.cumsum(z_valid, 0)
    r = ct.astype(col_idx, ct.float32) + one_b
    t_vec = (cssv - one_b) / r
    support = (z_sorted > t_vec) & valid_mask

    # Support size k, clamped to >= 1.
    k_int = ct.maximum(ct.sum(ct.astype(support, ct.int32), 0, keepdims=True), ct.full((1,), 1, ct.int32))
    k = ct.astype(k_int, ct.float32)
    s = ct.sum(ct.where(support, z_sorted, zero_b), 0, keepdims=True)
    tau = (s - ct.full((1,), 1.0, ct.float32)) / k

    x_row = ct.astype(
        ct.gather(x_input, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
        ct.float32,
    )
    y = ct.maximum(x_row - tau, ct.full((BLOCK_SIZE,), 0.0, ct.float32))
    ct.scatter(y_output, (row_idx, col_idx), ct.astype(y, y_output.dtype), check_bounds=True)


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
    """Exact, sort-based sparsemax forward.

    Sort each row descending (torch.sort), then one kernel computes the exact threshold tau
    via prefix sums and applies max(x - tau, 0). The whole row must fit in one tile
    (BLOCK = next_pow2(n_cols)) so ct.cumsum yields the running prefix sum.
    """
    if dim < 0:
        dim += x.dim()
    x_sw = x.transpose(dim, -1).contiguous()
    n_cols = x_sw.size(-1)
    n_rows = x_sw.numel() // n_cols
    x_flat = x_sw.view(n_rows, n_cols)
    x_sorted = torch.sort(x_flat.float(), dim=-1, descending=True).values

    BLOCK_SIZE = _next_power_of_2(n_cols)  # whole row in one tile (required for the cumsum)
    out_flat = torch.empty_like(x_flat)
    ct.launch(
        torch.cuda.current_stream(),
        (n_rows, 1, 1),
        _sparsemax_fwd_kernel_ct,
        (out_flat, x_flat, x_sorted, int(n_cols), int(BLOCK_SIZE)),
    )

    return out_flat.view_as(x_sw).transpose(dim, -1).contiguous(), out_flat


def _sparsemax_backward(grad_out: torch.Tensor, out_flat: torch.Tensor, dim: int):
    grad_sw = grad_out.transpose(dim, -1).contiguous()
    n_cols = grad_sw.size(-1)
    n_rows = grad_sw.numel() // n_cols
    go_flat = grad_sw.view(n_rows, n_cols).contiguous()

    BLOCK_SIZE = min(_next_power_of_2(n_cols), 4096)
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
