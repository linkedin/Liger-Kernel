import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _sparsemax_forward_kernel(
    x_ptr,
    x_stride_row,
    sorted_x_ptr,
    sorted_x_stride_row,
    o_ptr,
    o_stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    ptr_x_data_row = x_ptr + pid_row * x_stride_row
    ptr_sorted_x_data_row = sorted_x_ptr + pid_row * sorted_x_stride_row
    ptr_output_row = o_ptr + pid_row * o_stride_row

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    z_sorted_block = tl.load(
        ptr_sorted_x_data_row + offs,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".ca",
    ).to(tl.float32)

    z_valid = tl.where(mask, z_sorted_block, 0.0)
    cssv = tl.cumsum(z_valid, 0)

    r = (offs + 1).to(tl.float32)
    safe_r = tl.where(mask, r, 1.0)

    t_vec = (cssv - 1.0) / safe_r

    support = (z_sorted_block > t_vec) & mask

    k_int = tl.sum(support.to(tl.int32), 0)
    k_clamped_int = tl.maximum(k_int, 1)
    k = k_clamped_int.to(tl.float32)

    s = tl.sum(tl.where(support, z_sorted_block, 0.0), 0)

    tau = (s - 1.0) / k

    x_block = tl.load(
        ptr_x_data_row + offs,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)

    y = tl.maximum(x_block - tau, 0.0)

    tl.store(
        ptr_output_row + offs,
        y.to(ptr_output_row.dtype.element_ty),
        mask=mask,
        cache_modifier=".cs",
    )


@triton.jit
def _sparsemax_forward_tiled_kernel(
    x_ptr,
    x_stride_row,
    sorted_x_ptr,
    sorted_x_stride_row,
    o_ptr,
    o_stride_row,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    x_row_ptr = x_ptr + row * x_stride_row
    sorted_row_ptr = sorted_x_ptr + row * sorted_x_stride_row
    out_row_ptr = o_ptr + row * o_stride_row

    offs = tl.arange(0, BLOCK_SIZE)

    # ------------------------------------------------------------------
    # Step 1: find k (support size)
    # ------------------------------------------------------------------
    running_sum = tl.zeros((), tl.float32)
    k = tl.zeros((), tl.int32)
    sum_support = tl.zeros((), tl.float32)

    for tile in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tile * BLOCK_SIZE + offs
        mask = idx < n_cols

        z = tl.load(sorted_row_ptr + idx, mask=mask, other=-float("inf")).to(tl.float32)
        z_valid = tl.where(mask, z, 0.0)

        # prefix sum inside this tile
        tile_cumsum = tl.cumsum(z_valid, axis=0)

        # global cumsum = running_sum + tile_cumsum
        cssv = tile_cumsum + running_sum

        # global rank r = idx+1
        r = (idx + 1).to(tl.float32)
        safe_r = tl.where(mask, r, 1.0)

        t = (cssv - 1.0) / safe_r
        support = (z > t) & mask

        # find the last valid support index in this tile
        # (largest r where support is true)
        r_int = (idx + 1).to(tl.int32)
        support_r = tl.where(support, r_int, 0)
        tile_k = tl.max(support_r, axis=0)

        # update global k
        k = tl.maximum(k, tile_k)

        # update running_sum for next tile
        running_sum += tl.sum(z_valid, axis=0)

    # ------------------------------------------------------------------
    # Step 2: compute tau using k
    # ------------------------------------------------------------------
    sum_support = tl.zeros((), tl.float32)

    for tile in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tile * BLOCK_SIZE + offs
        mask = idx < n_cols

        z = tl.load(sorted_row_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        in_support = (idx + 1) <= k
        sum_support += tl.sum(tl.where(in_support & mask, z, 0.0), axis=0)

    k_f = tl.maximum(k, 1).to(tl.float32)
    tau = (sum_support - 1.0) / k_f

    # ------------------------------------------------------------------
    # Step 3: write output y = max(x - tau, 0)
    # ------------------------------------------------------------------
    for tile in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tile * BLOCK_SIZE + offs
        mask = idx < n_cols

        x = tl.load(x_row_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - tau, 0.0)

        tl.store(out_row_ptr + idx, y.to(out_row_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sparsemax_backward_tiled_kernel(o_ptr, go_ptr, gi_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    o_row = o_ptr + row * stride
    go_row = go_ptr + row * stride
    gi_row = gi_ptr + row * stride

    offs = tl.arange(0, BLOCK_SIZE)

    supp_cnt = tl.zeros((), tl.float32)
    go_sum = tl.zeros((), tl.float32)

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0
        go_sum += tl.sum(tl.where(supp, go_val, 0.0))
        supp_cnt += tl.sum(supp.to(tl.float32))

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)

        supp = o_val > 0
        gi_val = tl.where(
            supp,
            go_val - tl.cast(go_sum / tl.maximum(supp_cnt, 1e-6), gi_row.dtype.element_ty).to(tl.float32),
            0.0,
        )
        tl.store(gi_row + offs_iter, gi_val.to(gi_row.dtype.element_ty), mask=mask_iter)


@triton.jit
def _sparsemax_backward_kernel(
    o_ptr,
    go_ptr,
    gi_ptr,
    stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    o_row = o_ptr + row * stride
    go_row = go_ptr + row * stride
    gi_row = gi_ptr + row * stride

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    o_val = tl.load(o_row + offs, mask=mask, other=0.0).to(tl.float32)
    go_val = tl.load(go_row + offs, mask=mask, other=0.0).to(tl.float32)
    supp = (o_val > 0.0) & mask

    go_sum = tl.sum(tl.where(supp, go_val, 0.0), axis=0)
    supp_cnt = tl.sum(supp.to(tl.float32), axis=0)

    gi_val = tl.where(
        supp,
        go_val - go_sum / tl.maximum(supp_cnt, 1.0),
        0.0,
    )
    tl.store(gi_row + offs, gi_val.to(gi_row.dtype.element_ty), mask=mask)


def sparsemax_forward(x, dim):
    if dim < 0:
        dim += x.dim()

    x_sw = x.transpose(dim, -1).contiguous()
    n_cols = x_sw.size(-1)
    n_rows = x_sw.numel() // n_cols
    x_flat = x_sw.view(n_rows, n_cols)

    x_sorted_flat = torch.sort(x_flat.float(), dim=-1, descending=True).values

    # tiling strategy
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=8,
        memory_multiplier=6.0,
        shapes=((n_cols,),),
        tiling_dims=(0,),
    )

    if tile_shapes and len(tile_shapes) > 0:
        BLOCK_SIZE = tile_shapes[0][0]
    else:
        BLOCK_SIZE = 2048

    out_flat = torch.empty_like(x_flat)
    grid = (n_rows,)

    if n_cols <= BLOCK_SIZE:
        # non-tiled kernel: single load covers whole row
        _sparsemax_forward_kernel[grid](
            x_flat,
            x_flat.stride(0),
            x_sorted_flat,
            x_sorted_flat.stride(0),
            out_flat,
            out_flat.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # tiled kernel: loop over tiles to avoid UB overflow
        _sparsemax_forward_tiled_kernel[grid](
            x_flat,
            x_flat.stride(0),
            x_sorted_flat,
            x_sorted_flat.stride(0),
            out_flat,
            out_flat.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    y = out_flat.view_as(x_sw).transpose(dim, -1)
    return y, out_flat


def sparsemax_backward(
    grad_out: torch.Tensor,
    out_flat: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    if dim < 0:
        dim += grad_out.dim()

    grad_sw = grad_out.transpose(dim, -1).contiguous()
    n_cols = grad_sw.size(-1)
    n_rows = grad_sw.numel() // n_cols
    go_flat = grad_sw.view(n_rows, n_cols)

    dx_flat = torch.empty_like(go_flat).contiguous()
    grid = (n_rows,)

    # use single-pass kernel when feasible
    if n_cols <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        _sparsemax_backward_kernel[grid](
            out_flat,
            go_flat,
            dx_flat,
            out_flat.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    else:
        # use tiling strategy for very large n_cols
        tile_shapes = compute_default_tiling_strategy(
            safety_margin=0.9,
            dtype_size=8,
            memory_multiplier=100.0,
            shapes=((n_cols,),),
            tiling_dims=(0,),
        )

        if tile_shapes and len(tile_shapes) > 0:
            BLOCK_SIZE = tile_shapes[0][0]
        else:
            BLOCK_SIZE = 256

        _sparsemax_backward_tiled_kernel[grid](
            out_flat,
            go_flat,
            dx_flat,
            out_flat.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    dx = dx_flat.view_as(grad_sw).transpose(dim, -1)
    return dx


class LigerSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, dim: int):
        y, out_flat = sparsemax_forward(x, dim)
        ctx.save_for_backward(out_flat)
        ctx.dim = dim
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor):
        (out_flat,) = ctx.saved_tensors
        dx = sparsemax_backward(grad_out, out_flat, ctx.dim)
        return dx, None
