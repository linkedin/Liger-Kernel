import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
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
    num_warps: tl.constexpr,
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
def _sparsemax_backward_kernel(
    o_ptr, go_ptr, gi_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr
):
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
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0, cache_modifier=".ca").to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0.0
        go_sum += tl.sum(tl.where(supp, go_val, 0.0))
        supp_cnt += tl.sum(supp.to(tl.float32))

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0, cache_modifier=".ca").to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0.0
        gi_val = tl.where(
            supp,
            go_val - tl.cast(go_sum / tl.maximum(supp_cnt, 1e-6), gi_row.dtype.element_ty).to(tl.float32),
            0.0,
        )
        tl.store(gi_row + offs_iter, gi_val.to(gi_row.dtype.element_ty), mask=mask_iter, cache_modifier=".wb")


class LigerSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, dim: int):
        if dim < 0:
            dim += x.dim()
        ctx.dim = dim

        x_sw = x.transpose(dim, -1).contiguous()
        n_cols = x_sw.size(-1)
        n_rows = x_sw.numel() // n_cols
        x_flat = x_sw.view(n_rows, n_cols)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        out_flat = torch.empty_like(x_flat)
        grid = (n_rows,)

        x_sorted_flat = torch.sort(x_flat.float(), dim=-1, descending=True).values

        _sparsemax_forward_kernel[grid](
            x_flat,
            x_flat.stride(0),
            x_sorted_flat,
            x_sorted_flat.stride(0),
            out_flat,
            out_flat.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(out_flat)
        return out_flat.view_as(x_sw).transpose(dim, -1)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor):
        (out_flat,) = ctx.saved_tensors
        dim = ctx.dim

        go_sw = grad_out.transpose(dim, -1).contiguous()
        n_cols = go_sw.size(-1)
        n_rows = go_sw.numel() // n_cols
        go_flat = go_sw.view(n_rows, n_cols)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        gi_flat = torch.empty_like(go_flat)
        grid = (n_rows,)

        _sparsemax_backward_kernel[grid](
            out_flat,
            go_flat,
            gi_flat,
            out_flat.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return gi_flat.view_as(go_sw).transpose(dim, -1), None
