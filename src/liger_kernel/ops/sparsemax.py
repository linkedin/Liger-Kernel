import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _sparsemax_forward_kernel(
    x_ptr,
    x_stride, 
    tau_ptr, 
    tau_stride, 
    o_ptr, 
    o_stride, 
    n_cols, 
    block_size: tl.constexpr, 
    num_warps: tl.constexpr
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_stride
    tau_row = tau_ptr + row * tau_stride
    o_row = o_ptr + row * o_stride

    tau = tl.load(tau_row).to(tl.float32)
    out_ty = o_row.dtype.element_ty

    for i in range(0, tl.cdiv(n_cols, block_size)):
        offs = i * block_size + tl.arange(0, block_size)
        mask = offs < n_cols

        x_fp32 = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = x_fp32 - tau
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(o_row + offs, y.to(out_ty), mask=mask)


@triton.jit
def _sparsemax_backward_kernel(
    o_ptr,
    o_stride,
    go_ptr,
    go_stride,
    gi_ptr,
    gi_stride,
    n_cols,
    block_size: tl.constexpr,
    num_warps: tl.constexpr
):
    row = tl.program_id(0)
    o_row = o_ptr + row * o_stride
    go_row = go_ptr + row * go_stride
    gi_row = gi_ptr + row * gi_stride

    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_cnt = tl.zeros((), dtype=tl.float32)
    out_ty = o_row.dtype.element_ty

    for i in range(0, tl.cdiv(n_cols, block_size)):
        offs = i * block_size + tl.arange(0, block_size)
        mask = offs < n_cols

        o_f32 = tl.load(o_row + offs, mask=mask, other=0.0).to(tl.float32)
        go_f32 = tl.load(go_row + offs, mask=mask, other=0.0).to(tl.float32)
        supp = o_f32 > 0.0

        acc_sum += tl.sum(tl.where(supp, go_f32, 0.0))
        acc_cnt += tl.sum(tl.where(supp, 1.0, 0.0))

    v_hat_fp32 = acc_sum / tl.maximum(acc_cnt, 1e-9)
    v_hat_q = tl.cast(v_hat_fp32, out_ty)
    v_hat = v_hat_q.to(tl.float32)

    for i in range(0, tl.cdiv(n_cols, block_size)):
        offs = i * block_size + tl.arange(0, block_size)
        mask = offs < n_cols

        o_f32 = tl.load(o_row + offs, mask=mask, other=0.0).to(tl.float32)
        go_f32 = tl.load(go_row + offs, mask=mask, other=0.0).to(tl.float32)
        supp = o_f32 > 0.0

        gi_f32 = tl.where(supp, go_f32 - v_hat, 0.0)
        tl.store(gi_row + offs, gi_f32.to(out_ty), mask=mask)


class LigerSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, dim: int):
        if dim < 0:
            dim = x.dim() + dim
        ctx.dim = dim
        ctx.orig_shape = x.shape

        x_sw = x.transpose(dim, -1)
        n_cols = x_sw.size(-1)
        n_rows = x_sw.numel() // n_cols
        x_flat = x_sw.reshape(n_rows, n_cols).contiguous()

        x_f32 = x_flat.to(torch.float32)
        x_sorted, _ = torch.sort(x_f32, dim=-1, descending=True)
        csum = torch.cumsum(x_sorted, dim=-1)
        r = torch.arange(1, n_cols + 1, device=x.device, dtype=torch.float32).view(1, -1)
        bound = 1 + r * x_sorted
        support = bound > csum
        k = support.sum(dim=-1, keepdim=True).clamp(min=1)
        s = (x_sorted * support).sum(dim=-1, keepdim=True)
        tau_f32 = (s - 1) / k

        tau_q = tau_f32.to(x_flat.dtype)
        tau_e = tau_q.expand(n_rows, n_cols).contiguous()

        out_flat = torch.empty_like(x_flat)
        grid = (n_rows,)
        block_size, num_warps = calculate_settings(n_cols)
        _sparsemax_forward_kernel[grid](
            x_flat,
            x_flat.stride(0),
            tau_e,
            tau_e.stride(0),
            out_flat,
            out_flat.stride(0),
            n_cols,
            block_size=block_size,
            num_warps=num_warps,
        )

        ctx.save_for_backward(out_flat)
        out = out_flat.reshape(x_sw.shape).transpose(dim, -1)
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor):
        (out_flat,) = ctx.saved_tensors
        dim = ctx.dim

        go_sw = grad_out.transpose(dim, -1)
        n_cols = go_sw.size(-1)
        n_rows = go_sw.numel() // n_cols
        go_flat = go_sw.reshape(n_rows, n_cols).contiguous()
        gi_flat = torch.empty_like(go_flat)

        grid = (n_rows,)
        block_size, num_warps = calculate_settings(n_cols)
        _sparsemax_backward_kernel[grid](
            out_flat,
            out_flat.stride(0),
            go_flat,
            go_flat.stride(0),
            gi_flat,
            gi_flat.stride(0),
            n_cols,
            block_size=block_size,
            num_warps=num_warps,
        )

        gi = gi_flat.reshape(go_sw.shape).transpose(dim, -1)
        return gi, None
