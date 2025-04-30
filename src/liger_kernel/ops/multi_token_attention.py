import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torch.nn.modules.utils import _pair

from liger_kernel.ops.softmax import _softmax_forward
from liger_kernel.ops.sparsemax import _sparsemax_backward
from liger_kernel.ops.sparsemax import _sparsemax_forward


@triton.jit
def _mask_inf_fwd_kernel(scores_ptr, out_ptr, stride_b, stride_m, stride_n, L, BLOCK: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)

    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)

    base = scores_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n

    vals = tl.load(base + offs, mask=in_bounds, other=0.0)

    future = col_idx[None, :] > row_idx[:, None]

    neg_inf = tl.full(vals.shape, -1e9, dtype=vals.dtype)
    out = tl.where(future, neg_inf, vals)

    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds)


@triton.jit
def _mask_inf_bwd_kernel(grad_in_ptr, out_ptr, stride_b, stride_m, stride_n, L, BLOCK: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)

    base = grad_in_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    grad_vals = tl.load(base + offs, mask=in_bounds, other=0.0)

    future = col_idx[None, :] > row_idx[:, None]
    zero = tl.zeros(grad_vals.shape, dtype=grad_vals.dtype)
    out = tl.where(future, zero, grad_vals)

    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds)


@triton.jit
def _mask_zero_fwd_kernel(scores_ptr, out_ptr, stride_b, stride_m, stride_n, L, BLOCK: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)

    base = scores_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    vals = tl.load(base + offs, mask=in_bounds, other=0.0)

    future = col_idx[None, :] > row_idx[:, None]
    zero = tl.zeros(vals.shape, dtype=vals.dtype)
    out = tl.where(future, zero, vals)

    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds)


@triton.jit
def _mask_zero_bwd_kernel(grad_in_ptr, out_ptr, stride_b, stride_m, stride_n, L, BLOCK: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)

    base = grad_in_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    grad_vals = tl.load(base + offs, mask=in_bounds, other=0.0)

    future = col_idx[None, :] > row_idx[:, None]
    zero = tl.zeros(grad_vals.shape, dtype=grad_vals.dtype)
    out = tl.where(future, zero, grad_vals)

    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds)


def _mask_inf_forward(scores: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.contiguous().view(N, L, L)
    out = torch.empty_like(scores_f)

    sb, sm, sn = scores_f.stride(0), scores_f.stride(1), scores_f.stride(2)
    grid = (triton.cdiv(L, 128), triton.cdiv(L, 128), N)
    _mask_inf_fwd_kernel[grid](scores_f, out, sb, sm, sn, L, BLOCK=128, num_warps=4)
    return out.view(*batch, L, L)


def _mask_inf_backward(grad: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = grad.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_f = grad.contiguous().view(N, L, L)
    out = torch.empty_like(grad_f)

    sb, sm, sn = grad_f.stride(0), grad_f.stride(1), grad_f.stride(2)
    grid = (triton.cdiv(L, 128), triton.cdiv(L, 128), N)
    _mask_inf_bwd_kernel[grid](grad_f, out, sb, sm, sn, L, BLOCK=128, num_warps=4)
    return out.view(*batch, L, L)


def _mask_zero_forward(scores: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.contiguous().view(N, L, L)
    out = torch.empty_like(scores_f)

    sb, sm, sn = scores_f.stride(0), scores_f.stride(1), scores_f.stride(2)
    grid = (triton.cdiv(L, 128), triton.cdiv(L, 128), N)
    _mask_zero_fwd_kernel[grid](scores_f, out, sb, sm, sn, L, BLOCK=128, num_warps=4)
    return out.view(*batch, L, L)


def _mask_zero_backward(grad: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = grad.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_f = grad.contiguous().view(N, L, L)
    out = torch.empty_like(grad_f)

    sb, sm, sn = grad_f.stride(0), grad_f.stride(1), grad_f.stride(2)
    grid = (triton.cdiv(L, 128), triton.cdiv(L, 128), N)
    _mask_zero_bwd_kernel[grid](grad_f, out, sb, sm, sn, L, BLOCK=128, num_warps=4)
    return out.view(*batch, L, L)


class LigerMultiTokenAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, sparse=False):
        scores_inf = _mask_inf_forward(scores)

        out_flat_sparse = None
        activation_output = None

        ctx.sparse = sparse

        if sparse:
            if scores_inf.dtype != torch.float32:
                raise RuntimeError("Liger sparse multi-token attention currently only supports fp32 input scores")
            probs_sparse, out_flat_sparse = _sparsemax_forward(scores_inf.contiguous(), dim=-1)
            activation_output = probs_sparse
            ctx.save_for_backward(scores_inf, activation_output, out_flat_sparse, weight, bias)
            ctx.out_flat_sparse_saved = True
        else:
            probs_softmax, _, _, _ = _softmax_forward(scores_inf.contiguous())
            activation_output = probs_softmax
            ctx.save_for_backward(scores_inf, activation_output, weight, bias)
            ctx.out_flat_sparse_saved = False

        out_conv = F.conv2d(
            activation_output.contiguous(),
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        out = _mask_zero_forward(out_conv)

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.dim = -1

        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.out_flat_sparse_saved:
            scores_inf, activation_output, out_flat_sparse, weight, bias = ctx.saved_tensors
        else:
            scores_inf, activation_output, weight, bias = ctx.saved_tensors
            out_flat_sparse = None

        use_sparsemax = ctx.sparse
        dim = ctx.dim
        stride, padding, dilation, groups = (ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        grad_conv = _mask_zero_backward(grad_out.contiguous())

        grad_probs = F.conv_transpose2d(
            grad_conv, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

        grad_weight = torch.nn.grad.conv2d_weight(
            input=activation_output.contiguous(),
            weight_size=weight.shape,
            grad_output=grad_conv,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        grad_bias = None
        if bias is not None:
            grad_bias = grad_conv.contiguous().sum(dim=(0, 2, 3))

        grad_scores_inf = None
        if use_sparsemax:
            if not ctx.out_flat_sparse_saved or out_flat_sparse is None:
                raise RuntimeError("Internal error: Sparse flag is set but sparse tensor was not saved.")
            grad_scores_inf = _sparsemax_backward(grad_probs.contiguous(), out_flat_sparse.contiguous(), dim=dim)
        else:
            grad_probs_cont = grad_probs.contiguous()
            probs_cont = activation_output.contiguous()
            dot = (grad_probs_cont * probs_cont).sum(dim=-1, keepdim=True)
            grad_scores_inf = probs_cont * (grad_probs_cont - dot)

        grad_scores = _mask_inf_backward(grad_scores_inf.contiguous())

        return (grad_scores, grad_weight, grad_bias, None, None, None, None, None)
