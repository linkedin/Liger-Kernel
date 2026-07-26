# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
import cuda.tile as ct
import torch
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

from liger_kernel.ops.cutile.ops.sparsemax import _sparsemax_backward as _sparsemax_backward_ct
from liger_kernel.ops.cutile.ops.sparsemax import _sparsemax_forward as _sparsemax_forward_ct
from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

_MASK_INF_VAL = -1e9  # large negative; -inf breaks multiply-accumulate pattern ((-inf)*0 = NaN)


def _select_block_size(L: int) -> int:
    return min(_next_power_of_2(L), 128)


@ct.kernel
def _mask_inf_fwd_kernel_ct(
    scores_2d,
    output_2d,
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)
    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    for ci in range(n_chunks):
        col_start = ci * BLOCK
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + col_start
        src_tile = ct.load(scores_2d, index=(row_idx, ci), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BLOCK,)
        )
        is_future_f = ct.astype(col_idx > actual_row, ct.float32)
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        out_tile = (
            ct.astype(src_tile, ct.float32) * is_past_f + ct.full((BLOCK,), _MASK_INF_VAL, ct.float32) * is_future_f
        )
        ct.store(output_2d, index=(row_idx, ci), tile=ct.astype(out_tile, output_2d.dtype).reshape((1, BLOCK)))


@ct.kernel
def _mask_zero_fwd_kernel_ct(
    scores_2d,
    output_2d,
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)
    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    for ci in range(n_chunks):
        col_start = ci * BLOCK
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + col_start
        src_tile = ct.load(scores_2d, index=(row_idx, ci), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BLOCK,)
        )
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        out_tile = ct.astype(src_tile, ct.float32) * is_past_f
        ct.store(output_2d, index=(row_idx, ci), tile=ct.astype(out_tile, output_2d.dtype).reshape((1, BLOCK)))


@ct.kernel
def _mask_bwd_kernel_ct(
    grad_2d,
    output_2d,
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)
    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    for ci in range(n_chunks):
        col_start = ci * BLOCK
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + col_start
        grad_tile = ct.load(grad_2d, index=(row_idx, ci), shape=(1, BLOCK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BLOCK,)
        )
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        out_tile = ct.astype(grad_tile, ct.float32) * is_past_f
        ct.store(output_2d, index=(row_idx, ci), tile=ct.astype(out_tile, output_2d.dtype).reshape((1, BLOCK)))


@ct.kernel
def _fused_softmax_zeromask_bwd_kernel_ct(
    probs_2d,
    grad_probs_2d,
    output_2d,
    L: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """Fused softmax backward + causal zero-mask: dx = p*(dp - dot(p,dp)); zero col>row."""
    actual_row = ct.bid(0)
    batch_id = ct.bid(1)
    row_idx = batch_id * L + actual_row
    n_chunks = (L + BLOCK - 1) // BLOCK

    dot_tile = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + ci * BLOCK
        p_tile = ct.astype(
            ct.gather(probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dp_tile = ct.astype(
            ct.gather(grad_probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dot_tile = dot_tile + p_tile * dp_tile
    dot = ct.sum(dot_tile, 0, keepdims=False)

    for ci in range(n_chunks):
        col_idx = ct.arange(BLOCK, dtype=ct.int32) + ci * BLOCK
        p_tile = ct.astype(
            ct.gather(probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dp_tile = ct.astype(
            ct.gather(grad_probs_2d, (row_idx, col_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        dx_tile = p_tile * (dp_tile - dot)
        is_past_f = ct.astype(col_idx <= actual_row, ct.float32)
        ct.scatter(output_2d, (row_idx, col_idx), ct.astype(dx_tile * is_past_f, output_2d.dtype), check_bounds=True)


def _mask_launch(tensor: torch.Tensor, kernel) -> torch.Tensor:
    *batch, L, _ = tensor.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    t_f = tensor.reshape(N * L, L).contiguous()
    out = torch.empty_like(t_f)
    BLOCK = _select_block_size(L)
    ct.launch(torch.cuda.current_stream(), (L, N, 1), kernel, (t_f, out, int(L), int(BLOCK)))
    return out.reshape(*batch, L, L)


def _mask_inf_forward_ct(scores: torch.Tensor) -> torch.Tensor:
    return _mask_launch(scores, _mask_inf_fwd_kernel_ct)


def _mask_zero_forward_ct(scores: torch.Tensor) -> torch.Tensor:
    return _mask_launch(scores, _mask_zero_fwd_kernel_ct)


def _mask_backward_ct(grad: torch.Tensor) -> torch.Tensor:
    return _mask_launch(grad, _mask_bwd_kernel_ct)


def _fused_softmax_zeromask_bwd_ct_launch(probs: torch.Tensor, grad_probs: torch.Tensor) -> torch.Tensor:
    *batch, L, _ = probs.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    p_f = probs.reshape(N * L, L).contiguous()
    dp_f = grad_probs.reshape(N * L, L).contiguous()
    out = torch.empty_like(p_f)

    BLOCK = _select_block_size(L)
    grid = (L, N, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _fused_softmax_zeromask_bwd_kernel_ct,
        (p_f, dp_f, out, int(L), int(BLOCK)),
    )
    return out.reshape(*batch, L, L)


def _conv1x1_backward(grad_out: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor):
    """mm-based 1x1 conv backward -- bypasses cuDNN dispatch overhead.

    For a kernel_size=1 conv:
      grad_input[b,cin,h,w]  = sum_cout(W[cout,cin] * dout[b,cout,h,w])
      grad_weight[cout,cin]  = sum_{b,h,w}(dout[b,cout,h,w] * inp[b,cin,h,w])

    Both reduce to matrix multiplications on the (B*H*W, C) reshape, letting
    cuBLAS SGEMM handle the compute.  On B200 for CH=1, L=128 this is ~1.69x
    faster than F.conv_transpose2d + torch.nn.grad.conv2d_weight because it
    bypasses cuDNN's per-call dispatch overhead for this tiny shape.
    """
    B, C_out, H, W = grad_out.shape
    C_in = inp.shape[1]
    N = B * H * W
    go_2d = grad_out.permute(0, 2, 3, 1).reshape(N, C_out)
    in_2d = inp.permute(0, 2, 3, 1).reshape(N, C_in)
    w_2d = weight.view(C_out, C_in)
    grad_input = torch.mm(go_2d, w_2d).reshape(B, H, W, C_in).permute(0, 3, 1, 2).contiguous()
    grad_weight = torch.mm(go_2d.t(), in_2d).view(weight.shape)
    return grad_input, grad_weight


class LigerMultiTokenAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, sparse=False):
        scores = scores.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        ctx.sparse = sparse

        if sparse:
            if scores.dtype != torch.float32:
                raise RuntimeError(
                    f"CuTile sparse multi-token attention only supports fp32 input scores. Got dtype={scores.dtype}."
                )
            compute_dtype = torch.float32
            weight_c, bias_c = weight, bias

            scores_inf = _mask_inf_forward_ct(scores)
            probs, out_flat_sparse = _sparsemax_forward_ct(scores_inf, dim=-1)
            out_conv = F.conv2d(
                probs, weight_c, bias_c, stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            out = _mask_zero_forward_ct(out_conv)
            ctx.save_for_backward(scores_inf, probs, out_flat_sparse, weight_c, bias_c)
        else:
            compute_dtype = scores.dtype
            # fp16: promote to float32 for TF32 conv+softmax — avoids backward regression on small shapes (L≤128).
            if compute_dtype == torch.float16:
                scores = scores.float()
                weight_c = weight.float()
                bias_c = bias.float() if bias is not None else None
            else:
                weight_c, bias_c = weight, bias

            scores_inf = _mask_inf_forward_ct(scores)
            probs = torch.softmax(scores_inf, dim=-1)
            out_conv = F.conv2d(
                probs, weight_c, bias_c, stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            out = _mask_zero_forward_ct(out_conv)
            ctx.save_for_backward(scores_inf, probs, weight_c, bias_c)

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.compute_dtype = compute_dtype

        return out.to(compute_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        stride, padding, dilation, groups = (ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        sparse = ctx.sparse

        if sparse:
            scores_inf, probs, out_flat_sparse, weight, bias = ctx.saved_tensors
        else:
            scores_inf, probs, weight, bias = ctx.saved_tensors

        # .contiguous() is required: PyTorch's sum().backward() passes a broadcast
        # tensor (strides=0), which would cause CuTile gather to read invalid offsets.
        grad_out_c = grad_out.to(probs.dtype).contiguous()

        grad_conv = _mask_backward_ct(grad_out_c)

        # conv backward: mm-based 1x1 shortcut or cuDNN fallback
        if stride == (1, 1) and padding == (0, 0) and dilation == (1, 1) and groups == 1:
            grad_probs, grad_weight = _conv1x1_backward(grad_conv, probs, weight)
        else:
            # NOTE: we intentionally do NOT force torch.backends.cudnn.flags(benchmark=True)
            # here, so this conv-backward runs under the same cuDNN heuristic as the Triton
            # reference path (apples-to-apples comparison). For large spatial dims (e.g.
            # L>=4096), enabling cudnn.benchmark=True may perform better here: cuDNN's
            # default heuristic can otherwise pick a slower weight-gradient algorithm. If
            # you target long sequences, consider enabling it globally (or wrapping this call).
            grad_probs, grad_weight, _ = torch.ops.aten.convolution_backward(
                grad_conv,
                probs,
                weight,
                None,
                list(stride),
                list(padding),
                list(dilation),
                False,
                [0, 0],
                groups,
                [True, True, False],
            )

        grad_bias = None
        if bias is not None:
            grad_bias = grad_conv.sum(dim=(0, 2, 3))

        if sparse:
            grad_scores_inf = _sparsemax_backward_ct(grad_probs.contiguous(), out_flat_sparse, dim=-1)
            grad_scores = _mask_backward_ct(grad_scores_inf.to(probs.dtype).contiguous())
        else:
            grad_scores = _fused_softmax_zeromask_bwd_ct_launch(probs, grad_probs)

        orig = ctx.compute_dtype
        return (
            grad_scores.to(orig),
            grad_weight.to(orig),
            grad_bias.to(orig) if grad_bias is not None else None,
            None,
            None,
            None,
            None,
            None,
        )
