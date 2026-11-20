"""Fused LoRA linear: y = x @ W^T + scale * (x @ A^T) @ B^T (+ bias).

The two compute-heavy frozen-shaped GEMMs (``x @ W^T`` and ``dY @ W``) stay on
cuBLAS; only the LoRA epilogues are fused into Triton kernels:

    forward:  Y = fused_expand_add(base, scale * S @ B^T),  S = x @ A^T
    backward: dX = fused_dx_add(dY @ W, dS @ A),           dS = scale * (dY @ B)

This removes the materialization of the full-size ``delta`` / ``dx_lora``
tensors and two elementwise passes, while leaving the base GEMM tiling to
cuBLAS. Measurements for this graph split are reproducible with
``benchmark/scripts/benchmark_fused_lora_linear.py``.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _fused_lora_linear_forward_kernel(
    Y_ptr,
    base_ptr,
    S_ptr,
    B_ptr,
    M,
    N,
    R,
    scale,
    stride_bm,
    stride_bn,
    stride_sm,
    stride_sr,
    stride_Bn,
    stride_Br,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    scale = scale.to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r0 in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r0 * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        s = tl.load(
            S_ptr + offs_m[:, None] * stride_sm + offs_r[None, :] * stride_sr,
            mask=mask_m[:, None] & mask_r[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_Bn + offs_r[None, :] * stride_Br,
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        acc = tl.dot(s, tl.trans(b), acc, input_precision="ieee")

    base = tl.load(
        base_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    out = base + acc * scale
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out.to(Y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _fused_lora_linear_backward_kernel(
    dX_ptr,
    base_dX_ptr,
    dS_ptr,
    A_ptr,
    M,
    K,
    R,
    stride_bm,
    stride_bk,
    stride_sm,
    stride_sr,
    stride_ar,
    stride_ak,
    stride_xm,
    stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for r0 in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r0 * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        ds = tl.load(
            dS_ptr + offs_m[:, None] * stride_sm + offs_r[None, :] * stride_sr,
            mask=mask_m[:, None] & mask_r[None, :],
            other=0.0,
        )
        a = tl.load(
            A_ptr + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak,
            mask=mask_r[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(ds, a, acc, input_precision="ieee")

    base = tl.load(
        base_dX_ptr + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    out = base + acc
    tl.store(
        dX_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        out.to(dX_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_k[None, :],
    )


def _block_r(R):
    return max(16, min(triton.next_power_of_2(R), 64))


def fused_lora_linear_forward(x, W, A, B, bias=None, scale=1.0):
    """Y = x @ W^T + scale * (x @ A^T) @ B^T + bias; returns (Y, S)."""
    M, K = x.shape
    N, R = B.shape

    base = F.linear(x, W, bias)
    s = F.linear(x, A)
    y = torch.empty_like(base)

    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    with device_context(x.device):
        _fused_lora_linear_forward_kernel[grid](
            y,
            base,
            s,
            B,
            M,
            N,
            R,
            scale,
            base.stride(0),
            base.stride(1),
            s.stride(0),
            s.stride(1),
            B.stride(0),
            B.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_R=_block_r(R),
            num_warps=4,
            num_stages=2,
        )
    return y, s


def fused_lora_linear_backward(dy, x, W, A, dS):
    """dX = dY @ W + dS @ A; the base term stays on cuBLAS, only the add is fused."""
    M, K = x.shape

    base_dx = torch.mm(dy, W)
    dx = torch.empty_like(x)
    BLOCK_M, BLOCK_K = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    with device_context(x.device):
        _fused_lora_linear_backward_kernel[grid](
            dx,
            base_dx,
            dS,
            A,
            M,
            K,
            dS.shape[1],
            base_dx.stride(0),
            base_dx.stride(1),
            dS.stride(0),
            dS.stride(1),
            A.stride(0),
            A.stride(1),
            dx.stride(0),
            dx.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_R=_block_r(dS.shape[1]),
            num_warps=4,
            num_stages=2,
        )
    return dx


class LigerFusedLoRALinearFunction(torch.autograd.Function):
    """Autograd wrapper for the fused LoRA linear layer.

    Inputs: x (..., K), W (N, K) frozen base weight, A (R, K), B (N, R),
    bias (N,) or None, scale (float, typically lora_alpha / R).
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, W, A, B, bias, scale):
        ori_shape = x.shape
        K = ori_shape[-1]
        x = x.reshape(-1, K)
        y, s = fused_lora_linear_forward(x, W, A, B, bias, scale)

        ctx.save_for_backward(x, W, A, B, s)
        ctx.ori_shape = ori_shape
        ctx.has_bias = bias is not None
        ctx.scale = scale
        return y.view(*ori_shape[:-1], W.shape[0])

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dy):
        x, W, A, B, s = ctx.saved_tensors
        ori_shape = ctx.ori_shape
        scale = ctx.scale
        needs_x, needs_w, needs_a, needs_b, needs_bias, _ = ctx.needs_input_grad

        dy = dy.reshape(-1, dy.shape[-1])

        dx = dw = da = db = dbias = None

        if needs_b:
            db = torch.mm(dy.t(), s) * scale

        needs_ds = needs_x or needs_a
        ds = torch.mm(dy, B) * scale if needs_ds else None

        if needs_a:
            da = torch.mm(ds.t(), x)

        if needs_x:
            dx = fused_lora_linear_backward(dy, x, W, A, ds).view(ori_shape)

        if needs_w:
            dw = torch.mm(dy.t(), x)

        if needs_bias and ctx.has_bias:
            dbias = dy.sum(dim=0)

        return dx, dw, da, db, dbias, None
