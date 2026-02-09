import math

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


def _post_res_default_meta(c: int) -> tuple[int, int, int, int]:
    """
    Returns default (block_n, block_c, num_warps, num_stages) for post_res kernels.
    Tuned for different hidden dimensions on NVIDIA GPUs.
    """
    if c >= 4096:
        return 32, 128, 8, 3  # (block_n, block_c, num_warps, num_stages)
    if c >= 2048:
        return 32, 128, 4, 2
    if c >= 1024:
        return 32, 64, 4, 2
    return 32, 64, 2, 2


def _post_res_meta(
    c: int,
    block_n: Optional[int],
    block_c: Optional[int],
    num_warps: Optional[int],
    num_stages: Optional[int],
) -> tuple[int, int, int, int]:
    bn, bc, nw, ns = _post_res_default_meta(c)
    return (
        bn if block_n is None else int(block_n),
        bc if block_c is None else int(block_c),
        nw if num_warps is None else int(num_warps),
        ns if num_stages is None else int(num_stages),
    )


# -------------------------------------------------------------------------------------------------
# (1) Coefficients: fused matmul + RMS scalar (Eq. 14–15)
#   mix = (x @ phi) * rsqrt(mean(x^2) + eps)
#
# We provide two paths:
#   - TC path: x BF16/FP16 and phi BF16/FP16 (Tensor Cores)
#   - TF32-ish path: x cast to FP32 and phi FP32 (relies on Triton/arch for TF32)
# -------------------------------------------------------------------------------------------------


@triton.jit
def _mhc_mm_norm_fwd_kernel(
    x_ptr,
    phi_ptr,
    mix_ptr,
    invr_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_phik: tl.constexpr,
    stride_phim: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CAST_FP32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    sumsq = tl.zeros((BLOCK_N,), tl.float32)

    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)

        x = tl.load(
            x_ptr + n_offs[:, None] * stride_xn + k_offs[None, :] * stride_xk,
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )
        if CAST_FP32:
            x = x.to(tl.float32)
            sumsq += tl.sum(x * x, axis=1)
        else:
            x_f = x.to(tl.float32)
            sumsq += tl.sum(x_f * x_f, axis=1)

        phi = tl.load(
            phi_ptr + k_offs[:, None] * stride_phik + m_offs[None, :] * stride_phim,
            mask=(k_offs[:, None] < K) & (m_offs[None, :] < M),
            other=0.0,
        )
        if CAST_FP32:
            phi = phi.to(tl.float32)

        acc += tl.dot(x, phi)

    invr = tl.rsqrt(sumsq / K + eps)
    out = acc * invr[:, None]

    tl.store(
        mix_ptr + n_offs[:, None] * stride_mn + m_offs[None, :] * stride_mm,
        out,
        mask=(n_offs[:, None] < N) & (m_offs[None, :] < M),
    )
    if pid_m == 0:
        tl.store(invr_ptr + n_offs, invr, mask=n_offs < N)


def mhc_mm_norm_fwd(
    x: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
    *,
    out_mix: Optional[torch.Tensor] = None,
    out_invr: Optional[torch.Tensor] = None,
    block_n: int = 32,
    block_k: int = 256,
    block_m: int = 32,
    num_warps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused (x @ phi) + invr = rsqrt(mean(x^2)+eps) and returns mix=(x@phi)*invr.

    Args:
        x: [N, K] contiguous
        phi: [K, M] contiguous
        eps: float
    Returns:
        mix: [N, M] float32
        invr: [N] float32
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert phi.is_contiguous(), "phi must be contiguous"

    N, K = x.shape
    K2, M = phi.shape
    assert K2 == K, f"phi.shape[0] must match K: got {K2} vs {K}"

    if out_mix is None:
        out_mix = torch.empty((N, M), device=x.device, dtype=torch.float32)
    if out_invr is None:
        out_invr = torch.empty((N,), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m))

    use_tc = (x.dtype == phi.dtype) and (x.dtype in (torch.float16, torch.bfloat16))

    _mhc_mm_norm_fwd_kernel[grid](
        x,
        phi,
        out_mix,
        out_invr,
        N=N,
        K=K,
        M=M,
        stride_xn=x.stride(0),
        stride_xk=x.stride(1),
        stride_phik=phi.stride(0),
        stride_phim=phi.stride(1),
        stride_mn=out_mix.stride(0),
        stride_mm=out_mix.stride(1),
        eps=eps,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        CAST_FP32=not use_tc,
        num_warps=num_warps,
    )
    return out_mix, out_invr


# -------------------------------------------------------------------------------------------------
# Backward for fused (x @ phi) + RMS scalar
#
# mix = (x @ phi) * invr
# invr = rsqrt(mean(x^2) + eps)
#
# Given grad_mix, compute:
#   grad_z   = grad_mix * invr
#   g        = sum(grad_mix * (mix / invr)) = sum(grad_mix * mix) / invr
#   factor   = -(g / K) * invr^3
#   grad_x   = grad_z @ phi^T + factor * x
#   grad_phi = x^T @ grad_z
#
# grad_phi is accumulated into FP32 with atomic adds (split over N-chunks).
# -------------------------------------------------------------------------------------------------


@triton.jit
def _mhc_mm_norm_bwd_fused_kernel(
    x_ptr,
    phi_ptr,
    mix_ptr,
    invr_ptr,
    grad_mix_ptr,
    grad_x_ptr,
    grad_phi_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_phik: tl.constexpr,
    stride_phim: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_invr: tl.constexpr,
    stride_gmn: tl.constexpr,
    stride_gmm: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxk: tl.constexpr,
    stride_gpk: tl.constexpr,
    stride_gpm: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CAST_FP32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    invr = tl.load(invr_ptr + n_offs * stride_invr, mask=n_offs < N, other=0.0).to(tl.float32)

    x = tl.load(
        x_ptr + n_offs[:, None] * stride_xn + k_offs[None, :] * stride_xk,
        mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
        other=0.0,
    )
    if CAST_FP32:
        x = x.to(tl.float32)
        x_f = x
    else:
        x_f = x.to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    g_acc = tl.zeros((BLOCK_N,), tl.float32)

    for m0 in tl.static_range(0, M, BLOCK_M):
        m_offs = m0 + tl.arange(0, BLOCK_M)

        grad_mix = tl.load(
            grad_mix_ptr + n_offs[:, None] * stride_gmn + m_offs[None, :] * stride_gmm,
            mask=(n_offs[:, None] < N) & (m_offs[None, :] < M),
            other=0.0,
        ).to(tl.float32)

        mix = tl.load(
            mix_ptr + n_offs[:, None] * stride_mn + m_offs[None, :] * stride_mm,
            mask=(n_offs[:, None] < N) & (m_offs[None, :] < M),
            other=0.0,
        ).to(tl.float32)

        g_acc += tl.sum(grad_mix * mix, axis=1)

        phi = tl.load(
            phi_ptr + k_offs[:, None] * stride_phik + m_offs[None, :] * stride_phim,
            mask=(k_offs[:, None] < K) & (m_offs[None, :] < M),
            other=0.0,
        )
        if CAST_FP32:
            phi = phi.to(tl.float32)
            grad_z = grad_mix * invr[:, None]
        else:
            grad_z = (grad_mix * invr[:, None]).to(phi.dtype)

        acc += tl.dot(grad_z, tl.trans(phi))

        dphi = tl.dot(tl.trans(x), grad_z)
        tl.atomic_add(
            grad_phi_ptr + k_offs[:, None] * stride_gpk + m_offs[None, :] * stride_gpm,
            dphi,
            mask=(k_offs[:, None] < K) & (m_offs[None, :] < M),
        )

    g = g_acc / invr
    invr3 = invr * invr * invr
    factor = (-g * invr3) / K

    gx = acc + x_f * factor[:, None]

    if CAST_FP32:
        tl.store(
            grad_x_ptr + n_offs[:, None] * stride_gxn + k_offs[None, :] * stride_gxk,
            gx,
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
        )
    else:
        tl.store(
            grad_x_ptr + n_offs[:, None] * stride_gxn + k_offs[None, :] * stride_gxk,
            gx.to(x.dtype),
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
        )


def mhc_mm_norm_bwd(
    x: torch.Tensor,
    phi: torch.Tensor,
    mix: torch.Tensor,
    invr: torch.Tensor,
    grad_mix: torch.Tensor,
    *,
    out_grad_x: Optional[torch.Tensor] = None,
    out_grad_phi: Optional[torch.Tensor] = None,
    block_n: int = 32,
    block_k: int = 256,
    block_m: int = 32,
    num_warps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton backward for `mhc_mm_norm_fwd`.

    Returns:
        grad_x: [N, K] same dtype as x
        grad_phi: [K, M] FP32 (safe for atomic adds; cast on return if needed)

    Note:
        grad_phi is accumulated via atomic_add in FP32. For very large N
        (batch * sequence length > 1M), accumulated rounding errors may
        become noticeable. This is typically not an issue for standard
        training configurations.
    """
    assert (
        x.is_contiguous()
        and phi.is_contiguous()
        and mix.is_contiguous()
        and invr.is_contiguous()
        and grad_mix.is_contiguous()
    )

    N, K = x.shape
    K2, M = phi.shape
    assert K2 == K
    assert mix.shape == (N, M)
    assert grad_mix.shape == (N, M)
    assert invr.shape == (N,)

    if out_grad_x is None:
        out_grad_x = torch.empty_like(x)
    if out_grad_phi is None:
        out_grad_phi = torch.zeros((K, M), device=x.device, dtype=torch.float32)

    use_tc = (x.dtype == phi.dtype) and (x.dtype in (torch.float16, torch.bfloat16))

    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    _mhc_mm_norm_bwd_fused_kernel[grid](
        x,
        phi,
        mix,
        invr,
        grad_mix,
        out_grad_x,
        out_grad_phi,
        N=N,
        K=K,
        M=M,
        stride_xn=x.stride(0),
        stride_xk=x.stride(1),
        stride_phik=phi.stride(0),
        stride_phim=phi.stride(1),
        stride_mn=mix.stride(0),
        stride_mm=mix.stride(1),
        stride_invr=invr.stride(0),
        stride_gmn=grad_mix.stride(0),
        stride_gmm=grad_mix.stride(1),
        stride_gxn=out_grad_x.stride(0),
        stride_gxk=out_grad_x.stride(1),
        stride_gpk=out_grad_phi.stride(0),
        stride_gpm=out_grad_phi.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        CAST_FP32=not use_tc,
        num_warps=num_warps,
    )

    if out_grad_phi.dtype != phi.dtype:
        out_grad_phi = out_grad_phi.to(phi.dtype)
    return out_grad_x, out_grad_phi


# -------------------------------------------------------------------------------------------------
# Sinkhorn-Knopp forward/backward for H_res (Eq. 19)
# -------------------------------------------------------------------------------------------------


@triton.jit
def _mhc_split_sinkhorn_fwd_kernel(
    mix_ptr,
    b_ptr,
    hpre_ptr,
    hpost_ptr,
    hres_ptr,
    hist_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    M: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_hp_n: tl.constexpr,
    stride_hp_h: tl.constexpr,
    stride_hq_n: tl.constexpr,
    stride_hq_h: tl.constexpr,
    stride_hr_n: tl.constexpr,
    stride_hr_i: tl.constexpr,
    stride_hr_j: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_ht: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_hj: tl.constexpr,
    alpha_pre_ptr,
    alpha_post_ptr,
    alpha_res_ptr,
    pre_eps: tl.constexpr,
    sinkhorn_eps: tl.constexpr,
    post_mult: tl.constexpr,
    TMAX: tl.constexpr,
    STORE_HIST: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load scalar alpha parameters from GPU memory (avoids CPU sync)
    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    # --- Pre/post logits
    j = tl.arange(0, HC)
    mix_pre = tl.load(mix_ptr + pid * stride_mn + j * stride_mm).to(tl.float32)
    mix_post = tl.load(mix_ptr + pid * stride_mn + (HC + j) * stride_mm).to(tl.float32)

    b_pre = tl.load(b_ptr + j).to(tl.float32)
    b_post = tl.load(b_ptr + (HC + j)).to(tl.float32)

    pre_logits = mix_pre * alpha_pre + b_pre
    post_logits = mix_post * alpha_post + b_post

    pre = tl.sigmoid(pre_logits) + pre_eps
    post = tl.sigmoid(post_logits) * post_mult

    tl.store(hpre_ptr + pid * stride_hp_n + j * stride_hp_h, pre)
    tl.store(hpost_ptr + pid * stride_hq_n + j * stride_hq_h, post)

    # --- Residual logits matrix [HC, HC]
    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    flat = rows * HC + cols  # [HC,HC]

    mix_res = tl.load(mix_ptr + pid * stride_mn + (2 * HC + flat) * stride_mm).to(tl.float32)
    b_res = tl.load(b_ptr + (2 * HC + flat)).to(tl.float32)

    logits = mix_res * alpha_res + b_res

    # Sinkhorn: initial row-softmax (stable) then alternating row/col norms.
    row_max = tl.max(logits, axis=1)
    e = tl.exp(logits - row_max[:, None])
    row_sum = tl.sum(e, axis=1)
    mat = e / row_sum[:, None] + sinkhorn_eps

    col_sum = tl.sum(mat, axis=0)
    mat = mat / (col_sum[None, :] + sinkhorn_eps)

    if STORE_HIST:
        tl.store(
            hist_ptr + pid * stride_hn + 0 * stride_ht + rows * stride_hi + cols * stride_hj,
            mat,
        )

    for t in tl.static_range(0, TMAX - 1):
        row_sum = tl.sum(mat, axis=1)
        mat = mat / (row_sum[:, None] + sinkhorn_eps)
        col_sum = tl.sum(mat, axis=0)
        mat = mat / (col_sum[None, :] + sinkhorn_eps)
        if STORE_HIST:
            tl.store(
                hist_ptr + pid * stride_hn + (t + 1) * stride_ht + rows * stride_hi + cols * stride_hj,
                mat,
            )

    # Store h_res [N, HC, HC] (row-major: out, in)
    tl.store(hres_ptr + pid * stride_hr_n + rows * stride_hr_i + cols * stride_hr_j, mat)


@triton.jit
def _mhc_sinkhorn_bwd_kernel(
    mix_ptr,
    b_ptr,
    grad_out_ptr,
    grad_logits_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_go_n: tl.constexpr,
    stride_go_i: tl.constexpr,
    stride_go_j: tl.constexpr,
    stride_gl_n: tl.constexpr,
    stride_gl_i: tl.constexpr,
    stride_gl_j: tl.constexpr,
    alpha_res_ptr,
    sinkhorn_eps: tl.constexpr,
    TMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    flat = rows * HC + cols

    # Rebuild logits
    mix_res = tl.load(mix_ptr + pid * stride_mn + (2 * HC + flat) * stride_mm).to(tl.float32)
    b_res = tl.load(b_ptr + (2 * HC + flat)).to(tl.float32)
    logits = mix_res * alpha_res + b_res

    # Forward recompute (no lists) and backward with recompute per step.
    row_max = tl.max(logits, axis=1)
    e = tl.exp(logits - row_max[:, None])
    row_sum0 = tl.sum(e, axis=1)
    p = e / row_sum0[:, None]  # softmax, row-wise
    p_eps = p + sinkhorn_eps

    col_sum0 = tl.sum(p_eps, axis=0)
    mat0 = p_eps / (col_sum0[None, :] + sinkhorn_eps)

    # Start backward from grad_out
    g = tl.load(
        grad_out_ptr + pid * stride_go_n + rows * stride_go_i + cols * stride_go_j,
        mask=True,
        other=0.0,
    ).to(tl.float32)

    # Reverse iterations (TMAX-1 .. 1), recomputing mat_t, rs_t, cs_t
    for t in tl.static_range(TMAX - 1, 0, -1):
        mat = mat0
        rs_t = row_sum0
        cs_t = col_sum0
        mat_t = mat0

        for s in tl.static_range(1, TMAX):
            rs = tl.sum(mat, axis=1)
            mat = mat / (rs[:, None] + sinkhorn_eps)
            cs = tl.sum(mat, axis=0)
            mat = mat / (cs[None, :] + sinkhorn_eps)
            if s == t:
                mat_t = mat
                rs_t = rs
                cs_t = cs

        denom_col = cs_t + sinkhorn_eps  # [HC]
        dot_col = tl.sum(g * mat_t, axis=0)  # [HC]
        g_row = (g - dot_col[None, :]) / denom_col[None, :]

        m_row = mat_t * denom_col[None, :]  # invert col norm: m_row = m_out * denom
        denom_row = rs_t + sinkhorn_eps
        dot_row = tl.sum(g_row * m_row, axis=1)
        g = (g_row - dot_row[:, None]) / denom_row[:, None]

    # Undo initial col norm (t=0)
    denom_col0 = col_sum0 + sinkhorn_eps
    dot_col0 = tl.sum(g * mat0, axis=0)
    g_p = (g - dot_col0[None, :]) / denom_col0[None, :]

    # Softmax backward on rows: p * (g_p - sum(g_p * p))
    dot_soft = tl.sum(g_p * p, axis=1)
    grad_logits = p * (g_p - dot_soft[:, None])

    tl.store(grad_logits_ptr + pid * stride_gl_n + rows * stride_gl_i + cols * stride_gl_j, grad_logits)


@triton.jit
def _mhc_sinkhorn_bwd_hist_kernel(
    mix_ptr,
    b_ptr,
    hist_ptr,
    grad_out_ptr,
    grad_logits_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_ht: tl.constexpr,
    stride_hi: tl.constexpr,
    stride_hj: tl.constexpr,
    stride_go_n: tl.constexpr,
    stride_go_i: tl.constexpr,
    stride_go_j: tl.constexpr,
    stride_gl_n: tl.constexpr,
    stride_gl_i: tl.constexpr,
    stride_gl_j: tl.constexpr,
    alpha_res_ptr,
    sinkhorn_eps: tl.constexpr,
    TMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    flat = rows * HC + cols

    # Rebuild logits
    mix_res = tl.load(mix_ptr + pid * stride_mn + (2 * HC + flat) * stride_mm).to(tl.float32)
    b_res = tl.load(b_ptr + (2 * HC + flat)).to(tl.float32)
    logits = mix_res * alpha_res + b_res

    # Initial row-softmax
    row_max = tl.max(logits, axis=1)
    e = tl.exp(logits - row_max[:, None])
    row_sum0 = tl.sum(e, axis=1)
    p = e / row_sum0[:, None]
    p_eps = p + sinkhorn_eps

    col_sum0 = tl.sum(p_eps, axis=0)
    mat0 = p_eps / (col_sum0[None, :] + sinkhorn_eps)

    # Start backward from grad_out
    g = tl.load(
        grad_out_ptr + pid * stride_go_n + rows * stride_go_i + cols * stride_go_j,
        mask=True,
        other=0.0,
    ).to(tl.float32)

    # Reverse iterations (TMAX-1 .. 1) using stored mats
    for t in tl.static_range(TMAX - 1, 0, -1):
        mat_t = tl.load(hist_ptr + pid * stride_hn + t * stride_ht + rows * stride_hi + cols * stride_hj).to(tl.float32)
        mat_prev = tl.load(hist_ptr + pid * stride_hn + (t - 1) * stride_ht + rows * stride_hi + cols * stride_hj).to(
            tl.float32
        )

        row_sum = tl.sum(mat_prev, axis=1)
        mat_row = mat_prev / (row_sum[:, None] + sinkhorn_eps)
        col_sum = tl.sum(mat_row, axis=0)
        denom_col = col_sum + sinkhorn_eps

        dot_col = tl.sum(g * mat_t, axis=0)
        g_row = (g - dot_col[None, :]) / denom_col[None, :]

        m_row = mat_t * denom_col[None, :]
        denom_row = row_sum + sinkhorn_eps
        dot_row = tl.sum(g_row * m_row, axis=1)
        g = (g_row - dot_row[:, None]) / denom_row[:, None]

    # Undo initial col norm (t=0)
    denom_col0 = col_sum0 + sinkhorn_eps
    dot_col0 = tl.sum(g * mat0, axis=0)
    g_p = (g - dot_col0[None, :]) / denom_col0[None, :]

    # Softmax backward on rows: p * (g_p - sum(g_p * p))
    dot_soft = tl.sum(g_p * p, axis=1)
    grad_logits = p * (g_p - dot_soft[:, None])

    tl.store(grad_logits_ptr + pid * stride_gl_n + rows * stride_gl_i + cols * stride_gl_j, grad_logits)


def mhc_split_sinkhorn_fwd(
    mix: torch.Tensor,
    b: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    *,
    tmax: int,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    out_hpre: Optional[torch.Tensor] = None,
    out_hpost: Optional[torch.Tensor] = None,
    out_hres: Optional[torch.Tensor] = None,
    out_hist: Optional[torch.Tensor] = None,
    return_hist: bool = False,
    num_warps: int = 1,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Compute h_pre, h_post, h_res from `mix` (already normalized by RMS scalar).

    mix: [N, M] float32 where M = HC*HC + 2*HC
    b: [M] float32
    """
    assert mix.is_contiguous() and b.is_contiguous()

    N, M = mix.shape
    assert M == b.numel()
    # infer HC from M = HC*HC + 2*HC
    # Solve HC^2 + 2HC - M = 0
    HC = int((math.isqrt(4 + 4 * M) - 2) // 2)
    assert HC * HC + 2 * HC == M, f"Invalid M for mHC: M={M}"

    if out_hpre is None:
        out_hpre = torch.empty((N, HC), device=mix.device, dtype=torch.float32)
    if out_hpost is None:
        out_hpost = torch.empty((N, HC), device=mix.device, dtype=torch.float32)
    if out_hres is None:
        out_hres = torch.empty((N, HC, HC), device=mix.device, dtype=torch.float32)
    if return_hist:
        if out_hist is None:
            out_hist = torch.empty((N, tmax, HC, HC), device=mix.device, dtype=torch.float32)
    else:
        if out_hist is None:
            out_hist = torch.empty((1,), device=mix.device, dtype=torch.float32)

    grid = (N,)

    _mhc_split_sinkhorn_fwd_kernel[grid](
        mix,
        b,
        out_hpre,
        out_hpost,
        out_hres,
        out_hist,
        N=N,
        HC=HC,
        M=M,
        stride_mn=mix.stride(0),
        stride_mm=mix.stride(1),
        stride_hp_n=out_hpre.stride(0),
        stride_hp_h=out_hpre.stride(1),
        stride_hq_n=out_hpost.stride(0),
        stride_hq_h=out_hpost.stride(1),
        stride_hr_n=out_hres.stride(0),
        stride_hr_i=out_hres.stride(1),
        stride_hr_j=out_hres.stride(2),
        stride_hn=out_hist.stride(0) if out_hist.ndim > 1 else 0,
        stride_ht=out_hist.stride(1) if out_hist.ndim > 1 else 0,
        stride_hi=out_hist.stride(2) if out_hist.ndim > 1 else 0,
        stride_hj=out_hist.stride(3) if out_hist.ndim > 1 else 0,
        alpha_pre_ptr=alpha_pre.contiguous(),
        alpha_post_ptr=alpha_post.contiguous(),
        alpha_res_ptr=alpha_res.contiguous(),
        pre_eps=pre_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_mult=post_mult,
        TMAX=tmax,
        STORE_HIST=return_hist,
        num_warps=num_warps,
    )
    if return_hist:
        return out_hpre, out_hpost, out_hres, out_hist
    return out_hpre, out_hpost, out_hres


def mhc_sinkhorn_bwd(
    mix: torch.Tensor,
    b: torch.Tensor,
    alpha_res: torch.Tensor,
    grad_hres: torch.Tensor,
    *,
    tmax: int,
    sinkhorn_eps: float,
    hist: Optional[torch.Tensor] = None,
    out_grad_logits: Optional[torch.Tensor] = None,
    num_warps: int = 1,
) -> torch.Tensor:
    """
    Backward for Sinkhorn: returns grad_logits (same shape as h_res).

    mix: [N, M] float32
    b: [M] float32
    grad_hres: [N, HC, HC] float32
    """
    assert mix.is_contiguous() and b.is_contiguous() and grad_hres.is_contiguous()

    N, M = mix.shape
    HC = grad_hres.shape[1]
    assert grad_hres.shape == (N, HC, HC)
    assert M == HC * HC + 2 * HC

    if out_grad_logits is None:
        out_grad_logits = torch.empty((N, HC, HC), device=mix.device, dtype=torch.float32)

    grid = (N,)

    alpha_res_c = alpha_res.contiguous()

    if hist is not None:
        assert hist.is_contiguous()
        assert hist.shape == (N, tmax, HC, HC)
        _mhc_sinkhorn_bwd_hist_kernel[grid](
            mix,
            b,
            hist,
            grad_hres,
            out_grad_logits,
            N=N,
            HC=HC,
            stride_mn=mix.stride(0),
            stride_mm=mix.stride(1),
            stride_hn=hist.stride(0),
            stride_ht=hist.stride(1),
            stride_hi=hist.stride(2),
            stride_hj=hist.stride(3),
            stride_go_n=grad_hres.stride(0),
            stride_go_i=grad_hres.stride(1),
            stride_go_j=grad_hres.stride(2),
            stride_gl_n=out_grad_logits.stride(0),
            stride_gl_i=out_grad_logits.stride(1),
            stride_gl_j=out_grad_logits.stride(2),
            alpha_res_ptr=alpha_res_c,
            sinkhorn_eps=sinkhorn_eps,
            TMAX=tmax,
            num_warps=num_warps,
        )
    else:
        _mhc_sinkhorn_bwd_kernel[grid](
            mix,
            b,
            grad_hres,
            out_grad_logits,
            N=N,
            HC=HC,
            stride_mn=mix.stride(0),
            stride_mm=mix.stride(1),
            stride_go_n=grad_hres.stride(0),
            stride_go_i=grad_hres.stride(1),
            stride_go_j=grad_hres.stride(2),
            stride_gl_n=out_grad_logits.stride(0),
            stride_gl_i=out_grad_logits.stride(1),
            stride_gl_j=out_grad_logits.stride(2),
            alpha_res_ptr=alpha_res_c,
            sinkhorn_eps=sinkhorn_eps,
            TMAX=tmax,
            num_warps=num_warps,
        )
    return out_grad_logits


# -------------------------------------------------------------------------------------------------
# Apply kernels: mhc_pre and mhc_post_res (forward + backward)
# -------------------------------------------------------------------------------------------------


@triton.jit
def _mhc_pre_fwd_kernel(
    x_ptr,
    hpre_ptr,
    out_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    C: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_hh: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oc: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)
    for s in tl.static_range(0, HC):
        h_s = tl.load(
            hpre_ptr + n_offs * stride_hn + s * stride_hh,
            mask=(n_offs < N),
            other=0.0,
        ).to(tl.float32)
        xs = tl.load(
            x_ptr + n_offs[:, None] * stride_xn + s * stride_xh + c_offs[None, :] * stride_xc,
            mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        acc += xs * h_s[:, None]

    tl.store(
        out_ptr + n_offs[:, None] * stride_on + c_offs[None, :] * stride_oc,
        acc,
        mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
    )


@triton.jit
def _mhc_pre_bwd_kernel(
    x_ptr,
    hpre_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_h_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    C: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_hh: tl.constexpr,
    stride_gon: tl.constexpr,
    stride_goc: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxh: tl.constexpr,
    stride_gxc: tl.constexpr,
    stride_ghn: tl.constexpr,
    stride_ghh: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    go = tl.load(
        grad_out_ptr + n_offs[:, None] * stride_gon + c_offs[None, :] * stride_goc,
        mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
        other=0.0,
    ).to(tl.float32)

    # grad_x = grad_out * hpre
    for s in tl.static_range(0, HC):
        h_s = tl.load(
            hpre_ptr + n_offs * stride_hn + s * stride_hh,
            mask=(n_offs < N),
            other=0.0,
        ).to(tl.float32)
        gx = go * h_s[:, None]
        tl.store(
            grad_x_ptr + n_offs[:, None] * stride_gxn + s * stride_gxh + c_offs[None, :] * stride_gxc,
            gx,
            mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
        )

        # grad_hpre: dot(go, x_s) over C -> atomic add
        xs = tl.load(
            x_ptr + n_offs[:, None] * stride_xn + s * stride_xh + c_offs[None, :] * stride_xc,
            mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        part = tl.sum(go * xs, axis=1)
        tl.atomic_add(
            grad_h_ptr + n_offs * stride_ghn + s * stride_ghh,
            part,
            mask=n_offs < N,
        )


def mhc_pre_fwd(
    x: torch.Tensor,
    h_pre: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    block_n: int = 32,
    block_c: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    assert x.is_contiguous() and h_pre.is_contiguous()
    N, HC, C = x.shape
    assert h_pre.shape == (N, HC)

    if out is None:
        out = torch.empty((N, C), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(N, block_n), triton.cdiv(C, block_c))
    _mhc_pre_fwd_kernel[grid](
        x,
        h_pre,
        out,
        N=N,
        HC=HC,
        C=C,
        stride_xn=x.stride(0),
        stride_xh=x.stride(1),
        stride_xc=x.stride(2),
        stride_hn=h_pre.stride(0),
        stride_hh=h_pre.stride(1),
        stride_on=out.stride(0),
        stride_oc=out.stride(1),
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out


def mhc_pre_bwd(
    x: torch.Tensor,
    h_pre: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    out_grad_x: Optional[torch.Tensor] = None,
    out_grad_h: Optional[torch.Tensor] = None,
    block_n: int = 32,
    block_c: int = 128,
    num_warps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and h_pre.is_contiguous() and grad_out.is_contiguous()
    N, HC, C = x.shape
    assert grad_out.shape == (N, C)

    if out_grad_x is None:
        out_grad_x = torch.empty_like(x, dtype=torch.float32)
    if out_grad_h is None:
        out_grad_h = torch.zeros((N, HC), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(N, block_n), triton.cdiv(C, block_c))
    _mhc_pre_bwd_kernel[grid](
        x,
        h_pre,
        grad_out,
        out_grad_x,
        out_grad_h,
        N=N,
        HC=HC,
        C=C,
        stride_xn=x.stride(0),
        stride_xh=x.stride(1),
        stride_xc=x.stride(2),
        stride_hn=h_pre.stride(0),
        stride_hh=h_pre.stride(1),
        stride_gon=grad_out.stride(0),
        stride_goc=grad_out.stride(1),
        stride_gxn=out_grad_x.stride(0),
        stride_gxh=out_grad_x.stride(1),
        stride_gxc=out_grad_x.stride(2),
        stride_ghn=out_grad_h.stride(0),
        stride_ghh=out_grad_h.stride(1),
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out_grad_x, out_grad_h


@triton.jit
def _mhc_post_res_fwd_kernel(
    x_ptr,
    f_ptr,
    hpost_ptr,
    hres_ptr,
    out_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    C: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_fn: tl.constexpr,
    stride_fc: tl.constexpr,
    stride_hpn: tl.constexpr,
    stride_hph: tl.constexpr,
    stride_hrn: tl.constexpr,
    stride_hri: tl.constexpr,
    stride_hrj: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_oc: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    f = tl.load(
        f_ptr + n_offs[:, None] * stride_fn + c_offs[None, :] * stride_fc,
        mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
        other=0.0,
    ).to(tl.float32)

    o2 = tl.arange(0, HC)[:, None]  # [HC,1]
    hpost = tl.load(
        hpost_ptr + n_offs[None, :] * stride_hpn + o2 * stride_hph,
        mask=(n_offs[None, :] < N),
        other=0.0,
    ).to(tl.float32)  # [HC, BN]

    acc = f[None, :, :] * hpost[:, :, None]  # [HC, BN, BC]

    # residual mixing: sum_i hres[o,i] * x_i
    for i in tl.static_range(0, HC):
        xs = tl.load(
            x_ptr + n_offs[:, None] * stride_xn + i * stride_xh + c_offs[None, :] * stride_xc,
            mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
            other=0.0,
        ).to(tl.float32)  # [BN, BC]
        w = tl.load(
            hres_ptr + n_offs[None, :] * stride_hrn + o2 * stride_hri + i * stride_hrj,
            mask=(n_offs[None, :] < N),
            other=0.0,
        ).to(tl.float32)  # [HC, BN]
        acc += xs[None, :, :] * w[:, :, None]

    o3 = tl.arange(0, HC)[:, None, None]
    n3 = n_offs[None, :, None]
    c3 = c_offs[None, None, :]
    tl.store(
        out_ptr + n3 * stride_on + o3 * stride_oh + c3 * stride_oc,
        acc,
        mask=(n3 < N) & (c3 < C),
    )


@triton.jit
def _mhc_post_res_bwd_kernel(
    x_ptr,
    f_ptr,
    hpost_ptr,
    hres_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_f_ptr,
    grad_hpost_ptr,
    grad_hres_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    C: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_fn: tl.constexpr,
    stride_fc: tl.constexpr,
    stride_hpn: tl.constexpr,
    stride_hph: tl.constexpr,
    stride_hrn: tl.constexpr,
    stride_hri: tl.constexpr,
    stride_hrj: tl.constexpr,
    stride_gon: tl.constexpr,
    stride_goh: tl.constexpr,
    stride_goc: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxh: tl.constexpr,
    stride_gxc: tl.constexpr,
    stride_gfn: tl.constexpr,
    stride_gfc: tl.constexpr,
    stride_ghpn: tl.constexpr,
    stride_ghph: tl.constexpr,
    stride_ghrn: tl.constexpr,
    stride_ghri: tl.constexpr,
    stride_ghrj: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    f = tl.load(
        f_ptr + n_offs[:, None] * stride_fn + c_offs[None, :] * stride_fc,
        mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
        other=0.0,
    ).to(tl.float32)

    o2 = tl.arange(0, HC)[:, None]  # [HC,1]
    hpost = tl.load(
        hpost_ptr + n_offs[None, :] * stride_hpn + o2 * stride_hph,
        mask=(n_offs[None, :] < N),
        other=0.0,
    ).to(tl.float32)  # [HC, BN]

    o3 = tl.arange(0, HC)[:, None, None]
    n3 = n_offs[None, :, None]
    c3 = c_offs[None, None, :]
    go = tl.load(
        grad_out_ptr + n3 * stride_gon + o3 * stride_goh + c3 * stride_goc,
        mask=(n3 < N) & (c3 < C),
        other=0.0,
    ).to(tl.float32)  # [HC, BN, BC]

    # grad_f: sum_o go[o] * hpost[o]
    gf = tl.sum(go * hpost[:, :, None], axis=0)
    tl.store(
        grad_f_ptr + n_offs[:, None] * stride_gfn + c_offs[None, :] * stride_gfc,
        gf,
        mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
    )

    # grad_hpost: dot(go[o], f) over C  (atomic over C blocks)
    part_hpost = tl.sum(go * f[None, :, :], axis=2)  # [HC, BN]
    tl.atomic_add(
        grad_hpost_ptr + n_offs[None, :] * stride_ghpn + o2 * stride_ghph,
        part_hpost,
        mask=(n_offs[None, :] < N),
    )

    # grad_x: hres^T @ go  (in-stream i gets sum_o hres[o,i] * go[o])
    for i in tl.static_range(0, HC):
        w = tl.load(
            hres_ptr + n_offs[None, :] * stride_hrn + o2 * stride_hri + i * stride_hrj,
            mask=(n_offs[None, :] < N),
            other=0.0,
        ).to(tl.float32)  # [HC, BN]
        gx = tl.sum(go * w[:, :, None], axis=0)  # [BN, BC]
        tl.store(
            grad_x_ptr + n_offs[:, None] * stride_gxn + i * stride_gxh + c_offs[None, :] * stride_gxc,
            gx,
            mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
        )

    # grad_hres[o,i]: dot(go[o], x[i]) over C (atomic)
    for i in tl.static_range(0, HC):
        xi = tl.load(
            x_ptr + n_offs[:, None] * stride_xn + i * stride_xh + c_offs[None, :] * stride_xc,
            mask=(n_offs[:, None] < N) & (c_offs[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        part_hres = tl.sum(go * xi[None, :, :], axis=2)  # [HC, BN]
        tl.atomic_add(
            grad_hres_ptr + n_offs[None, :] * stride_ghrn + o2 * stride_ghri + i * stride_ghrj,
            part_hres,
            mask=(n_offs[None, :] < N),
        )


def mhc_post_res_fwd(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    block_n: Optional[int] = None,
    block_c: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    assert x.is_contiguous() and f_out.is_contiguous() and h_post.is_contiguous() and h_res.is_contiguous()

    N, HC, C = x.shape
    assert f_out.shape == (N, C)
    assert h_post.shape == (N, HC)
    assert h_res.shape == (N, HC, HC)

    if out is None:
        out = torch.empty((N, HC, C), device=x.device, dtype=torch.float32)

    block_n, block_c, num_warps, num_stages = _post_res_meta(C, block_n, block_c, num_warps, num_stages)

    grid = (triton.cdiv(N, block_n), triton.cdiv(C, block_c))
    _mhc_post_res_fwd_kernel[grid](
        x,
        f_out,
        h_post,
        h_res,
        out,
        N=N,
        HC=HC,
        C=C,
        stride_xn=x.stride(0),
        stride_xh=x.stride(1),
        stride_xc=x.stride(2),
        stride_fn=f_out.stride(0),
        stride_fc=f_out.stride(1),
        stride_hpn=h_post.stride(0),
        stride_hph=h_post.stride(1),
        stride_hrn=h_res.stride(0),
        stride_hri=h_res.stride(1),
        stride_hrj=h_res.stride(2),
        stride_on=out.stride(0),
        stride_oh=out.stride(1),
        stride_oc=out.stride(2),
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def mhc_post_res_bwd(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    out_grad_x: Optional[torch.Tensor] = None,
    out_grad_f: Optional[torch.Tensor] = None,
    out_grad_hpost: Optional[torch.Tensor] = None,
    out_grad_hres: Optional[torch.Tensor] = None,
    block_n: Optional[int] = None,
    block_c: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        x.is_contiguous()
        and f_out.is_contiguous()
        and h_post.is_contiguous()
        and h_res.is_contiguous()
        and grad_out.is_contiguous()
    )

    N, HC, C = x.shape
    assert grad_out.shape == (N, HC, C)

    if out_grad_x is None:
        out_grad_x = torch.empty_like(x, dtype=torch.float32)
    if out_grad_f is None:
        out_grad_f = torch.empty_like(f_out, dtype=torch.float32)
    if out_grad_hpost is None:
        out_grad_hpost = torch.zeros((N, HC), device=x.device, dtype=torch.float32)
    if out_grad_hres is None:
        out_grad_hres = torch.zeros((N, HC, HC), device=x.device, dtype=torch.float32)

    block_n, block_c, num_warps, num_stages = _post_res_meta(C, block_n, block_c, num_warps, num_stages)

    grid = (triton.cdiv(N, block_n), triton.cdiv(C, block_c))
    _mhc_post_res_bwd_kernel[grid](
        x,
        f_out,
        h_post,
        h_res,
        grad_out,
        out_grad_x,
        out_grad_f,
        out_grad_hpost,
        out_grad_hres,
        N=N,
        HC=HC,
        C=C,
        stride_xn=x.stride(0),
        stride_xh=x.stride(1),
        stride_xc=x.stride(2),
        stride_fn=f_out.stride(0),
        stride_fc=f_out.stride(1),
        stride_hpn=h_post.stride(0),
        stride_hph=h_post.stride(1),
        stride_hrn=h_res.stride(0),
        stride_hri=h_res.stride(1),
        stride_hrj=h_res.stride(2),
        stride_gon=grad_out.stride(0),
        stride_goh=grad_out.stride(1),
        stride_goc=grad_out.stride(2),
        stride_gxn=out_grad_x.stride(0),
        stride_gxh=out_grad_x.stride(1),
        stride_gxc=out_grad_x.stride(2),
        stride_gfn=out_grad_f.stride(0),
        stride_gfc=out_grad_f.stride(1),
        stride_ghpn=out_grad_hpost.stride(0),
        stride_ghph=out_grad_hpost.stride(1),
        stride_ghrn=out_grad_hres.stride(0),
        stride_ghri=out_grad_hres.stride(1),
        stride_ghrj=out_grad_hres.stride(2),
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_grad_x, out_grad_f, out_grad_hpost, out_grad_hres


def _flatten_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Flattens leading dimensions so x becomes [N, HC, C].
    Returns (x_flat, outer_shape).
    """
    assert x.dim() >= 3, "x must be [..., HC, C]"
    outer = tuple(x.shape[:-2])
    return x.contiguous().view(-1, x.shape[-2], x.shape[-1]), outer


def _unflatten_tokens(y: torch.Tensor, outer: Tuple[int, ...]) -> torch.Tensor:
    return y.view(*outer, *y.shape[1:])


class LigerMHCCoeffsFunction(torch.autograd.Function):
    """
    Autograd function for mHC coefficient computation.

    Memory/Compute Trade-off:
        When gradients are needed, Sinkhorn iteration history (hist) is saved
        during forward to avoid recomputation in backward. This increases
        memory usage by O(N * tmax * HC^2) but reduces backward compute.
    """

    @staticmethod
    @ensure_contiguous
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,  # [..., HC, C] bf16/fp16 (or fp32 if allow_fp32)
        phi: torch.Tensor,  # [HC*C, M]
        b: torch.Tensor,  # [M]
        alpha_pre: torch.Tensor,  # scalar
        alpha_post: torch.Tensor,  # scalar
        alpha_res: torch.Tensor,  # scalar
        allow_fp32: bool,
        tmax: int,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if allow_fp32:
            assert x.dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ), "x should be BF16/FP16/FP32 when allow_fp32=True"
        else:
            assert x.dtype in (torch.bfloat16, torch.float16), "x should be BF16/FP16 (set allow_fp32=True for FP32)"
        x_flat, outer = _flatten_tokens(x)
        N, HC, C = x_flat.shape
        K = HC * C
        x_mat = x_flat.view(-1, K)

        assert phi.dim() == 2 and phi.shape[0] == K, f"phi must be [HC*C, M], got {tuple(phi.shape)}"
        M = int(phi.shape[1])
        assert b.shape == (M,), f"b must be [M], got {tuple(b.shape)}"

        # (1) fused coeff matmul + norm
        mix, invr = mhc_mm_norm_fwd(x_mat, phi, eps=float(rms_eps))

        # (2) split + sigmoid + sinkhorn
        need_hist = any(ctx.needs_input_grad)
        if need_hist:
            h_pre, h_post, h_res, hist = mhc_split_sinkhorn_fwd(
                mix,
                b,
                alpha_pre,
                alpha_post,
                alpha_res,
                tmax=int(tmax),
                pre_eps=float(pre_eps),
                sinkhorn_eps=float(sinkhorn_eps),
                post_mult=float(post_mult),
                return_hist=True,
            )
        else:
            h_pre, h_post, h_res = mhc_split_sinkhorn_fwd(
                mix,
                b,
                alpha_pre,
                alpha_post,
                alpha_res,
                tmax=int(tmax),
                pre_eps=float(pre_eps),
                sinkhorn_eps=float(sinkhorn_eps),
                post_mult=float(post_mult),
            )
            hist = None

        # Save for backward
        if hist is not None:
            ctx.save_for_backward(x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res, hist)
        else:
            ctx.save_for_backward(x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res)
        ctx.meta = (
            outer,
            HC,
            C,
            int(tmax),
            float(rms_eps),
            float(pre_eps),
            float(sinkhorn_eps),
            float(post_mult),
            hist is not None,
        )

        return (
            _unflatten_tokens(h_pre, outer),
            _unflatten_tokens(h_post, outer),
            _unflatten_tokens(h_res, outer),
        )

    @staticmethod
    @ensure_contiguous
    def backward(
        ctx: Any,
        grad_h_pre: torch.Tensor | None,
        grad_h_post: torch.Tensor | None,
        grad_h_res: torch.Tensor | None,
    ):
        saved = ctx.saved_tensors
        outer, HC, C, tmax, rms_eps, pre_eps, sinkhorn_eps, post_mult, has_hist = ctx.meta
        if has_hist:
            x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res, hist = saved
        else:
            x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res = saved
            hist = None
        N = x_mat.shape[0]
        M = mix.shape[1]
        assert M == HC * HC + 2 * HC

        need_pre = grad_h_pre is not None
        need_post = grad_h_post is not None
        need_res = grad_h_res is not None

        # flatten grads (None -> zeros)
        if need_pre:
            gh_pre = grad_h_pre.contiguous().view(-1, HC).to(torch.float32)
        else:
            gh_pre = torch.zeros((N, HC), device=mix.device, dtype=torch.float32)
        if need_post:
            gh_post = grad_h_post.contiguous().view(-1, HC).to(torch.float32)
        else:
            gh_post = torch.zeros((N, HC), device=mix.device, dtype=torch.float32)
        if need_res:
            gh_res = grad_h_res.contiguous().view(-1, HC, HC).to(torch.float32)
        else:
            gh_res = torch.zeros((N, HC, HC), device=mix.device, dtype=torch.float32)

        # --- Sinkhorn backward -> grad logits for residual matrix
        if need_res:
            grad_res_logits = mhc_sinkhorn_bwd(
                mix,
                b,
                alpha_res,
                gh_res,
                tmax=tmax,
                sinkhorn_eps=sinkhorn_eps,
                hist=hist,
            )  # [N, HC, HC] fp32
        else:
            grad_res_logits = gh_res

        # --- Pre/post derivatives (sigmoid)
        mix_pre = mix[:, :HC]
        mix_post = mix[:, HC : 2 * HC]
        mix_res = mix[:, 2 * HC :]

        b_pre = b[:HC]
        b_post = b[HC : 2 * HC]
        if need_pre:
            pre_logits = mix_pre * alpha_pre + b_pre
            pre_sig = torch.sigmoid(pre_logits)
            grad_pre_logits = gh_pre * (pre_sig * (1.0 - pre_sig))  # [N,HC]
        else:
            grad_pre_logits = gh_pre

        if need_post:
            post_logits = mix_post * alpha_post + b_post
            post_sig = torch.sigmoid(post_logits)
            grad_post_logits = gh_post * (post_mult * post_sig * (1.0 - post_sig))  # [N,HC]
        else:
            grad_post_logits = gh_post

        grad_res_logits_flat = grad_res_logits.reshape(N, HC * HC)

        # --- Grad w.r.t mix
        grad_mix = torch.empty_like(mix)
        grad_mix[:, :HC] = grad_pre_logits * alpha_pre
        grad_mix[:, HC : 2 * HC] = grad_post_logits * alpha_post
        grad_mix[:, 2 * HC :] = grad_res_logits_flat * alpha_res

        # --- Grad w.r.t b
        grad_b = torch.zeros_like(b, dtype=torch.float32)
        if need_pre:
            grad_b[:HC] = grad_pre_logits.sum(dim=0)
        if need_post:
            grad_b[HC : 2 * HC] = grad_post_logits.sum(dim=0)
        if need_res:
            grad_b[2 * HC :] = grad_res_logits_flat.sum(dim=0)

        # --- Grad w.r.t alphas
        if need_pre:
            grad_alpha_pre = (grad_pre_logits * mix_pre).sum()
        else:
            grad_alpha_pre = torch.zeros((), device=mix.device, dtype=torch.float32)
        if need_post:
            grad_alpha_post = (grad_post_logits * mix_post).sum()
        else:
            grad_alpha_post = torch.zeros((), device=mix.device, dtype=torch.float32)
        if need_res:
            grad_alpha_res = (grad_res_logits_flat * mix_res).sum()
        else:
            grad_alpha_res = torch.zeros((), device=mix.device, dtype=torch.float32)

        # --- Grad w.r.t x and phi via fused mm+norm backward
        grad_x_mat, grad_phi = mhc_mm_norm_bwd(
            x_mat,
            phi,
            mix,
            invr,
            grad_mix,
        )

        grad_x = grad_x_mat.view(-1, HC, C)
        grad_x = _unflatten_tokens(grad_x, outer)

        # Return grads for each forward input
        return (
            grad_x,  # x
            grad_phi,  # phi
            grad_b,  # b
            grad_alpha_pre,  # alpha_pre
            grad_alpha_post,  # alpha_post
            grad_alpha_res,  # alpha_res
            None,  # allow_fp32
            None,
            None,
            None,
            None,
            None,  # config scalars
        )


class LigerMHCPreFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx: Any, x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        x_flat, outer = _flatten_tokens(x)
        h_pre_flat = h_pre.contiguous().view(-1, x_flat.shape[1]).to(torch.float32)
        out = mhc_pre_fwd(x_flat, h_pre_flat)  # [N,C] fp32
        ctx.save_for_backward(x_flat, h_pre_flat)
        ctx.outer = outer
        out = out.to(x_flat.dtype)
        return _unflatten_tokens(out, outer)

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_out: torch.Tensor):
        x_flat, h_pre_flat = ctx.saved_tensors
        outer = ctx.outer
        N, HC, C = x_flat.shape
        go = grad_out.contiguous().view(-1, C).to(torch.float32)
        grad_x, grad_h = mhc_pre_bwd(x_flat, h_pre_flat, go)
        grad_x = grad_x.to(x_flat.dtype)
        return _unflatten_tokens(grad_x.view(-1, HC, C), outer), _unflatten_tokens(grad_h.view(-1, HC), outer)


class LigerMHCPostResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx: Any, x: torch.Tensor, f_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor
    ) -> torch.Tensor:
        x_flat, outer = _flatten_tokens(x)
        N, HC, C = x_flat.shape
        f_flat = f_out.contiguous().view(-1, C)
        h_post_flat = h_post.contiguous().view(-1, HC).to(torch.float32)
        h_res_flat = h_res.contiguous().view(-1, HC, HC).to(torch.float32)
        out = mhc_post_res_fwd(x_flat, f_flat, h_post_flat, h_res_flat)  # [N,HC,C] fp32
        ctx.save_for_backward(x_flat, f_flat, h_post_flat, h_res_flat)
        ctx.outer = outer
        out = out.to(x_flat.dtype)
        return _unflatten_tokens(out, outer)

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_out: torch.Tensor):
        x_flat, f_flat, h_post_flat, h_res_flat = ctx.saved_tensors
        outer = ctx.outer
        N, HC, C = x_flat.shape
        go = grad_out.contiguous().view(-1, HC, C).to(torch.float32)

        grad_x, grad_f, grad_hpost, grad_hres = mhc_post_res_bwd(x_flat, f_flat, h_post_flat, h_res_flat, go)

        return (
            _unflatten_tokens(grad_x.to(x_flat.dtype).view(-1, HC, C), outer),
            _unflatten_tokens(grad_f.to(f_flat.dtype).view(-1, C), outer),
            _unflatten_tokens(grad_hpost.view(-1, HC), outer),
            _unflatten_tokens(grad_hres.view(-1, HC, HC), outer),
        )
