"""
Ascend NPU implementation of Manifold-Constrained Hyper-Connections (mHC).

This module provides NPU-optimized Triton kernels for the three mHC sub-operators:
  1. Coefficients: fused matmul + RMS normalization + Sinkhorn routing
  2. Pre-aggregate: weighted sum across residual streams
  3. Post + residual: residual mixing with post-scaling

All kernels use the unified UB tiling strategy from ``ub_manager`` and follow
Ascend NPU best practices (persistent grid-stride loops, cache modifiers).
"""

import math

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# ---------------------------------------------------------------------------
# UB-aware block size helpers  (via unified compute_default_tiling_strategy)
# ---------------------------------------------------------------------------


def _mhc_mm_norm_block_sizes(N, K, M, dtype_size=4):
    """
    Compute UB-safe (BLOCK_N, BLOCK_K, BLOCK_M) for the matmul+norm kernel.

    Adaptive strategy:
      - BLOCK_M covers all M columns when M is small (typical for mHC where
        M = HC²+2HC ≈ 24-80), avoiding M-dimension tiling entirely.
      - BLOCK_N is chosen to ensure enough tiles for full NPU core utilisation.
      - BLOCK_K fills remaining UB budget.
    """
    # 1. BLOCK_M: cover all M in one block when feasible
    block_m = min(triton.next_power_of_2(M), 64)

    # 2. BLOCK_N: ensure enough tiles (≥ num_cores) for parallelism
    num_cores = get_npu_core_count()
    tiles_m = triton.cdiv(M, block_m)
    block_n = 16
    for bn in [128, 64, 32, 16, 8, 4]:
        if triton.cdiv(N, bn) * tiles_m >= num_cores:
            block_n = bn
            break
    else:
        block_n = 4

    # 3. BLOCK_K: fill UB budget given chosen BN and BM
    #    UB per tile: resident  acc[BN,BM]*4 + sumsq[BN]*4
    #                 streaming 2*(x[BN,BK]*4 + phi[BK,BM]*4)
    #    Solve for BK: 8*(BN+BM)*BK ≤ UB*0.9 - 4*BN*(BM+1)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.90,
        dtype_size=4,
        memory_multiplier=float(2 * (block_n + block_m)) / max(block_n, 1),
        shapes=((block_n, K),),
        tiling_dims=(1,),
    )
    block_k = max(32, tile_shapes[0][1]) if tile_shapes else 128
    block_k = min(block_k, K, 256)
    return block_n, block_k, block_m


def _mhc_pre_post_block_c(C, HC, block_n=4, is_post=False, is_bwd=False):
    """
    Compute UB-safe BLOCK_C for pre / post_res kernels via unified tiling.

    Estimates memory_multiplier from the number of [BN, BC]-sized float32 tiles
    simultaneously resident in UB (with 2× multi-buffer factor).
    """
    if is_post:
        # fwd: only acc[BN,BC] + xs[BN,BC] + f[BN,BC] live at once (j-loop writes immediately)
        # bwd: still has the full [HC,BN,BC] accumulator structure
        multiplier = float(2 * (HC + 4)) if is_bwd else 6.0
        # multiplier = 14.0
    else:
        multiplier = 4.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.90,
        dtype_size=4,
        memory_multiplier=multiplier,
        shapes=((block_n, C),),
        tiling_dims=(1,),
    )
    block_c = tile_shapes[0][1] if tile_shapes else 128
    return max(32, min(block_c, C))


@triton.jit
def _mhc_mm_norm_fwd_kernel_npu(
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
):
    """NPU matmul+RMS-norm forward.  Persistent grid-stride loop."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = tl.cdiv(N, BLOCK_N) * tl.cdiv(M, BLOCK_M)
    tiles_per_row = tl.cdiv(M, BLOCK_M)

    for tile_id in tl.range(pid, total_tiles, num_progs):
        pid_n = tile_id // tiles_per_row
        pid_m = tile_id % tiles_per_row

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        sumsq = tl.zeros((BLOCK_N,), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)

            x = tl.load(
                x_ptr + n_offs[:, None] * stride_xn + k_offs[None, :] * stride_xk,
                mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                other=0.0,
            ).to(tl.float32)

            sumsq += tl.sum(x * x, axis=1)

            phi = tl.load(
                phi_ptr + k_offs[:, None] * stride_phik + m_offs[None, :] * stride_phim,
                mask=(k_offs[:, None] < K) & (m_offs[None, :] < M),
                other=0.0,
            ).to(tl.float32)

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and phi.is_contiguous()
    N, K = x.shape
    _, M = phi.shape

    if out_mix is None:
        out_mix = torch.empty((N, M), device=x.device, dtype=torch.float32)
    if out_invr is None:
        out_invr = torch.empty((N,), device=x.device, dtype=torch.float32)

    dtype_sz = x.element_size()
    block_n, block_k, block_m = _mhc_mm_norm_block_sizes(N, K, M, dtype_sz)

    num_cores = get_npu_core_count()
    total_tiles = triton.cdiv(N, block_n) * triton.cdiv(M, block_m)
    grid = (min(total_tiles, num_cores),)
    _mhc_mm_norm_fwd_kernel_npu[grid](
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
    )
    return out_mix, out_invr


@triton.jit
def _mhc_mm_norm_bwd_kernel_npu(
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
    stride_gmn: tl.constexpr,
    stride_gmm: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxk: tl.constexpr,
    stride_gpk: tl.constexpr,
    stride_gpm: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """NPU backward for matmul+RMS norm.  Persistent grid-stride loop."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = tl.cdiv(N, BLOCK_N) * tl.cdiv(K, BLOCK_K)
    tiles_per_row = tl.cdiv(K, BLOCK_K)

    for tile_id in tl.range(pid, total_tiles, num_progs):
        pid_n = tile_id // tiles_per_row
        pid_k = tile_id % tiles_per_row

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        invr = tl.load(invr_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)

        x = tl.load(
            x_ptr + n_offs[:, None] * stride_xn + k_offs[None, :] * stride_xk,
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        g_acc = tl.zeros((BLOCK_N,), tl.float32)

        for m0 in range(0, M, BLOCK_M):
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
            ).to(tl.float32)

            grad_z = grad_mix * invr[:, None]
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

        gx = acc + x * factor[:, None]

        tl.store(
            grad_x_ptr + n_offs[:, None] * stride_gxn + k_offs[None, :] * stride_gxk,
            gx,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        x.is_contiguous()
        and phi.is_contiguous()
        and mix.is_contiguous()
        and invr.is_contiguous()
        and grad_mix.is_contiguous()
    )
    N, K = x.shape
    _, M = phi.shape

    if out_grad_x is None:
        out_grad_x = torch.empty((N, K), device=x.device, dtype=torch.float32)
    if out_grad_phi is None:
        out_grad_phi = torch.zeros((K, M), device=x.device, dtype=torch.float32)

    dtype_sz = x.element_size()
    block_n, block_k, block_m = _mhc_mm_norm_block_sizes(N, K, M, dtype_sz)

    num_cores = get_npu_core_count()
    total_tiles = triton.cdiv(N, block_n) * triton.cdiv(K, block_k)
    grid = (min(total_tiles, num_cores),)
    _mhc_mm_norm_bwd_kernel_npu[grid](
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
        stride_gmn=grad_mix.stride(0),
        stride_gmm=grad_mix.stride(1),
        stride_gxn=out_grad_x.stride(0),
        stride_gxk=out_grad_x.stride(1),
        stride_gpk=out_grad_phi.stride(0),
        stride_gpm=out_grad_phi.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=block_m,
    )
    if out_grad_phi.dtype != phi.dtype:
        out_grad_phi = out_grad_phi.to(phi.dtype)
    return out_grad_x, out_grad_phi


@triton.jit
def _mhc_split_sinkhorn_fwd_kernel_npu(
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
    """One program per token – processes (h_pre, h_post, h_res) for one sample."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    for sample_id in tl.range(pid, N, num_progs):
        j = tl.arange(0, HC)
        mix_pre = tl.load(mix_ptr + sample_id * stride_mn + j * stride_mm).to(tl.float32)
        mix_post = tl.load(mix_ptr + sample_id * stride_mn + (HC + j) * stride_mm).to(tl.float32)

        b_pre = tl.load(b_ptr + j).to(tl.float32)
        b_post = tl.load(b_ptr + (HC + j)).to(tl.float32)

        pre_logits = mix_pre * alpha_pre + b_pre
        post_logits = mix_post * alpha_post + b_post

        pre = tl.sigmoid(pre_logits) + pre_eps
        post = tl.sigmoid(post_logits) * post_mult

        tl.store(hpre_ptr + sample_id * stride_hp_n + j * stride_hp_h, pre)
        tl.store(hpost_ptr + sample_id * stride_hq_n + j * stride_hq_h, post)

        # Residual logits [HC, HC]
        rows = tl.arange(0, HC)[:, None]
        cols = tl.arange(0, HC)[None, :]
        flat = rows * HC + cols

        mix_res = tl.load(mix_ptr + sample_id * stride_mn + (2 * HC + flat) * stride_mm).to(tl.float32)
        b_res = tl.load(b_ptr + (2 * HC + flat)).to(tl.float32)
        logits = mix_res * alpha_res + b_res

        # Sinkhorn
        row_max = tl.max(logits, axis=1)
        e = tl.exp(logits - row_max[:, None])
        row_sum = tl.sum(e, axis=1)
        mat = e / row_sum[:, None] + sinkhorn_eps

        col_sum = tl.sum(mat, axis=0)
        mat = mat / (col_sum[None, :] + sinkhorn_eps)

        if STORE_HIST:
            tl.store(
                hist_ptr + sample_id * stride_hn + 0 * stride_ht + rows * stride_hi + cols * stride_hj,
                mat,
            )

        for t in range(1, TMAX):
            row_sum = tl.sum(mat, axis=1)
            mat = mat / (row_sum[:, None] + sinkhorn_eps)
            col_sum = tl.sum(mat, axis=0)
            mat = mat / (col_sum[None, :] + sinkhorn_eps)
            if STORE_HIST:
                tl.store(
                    hist_ptr + sample_id * stride_hn + t * stride_ht + rows * stride_hi + cols * stride_hj,
                    mat,
                )

        tl.store(hres_ptr + sample_id * stride_hr_n + rows * stride_hr_i + cols * stride_hr_j, mat)


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
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    assert mix.is_contiguous() and b.is_contiguous()
    N, M = mix.shape
    HC = int((math.isqrt(4 + 4 * M) - 2) // 2)
    assert HC * HC + 2 * HC == M

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

    num_cores = get_npu_core_count()
    grid = (min(N, num_cores),)

    _mhc_split_sinkhorn_fwd_kernel_npu[grid](
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
    )
    if return_hist:
        return out_hpre, out_hpost, out_hres, out_hist
    return out_hpre, out_hpost, out_hres


@triton.jit
def _mhc_sinkhorn_bwd_hist_kernel_npu(
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
    num_progs = tl.num_programs(0)

    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    flat = rows * HC + cols

    for sample_id in tl.range(pid, N, num_progs):
        # Rebuild logits
        mix_res = tl.load(mix_ptr + sample_id * stride_mn + (2 * HC + flat) * stride_mm).to(tl.float32)
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
            grad_out_ptr + sample_id * stride_go_n + rows * stride_go_i + cols * stride_go_j,
        ).to(tl.float32)

        # Reverse iterations (TMAX-1 .. 1) using stored mats
        for t in tl.static_range(TMAX - 1, 0, -1):
            mat_t = tl.load(hist_ptr + sample_id * stride_hn + t * stride_ht + rows * stride_hi + cols * stride_hj).to(
                tl.float32
            )
            mat_prev = tl.load(
                hist_ptr + sample_id * stride_hn + (t - 1) * stride_ht + rows * stride_hi + cols * stride_hj
            ).to(tl.float32)

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

        tl.store(grad_logits_ptr + sample_id * stride_gl_n + rows * stride_gl_i + cols * stride_gl_j, grad_logits)


@triton.jit
def _mhc_sinkhorn_bwd_kernel_npu(
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
    """Backward without history – recompute forward per step."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    flat = rows * HC + cols

    for sample_id in tl.range(pid, N, num_progs):
        mix_res = tl.load(mix_ptr + sample_id * stride_mn + (2 * HC + flat) * stride_mm).to(tl.float32)
        b_res = tl.load(b_ptr + (2 * HC + flat)).to(tl.float32)
        logits = mix_res * alpha_res + b_res

        row_max = tl.max(logits, axis=1)
        e = tl.exp(logits - row_max[:, None])
        row_sum0 = tl.sum(e, axis=1)
        p = e / row_sum0[:, None]
        p_eps = p + sinkhorn_eps

        col_sum0 = tl.sum(p_eps, axis=0)
        mat0 = p_eps / (col_sum0[None, :] + sinkhorn_eps)

        g = tl.load(
            grad_out_ptr + sample_id * stride_go_n + rows * stride_go_i + cols * stride_go_j,
        ).to(tl.float32)

        # Reverse with recompute
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

            denom_col = cs_t + sinkhorn_eps
            dot_col = tl.sum(g * mat_t, axis=0)
            g_row = (g - dot_col[None, :]) / denom_col[None, :]

            m_row = mat_t * denom_col[None, :]
            denom_row = rs_t + sinkhorn_eps
            dot_row = tl.sum(g_row * m_row, axis=1)
            g = (g_row - dot_row[:, None]) / denom_row[:, None]

        denom_col0 = col_sum0 + sinkhorn_eps
        dot_col0 = tl.sum(g * mat0, axis=0)
        g_p = (g - dot_col0[None, :]) / denom_col0[None, :]

        dot_soft = tl.sum(g_p * p, axis=1)
        grad_logits = p * (g_p - dot_soft[:, None])

        tl.store(
            grad_logits_ptr + sample_id * stride_gl_n + rows * stride_gl_i + cols * stride_gl_j,
            grad_logits,
        )


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
) -> torch.Tensor:
    assert mix.is_contiguous() and b.is_contiguous() and grad_hres.is_contiguous()
    N, M = mix.shape
    HC = grad_hres.shape[1]
    assert grad_hres.shape == (N, HC, HC)

    if out_grad_logits is None:
        out_grad_logits = torch.empty((N, HC, HC), device=mix.device, dtype=torch.float32)

    num_cores = get_npu_core_count()
    grid = (min(N, num_cores),)
    alpha_res_c = alpha_res.contiguous()

    if hist is not None:
        assert hist.is_contiguous() and hist.shape == (N, tmax, HC, HC)
        _mhc_sinkhorn_bwd_hist_kernel_npu[grid](
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
        )
    else:
        _mhc_sinkhorn_bwd_kernel_npu[grid](
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
        )
    return out_grad_logits


@triton.jit
def _mhc_pre_fwd_kernel_npu(
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
def _mhc_pre_bwd_kernel_npu(
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


def _select_pre_post_blocks(N, C, HC, is_post, is_bwd):
    """Jointly select (BLOCK_N, BLOCK_C) that fits UB and maximises core usage.

    Iterates from large BLOCK_N downward; for each candidate, computes the
    matching UB-safe BLOCK_C and checks whether the resulting tile count
    is enough to keep all NPU cores busy.
    """
    num_cores = get_npu_core_count()
    for bn in [32, 16, 8, 4, 2, 1]:
        bc = _mhc_pre_post_block_c(C, HC, block_n=bn, is_post=is_post, is_bwd=is_bwd)
        if triton.cdiv(N, bn) * triton.cdiv(C, bc) >= num_cores:
            return bn, bc
    # Fallback: smallest BN, matching BC
    return 1, _mhc_pre_post_block_c(C, HC, block_n=1, is_post=is_post, is_bwd=is_bwd)


def mhc_pre_fwd(
    x: torch.Tensor,
    h_pre: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.is_contiguous() and h_pre.is_contiguous()
    N, HC, C = x.shape
    assert h_pre.shape == (N, HC)

    if out is None:
        out = torch.empty((N, C), device=x.device, dtype=torch.float32)

    num_cores = get_npu_core_count()
    block_n, block_c = _select_pre_post_blocks(N, C, HC, is_post=False, is_bwd=False)
    grid = (num_cores,)

    _mhc_pre_fwd_kernel_npu[grid](
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
    )
    return out


def mhc_pre_bwd(
    x: torch.Tensor,
    h_pre: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    out_grad_x: Optional[torch.Tensor] = None,
    out_grad_h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and h_pre.is_contiguous() and grad_out.is_contiguous()
    N, HC, C = x.shape

    if out_grad_x is None:
        out_grad_x = torch.empty_like(x, dtype=torch.float32)
    if out_grad_h is None:
        out_grad_h = torch.zeros((N, HC), device=x.device, dtype=torch.float32)

    num_cores = get_npu_core_count()
    block_n, block_c = _select_pre_post_blocks(N, C, HC, is_post=False, is_bwd=True)
    grid = (num_cores,)

    _mhc_pre_bwd_kernel_npu[grid](
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
    )
    return out_grad_x, out_grad_h


@triton.jit
def _mhc_post_res_fwd_kernel_npu(
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
def _mhc_post_res_bwd_kernel_npu(
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
) -> torch.Tensor:
    assert x.is_contiguous() and f_out.is_contiguous() and h_post.is_contiguous() and h_res.is_contiguous()
    N, HC, C = x.shape

    if out is None:
        out = torch.empty((N, HC, C), device=x.device, dtype=torch.float32)

    num_cores = get_npu_core_count()
    block_n, block_c = _select_pre_post_blocks(N, C, HC, is_post=True, is_bwd=False)
    grid = (num_cores,)

    _mhc_post_res_fwd_kernel_npu[grid](
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        x.is_contiguous()
        and f_out.is_contiguous()
        and h_post.is_contiguous()
        and h_res.is_contiguous()
        and grad_out.is_contiguous()
    )
    N, HC, C = x.shape

    if out_grad_x is None:
        out_grad_x = torch.empty_like(x, dtype=torch.float32)
    if out_grad_f is None:
        out_grad_f = torch.empty_like(f_out, dtype=torch.float32)
    if out_grad_hpost is None:
        out_grad_hpost = torch.zeros((N, HC), device=x.device, dtype=torch.float32)
    if out_grad_hres is None:
        out_grad_hres = torch.zeros((N, HC, HC), device=x.device, dtype=torch.float32)

    num_cores = get_npu_core_count()
    block_n, block_c = _select_pre_post_blocks(N, C, HC, is_post=True, is_bwd=True)
    grid = (num_cores,)

    _mhc_post_res_bwd_kernel_npu[grid](
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
    )
    return out_grad_x, out_grad_f, out_grad_hpost, out_grad_hres


@triton.jit
def _mhc_coeffs_bwd_assemble_kernel_npu(
    mix_ptr,
    b_ptr,
    gh_pre_ptr,
    gh_post_ptr,
    grad_res_logits_ptr,
    alpha_pre_ptr,
    alpha_post_ptr,
    alpha_res_ptr,
    grad_mix_ptr,
    grad_b_ptr,
    grad_alpha_pre_ptr,
    grad_alpha_post_ptr,
    grad_alpha_res_ptr,
    N: tl.constexpr,
    HC: tl.constexpr,
    post_mult: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_ghn: tl.constexpr,
    stride_grn: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_RES: tl.constexpr,
):
    """
    Fuse sigmoid backward + grad_mix assembly + grad_b / grad_alpha reductions.

    Grid: (min(N, num_cores),) with grid-stride loop over rows.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)

    j = tl.arange(0, BLOCK_HC)
    j_mask = j < HC
    b_pre = tl.load(b_ptr + j, mask=j_mask, other=0.0).to(tl.float32)
    b_post = tl.load(b_ptr + HC + j, mask=j_mask, other=0.0).to(tl.float32)

    HC_SQ = HC * HC
    r = tl.arange(0, BLOCK_RES)
    r_mask = r < HC_SQ

    # Local accumulators — reduced via atomic_add after the loop
    acc_gb_pre = tl.zeros((BLOCK_HC,), tl.float32)
    acc_gb_post = tl.zeros((BLOCK_HC,), tl.float32)
    acc_gb_res = tl.zeros((BLOCK_RES,), tl.float32)
    acc_ga_pre = tl.zeros((BLOCK_HC,), tl.float32)
    acc_ga_post = tl.zeros((BLOCK_HC,), tl.float32)
    acc_ga_res = tl.zeros((BLOCK_RES,), tl.float32)

    for n in tl.range(pid, N, num_progs):
        row_mix = n * stride_mn

        # --- Pre segment ---
        mix_pre = tl.load(mix_ptr + row_mix + j, mask=j_mask, other=0.0).to(tl.float32)
        gh_pre_v = tl.load(gh_pre_ptr + n * stride_ghn + j, mask=j_mask, other=0.0).to(tl.float32)
        pre_logits = mix_pre * alpha_pre + b_pre
        pre_sig = tl.sigmoid(pre_logits)
        grad_pre = gh_pre_v * (pre_sig * (1.0 - pre_sig))
        tl.store(grad_mix_ptr + row_mix + j, grad_pre * alpha_pre, mask=j_mask)
        acc_gb_pre += grad_pre
        acc_ga_pre += grad_pre * mix_pre

        # --- Post segment ---
        mix_post = tl.load(mix_ptr + row_mix + HC + j, mask=j_mask, other=0.0).to(tl.float32)
        gh_post_v = tl.load(gh_post_ptr + n * stride_ghn + j, mask=j_mask, other=0.0).to(tl.float32)
        post_logits = mix_post * alpha_post + b_post
        post_sig = tl.sigmoid(post_logits)
        grad_post = gh_post_v * (post_mult * post_sig * (1.0 - post_sig))
        tl.store(grad_mix_ptr + row_mix + HC + j, grad_post * alpha_post, mask=j_mask)
        acc_gb_post += grad_post
        acc_ga_post += grad_post * mix_post

        # --- Res segment (no sigmoid – sinkhorn bwd already applied) ---
        mix_res = tl.load(mix_ptr + row_mix + 2 * HC + r, mask=r_mask, other=0.0).to(tl.float32)
        grad_res = tl.load(grad_res_logits_ptr + n * stride_grn + r, mask=r_mask, other=0.0).to(tl.float32)
        tl.store(grad_mix_ptr + row_mix + 2 * HC + r, grad_res * alpha_res, mask=r_mask)
        acc_gb_res += grad_res
        acc_ga_res += grad_res * mix_res

    # Atomic reduce grad_b
    tl.atomic_add(grad_b_ptr + j, acc_gb_pre, mask=j_mask)
    tl.atomic_add(grad_b_ptr + HC + j, acc_gb_post, mask=j_mask)
    tl.atomic_add(grad_b_ptr + 2 * HC + r, acc_gb_res, mask=r_mask)

    # Atomic reduce grad_alpha (sum per-element vectors, then scalar atomic)
    tl.atomic_add(grad_alpha_pre_ptr, tl.sum(acc_ga_pre))
    tl.atomic_add(grad_alpha_post_ptr, tl.sum(acc_ga_post))
    tl.atomic_add(grad_alpha_res_ptr, tl.sum(acc_ga_res))


def mhc_coeffs_bwd_assemble(
    mix: torch.Tensor,
    b: torch.Tensor,
    gh_pre: torch.Tensor,
    gh_post: torch.Tensor,
    grad_res_logits: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    post_mult: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused backward assembly: produces grad_mix, grad_b, grad_alpha_{pre,post,res}.

    Replaces ~15 PyTorch kernel launches with 1 Triton kernel.
    """
    N, M = mix.shape
    HC = gh_pre.shape[1]

    grad_mix = torch.empty_like(mix, dtype=torch.float32)
    grad_b = torch.zeros_like(b, dtype=torch.float32)
    grad_alpha_pre_t = torch.zeros(1, device=mix.device, dtype=torch.float32)
    grad_alpha_post_t = torch.zeros(1, device=mix.device, dtype=torch.float32)
    grad_alpha_res_t = torch.zeros(1, device=mix.device, dtype=torch.float32)

    # Flatten grad_res_logits from [N, HC, HC] to [N, HC*HC]
    grad_res_flat = grad_res_logits.reshape(N, HC * HC).contiguous()

    BLOCK_HC = triton.next_power_of_2(HC)
    BLOCK_RES = triton.next_power_of_2(HC * HC)

    num_cores = get_npu_core_count()
    grid = (min(N, num_cores),)

    _mhc_coeffs_bwd_assemble_kernel_npu[grid](
        mix,
        b,
        gh_pre,
        gh_post,
        grad_res_flat,
        alpha_pre,
        alpha_post,
        alpha_res,
        grad_mix,
        grad_b,
        grad_alpha_pre_t,
        grad_alpha_post_t,
        grad_alpha_res_t,
        N=N,
        HC=HC,
        post_mult=post_mult,
        stride_mn=mix.stride(0),
        stride_ghn=gh_pre.stride(0),
        stride_grn=grad_res_flat.stride(0),
        BLOCK_HC=BLOCK_HC,
        BLOCK_RES=BLOCK_RES,
    )
    return (
        grad_mix,
        grad_b,
        grad_alpha_pre_t.squeeze(0),
        grad_alpha_post_t.squeeze(0),
        grad_alpha_res_t.squeeze(0),
    )


def _flatten_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    assert x.dim() >= 3
    return x.contiguous().view(-1, x.shape[-2], x.shape[-1]), x.shape


class LigerMHCCoeffsFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx: Any,
        x: torch.Tensor,
        phi: torch.Tensor,
        b: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        allow_fp32: bool,
        tmax: int,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if allow_fp32:
            assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        else:
            assert x.dtype in (torch.bfloat16, torch.float16)

        x_shape = x.shape
        x_flat, _ = _flatten_tokens(x)
        N, HC, C = x_flat.shape
        K = HC * C
        x_mat = x_flat.view(-1, K)

        assert phi.dim() == 2 and phi.shape[0] == K
        M = int(phi.shape[1])
        assert b.shape == (M,)

        mix, invr = mhc_mm_norm_fwd(x_mat, phi, eps=float(rms_eps))

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

        if hist is not None:
            ctx.save_for_backward(x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res, hist)
        else:
            ctx.save_for_backward(x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res)
        ctx.meta = (x_shape, HC, C, int(tmax), float(sinkhorn_eps), float(post_mult), hist is not None)

        outer = x_shape[:-2]
        return (
            h_pre.view(*outer, HC),
            h_post.view(*outer, HC),
            h_res.view(*outer, HC, HC),
        )

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_h_pre, grad_h_post, grad_h_res):
        saved = ctx.saved_tensors
        x_shape, HC, C, tmax, sinkhorn_eps, post_mult, has_hist = ctx.meta
        if has_hist:
            x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res, hist = saved
        else:
            x_mat, phi, b, mix, invr, alpha_pre, alpha_post, alpha_res = saved
            hist = None
        N = x_mat.shape[0]

        gh_pre = (
            grad_h_pre.view(-1, HC).to(torch.float32)
            if grad_h_pre is not None
            else torch.zeros((N, HC), device=mix.device, dtype=torch.float32)
        )
        gh_post = (
            grad_h_post.view(-1, HC).to(torch.float32)
            if grad_h_post is not None
            else torch.zeros((N, HC), device=mix.device, dtype=torch.float32)
        )
        gh_res = (
            grad_h_res.view(-1, HC, HC).to(torch.float32)
            if grad_h_res is not None
            else torch.zeros((N, HC, HC), device=mix.device, dtype=torch.float32)
        )

        # (a) Sinkhorn backward → grad_res_logits [N, HC, HC]
        if grad_h_res is not None:
            grad_res_logits = mhc_sinkhorn_bwd(
                mix, b, alpha_res, gh_res, tmax=tmax, sinkhorn_eps=sinkhorn_eps, hist=hist
            )
        else:
            grad_res_logits = gh_res

        # (b) Fused sigmoid bwd + assembly + reductions (1 kernel, replaces ~15 ops)
        grad_mix, grad_b, grad_alpha_pre, grad_alpha_post, grad_alpha_res = mhc_coeffs_bwd_assemble(
            mix,
            b,
            gh_pre,
            gh_post,
            grad_res_logits,
            alpha_pre,
            alpha_post,
            alpha_res,
            post_mult=post_mult,
        )

        # (c) Matmul+norm backward
        grad_x_mat, grad_phi = mhc_mm_norm_bwd(x_mat, phi, mix, invr, grad_mix)
        grad_x = grad_x_mat.view(x_shape)

        return (
            grad_x,
            grad_phi,
            grad_b,
            grad_alpha_pre,
            grad_alpha_post,
            grad_alpha_res,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LigerMHCPreFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx: Any, x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_flat, _ = _flatten_tokens(x)
        h_pre_flat = h_pre.view(-1, x_flat.shape[1]).to(torch.float32)
        out = mhc_pre_fwd(x_flat, h_pre_flat)
        ctx.save_for_backward(x_flat, h_pre_flat)
        ctx.x_shape = x_shape
        out = out.to(x_flat.dtype)
        return out.view(*x_shape[:-2], out.shape[-1])

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_out: torch.Tensor):
        x_flat, h_pre_flat = ctx.saved_tensors
        x_shape = ctx.x_shape
        N, HC, C = x_flat.shape
        go = grad_out.view(-1, C).to(torch.float32)
        grad_x, grad_h = mhc_pre_bwd(x_flat, h_pre_flat, go)
        grad_x = grad_x.to(x_flat.dtype)
        return grad_x.view(*x_shape), grad_h.view(*x_shape[:-1])


class LigerMHCPostResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx: Any, x: torch.Tensor, f_out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor
    ) -> torch.Tensor:
        x_shape = x.shape
        x_flat, _ = _flatten_tokens(x)
        N, HC, C = x_flat.shape
        f_flat = f_out.view(-1, C)
        h_post_flat = h_post.view(-1, HC).to(torch.float32)
        h_res_flat = h_res.view(-1, HC, HC).to(torch.float32)
        out = mhc_post_res_fwd(x_flat, f_flat, h_post_flat, h_res_flat)
        ctx.save_for_backward(x_flat, f_flat, h_post_flat, h_res_flat)
        ctx.x_shape = x_shape
        out = out.to(x_flat.dtype)
        return out.view(*x_shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx: Any, grad_out: torch.Tensor):
        x_flat, f_flat, h_post_flat, h_res_flat = ctx.saved_tensors
        x_shape = ctx.x_shape
        N, HC, C = x_flat.shape
        go = grad_out.view(-1, HC, C).to(torch.float32)

        grad_x, grad_f, grad_hpost, grad_hres = mhc_post_res_bwd(x_flat, f_flat, h_post_flat, h_res_flat, go)

        outer = x_shape[:-2]
        return (
            grad_x.to(x_flat.dtype).view(*x_shape),
            grad_f.to(f_flat.dtype).view(*outer, C),
            grad_hpost.view(*outer, HC),
            grad_hres.view(*outer, HC, HC),
        )
