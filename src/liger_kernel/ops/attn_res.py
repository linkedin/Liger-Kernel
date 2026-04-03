"""
Attention Residuals (AttnRes) - Kimi/Moonshot AI

Replaces standard residual connections h_l = h_{l-1} + f_l(RMSNorm(h_{l-1}))
with softmax attention over depth for dynamic weighting:

  V = stack(blocks)           # [N, B, T, D]
  K = RMSNorm(V)              # per-block normalize
  scores = einsum(w, K)       # [N, B, T] — w is [D] learned query
  alpha = softmax(scores, 0)  # over block dim
  h = einsum(alpha, V)        # [B, T, D] — weighted sum

Solves PreNorm dilution: deep layer contributions being diluted.
Paper: https://arxiv.org/abs/2603.15031

Triton optimizations:
1. Single kernel fuses RMSNorm + dot + softmax + weighted sum
2. Each program handles one (batch, token) position
3. N is small (≤16 blocks), scores fit in registers
"""

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous

# ============================================================================
# Forward Kernel
# ============================================================================


@triton.jit
def _attn_res_fwd_kernel(
    V_ptr,  # [N, B*T, D] stacked block values
    W_query_ptr,  # [D] learned pseudo-query
    W_norm_ptr,  # [D] RMSNorm weight for keys
    Out_ptr,  # [B*T, D] output
    Alpha_ptr,  # [B*T, N] attention weights (saved for bwd)
    RSTD_ptr,  # [B*T, N] rstd per (token, block)
    n_blocks,  # N
    n_tokens,  # B*T
    D,  # hidden dim
    eps,
    BLOCK_D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Forward: one program per token position."""
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < D

    # Load shared vectors
    w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_norm = tl.load(W_norm_ptr + cols, mask=d_mask, other=0.0)

    # Pass 1: compute scores = dot(w_query, RMSNorm(v_i)) for each block
    # scores[i] is stored at register position i via tl.where (workaround for
    # Triton lacking true scalar indexing into a register-held vector; the
    # tl.where pattern compiles to a predicated move and keeps everything in
    # registers without spilling to global memory).
    scores = tl.zeros((MAX_BLOCKS,), dtype=tl.float32) + float("-inf")
    score_max = tl.full((), float("-inf"), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            # RMSNorm: k = v * rstd * w_norm
            ms = tl.sum(v * v, axis=0) / D
            rstd = tl.rsqrt(ms + eps)
            # Alpha/RSTD layout: [B*T, N] — contiguous along N for each token
            tl.store(RSTD_ptr + tok * n_blocks + i, rstd)

            k = (v * rstd).to(w_norm.dtype) * w_norm

            # score = dot(w_query, k)
            sc = tl.sum(w_query * k.to(tl.float32), axis=0)
            scores = tl.where(tl.arange(0, MAX_BLOCKS) == i, sc, scores)
            score_max = tl.maximum(score_max, sc)

    # Softmax over blocks
    exp_scores = tl.where(
        tl.arange(0, MAX_BLOCKS) < n_blocks,
        tl.exp(scores - score_max),
        0.0,
    )
    sum_exp = tl.sum(exp_scores, axis=0)
    alpha = exp_scores / sum_exp  # [MAX_BLOCKS]

    # Store alpha for backward — layout [B*T, N], contiguous along N
    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            a_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, alpha, 0.0))
            tl.store(Alpha_ptr + tok * n_blocks + i, a_i)

    # Pass 2: weighted sum h = sum(alpha_i * v_i)
    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, alpha, 0.0))
            h += a_i * v

    # tl.store handles implicit dtype conversion
    tl.store(Out_ptr + tok * D + cols, h, mask=d_mask)


# ============================================================================
# Backward Kernel
# ============================================================================


@triton.jit
def _attn_res_bwd_kernel(
    dOut_ptr,  # [B*T, D] upstream gradient
    V_ptr,  # [N, B*T, D]
    W_query_ptr,  # [D]
    W_norm_ptr,  # [D]
    Alpha_ptr,  # [B*T, N] saved from forward
    RSTD_ptr,  # [B*T, N] saved from forward
    dV_ptr,  # [N, B*T, D] output gradients
    dW_query_ptr,  # [D] atomic accumulate
    dW_norm_ptr,  # [D] atomic accumulate
    n_blocks,
    n_tokens,
    D,
    eps,
    BLOCK_D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Backward: one program per token."""
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < D

    dh = tl.load(dOut_ptr + tok * D + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_norm = tl.load(W_norm_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)

    # Load alpha for all blocks — layout [B*T, N], contiguous load
    d_alpha = tl.zeros((MAX_BLOCKS,), dtype=tl.float32)
    alpha = tl.zeros((MAX_BLOCKS,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.load(Alpha_ptr + tok * n_blocks + i)

            da_i = tl.sum(dh * v, axis=0)
            d_alpha = tl.where(tl.arange(0, MAX_BLOCKS) == i, da_i, d_alpha)
            alpha = tl.where(tl.arange(0, MAX_BLOCKS) == i, a_i, alpha)

    # Softmax backward: d_score_i = alpha_i * (d_alpha_i - sum_j(alpha_j * d_alpha_j))
    sum_a_da = tl.sum(alpha * d_alpha, axis=0)
    d_scores = alpha * (d_alpha - sum_a_da)

    # For each block: compute dV_i and accumulate dW_query, dW_norm
    dw_query_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dw_norm_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * D + tok * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, alpha, 0.0))
            ds_i = tl.sum(tl.where(tl.arange(0, MAX_BLOCKS) == i, d_scores, 0.0))
            rstd = tl.load(RSTD_ptr + tok * n_blocks + i)

            # dV_i from weighted sum: alpha_i * dh
            dv_from_sum = a_i * dh

            # dV_i from score path: d_score_i * d(score_i)/d(v_i)
            # score_i = dot(w_query, RMSNorm(v_i) * w_norm)
            # d(score_i)/d(v_i) = d(score_i)/d(k_i) * d(k_i)/d(v_i)
            # where k_i = RMSNorm(v_i) * w_norm
            # d(score_i)/d(k_i) = w_query
            # d(k_i)/d(v_i) = w_norm * d(RMSNorm)/d(v_i)

            # RMSNorm backward: d(v*rstd)/dv = rstd * (I - (1/D) * rstd^2 * v * v^T)
            v_norm = v * rstd
            dk = ds_i * w_query * w_norm  # [D]
            sum_dk_v = tl.sum(dk * v, axis=0)
            dv_from_score = rstd * dk - rstd * rstd * rstd * (sum_dk_v / D) * v

            dv_total = dv_from_sum + dv_from_score
            # tl.store handles implicit dtype conversion
            tl.store(dV_ptr + v_off + cols, dv_total, mask=d_mask)

            # dW_query += d_score_i * k_i
            k_i = v_norm * w_norm
            dw_query_acc += ds_i * k_i

            # dW_norm += d_score_i * w_query * v_norm (element-wise)
            dw_norm_acc += ds_i * w_query * v_norm

    tl.atomic_add(dW_query_ptr + cols, dw_query_acc, mask=d_mask)
    tl.atomic_add(dW_norm_ptr + cols, dw_norm_acc, mask=d_mask)


# ============================================================================
# Python wrappers
# ============================================================================


def _next_pow2(n):
    return triton.next_power_of_2(n)


def _get_max_blocks(n_blocks):
    """Round up to constexpr-friendly value."""
    for mb in [4, 8, 16, 32]:
        if n_blocks <= mb:
            return mb
    return 32


def attn_res_forward(blocks, w_query, w_norm, eps=1e-6):
    """
    Args:
        blocks: list of N tensors [B, T, D] or stacked [N, B, T, D]
        w_query: [D] learned pseudo-query
        w_norm: [D] RMSNorm weight for keys
    Returns:
        h: [B, T, D] weighted output
        V: [N, B*T, D] stacked (saved for bwd)
        alpha: [B*T, N] attention weights
        rstd: [B*T, N] per-token rstd
    """
    if isinstance(blocks, (list, tuple)):
        V = torch.stack(blocks)  # [N, B, T, D]
    else:
        V = blocks
    orig_shape = V.shape  # [N, B, T, D] or [N, B*T, D]
    N = V.shape[0]
    D = V.shape[-1]

    # Flatten to [N, B*T, D]
    V_3d = V.reshape(N, -1, D).contiguous()
    n_tokens = V_3d.shape[1]

    w_query = w_query.contiguous()
    w_norm = w_norm.contiguous()

    Out = torch.empty(n_tokens, D, device=V.device, dtype=V.dtype)
    # Layout [B*T, N] for coalesced access per token
    Alpha = torch.empty(n_tokens, N, device=V.device, dtype=torch.float32)
    RSTD = torch.empty(n_tokens, N, device=V.device, dtype=torch.float32)

    BLOCK_D = _next_pow2(D)
    MAX_BLOCKS = _get_max_blocks(N)
    nw = 4
    if BLOCK_D >= 2048:
        nw = 8
    if BLOCK_D >= 8192:
        nw = 16

    _attn_res_fwd_kernel[(n_tokens,)](
        V_3d,
        w_query,
        w_norm,
        Out,
        Alpha,
        RSTD,
        N,
        n_tokens,
        D,
        eps,
        BLOCK_D=BLOCK_D,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=nw,
    )

    # Reshape output to match input spatial dims
    out_shape = list(orig_shape[1:])  # [B, T, D] or [B*T, D]
    return Out.view(out_shape), V_3d, Alpha, RSTD


def attn_res_backward(dh, V_3d, w_query, w_norm, Alpha, RSTD, eps=1e-6):
    """
    Returns: dV [N, B*T, D], dW_query [D], dW_norm [D]
    """
    dh = dh.contiguous()
    N, n_tokens, D = V_3d.shape
    dh_2d = dh.reshape(n_tokens, D)

    dV = torch.empty_like(V_3d)
    dW_query = torch.zeros(D, dtype=torch.float32, device=dh.device)
    dW_norm = torch.zeros(D, dtype=torch.float32, device=dh.device)

    BLOCK_D = _next_pow2(D)
    MAX_BLOCKS = _get_max_blocks(N)
    nw = 4
    if BLOCK_D >= 2048:
        nw = 8
    if BLOCK_D >= 8192:
        nw = 16

    _attn_res_bwd_kernel[(n_tokens,)](
        dh_2d,
        V_3d,
        w_query,
        w_norm,
        Alpha,
        RSTD,
        dV,
        dW_query,
        dW_norm,
        N,
        n_tokens,
        D,
        eps,
        BLOCK_D=BLOCK_D,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=nw,
    )

    return dV, dW_query.to(w_query.dtype), dW_norm.to(w_norm.dtype)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================


class LigerAttnResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(V_stacked, w_query, w_norm, eps):
        h, V_3d, Alpha, RSTD = attn_res_forward(V_stacked, w_query, w_norm, eps)
        return h, V_3d, Alpha, RSTD

    @staticmethod
    def setup_context(ctx, inputs, output):
        V_stacked, w_query, w_norm, eps = inputs
        h, V_3d, Alpha, RSTD = output
        ctx.save_for_backward(V_3d, w_query, w_norm, Alpha, RSTD)
        ctx.eps = eps
        ctx.orig_shape = V_stacked.shape  # [N, B, T, D] or [N, B*T, D]

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dh, _grad_V_3d, _grad_Alpha, _grad_RSTD):
        V_3d, w_query, w_norm, Alpha, RSTD = ctx.saved_tensors
        dV, dW_query, dW_norm = attn_res_backward(dh, V_3d, w_query, w_norm, Alpha, RSTD, ctx.eps)
        # Reshape dV back to original input shape
        dV = dV.view(ctx.orig_shape)
        return dV, dW_query, dW_norm, None
