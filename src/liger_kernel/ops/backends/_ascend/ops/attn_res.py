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
from liger_kernel.ops.utils import get_npu_core_count

# ============================================================================
# Forward Kernel
# ============================================================================


@triton.jit
def _attn_res_fwd_kernel(
    V_ptr,              # [N, B*T, D]
    W_query_ptr,        # [D]
    W_norm_ptr,         # [D]
    Out_ptr,            # [B*T, D]
    Alpha_ptr,          # [B*T, N]
    RSTD_ptr,           # [B*T, N]
    n_tokens,
    D,
    eps,
    n_blocks: tl.constexpr,  
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_idx in range(pid, n_tokens, num_programs):
        cols = tl.arange(0, BLOCK_D)
        d_mask = cols < D

        # Load shared vectors
        w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
        w_norm  = tl.load(W_norm_ptr  + cols, mask=d_mask, other=0.0)

        # -----------------------------
        # Pass 1: compute scores
        # -----------------------------
        scores = tl.zeros((n_blocks,), dtype=tl.float32)
        score_max = tl.full((), float("-inf"), dtype=tl.float32)

        for i in tl.static_range(0, n_blocks):
            v_off = i * n_tokens * D + row_idx * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            # RMSNorm
            ms = tl.sum(v * v, axis=0) / D
            rstd = tl.rsqrt(ms + eps)
            tl.store(RSTD_ptr + row_idx * n_blocks + i, rstd)

            k = (v * rstd).to(w_norm.dtype) * w_norm

            sc = tl.sum(w_query * k.to(tl.float32), axis=0)

            scores = tl.where(tl.arange(0, n_blocks) == i, sc, scores)
            score_max = tl.maximum(score_max, sc)

        # -----------------------------
        # Softmax
        # -----------------------------
        exp_scores = tl.exp(scores - score_max)
        sum_exp = tl.sum(exp_scores, axis=0)
        alpha = exp_scores / sum_exp   # [n_blocks]
        h = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Store alpha
        for i in tl.static_range(0, n_blocks):
            a_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, alpha, 0.0))
            tl.store(Alpha_ptr + row_idx * n_blocks + i, a_i)

            v_off = i * n_tokens * D + row_idx * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            # a_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, alpha, 0.0))
            h += a_i * v

        tl.store(Out_ptr + row_idx * D + cols, h, mask=d_mask)


# ============================================================================
# Backward Kernel
# ============================================================================


@triton.jit
def _attn_res_bwd_kernel(
    dOut_ptr,          # [B*T, D]
    V_ptr,             # [N, B*T, D]
    W_query_ptr,       # [D]
    W_norm_ptr,        # [D]
    Alpha_ptr,         # [B*T, N]
    RSTD_ptr,          # [B*T, N]
    dV_ptr,            # [N, B*T, D]
    dW_query_ptr,      # [D]
    dW_norm_ptr,       # [D]
    n_tokens,
    D,
    n_blocks: tl.constexpr, 
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_idx in range(pid, n_tokens, num_programs):
        cols = tl.arange(0, BLOCK_D)
        d_mask = cols < D

        # Load shared vectors
        dh = tl.load(dOut_ptr + row_idx * D + cols, mask=d_mask, other=0.0).to(tl.float32)
        w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
        w_norm  = tl.load(W_norm_ptr  + cols, mask=d_mask, other=0.0).to(tl.float32)

        # ---------------------------------
        # Load alpha and compute d_alpha
        # ---------------------------------
        alpha = tl.zeros((n_blocks,), dtype=tl.float32)
        d_alpha = tl.zeros((n_blocks,), dtype=tl.float32)

        for i in tl.static_range(0, n_blocks):
            v_off = i * n_tokens * D + row_idx * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            a_i = tl.load(Alpha_ptr + row_idx * n_blocks + i)
            da_i = tl.sum(dh * v, axis=0)

            alpha = tl.where(tl.arange(0, n_blocks) == i, a_i, alpha)
            d_alpha = tl.where(tl.arange(0, n_blocks) == i, da_i, d_alpha)

        # ---------------------------------
        # Softmax backward
        # ---------------------------------
        sum_a_da = tl.sum(alpha * d_alpha, axis=0)
        d_scores = alpha * (d_alpha - sum_a_da)

        # ---------------------------------
        # Accumulators
        # ---------------------------------
        dw_query_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dw_norm_acc  = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # ---------------------------------
        # Main loop
        # ---------------------------------
        for i in tl.static_range(0, n_blocks):
            v_off = i * n_tokens * D + row_idx * D
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            a_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, alpha, 0.0))
            ds_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, d_scores, 0.0))
            rstd = tl.load(RSTD_ptr + row_idx * n_blocks + i)

            # ---- dV (two paths) ----
            dv_from_sum = a_i * dh

            v_norm = v * rstd
            dk = ds_i * w_query * w_norm

            sum_dk_v = tl.sum(dk * v, axis=0)
            dv_from_score = rstd * dk - rstd * rstd * rstd * (sum_dk_v / D) * v

            dv_total = dv_from_sum + dv_from_score
            tl.store(dV_ptr + v_off + cols, dv_total, mask=d_mask)

            # ---- parameter grads ----
            k_i = v_norm * w_norm
            dw_query_acc += ds_i * k_i
            dw_norm_acc  += ds_i * w_query * v_norm

        # ---------------------------------
        # Atomics
        # ---------------------------------
        tl.atomic_add(dW_query_ptr + cols, dw_query_acc, mask=d_mask)
        tl.atomic_add(dW_norm_ptr  + cols, dw_norm_acc, mask=d_mask)


# ============================================================================
# Python wrappers
# ============================================================================


def _next_pow2(n):
    return triton.next_power_of_2(n)


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
    # import pdb; pdb.set_trace() 

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
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, n_tokens)
    # import pdb; pdb.set_trace()

    _attn_res_fwd_kernel[(grid_size,)](
        V_3d,
        w_query,
        w_norm,
        Out,
        Alpha,
        RSTD,
        n_tokens,
        D,
        eps,
        BLOCK_D=BLOCK_D,
        n_blocks=N,
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
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, n_tokens)

    _attn_res_bwd_kernel[(grid_size,)](
        dh_2d,
        V_3d,
        w_query,
        w_norm,
        Alpha,
        RSTD,
        dV,
        dW_query,
        dW_norm,
        n_tokens,
        D,
        BLOCK_D=BLOCK_D,
        n_blocks=N,
    )

    return dV, dW_query.to(w_query.dtype), dW_norm.to(w_norm.dtype)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================


class LigerAttnResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, V_stacked, w_query, w_norm, eps):
        ctx.orig_shape = V_stacked.shape  # [N, B, T, D] or [N, B*T, D]
        h, V_3d, Alpha, RSTD = attn_res_forward(V_stacked, w_query, w_norm, eps)
        ctx.save_for_backward(V_3d, w_query, w_norm, Alpha, RSTD)
        ctx.eps = eps
        return h

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dh):
        V_3d, w_query, w_norm, Alpha, RSTD = ctx.saved_tensors
        dV, dW_query, dW_norm = attn_res_backward(dh, V_3d, w_query, w_norm, Alpha, RSTD, ctx.eps)
        # Reshape dV back to original input shape
        dV = dV.view(ctx.orig_shape)
        return dV, dW_query, dW_norm, None
