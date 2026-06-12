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

Ascend/NPU optimizations:
1. Fused Triton kernel: RMSNorm + dot + softmax + weighted sum
2. Grid-stride launch sized to NPU core count
3. UB-aware tiling along D for large hidden dimensions
4. Feature dim padded to _FEAT_ALIGN for aligned vector loads
"""

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# Pad the last dim to a multiple of _FEAT_ALIGN for aligned vector UB loads.
# Kernels use logical D for math/masks and d_stride for storage row pitch.
_FEAT_ALIGN = 16


def _pad_features_aligned(V: torch.Tensor) -> torch.Tensor:
    d = V.shape[-1]
    pad = (-d) % _FEAT_ALIGN
    return torch.nn.functional.pad(V, (0, pad)) if pad else V


def _get_max_blocks(n_blocks: int) -> int:
    """Round block count up to a constexpr-friendly upper bound."""
    for mb in (4, 8, 16, 32):
        if n_blocks <= mb:
            return mb
    return 32


def _get_optimal_block_d(D: int, is_forward: bool) -> int:
    """Pick tile size along D to stay within UB limits."""
    if D <= 1024:
        return triton.next_power_of_2(D)

    memory_multiplier = 12.0 if is_forward else 16.0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.85,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((D,),),
        tiling_dims=(0,),
    )
    if tile_shapes:
        return max(512, tile_shapes[0][0])
    return 512


def _launch_grid(n_tokens: int) -> int:
    return min(get_npu_core_count(), n_tokens)


@triton.jit
def _attn_res_fwd_kernel(
    V_ptr,  # [N, B*T, D]
    W_query_ptr,  # [D]
    W_norm_ptr,  # [D]
    Out_ptr,  # [B*T, D]
    Alpha_ptr,  # [B*T, N]
    RSTD_ptr,  # [B*T, N]
    n_tokens,
    D,
    d_stride,
    eps,
    n_blocks: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward (small D): one row per program via grid-stride over tokens."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_idx in range(pid, n_tokens, num_programs):
        cols = tl.arange(0, BLOCK_D)
        d_mask = cols < D

        # Load shared vectors
        w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
        w_norm = tl.load(W_norm_ptr + cols, mask=d_mask, other=0.0)

        # Compute scores
        scores = tl.zeros((n_blocks,), dtype=tl.float32)
        score_max = tl.full((), float("-inf"), dtype=tl.float32)

        for i in tl.static_range(0, n_blocks):
            v_off = i * n_tokens * d_stride + row_idx * d_stride
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            # RMSNorm
            ms = tl.sum(v * v, axis=0) / D
            rstd = tl.rsqrt(ms + eps)
            tl.store(RSTD_ptr + row_idx * n_blocks + i, rstd)

            k = (v * rstd).to(w_norm.dtype) * w_norm

            sc = tl.sum(w_query * k.to(tl.float32), axis=0)

            scores = tl.where(tl.arange(0, n_blocks) == i, sc, scores)
            score_max = tl.maximum(score_max, sc)

        # Softmax
        exp_scores = tl.exp(scores - score_max)
        alpha = exp_scores / tl.sum(exp_scores, axis=0)
        h = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Store alpha
        for i in tl.static_range(0, n_blocks):
            a_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, alpha, 0.0))
            tl.store(Alpha_ptr + row_idx * n_blocks + i, a_i)

            v_off = i * n_tokens * d_stride + row_idx * d_stride
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)

            h += a_i * v

        tl.store(Out_ptr + row_idx * d_stride + cols, h, mask=d_mask)


@triton.jit
def _attn_res_fwd_kernel_tiled(
    V_ptr,  # [N, T, D]
    W_query_ptr,  # [D]
    W_norm_ptr,  # [D]
    Out_ptr,  # [T, D]
    Alpha_ptr,  # [T, N]
    RSTD_ptr,  # [T, N]
    n_tokens,
    D,
    d_stride,
    eps,
    n_blocks: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward (large D): tile along feature dim to fit in UB."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for tok in tl.range(pid, n_tokens, num_programs):
        # Compute scores + rstd
        scores = tl.zeros((n_blocks,), dtype=tl.float32)
        score_max = -float("inf")

        for i in tl.static_range(0, n_blocks):
            # RMSNorm
            sum_sq = 0.0
            for d in range(0, D, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                mask = cols < D

                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )

                sum_sq += tl.sum(v * v, axis=0)

            rstd = tl.rsqrt(sum_sq / D + eps)
            tl.store(RSTD_ptr + tok * n_blocks + i, rstd)

            # Compute score
            sc = 0.0
            for d in range(0, D, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                mask = cols < D

                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )

                wq = tl.load(W_query_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                wn = tl.load(W_norm_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                sc += tl.sum(wq * v * rstd * wn, axis=0)

            scores = tl.where(tl.arange(0, n_blocks) == i, sc, scores)
            score_max = tl.maximum(score_max, sc)

        # Softmax
        exp_scores = tl.exp(scores - score_max)
        alpha = exp_scores / tl.sum(exp_scores, axis=0)

        # store alpha
        for i in tl.static_range(0, n_blocks):
            a_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, alpha, 0.0))
            tl.store(Alpha_ptr + tok * n_blocks + i, a_i)

        # Output accumulation (tiled over D)
        for d in range(0, D, BLOCK_D):
            cols = d + tl.arange(0, BLOCK_D)
            mask = cols < D

            h = tl.zeros((BLOCK_D,), dtype=tl.float32)

            for i in tl.static_range(0, n_blocks):
                a_i = tl.sum(tl.where(tl.arange(0, n_blocks) == i, alpha, 0.0))

                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )

                h += a_i * v

            tl.store(Out_ptr + tok * d_stride + cols, h, mask=mask)


@triton.jit
def _attn_res_bwd_kernel(
    dOut_ptr,  # [B*T, D]
    V_ptr,  # [N, B*T, D]
    W_query_ptr,  # [D]
    W_norm_ptr,  # [D]
    Alpha_ptr,  # [B*T, N]
    RSTD_ptr,  # [B*T, N]
    dV_ptr,  # [N, B*T, D]
    dW_query_ptr,  # [D]
    dW_norm_ptr,  # [D]
    n_blocks,
    n_tokens,
    D,
    d_stride,
    BLOCK_D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Backward (small D): one program per token."""
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < D

    dh = tl.load(dOut_ptr + tok * d_stride + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_query = tl.load(W_query_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)
    w_norm = tl.load(W_norm_ptr + cols, mask=d_mask, other=0.0).to(tl.float32)

    # Softmax backward: d_score_i = alpha_i * (d_alpha_i - sum_j alpha_j * d_alpha_j)
    sum_a_da = 0.0
    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * d_stride + tok * d_stride
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.load(Alpha_ptr + tok * n_blocks + i)
            sum_a_da += a_i * tl.sum(dh * v, axis=0)

    dw_query_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dw_norm_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            v_off = i * n_tokens * d_stride + tok * d_stride
            v = tl.load(V_ptr + v_off + cols, mask=d_mask, other=0.0).to(tl.float32)
            a_i = tl.load(Alpha_ptr + tok * n_blocks + i)
            da_i = tl.sum(dh * v, axis=0)
            ds_i = a_i * (da_i - sum_a_da)
            rstd = tl.load(RSTD_ptr + tok * n_blocks + i)

            dv_from_sum = a_i * dh
            dk = ds_i * w_query * w_norm
            sum_dk_v = tl.sum(dk * v, axis=0)
            dv_from_score = rstd * dk - rstd * rstd * rstd * (sum_dk_v / D) * v
            tl.store(dV_ptr + v_off + cols, dv_from_sum + dv_from_score, mask=d_mask)

            v_norm = v * rstd
            dw_query_acc += ds_i * v_norm * w_norm
            dw_norm_acc += ds_i * w_query * v_norm

    tl.atomic_add(dW_query_ptr + cols, dw_query_acc, mask=d_mask)
    tl.atomic_add(dW_norm_ptr + cols, dw_norm_acc, mask=d_mask)


@triton.jit
def _attn_res_bwd_kernel_tiled(
    dOut_ptr,
    V_ptr,
    W_query_ptr,
    W_norm_ptr,
    Alpha_ptr,
    RSTD_ptr,
    dV_ptr,
    dW_query_ptr,
    dW_norm_ptr,
    n_blocks,
    n_tokens,
    D,
    d_stride,
    BLOCK_D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Backward (large D): tile along feature dim to fit in UB."""
    tok = tl.program_id(0)

    sum_a_da = 0.0
    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            a_i = tl.load(Alpha_ptr + tok * n_blocks + i)
            da_i = 0.0
            for d in range(0, D, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                mask = cols < D
                dh = tl.load(dOut_ptr + tok * d_stride + cols, mask=mask, other=0.0).to(tl.float32)
                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                da_i += tl.sum(dh * v, axis=0)
            sum_a_da += a_i * da_i

    for i in tl.static_range(0, MAX_BLOCKS):
        if i < n_blocks:
            a_i = tl.load(Alpha_ptr + tok * n_blocks + i)
            da_i = 0.0
            for d in range(0, D, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                mask = cols < D
                dh = tl.load(dOut_ptr + tok * d_stride + cols, mask=mask, other=0.0).to(tl.float32)
                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                da_i += tl.sum(dh * v, axis=0)

            ds_i = a_i * (da_i - sum_a_da)
            rstd = tl.load(RSTD_ptr + tok * n_blocks + i)

            sum_dk_v = 0.0
            for d in range(0, D, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                mask = cols < D
                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                wq = tl.load(W_query_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                wn = tl.load(W_norm_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                sum_dk_v += tl.sum(ds_i * wq * wn * v, axis=0)

            # Compute per-D outputs
            for d in range(0, D, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                mask = cols < D

                dh = tl.load(dOut_ptr + tok * d_stride + cols, mask=mask, other=0.0).to(tl.float32)
                v = tl.load(V_ptr + i * n_tokens * d_stride + tok * d_stride + cols, mask=mask, other=0.0).to(
                    tl.float32
                )

                wq = tl.load(W_query_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                wn = tl.load(W_norm_ptr + cols, mask=mask, other=0.0).to(tl.float32)

                dk = ds_i * wq * wn
                dv_from_score = rstd * dk - (rstd * rstd * rstd) * (sum_dk_v / D) * v
                tl.store(
                    dV_ptr + i * n_tokens * d_stride + tok * d_stride + cols,
                    a_i * dh + dv_from_score,
                    mask=mask,
                )

                v_norm = v * rstd
                tl.atomic_add(dW_query_ptr + cols, ds_i * v_norm * wn, mask=mask)
                tl.atomic_add(dW_norm_ptr + cols, ds_i * wq * v_norm, mask=mask)


def _stack_blocks(blocks):
    if isinstance(blocks, (list, tuple)):
        return torch.stack(blocks)
    return blocks


def _launch_fwd_kernel(V_3d, w_query, w_norm, Out, Alpha, RSTD, n_tokens, D, d_stride, eps, n_blocks, block_d):
    grid = (_launch_grid(n_tokens),)
    args = (V_3d, w_query, w_norm, Out, Alpha, RSTD, n_tokens, D, d_stride, eps)
    constexpr = dict(BLOCK_D=block_d, n_blocks=n_blocks)
    if block_d >= D:
        _attn_res_fwd_kernel[grid](*args, **constexpr)
    else:
        _attn_res_fwd_kernel_tiled[grid](*args, **constexpr)


def _launch_bwd_kernel(dh_2d, V_3d, w_query, w_norm, Alpha, RSTD, dV, dW_query, dW_norm, n_blocks, D, block_d):
    _, n_tokens, d_stride = V_3d.shape
    args = (dh_2d, V_3d, w_query, w_norm, Alpha, RSTD, dV, dW_query, dW_norm, n_blocks, n_tokens, D, d_stride)
    constexpr = dict(BLOCK_D=block_d, MAX_BLOCKS=_get_max_blocks(n_blocks))
    if block_d >= D:
        _attn_res_bwd_kernel[(n_tokens,)](*args, **constexpr)
    else:
        _attn_res_bwd_kernel_tiled[(n_tokens,)](*args, **constexpr)


def attn_res_forward(blocks, w_query, w_norm, eps=1e-6):
    """
    Args:
        blocks: list of N tensors [B, T, D] or stacked [N, B, T, D]
        w_query: [D] learned pseudo-query
        w_norm: [D] RMSNorm weight for keys
        eps: RMSNorm epsilon
    Returns:
        h: weighted output with same spatial shape as each block
        V_3d: [N, B*T, D] stacked values (saved for backward)
        alpha: [B*T, N] attention weights
        rstd: [B*T, N] per-token RMSNorm rstd
    """
    V = _stack_blocks(blocks)
    orig_shape = V.shape
    n_blocks, D = V.shape[0], V.shape[-1]

    V = _pad_features_aligned(V)
    V_3d = V.reshape(n_blocks, -1, V.shape[-1]).contiguous()
    d_stride = V_3d.shape[-1]
    n_tokens = V_3d.shape[1]

    w_query = w_query.contiguous()
    w_norm = w_norm.contiguous()

    Out = torch.empty(n_tokens, d_stride, device=V.device, dtype=V.dtype)
    Alpha = torch.empty(n_tokens, n_blocks, device=V.device, dtype=torch.float32)
    RSTD = torch.empty(n_tokens, n_blocks, device=V.device, dtype=torch.float32)

    block_d = _get_optimal_block_d(D, is_forward=True)
    _launch_fwd_kernel(V_3d, w_query, w_norm, Out, Alpha, RSTD, n_tokens, D, d_stride, eps, n_blocks, block_d)

    if d_stride != D:
        Out = Out[:, :D].contiguous()

    return Out.view(*orig_shape[1:]), V_3d, Alpha, RSTD


def attn_res_backward(dh, V_3d, w_query, w_norm, Alpha, RSTD):
    """
    Args:
        dh: upstream gradient matching forward output shape
        V_3d: [N, B*T, D] from forward
        w_query, w_norm: forward weights
        Alpha, RSTD: saved forward intermediates
    Returns:
        dV [N, B*T, D], dW_query [D], dW_norm [D]
    """
    dh = dh.contiguous()
    _, n_tokens, d_stride = V_3d.shape
    D = dh.shape[-1]
    dh_2d = dh.reshape(n_tokens, D)
    if d_stride != D:
        dh_2d = torch.nn.functional.pad(dh_2d, (0, d_stride - D))

    dV = torch.empty_like(V_3d)
    dW_query = torch.zeros(D, dtype=torch.float32, device=dh.device)
    dW_norm = torch.zeros(D, dtype=torch.float32, device=dh.device)

    block_d = _get_optimal_block_d(D, is_forward=False)
    _launch_bwd_kernel(dh_2d, V_3d, w_query, w_norm, Alpha, RSTD, dV, dW_query, dW_norm, V_3d.shape[0], D, block_d)

    if d_stride != D:
        dV = dV[:, :, :D].contiguous()

    return dV, dW_query.to(w_query.dtype), dW_norm.to(w_norm.dtype)


class LigerAttnResFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, V_stacked, w_query, w_norm, eps):
        ctx.orig_shape = V_stacked.shape  # [N, B, T, D] or [N, B*T, D]
        h, V_3d, Alpha, RSTD = attn_res_forward(V_stacked, w_query, w_norm, eps)
        ctx.save_for_backward(V_3d, w_query, w_norm, Alpha, RSTD)
        return h

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dh):
        V_3d, w_query, w_norm, Alpha, RSTD = ctx.saved_tensors
        dV, dW_query, dW_norm = attn_res_backward(dh, V_3d, w_query, w_norm, Alpha, RSTD)
        return dV.view(ctx.orig_shape), dW_query, dW_norm, None
