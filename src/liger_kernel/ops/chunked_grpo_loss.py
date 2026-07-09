"""Chunked selective log-softmax through the lm_head for GRPO-style losses.

Computes per-token log-probabilities (and logsumexp) directly from hidden
states and the lm_head weight without ever materializing the (N, V) logits
tensor, giving the memory profile of the chunked fused-linear GRPO loss with
Triton-kernel speed:

  forward:  one fused kernel streams vocab tiles per 128-token program,
            computing each logits tile with tl.dot GEMMs into a fresh
            accumulator and maintaining an online logsumexp (flash-attention
            style). Only per-token logp / lse (fp32) are written to HBM.
  backward: a fused kernel recomputes logits tiles and emits grad_logits for
            one sequence chunk into a reusable buffer in the input dtype; the
            two large grad GEMMs (grad_hidden, grad_weight) run in cuBLAS with
            an fp32 cross-chunk grad_weight accumulator. No atomics.

sm_103 (B300) notes:
  - triton-lang/triton#10821: two tl.dot calls chained through one accumulator
    in a K-loop miscompile on tcgen05 tiles (BLOCK_M >= 64). Every K-loop here
    issues exactly ONE tl.dot per accumulator per iteration and each vocab tile
    gets a fresh accumulator, so the buggy pattern never occurs.
  - Launch configs are fixed (no autotune) and there are no atomics, so results
    are bitwise reproducible run-to-run and rank-to-rank.
"""

import torch
import triton
import triton.language as tl

# Tile sizes. BLOCK_M >= 64 selects the tcgen05 MMA path on sm_103, which is
# safe given the single-dot-per-accumulator structure (see module docstring).
# Chosen by a manual sweep on B300 at V=248320, H=2048, 65k tokens: 249 ms
# fwd+bwd vs 377 ms for (64, 128, 128); larger BN/BK exhaust shared memory at
# num_stages=3, and num_warps=16 regresses ~2x.
_BM = 128
_BN = 256
_BK = 64
_NUM_WARPS = 8
_NUM_STAGES = 3
# fp32 inputs double the tile footprint; 3 stages exceeds sm_103's 228 KB SMEM
# (needs ~288 KB), so drop to 2 pipeline stages for fp32.
_NUM_STAGES_FP32 = 2


def _num_stages(dtype: torch.dtype) -> int:
    return _NUM_STAGES_FP32 if dtype == torch.float32 else _NUM_STAGES
# Sequence-chunk size for the backward grad_logits buffer
# (4096 x 248320 bf16 ~= 1.9 GiB).
_BWD_SEQ_CHUNK = 4096


@triton.jit
def _chunked_selective_logp_fwd_kernel(
    HIDDEN,
    W,
    TARGETS,
    LOGP,
    LSE,
    N,
    V,
    stride_hn,
    stride_wv,
    inv_temp,
    H: tl.constexpr,
    V_CEIL: tl.constexpr,
    EVEN_V: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    row_mask = rows < N
    tgt = tl.load(TARGETS + rows, mask=row_mask, other=-1)

    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    t_logit = tl.zeros((BM,), tl.float32)

    for v0 in tl.range(0, V_CEIL, BN):
        cols = v0 + tl.arange(0, BN)
        col_mask = cols < V
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in tl.range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            a = tl.load(
                HIDDEN + rows[:, None] * stride_hn + k[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            if EVEN_V:
                b = tl.load(W + cols[:, None] * stride_wv + k[None, :])
            else:
                b = tl.load(
                    W + cols[:, None] * stride_wv + k[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
            acc = tl.dot(a, tl.trans(b), acc)  # single dot per acc per iteration
        logits = acc * inv_temp
        if not EVEN_V:
            logits = tl.where(col_mask[None, :], logits, float("-inf"))
        tile_max = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(logits - m_new[:, None]), axis=1)
        m_i = m_new
        is_tgt = tgt[:, None] == cols[None, :]
        t_logit += tl.sum(tl.where(is_tgt, logits, 0.0), axis=1)

    lse = m_i + tl.log(l_i)
    tl.store(LOGP + rows, t_logit - lse, mask=row_mask)
    tl.store(LSE + rows, lse, mask=row_mask)


@triton.jit
def _chunked_grad_logits_kernel(
    HIDDEN,
    W,
    TARGETS,
    LSE,
    GRAD_LOGP,
    GRAD_LOGITS,
    N,
    V,
    row0,
    stride_hn,
    stride_wv,
    stride_gl,
    inv_temp,
    H: tl.constexpr,
    EVEN_V: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = row0 + pid_m * BM + tl.arange(0, BM)
    row_mask = rows < N
    cols = pid_n * BN + tl.arange(0, BN)
    col_mask = cols < V

    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(
            HIDDEN + rows[:, None] * stride_hn + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        if EVEN_V:
            b = tl.load(W + cols[:, None] * stride_wv + k[None, :])
        else:
            b = tl.load(
                W + cols[:, None] * stride_wv + k[None, :],
                mask=col_mask[:, None],
                other=0.0,
            )
        acc = tl.dot(a, tl.trans(b), acc)  # single dot per acc per iteration
    logits = acc * inv_temp

    lse = tl.load(LSE + rows, mask=row_mask, other=0.0)
    p = tl.exp(logits - lse[:, None])
    if not EVEN_V:
        p = tl.where(col_mask[None, :], p, 0.0)
    tgt = tl.load(TARGETS + rows, mask=row_mask, other=-1)
    g = tl.load(GRAD_LOGP + rows, mask=row_mask, other=0.0)
    is_tgt = tgt[:, None] == cols[None, :]
    # d logp_target / d raw_logit_v = (delta_{v==target} - softmax_v) / T
    gl = (tl.where(is_tgt, 1.0, 0.0) - p) * (g * inv_temp)[:, None]

    lrow = pid_m * BM + tl.arange(0, BM)
    store_mask = row_mask[:, None] if EVEN_V else (row_mask[:, None] & col_mask[None, :])
    tl.store(
        GRAD_LOGITS + lrow[:, None] * stride_gl + cols[None, :],
        gl.to(GRAD_LOGITS.dtype.element_ty),
        mask=store_mask,
    )


class ChunkedSelectiveLogPFunction(torch.autograd.Function):
    """Per-token selective log-softmax through the lm_head, chunked over vocab.

    forward(hidden (N, H), weight (V, H), targets (N,)) -> logp (N,) fp32
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature):
        assert hidden.is_contiguous(), "hidden must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"
        n_tokens, h = hidden.shape
        v = weight.shape[0]
        assert h % _BK == 0, f"hidden size {h} must be divisible by {_BK}"
        even_v = v % _BN == 0
        v_ceil = triton.cdiv(v, _BN) * _BN

        logp = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
        lse = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
        grid = (triton.cdiv(n_tokens, _BM),)
        _chunked_selective_logp_fwd_kernel[grid](
            hidden,
            weight,
            targets,
            logp,
            lse,
            n_tokens,
            v,
            hidden.stride(0),
            weight.stride(0),
            1.0 / temperature,
            H=h,
            V_CEIL=v_ceil,
            EVEN_V=even_v,
            BM=_BM,
            BN=_BN,
            BK=_BK,
            num_warps=_NUM_WARPS,
            num_stages=_num_stages(hidden.dtype),
        )
        ctx.save_for_backward(hidden, weight, targets, lse)
        ctx.temperature = temperature
        return logp

    @staticmethod
    def backward(ctx, grad_logp):
        hidden, weight, targets, lse = ctx.saved_tensors
        n_tokens, h = hidden.shape
        v = weight.shape[0]
        even_v = v % _BN == 0
        grad_logp = grad_logp.contiguous()

        grad_hidden = torch.empty_like(hidden)
        # fp32 cross-chunk accumulator; the chunk GEMMs run in the input dtype.
        grad_weight = torch.zeros(v, h, dtype=torch.float32, device=weight.device)
        chunk = min(_BWD_SEQ_CHUNK, n_tokens)
        buf = torch.empty(chunk, v, dtype=hidden.dtype, device=hidden.device)

        for row0 in range(0, n_tokens, chunk):
            n = min(chunk, n_tokens - row0)
            grid = (triton.cdiv(n, _BM), triton.cdiv(v, _BN))
            _chunked_grad_logits_kernel[grid](
                hidden,
                weight,
                targets,
                lse,
                grad_logp,
                buf,
                n_tokens,
                v,
                row0,
                hidden.stride(0),
                weight.stride(0),
                buf.stride(0),
                1.0 / ctx.temperature,
                H=h,
                EVEN_V=even_v,
                BM=_BM,
                BN=_BN,
                BK=_BK,
                num_warps=_NUM_WARPS,
                num_stages=_num_stages(hidden.dtype),
            )
            gl = buf[:n]
            torch.matmul(gl, weight, out=grad_hidden[row0 : row0 + n])
            grad_weight.add_(gl.t() @ hidden[row0 : row0 + n])

        return grad_hidden, grad_weight.to(weight.dtype), None, None


@torch.no_grad()
def chunked_selective_log_softmax_with_lse(hidden, weight, targets, temperature=1.0):
    """No-grad variant returning (logp, lse), both (N,) fp32. For testing/inspection."""
    assert hidden.is_contiguous() and weight.is_contiguous()
    n_tokens, h = hidden.shape
    v = weight.shape[0]
    assert h % _BK == 0, f"hidden size {h} must be divisible by {_BK}"
    even_v = v % _BN == 0
    v_ceil = triton.cdiv(v, _BN) * _BN
    logp = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
    lse = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
    grid = (triton.cdiv(n_tokens, _BM),)
    _chunked_selective_logp_fwd_kernel[grid](
        hidden,
        weight,
        targets,
        logp,
        lse,
        n_tokens,
        v,
        hidden.stride(0),
        weight.stride(0),
        1.0 / temperature,
        H=h,
        V_CEIL=v_ceil,
        EVEN_V=even_v,
        BM=_BM,
        BN=_BN,
        BK=_BK,
        num_warps=_NUM_WARPS,
        num_stages=_num_stages(hidden.dtype),
    )
    return logp, lse


def chunked_selective_log_softmax(hidden, weight, targets, temperature=1.0):
    """Differentiable selective log-softmax through the lm_head without logits.

    Args:
        hidden: (N, H) hidden states (bf16/fp16/fp32, contiguous).
        weight: (V, H) lm_head weight.
        targets: (N,) token ids to select.
        temperature: softmax temperature applied to logits.

    Returns:
        (N,) fp32 per-token log-probabilities of the targets.
    """
    return ChunkedSelectiveLogPFunction.apply(hidden, weight, targets, temperature)
