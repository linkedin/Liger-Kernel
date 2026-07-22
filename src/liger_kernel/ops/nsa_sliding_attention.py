"""
Native Sparse Attention (arXiv 2502.11089) — sliding-window branch, Triton.

Causal sliding-window attention: each query attends to the ``window_size`` most
recent keys (inclusive of itself). FlashAttention-2 style: online softmax in the
forward, recompute-from-LSE in the backward. The attended key range is a
contiguous band, so each dK/dV output tile is written by exactly one program —
no ``tl.atomic_add``, fully deterministic, and multi-backend safe.

Algorithm: FlashAttention-2 (arXiv 2307.08691). Kernel structure informed by the
public NSA references XunhaoLai/native-sparse-attention-triton (Apache-2.0) and
fla-org/native-sparse-attention (MIT); no code copied verbatim.
"""

import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import exp2
        from triton.language.extra.libdevice import log2
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import exp2
        from triton.language.extra.cuda.libdevice import log2
else:
    from triton.language.math import exp2
    from triton.language.math import log2

LOG2E: tl.constexpr = 1.4426950408889634


def _num_warps(block: int) -> int:
    # Portable warp count: cap on HIP (warp=64) so block-threads stay <= 1024.
    if block >= 128:
        return 8 if not is_hip() else 4
    return 4


@triton.jit
def _sliding_fwd_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    seq_len,
    window,
    qk_scale,  # scale * log2(e)
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)

    row = pid_bh * seq_len * HEAD_DIM
    lse_row = pid_bh * seq_len

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM

    q_ptrs = Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    m_start = pid_m * BLOCK_M
    lo = m_start - window + 1
    lo = tl.maximum(lo, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(m_start + BLOCK_M, seq_len)

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & d_mask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale  # [BLOCK_M, BLOCK_N] fp32

        valid = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] > offs_m[:, None] - window)
        valid = valid & (offs_n[None, :] < seq_len)
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Rows whose entire key block is masked keep m_new = -inf; using a safe 0
        # in the exponentials makes alpha=p=0 for them (no-op) and avoids NaN. A later
        # in-band block (every real query has its diagonal in-band) resets via alpha.
        m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = exp2(m_i - m_new_safe)
        p = exp2(qk - m_new_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = V + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & d_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = m_i + log2(l_safe)

    o_ptrs = Out + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len) & d_mask[None, :])
    tl.store(LSE + lse_row + offs_m, lse, mask=offs_m < seq_len)


@triton.jit
def _sliding_bwd_preprocess(
    Out,
    DO,
    DELTA,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    row = pid_bh * seq_len * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM
    mask = (offs_m[:, None] < seq_len) & d_mask[None, :]
    o = tl.load(Out + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(DELTA + pid_bh * seq_len + offs_m, delta, mask=offs_m < seq_len)


@triton.jit
def _sliding_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    seq_len,
    window,
    scale,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    row = pid_bh * seq_len * HEAD_DIM
    lse_row = pid_bh * seq_len

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM
    m_mask = (offs_m[:, None] < seq_len) & d_mask[None, :]

    q = tl.load(Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
    do = tl.load(DO + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
    lse = tl.load(LSE + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)
    delta = tl.load(DELTA + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    m_start = pid_m * BLOCK_M
    lo = tl.maximum(m_start - window + 1, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(m_start + BLOCK_M, seq_len)

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = (offs_n[:, None] < seq_len) & d_mask[None, :]
        k = tl.load(K + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)
        v = tl.load(V + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        valid = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] > offs_m[:, None] - window)
        valid = valid & (offs_n[None, :] < seq_len)
        p = tl.where(valid, exp2(qk - lse[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")  # [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")

    dq = dq * scale
    tl.store(DQ + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=m_mask)


@triton.jit
def _sliding_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    seq_len,
    window,
    scale,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    row = pid_bh * seq_len * HEAD_DIM
    lse_row = pid_bh * seq_len

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # key rows owned by this program
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM
    n_mask = (offs_n[:, None] < seq_len) & d_mask[None, :]

    k = tl.load(K + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)
    v = tl.load(V + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)

    # Queries touching key p: t in [p, p+window). For this key block, t in [n_start, n_end-1+window).
    n_start = pid_n * BLOCK_N
    lo = (n_start // BLOCK_M) * BLOCK_M
    hi = tl.minimum(n_start + BLOCK_N - 1 + window, seq_len)

    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = (offs_m[:, None] < seq_len) & d_mask[None, :]
        q = tl.load(Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
        do = tl.load(DO + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
        lse = tl.load(LSE + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)
        delta = tl.load(DELTA + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale  # [BLOCK_M, BLOCK_N]
        valid = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] > offs_m[:, None] - window)
        valid = valid & (offs_n[None, :] < seq_len) & (offs_m[:, None] < seq_len)
        p = tl.where(valid, exp2(qk - lse[:, None]), 0.0)  # [BLOCK_M, BLOCK_N]

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")  # [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None])  # [BLOCK_M, BLOCK_N]

        dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")  # [BLOCK_N, D]
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")  # [BLOCK_N, D]

    dk = dk * scale
    tl.store(DK + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask)
    tl.store(DV + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask)


def _shape_bh(x):
    b, h, s, d = x.shape
    return x.reshape(b * h, s, d), (b, h, s, d)


def sliding_attention_forward(q, k, v, window, scale):
    q2, shape = _shape_bh(q)
    k2, _ = _shape_bh(k)
    v2, _ = _shape_bh(v)
    bh, seq_len, head_dim = q2.shape
    block_dmodel = triton.next_power_of_2(head_dim)
    BLOCK_M = 64
    BLOCK_N = 64
    o = torch.empty_like(q2)
    lse = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
    grid = (bh, triton.cdiv(seq_len, BLOCK_M))
    _sliding_fwd_kernel[grid](
        q2,
        k2,
        v2,
        o,
        lse,
        seq_len,
        window,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    return o.view(shape), lse


def sliding_attention_backward(do, q, k, v, o, lse, window, scale):
    q2, shape = _shape_bh(q)
    k2, _ = _shape_bh(k)
    v2, _ = _shape_bh(v)
    o2, _ = _shape_bh(o)
    do2, _ = _shape_bh(do)
    bh, seq_len, head_dim = q2.shape
    block_dmodel = triton.next_power_of_2(head_dim)
    BLOCK_M = 64
    BLOCK_N = 64

    delta = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
    _sliding_bwd_preprocess[(bh, triton.cdiv(seq_len, BLOCK_M))](
        o2, do2, delta, seq_len, HEAD_DIM=head_dim, BLOCK_DMODEL=block_dmodel, BLOCK_M=BLOCK_M, num_warps=4
    )

    dq = torch.empty_like(q2)
    dk = torch.empty_like(k2)
    dv = torch.empty_like(v2)

    _sliding_bwd_dq_kernel[(bh, triton.cdiv(seq_len, BLOCK_M))](
        q2,
        k2,
        v2,
        do2,
        lse,
        delta,
        dq,
        seq_len,
        window,
        scale,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    _sliding_bwd_dkdv_kernel[(bh, triton.cdiv(seq_len, BLOCK_N))](
        q2,
        k2,
        v2,
        do2,
        lse,
        delta,
        dk,
        dv,
        seq_len,
        window,
        scale,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    return dq.view(shape), dk.view(shape), dv.view(shape)


class LigerNSASlidingAttentionFunction(torch.autograd.Function):
    """Causal sliding-window attention (MHA; expand KV for GQA before calling)."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, q, k, v, window_size, scale):
        o, lse = sliding_attention_forward(q, k, v, window_size, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.window_size = window_size
        ctx.scale = scale
        return o

    @staticmethod
    @ensure_contiguous
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = sliding_attention_backward(do, q, k, v, o, lse, ctx.window_size, ctx.scale)
        return dq, dk, dv, None, None


def nsa_sliding_attention(q, k, v, window_size, scale):
    """Sliding-window branch with GQA support.

    q: [B, Hq, S, D]; k, v: [B, Hkv, S, D] (Hkv divides Hq). Returns [B, Hq, S, D].
    KV heads are expanded to query heads (autograd reduces the duplicated grads).
    """
    num_q_heads, num_kv_heads = q.shape[1], k.shape[1]
    if num_kv_heads != num_q_heads:
        group = num_q_heads // num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    return LigerNSASlidingAttentionFunction.apply(q, k, v, window_size, scale)
