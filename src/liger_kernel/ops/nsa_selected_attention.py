"""
Native Sparse Attention (arXiv 2502.11089) — selected block-sparse branch, Triton.

This is *the* NSA kernel: exact causal attention restricted to the top-n selection
blocks chosen per query (paper Eq. 11-12). The block choice is discrete and shared
by every query head in a GQA group, which is precisely what lets the backward stay
atomic-free — the selection is a fixed boolean ``block_mask`` (no gradient), so
dK/dV can be computed KV-block-parallel with a single deterministic write.

FlashAttention-2 (arXiv 2307.08691) online-softmax forward + recompute-from-LSE
backward, mirroring ``nsa_sliding_attention.py``. Each key chunk is loaded only
when at least one query in the tile selects its block, giving the O(S*n*l') work
that beats the dense O(S^2) baseline.

Kernel structure informed by the public NSA references
XunhaoLai/native-sparse-attention-triton (Apache-2.0) and
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


def _pick_block_n(selection_block_size: int) -> int:
    """Largest kernel key-block in {64,32,16} that divides the selection block.

    A key chunk of ``BLOCK_N`` positions must lie entirely inside one selection
    block so its selection bit is well defined; requiring ``BLOCK_N | l'`` (and
    ``BLOCK_N >= 16`` for a portable ``tl.dot``) guarantees that.
    """
    for cand in (64, 32, 16):
        if selection_block_size % cand == 0:
            return cand
    raise ValueError(
        f"selection_block_size ({selection_block_size}) must be a multiple of 16, 32, or 64 "
        "for the selected-attention kernel."
    )


@triton.jit
def _selected_fwd_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    BM,  # block_mask [BH, seq_len, num_blocks], int8 (1 = selected)
    seq_len,
    num_blocks,
    qk_scale,  # scale * log2(e)
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)

    row = pid_bh * seq_len * HEAD_DIM
    lse_row = pid_bh * seq_len
    bm_row = pid_bh * seq_len * num_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM

    q = tl.load(
        Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    hi = tl.minimum(pid_m * BLOCK_M + BLOCK_M, seq_len)  # causal: no key beyond last query in tile
    for start_n in range(0, hi, BLOCK_N):
        j = start_n // SEL_BLOCK  # the (single) selection block this chunk lies in
        sel = tl.load(BM + bm_row + offs_m * num_blocks + j, mask=offs_m < seq_len, other=0)
        # Skip the whole chunk unless some query in the tile selects this block.
        if tl.sum(sel.to(tl.int32)) > 0:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & d_mask[None, :], other=0.0)
            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale

            valid = (sel[:, None] != 0) & (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < seq_len)
            qk = tl.where(valid, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # Rows selecting no key in this chunk keep m_new = -inf; a safe 0 in the
            # exponentials makes alpha=p=0 (no-op) and avoids NaN (see sliding kernel).
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
def _selected_bwd_preprocess(
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
def _selected_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    BM,
    seq_len,
    num_blocks,
    scale,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    row = pid_bh * seq_len * HEAD_DIM
    lse_row = pid_bh * seq_len
    bm_row = pid_bh * seq_len * num_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM
    m_mask = (offs_m[:, None] < seq_len) & d_mask[None, :]

    q = tl.load(Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
    do = tl.load(DO + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
    lse = tl.load(LSE + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)
    delta = tl.load(DELTA + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    hi = tl.minimum(pid_m * BLOCK_M + BLOCK_M, seq_len)
    for start_n in range(0, hi, BLOCK_N):
        j = start_n // SEL_BLOCK
        sel = tl.load(BM + bm_row + offs_m * num_blocks + j, mask=offs_m < seq_len, other=0)
        if tl.sum(sel.to(tl.int32)) > 0:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = (offs_n[:, None] < seq_len) & d_mask[None, :]
            k = tl.load(K + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)
            v = tl.load(V + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
            valid = (sel[:, None] != 0) & (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < seq_len)
            p = tl.where(valid, exp2(qk - lse[:, None]), 0.0)

            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")

    dq = dq * scale
    tl.store(DQ + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=m_mask)


@triton.jit
def _selected_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    BM,
    seq_len,
    num_blocks,
    scale,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    row = pid_bh * seq_len * HEAD_DIM
    lse_row = pid_bh * seq_len
    bm_row = pid_bh * seq_len * num_blocks

    n_start = pid_n * BLOCK_N
    j = n_start // SEL_BLOCK  # selection block owned by this key tile (BLOCK_N | SEL_BLOCK)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM
    n_mask = (offs_n[:, None] < seq_len) & d_mask[None, :]

    k = tl.load(K + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)
    v = tl.load(V + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)

    lo = (n_start // BLOCK_M) * BLOCK_M  # causal: only queries at or after this key tile
    for start_m in range(lo, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        sel = tl.load(BM + bm_row + offs_m * num_blocks + j, mask=offs_m < seq_len, other=0)
        if tl.sum(sel.to(tl.int32)) > 0:
            m_mask = (offs_m[:, None] < seq_len) & d_mask[None, :]
            q = tl.load(Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
            do = tl.load(DO + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
            lse = tl.load(LSE + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)
            delta = tl.load(DELTA + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)

            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
            valid = (sel[:, None] != 0) & (offs_n[None, :] <= offs_m[:, None])
            valid = valid & (offs_n[None, :] < seq_len) & (offs_m[:, None] < seq_len)
            p = tl.where(valid, exp2(qk - lse[:, None]), 0.0)

            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])

            dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
            dk += tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")

    dk = dk * scale
    tl.store(DK + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask)
    tl.store(DV + row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask)


def _shape_bh(x):
    b, h, s, d = x.shape
    return x.reshape(b * h, s, d), (b, h, s, d)


def selected_attention_forward(q, k, v, block_mask, selection_block_size, scale):
    q2, shape = _shape_bh(q)
    k2, _ = _shape_bh(k)
    v2, _ = _shape_bh(v)
    b, h, seq_len, num_blocks = block_mask.shape
    bm2 = block_mask.reshape(b * h, seq_len, num_blocks)

    bh, _, head_dim = q2.shape
    block_dmodel = triton.next_power_of_2(head_dim)
    BLOCK_M = 64
    BLOCK_N = _pick_block_n(selection_block_size)

    o = torch.empty_like(q2)
    lse = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
    grid = (bh, triton.cdiv(seq_len, BLOCK_M))
    _selected_fwd_kernel[grid](
        q2,
        k2,
        v2,
        o,
        lse,
        bm2,
        seq_len,
        num_blocks,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SEL_BLOCK=selection_block_size,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    return o.view(shape), lse


def selected_attention_backward(do, q, k, v, o, lse, block_mask, selection_block_size, scale):
    q2, shape = _shape_bh(q)
    k2, _ = _shape_bh(k)
    v2, _ = _shape_bh(v)
    o2, _ = _shape_bh(o)
    do2, _ = _shape_bh(do)
    b, h, seq_len, num_blocks = block_mask.shape
    bm2 = block_mask.reshape(b * h, seq_len, num_blocks)

    bh, _, head_dim = q2.shape
    block_dmodel = triton.next_power_of_2(head_dim)
    BLOCK_M = 64
    BLOCK_N = _pick_block_n(selection_block_size)

    delta = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
    _selected_bwd_preprocess[(bh, triton.cdiv(seq_len, BLOCK_M))](
        o2, do2, delta, seq_len, HEAD_DIM=head_dim, BLOCK_DMODEL=block_dmodel, BLOCK_M=BLOCK_M, num_warps=4
    )

    dq = torch.empty_like(q2)
    dk = torch.empty_like(k2)
    dv = torch.empty_like(v2)

    _selected_bwd_dq_kernel[(bh, triton.cdiv(seq_len, BLOCK_M))](
        q2,
        k2,
        v2,
        do2,
        lse,
        delta,
        dq,
        bm2,
        seq_len,
        num_blocks,
        scale,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SEL_BLOCK=selection_block_size,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    _selected_bwd_dkdv_kernel[(bh, triton.cdiv(seq_len, BLOCK_N))](
        q2,
        k2,
        v2,
        do2,
        lse,
        delta,
        dk,
        dv,
        bm2,
        seq_len,
        num_blocks,
        scale,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SEL_BLOCK=selection_block_size,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    return dq.view(shape), dk.view(shape), dv.view(shape)


class LigerNSASelectedAttentionFunction(torch.autograd.Function):
    """Selected block-sparse attention (MHA; expand KV + mask for GQA before calling)."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, q, k, v, block_mask, selection_block_size, scale):
        o, lse = selected_attention_forward(q, k, v, block_mask, selection_block_size, scale)
        ctx.save_for_backward(q, k, v, o, lse, block_mask)
        ctx.selection_block_size = selection_block_size
        ctx.scale = scale
        return o

    @staticmethod
    @ensure_contiguous
    def backward(ctx, do):
        q, k, v, o, lse, block_mask = ctx.saved_tensors
        dq, dk, dv = selected_attention_backward(do, q, k, v, o, lse, block_mask, ctx.selection_block_size, ctx.scale)
        return dq, dk, dv, None, None, None


def nsa_selected_attention(q, k, v, selected, selection_block_size, scale):
    """Selected block-sparse branch with GQA support.

    q: ``[B, Hq, S, D]``; k, v: ``[B, Hkv, S, D]`` (Hkv divides Hq); ``selected``:
    ``[B, Hkv, S, num_blocks]`` boolean block mask (already causal & forced-block
    aware, from :func:`select_blocks`). Returns ``[B, Hq, S, D]``.

    KV heads and the group-shared block mask are expanded to query heads so the
    kernel runs one head per program (autograd reduces the duplicated grads). The
    group-centric KV-reuse optimisation is a documented fast-follow.
    """
    num_q_heads, num_kv_heads = q.shape[1], k.shape[1]
    if num_kv_heads != num_q_heads:
        group = num_q_heads // num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        selected = selected.repeat_interleave(group, dim=1)
    block_mask = selected.to(torch.int8).contiguous()
    return LigerNSASelectedAttentionFunction.apply(q, k, v, block_mask, selection_block_size, scale)
