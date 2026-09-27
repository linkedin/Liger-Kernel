"""
Native Sparse Attention (arXiv 2502.11089) — compressed (coarse) branch, Triton.

Each query attends over the *compressed* key/value stream produced by the learnable
compression MLP phi (paper Eq. 7-8): ``k_cmp/v_cmp`` have one vector per length-``l``
strided block, so this axis is ``Sc = floor((S - l) / d) + 1`` long — far shorter
than ``S``. A compressed block ``j`` spans original positions
``[j*stride, j*stride + block_size - 1]`` and is visible to query ``t`` only once it
lies fully in the past (``j*stride + block_size - 1 <= t``).

FlashAttention-2 online-softmax forward + recompute-from-LSE backward, mirroring
``nsa_sliding_attention.py``. The compression MLP itself stays in PyTorch (small,
not the bottleneck); only this attention is kernelised. The selection-scoring
matrix ``p_cmp`` is recomputed in torch from the (cheap, ``[S, Sc]``) score matmul.

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


@triton.jit
def _compressed_fwd_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    seq_len,
    num_cmp,  # Sc: number of compressed blocks
    block_size,
    stride,
    qk_scale,  # scale * log2(e)
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)

    row = pid_bh * seq_len * HEAD_DIM
    cmp_row = pid_bh * num_cmp * HEAD_DIM
    lse_row = pid_bh * seq_len

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

    # Only compressed blocks whose last covered position <= max query in the tile
    # can be visible; block j needs j*stride + block_size - 1 <= offs_m.
    m_last = pid_m * BLOCK_M + BLOCK_M - 1
    hi = (m_last - (block_size - 1)) // stride + 1  # exclusive upper compressed-block bound
    hi = tl.minimum(tl.maximum(hi, 0), num_cmp)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < num_cmp) & d_mask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale

        key_last = offs_n[None, :] * stride + (block_size - 1)
        valid = (key_last <= offs_m[:, None]) & (offs_n[None, :] < num_cmp)
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Rows with no visible block in this chunk keep m_new = -inf; a safe 0 in the
        # exponentials makes alpha=p=0 (no-op) and avoids NaN (see sliding kernel).
        m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = exp2(m_i - m_new_safe)
        p = exp2(qk - m_new_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = V + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < num_cmp) & d_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = m_i + log2(l_safe)

    o_ptrs = Out + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len) & d_mask[None, :])
    tl.store(LSE + lse_row + offs_m, lse, mask=offs_m < seq_len)


@triton.jit
def _compressed_bwd_preprocess(
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
def _compressed_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    seq_len,
    num_cmp,
    block_size,
    stride,
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
    cmp_row = pid_bh * num_cmp * HEAD_DIM
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

    m_last = pid_m * BLOCK_M + BLOCK_M - 1
    hi = (m_last - (block_size - 1)) // stride + 1
    hi = tl.minimum(tl.maximum(hi, 0), num_cmp)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = (offs_n[:, None] < num_cmp) & d_mask[None, :]
        k = tl.load(K + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)
        v = tl.load(V + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        key_last = offs_n[None, :] * stride + (block_size - 1)
        valid = (key_last <= offs_m[:, None]) & (offs_n[None, :] < num_cmp)
        p = tl.where(valid, exp2(qk - lse[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")

    dq = dq * scale
    tl.store(DQ + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=m_mask)


@triton.jit
def _compressed_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    seq_len,
    num_cmp,
    block_size,
    stride,
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
    cmp_row = pid_bh * num_cmp * HEAD_DIM
    lse_row = pid_bh * seq_len

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # compressed-key rows owned here
    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < HEAD_DIM
    n_mask = (offs_n[:, None] < num_cmp) & d_mask[None, :]

    k = tl.load(K + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)
    v = tl.load(V + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], tl.float32)

    # Earliest query that can see the first key in this tile: n_start*stride + block_size - 1.
    n_start = pid_n * BLOCK_N
    first_q = n_start * stride + (block_size - 1)
    lo = (first_q // BLOCK_M) * BLOCK_M

    for start_m in range(lo, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = (offs_m[:, None] < seq_len) & d_mask[None, :]
        q = tl.load(Q + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
        do = tl.load(DO + row + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0)
        lse = tl.load(LSE + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)
        delta = tl.load(DELTA + lse_row + offs_m, mask=offs_m < seq_len, other=0.0)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        key_last = offs_n[None, :] * stride + (block_size - 1)
        valid = (key_last <= offs_m[:, None]) & (offs_n[None, :] < num_cmp) & (offs_m[:, None] < seq_len)
        p = tl.where(valid, exp2(qk - lse[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])

        dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")

    dk = dk * scale
    tl.store(DK + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=n_mask)
    tl.store(DV + cmp_row + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=n_mask)


def _shape_bh(x):
    b, h, s, d = x.shape
    return x.reshape(b * h, s, d), (b, h, s, d)


def compressed_attention_forward(q, k_cmp, v_cmp, block_size, stride, scale):
    q2, shape = _shape_bh(q)
    k2, _ = _shape_bh(k_cmp)
    v2, _ = _shape_bh(v_cmp)
    bh, seq_len, head_dim = q2.shape
    num_cmp = k2.shape[1]
    block_dmodel = triton.next_power_of_2(head_dim)
    BLOCK_M = 64
    BLOCK_N = 32

    o = torch.empty_like(q2)
    lse = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
    grid = (bh, triton.cdiv(seq_len, BLOCK_M))
    _compressed_fwd_kernel[grid](
        q2,
        k2,
        v2,
        o,
        lse,
        seq_len,
        num_cmp,
        block_size,
        stride,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    return o.view(shape), lse


def compressed_attention_backward(do, q, k_cmp, v_cmp, o, lse, block_size, stride, scale):
    q2, shape = _shape_bh(q)
    k2, cmp_shape = _shape_bh(k_cmp)
    v2, _ = _shape_bh(v_cmp)
    o2, _ = _shape_bh(o)
    do2, _ = _shape_bh(do)
    bh, seq_len, head_dim = q2.shape
    num_cmp = k2.shape[1]
    block_dmodel = triton.next_power_of_2(head_dim)
    BLOCK_M = 64
    BLOCK_N = 32

    delta = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32)
    _compressed_bwd_preprocess[(bh, triton.cdiv(seq_len, BLOCK_M))](
        o2, do2, delta, seq_len, HEAD_DIM=head_dim, BLOCK_DMODEL=block_dmodel, BLOCK_M=BLOCK_M, num_warps=4
    )

    dq = torch.empty_like(q2)
    dk = torch.empty_like(k2)
    dv = torch.empty_like(v2)

    _compressed_bwd_dq_kernel[(bh, triton.cdiv(seq_len, BLOCK_M))](
        q2,
        k2,
        v2,
        do2,
        lse,
        delta,
        dq,
        seq_len,
        num_cmp,
        block_size,
        stride,
        scale,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    _compressed_bwd_dkdv_kernel[(bh, triton.cdiv(num_cmp, BLOCK_N))](
        q2,
        k2,
        v2,
        do2,
        lse,
        delta,
        dk,
        dv,
        seq_len,
        num_cmp,
        block_size,
        stride,
        scale,
        scale * LOG2E,
        HEAD_DIM=head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_num_warps(block_dmodel),
        num_stages=2,
    )
    return dq.view(shape), dk.view(cmp_shape), dv.view(cmp_shape)


class LigerNSACompressedAttentionFunction(torch.autograd.Function):
    """Compressed (coarse) attention (MHA; expand compressed KV for GQA before calling)."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, q, k_cmp, v_cmp, block_size, stride, scale):
        o, lse = compressed_attention_forward(q, k_cmp, v_cmp, block_size, stride, scale)
        ctx.save_for_backward(q, k_cmp, v_cmp, o, lse)
        ctx.block_size = block_size
        ctx.stride = stride
        ctx.scale = scale
        return o

    @staticmethod
    @ensure_contiguous
    def backward(ctx, do):
        q, k_cmp, v_cmp, o, lse = ctx.saved_tensors
        dq, dk, dv = compressed_attention_backward(do, q, k_cmp, v_cmp, o, lse, ctx.block_size, ctx.stride, ctx.scale)
        return dq, dk, dv, None, None, None


def nsa_compressed_attention(q, k_cmp, v_cmp, block_size, stride, scale):
    """Compressed branch with GQA support.

    q: ``[B, Hq, S, D]``; k_cmp, v_cmp: ``[B, Hkv, Sc, D]`` (Hkv divides Hq). Returns
    ``[B, Hq, S, D]``. Compressed KV heads are expanded to query heads (autograd
    reduces the duplicated grads). Returns only the branch output; the selection
    score matrix ``p_cmp`` is recomputed in torch from the cheap ``[S, Sc]`` matmul.
    """
    num_q_heads, num_kv_heads = q.shape[1], k_cmp.shape[1]
    if num_kv_heads != num_q_heads:
        group = num_q_heads // num_kv_heads
        k_cmp = k_cmp.repeat_interleave(group, dim=1)
        v_cmp = v_cmp.repeat_interleave(group, dim=1)
    return LigerNSACompressedAttentionFunction.apply(q, k_cmp, v_cmp, block_size, stride, scale)
