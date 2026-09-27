"""Fused QK-Norm + RoPE Triton kernel.

Several recent architectures (e.g. Qwen3) apply a per-head RMSNorm to the query
and key projections *before* rotary positional embedding:

    q = q_proj(x).view(B, T, n_qh, hd)
    q = q_norm(q).transpose(1, 2)          # RMSNorm over the head_dim
    k = k_proj(x).view(B, T, n_kh, hd)
    k = k_norm(k).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

Both RMSNorm and RoPE are memory-bound elementwise/reduction ops that only touch
Q and K.  Running them as separate kernels round-trips the normalized Q/K through
HBM, and the ``.transpose(1, 2)`` in between forces the RoPE kernel to materialize
a ``.contiguous()`` copy.  This module fuses the whole ``RMSNorm -> RoPE`` chain
into a single Triton kernel that reads Q/K once and writes them once, absorbing
the transpose as a pure stride operation.

The RMSNorm follows the "llama" casting convention (reduction + rstd in fp32,
matching ``Qwen3RMSNorm``).  The RoPE follows the HuggingFace Llama/Qwen half-
rotation layout (first half / second half), identical to ``LigerRopeFunction``.
"""

import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _qk_norm_rope_forward_kernel(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    oq_ptr,
    oq_row_stride,
    ok_ptr,
    ok_row_stride,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    rstd_q_ptr,
    rstd_k_ptr,
    seq_len,
    eps,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    cos_bs: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd_half: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program instance == one token.

    q layout: (bsz, seq_len, n_qh, hd) contiguous, so the per-token row stride is
    ``n_qh * hd``.  We load the left / right halves of every head separately (as
    in the RoPE kernel) and reduce over the full head_dim for the RMSNorm.
    """
    pid = tl.program_id(0).to(tl.int64)

    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride
    oq_ptr = oq_ptr + pid * oq_row_stride
    ok_ptr = ok_ptr + pid * ok_row_stride

    # ---- locate cos/sin for this token (only the left half is needed) ----
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    cos_ptr = cos_ptr + tl.where(
        cos_bs == 1,
        seq_idx * cos_row_stride,
        batch_idx * (seq_len * cos_row_stride) + seq_idx * cos_row_stride,
    )
    sin_ptr = sin_ptr + tl.where(
        cos_bs == 1,
        seq_idx * sin_row_stride,
        batch_idx * (seq_len * sin_row_stride) + seq_idx * sin_row_stride,
    )
    half_cols = tl.arange(0, pad_hd_half)
    half_mask = half_cols < (hd // 2)
    cos_row = tl.load(cos_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    sin_row = tl.load(sin_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)

    # ---- RMSNorm weight (loaded once, broadcast over heads) ----
    wq1 = tl.load(wq_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    wq2 = tl.load(wq_ptr + (hd // 2) + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    wk1 = tl.load(wk_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    wk2 = tl.load(wk_ptr + (hd // 2) + half_cols, mask=half_mask, other=0.0).to(tl.float32)

    # ================= Q =================
    q_heads = tl.arange(0, pad_n_qh)
    q1_off = q_heads[:, None] * hd + half_cols[None, :]
    q2_off = q1_off + (hd // 2)
    q_mask = (q_heads[:, None] < n_qh) & half_mask[None, :]
    q1 = tl.load(q_ptr + q1_off, mask=q_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + q2_off, mask=q_mask, other=0.0).to(tl.float32)

    # RMSNorm over the full head_dim (llama casting: reduce in fp32)
    ms_q = (tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)) / hd
    rstd_q = rsqrt(ms_q + eps)  # (pad_n_qh,)
    tl.store(rstd_q_ptr + pid * n_qh + q_heads, rstd_q, mask=q_heads < n_qh)

    qn1 = q1 * rstd_q[:, None] * wq1[None, :]
    qn2 = q2 * rstd_q[:, None] * wq2[None, :]

    # RoPE: out1 = n1*cos - n2*sin ; out2 = n2*cos + n1*sin
    oq1 = qn1 * cos_row[None, :] - qn2 * sin_row[None, :]
    oq2 = qn2 * cos_row[None, :] + qn1 * sin_row[None, :]
    tl.store(oq_ptr + q1_off, oq1.to(OUT_DTYPE), mask=q_mask)
    tl.store(oq_ptr + q2_off, oq2.to(OUT_DTYPE), mask=q_mask)

    # ================= K =================
    k_heads = tl.arange(0, pad_n_kh)
    k1_off = k_heads[:, None] * hd + half_cols[None, :]
    k2_off = k1_off + (hd // 2)
    k_mask = (k_heads[:, None] < n_kh) & half_mask[None, :]
    k1 = tl.load(k_ptr + k1_off, mask=k_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptr + k2_off, mask=k_mask, other=0.0).to(tl.float32)

    ms_k = (tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) / hd
    rstd_k = rsqrt(ms_k + eps)
    tl.store(rstd_k_ptr + pid * n_kh + k_heads, rstd_k, mask=k_heads < n_kh)

    kn1 = k1 * rstd_k[:, None] * wk1[None, :]
    kn2 = k2 * rstd_k[:, None] * wk2[None, :]

    ok1 = kn1 * cos_row[None, :] - kn2 * sin_row[None, :]
    ok2 = kn2 * cos_row[None, :] + kn1 * sin_row[None, :]
    tl.store(ok_ptr + k1_off, ok1.to(OUT_DTYPE), mask=k_mask)
    tl.store(ok_ptr + k2_off, ok2.to(OUT_DTYPE), mask=k_mask)


@triton.jit
def _qk_norm_rope_backward_kernel(
    doq_ptr,
    doq_row_stride,
    dok_ptr,
    dok_row_stride,
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    dq_ptr,
    dq_row_stride,
    dk_ptr,
    dk_row_stride,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    rstd_q_ptr,
    rstd_k_ptr,
    dwq_ptr,
    dwk_ptr,
    dw_row_stride,
    n_rows,
    seq_len,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    cos_bs: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd_half: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Grid-strided over tokens; each program accumulates a partial dWq/dWk.

    ``dwq_ptr`` / ``dwk_ptr`` point to per-program partial buffers of shape
    ``(num_programs, hd)`` that the host reduces to the final weight gradients.
    """
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)

    half_cols = tl.arange(0, pad_hd_half)
    half_mask = half_cols < (hd // 2)

    wq1 = tl.load(wq_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    wq2 = tl.load(wq_ptr + (hd // 2) + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    wk1 = tl.load(wk_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    wk2 = tl.load(wk_ptr + (hd // 2) + half_cols, mask=half_mask, other=0.0).to(tl.float32)

    dwq1_acc = tl.zeros((pad_hd_half,), dtype=tl.float32)
    dwq2_acc = tl.zeros((pad_hd_half,), dtype=tl.float32)
    dwk1_acc = tl.zeros((pad_hd_half,), dtype=tl.float32)
    dwk2_acc = tl.zeros((pad_hd_half,), dtype=tl.float32)

    q_heads = tl.arange(0, pad_n_qh)
    k_heads = tl.arange(0, pad_n_kh)
    q1_off = q_heads[:, None] * hd + half_cols[None, :]
    q2_off = q1_off + (hd // 2)
    k1_off = k_heads[:, None] * hd + half_cols[None, :]
    k2_off = k1_off + (hd // 2)
    q_mask = (q_heads[:, None] < n_qh) & half_mask[None, :]
    k_mask = (k_heads[:, None] < n_kh) & half_mask[None, :]

    for token in range(pid, n_rows, num_programs):
        batch_idx = token // seq_len
        seq_idx = token % seq_len
        c_ptr = cos_ptr + tl.where(
            cos_bs == 1,
            seq_idx * cos_row_stride,
            batch_idx * (seq_len * cos_row_stride) + seq_idx * cos_row_stride,
        )
        s_ptr = sin_ptr + tl.where(
            cos_bs == 1,
            seq_idx * sin_row_stride,
            batch_idx * (seq_len * sin_row_stride) + seq_idx * sin_row_stride,
        )
        cos_row = tl.load(c_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)
        sin_row = tl.load(s_ptr + half_cols, mask=half_mask, other=0.0).to(tl.float32)

        # ---------------- Q ----------------
        doq1 = tl.load(doq_ptr + token * doq_row_stride + q1_off, mask=q_mask, other=0.0).to(tl.float32)
        doq2 = tl.load(doq_ptr + token * doq_row_stride + q2_off, mask=q_mask, other=0.0).to(tl.float32)
        # RoPE backward -> grad wrt normed value (dY)
        dqn1 = doq1 * cos_row[None, :] + doq2 * sin_row[None, :]
        dqn2 = doq2 * cos_row[None, :] - doq1 * sin_row[None, :]

        xq1 = tl.load(q_ptr + token * q_row_stride + q1_off, mask=q_mask, other=0.0).to(tl.float32)
        xq2 = tl.load(q_ptr + token * q_row_stride + q2_off, mask=q_mask, other=0.0).to(tl.float32)
        rstd_q = tl.load(rstd_q_ptr + token * n_qh + q_heads, mask=q_heads < n_qh, other=0.0)

        # RMSNorm backward (llama): dx = rstd*(m - (1/hd)*rstd^2*sum(m*x)*x), m = dY*w
        mq1 = dqn1 * wq1[None, :]
        mq2 = dqn2 * wq2[None, :]
        sum_mx_q = tl.sum(mq1 * xq1, axis=1) + tl.sum(mq2 * xq2, axis=1)
        coef_q = (1.0 / hd) * rstd_q * rstd_q * sum_mx_q  # (pad_n_qh,)
        dxq1 = rstd_q[:, None] * (mq1 - coef_q[:, None] * xq1)
        dxq2 = rstd_q[:, None] * (mq2 - coef_q[:, None] * xq2)
        tl.store(dq_ptr + token * dq_row_stride + q1_off, dxq1.to(OUT_DTYPE), mask=q_mask)
        tl.store(dq_ptr + token * dq_row_stride + q2_off, dxq2.to(OUT_DTYPE), mask=q_mask)

        # dW += dY * (x * rstd), reduce over heads (token accumulation happens in the loop)
        dwq1_acc += tl.sum(dqn1 * (xq1 * rstd_q[:, None]), axis=0)
        dwq2_acc += tl.sum(dqn2 * (xq2 * rstd_q[:, None]), axis=0)

        # ---------------- K ----------------
        dok1 = tl.load(dok_ptr + token * dok_row_stride + k1_off, mask=k_mask, other=0.0).to(tl.float32)
        dok2 = tl.load(dok_ptr + token * dok_row_stride + k2_off, mask=k_mask, other=0.0).to(tl.float32)
        dkn1 = dok1 * cos_row[None, :] + dok2 * sin_row[None, :]
        dkn2 = dok2 * cos_row[None, :] - dok1 * sin_row[None, :]

        xk1 = tl.load(k_ptr + token * k_row_stride + k1_off, mask=k_mask, other=0.0).to(tl.float32)
        xk2 = tl.load(k_ptr + token * k_row_stride + k2_off, mask=k_mask, other=0.0).to(tl.float32)
        rstd_k = tl.load(rstd_k_ptr + token * n_kh + k_heads, mask=k_heads < n_kh, other=0.0)

        mk1 = dkn1 * wk1[None, :]
        mk2 = dkn2 * wk2[None, :]
        sum_mx_k = tl.sum(mk1 * xk1, axis=1) + tl.sum(mk2 * xk2, axis=1)
        coef_k = (1.0 / hd) * rstd_k * rstd_k * sum_mx_k
        dxk1 = rstd_k[:, None] * (mk1 - coef_k[:, None] * xk1)
        dxk2 = rstd_k[:, None] * (mk2 - coef_k[:, None] * xk2)
        tl.store(dk_ptr + token * dk_row_stride + k1_off, dxk1.to(OUT_DTYPE), mask=k_mask)
        tl.store(dk_ptr + token * dk_row_stride + k2_off, dxk2.to(OUT_DTYPE), mask=k_mask)

        dwk1_acc += tl.sum(dkn1 * (xk1 * rstd_k[:, None]), axis=0)
        dwk2_acc += tl.sum(dkn2 * (xk2 * rstd_k[:, None]), axis=0)

    tl.store(dwq_ptr + pid * dw_row_stride + half_cols, dwq1_acc, mask=half_mask)
    tl.store(dwq_ptr + pid * dw_row_stride + (hd // 2) + half_cols, dwq2_acc, mask=half_mask)
    tl.store(dwk_ptr + pid * dw_row_stride + half_cols, dwk1_acc, mask=half_mask)
    tl.store(dwk_ptr + pid * dw_row_stride + (hd // 2) + half_cols, dwk2_acc, mask=half_mask)


def _num_warps(pad_n_qh, pad_n_kh, pad_hd_half):
    block = max(pad_n_qh, pad_n_kh) * pad_hd_half
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def qk_norm_rope_forward(q, k, q_weight, k_weight, cos, sin, eps):
    # q: (bsz, seq_len, n_qh, hd) ; k: (bsz, seq_len, n_kh, hd)  (pre-transpose layout)
    bsz, seq_len, n_qh, hd = q.shape
    n_kh = k.shape[2]
    assert hd % 2 == 0, "head_dim must be even for RoPE"

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    rstd_q = torch.empty((bsz * seq_len, n_qh), dtype=torch.float32, device=q.device)
    rstd_k = torch.empty((bsz * seq_len, n_kh), dtype=torch.float32, device=q.device)

    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd_half = triton.next_power_of_2(hd // 2)
    n_rows = bsz * seq_len

    out_dtype = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[q.dtype]

    _qk_norm_rope_forward_kernel[(n_rows,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        oq,
        oq.stride(1),
        ok,
        ok.stride(1),
        q_weight,
        k_weight,
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        rstd_q,
        rstd_k,
        seq_len,
        eps,
        n_qh,
        n_kh,
        hd,
        cos.shape[0],
        pad_n_qh,
        pad_n_kh,
        pad_hd_half,
        out_dtype,
        num_warps=_num_warps(pad_n_qh, pad_n_kh, pad_hd_half),
    )
    # absorb the transpose: return (bsz, n_head, seq_len, hd) as a strided view
    return oq.transpose(1, 2), ok.transpose(1, 2), q, k, rstd_q, rstd_k


def qk_norm_rope_backward(doq, dok, q, k, q_weight, k_weight, cos, sin, rstd_q, rstd_k, eps):
    # doq/dok arrive as (bsz, n_head, seq_len, hd) transposed views -> back to (bsz, seq_len, n_head, hd)
    doq = doq.transpose(1, 2).contiguous()
    dok = dok.transpose(1, 2).contiguous()

    bsz, seq_len, n_qh, hd = q.shape
    n_kh = k.shape[2]
    n_rows = bsz * seq_len

    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd_half = triton.next_power_of_2(hd // 2)

    sm_count = 1
    if q.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    n_programs = min(sm_count, n_rows)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    _dwq = torch.zeros((n_programs, hd), dtype=torch.float32, device=q.device)
    _dwk = torch.zeros((n_programs, hd), dtype=torch.float32, device=q.device)

    out_dtype = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[q.dtype]

    _qk_norm_rope_backward_kernel[(n_programs,)](
        doq,
        doq.stride(1),
        dok,
        dok.stride(1),
        q,
        q.stride(1),
        k,
        k.stride(1),
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        q_weight,
        k_weight,
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        rstd_q,
        rstd_k,
        _dwq,
        _dwk,
        _dwq.stride(0),
        n_rows,
        seq_len,
        n_qh,
        n_kh,
        hd,
        cos.shape[0],
        pad_n_qh,
        pad_n_kh,
        pad_hd_half,
        out_dtype,
        num_warps=_num_warps(pad_n_qh, pad_n_kh, pad_hd_half),
    )

    dq_weight = _dwq.sum(dim=0).to(q_weight.dtype)
    dk_weight = _dwk.sum(dim=0).to(k_weight.dtype)
    return dq, dk, dq_weight, dk_weight


class LigerQkNormRopeFunction(torch.autograd.Function):
    """Fused per-head RMSNorm(Q/K) + RoPE.

    Inputs (matching the pre-transpose projection layout used by Qwen3):
        q: (bsz, seq_len, n_q_head, head_dim)
        k: (bsz, seq_len, n_kv_head, head_dim)
        q_weight, k_weight: (head_dim,)
        cos, sin: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
    Returns:
        q, k: (bsz, n_head, seq_len, head_dim)  (transposed, ready for attention)
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, q, k, q_weight, k_weight, cos, sin, eps):
        oq, ok, q_saved, k_saved, rstd_q, rstd_k = qk_norm_rope_forward(q, k, q_weight, k_weight, cos, sin, eps)
        ctx.eps = eps
        ctx.save_for_backward(q_saved, k_saved, q_weight, k_weight, cos, sin, rstd_q, rstd_k)
        return oq, ok

    @staticmethod
    def backward(ctx, doq, dok):
        q, k, q_weight, k_weight, cos, sin, rstd_q, rstd_k = ctx.saved_tensors
        dq, dk, dq_weight, dk_weight = qk_norm_rope_backward(
            doq, dok, q, k, q_weight, k_weight, cos, sin, rstd_q, rstd_k, ctx.eps
        )
        return dq, dk, dq_weight, dk_weight, None, None, None
