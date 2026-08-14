# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Rotary Positional Embedding (RoPE) kernel (CuTile backend).
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _rope_4d_kernel_ct(
    Q,
    K,
    COS,
    SIN,
    seq_len: ConstInt,
    sin_sign: ct.Constant[float],
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    cos_bs = COS.shape[0]

    pid = ct.bid(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    # Stay in the input dtype throughout the math — triton's rope does the same.
    # The fp32 cast doubles register pressure and arithmetic cost on bf16 inputs
    # without meaningful precision gain for RoPE (rotation has unit magnitude).
    cos_row = ct.load(COS, index=(cos_batch_idx, seq_idx, 0), shape=(1, 1, TILE_HD)).reshape((1, TILE_HD))
    sin_row = ct.load(SIN, index=(cos_batch_idx, seq_idx, 0), shape=(1, 1, TILE_HD)).reshape((1, TILE_HD)) * sin_sign

    q_r = ct.load(Q, index=(batch_idx, 0, seq_idx, 0), shape=(1, TILE_QH, 1, TILE_HD)).reshape((TILE_QH, TILE_HD))
    q_i = ct.load(Q, index=(batch_idx, 0, seq_idx, 1), shape=(1, TILE_QH, 1, TILE_HD)).reshape((TILE_QH, TILE_HD))
    new_q_r = q_r * cos_row - q_i * sin_row
    new_q_i = q_i * cos_row + q_r * sin_row
    ct.store(Q, index=(batch_idx, 0, seq_idx, 0), tile=new_q_r.reshape((1, TILE_QH, 1, TILE_HD)))
    ct.store(Q, index=(batch_idx, 0, seq_idx, 1), tile=new_q_i.reshape((1, TILE_QH, 1, TILE_HD)))

    k_r = ct.load(K, index=(batch_idx, 0, seq_idx, 0), shape=(1, TILE_KH, 1, TILE_HD)).reshape((TILE_KH, TILE_HD))
    k_i = ct.load(K, index=(batch_idx, 0, seq_idx, 1), shape=(1, TILE_KH, 1, TILE_HD)).reshape((TILE_KH, TILE_HD))
    new_k_r = k_r * cos_row - k_i * sin_row
    new_k_i = k_i * cos_row + k_r * sin_row
    ct.store(K, index=(batch_idx, 0, seq_idx, 0), tile=new_k_r.reshape((1, TILE_KH, 1, TILE_HD)))
    ct.store(K, index=(batch_idx, 0, seq_idx, 1), tile=new_k_i.reshape((1, TILE_KH, 1, TILE_HD)))


@ct.kernel
def _rope_general_kernel_ct(
    Q,  # (bsz, seq_len, n_q_heads, head_dim) -- 4D natural layout
    K,  # (bsz, seq_len, n_k_heads, head_dim)
    COS,  # (cos_bs, seq_len, head_dim)
    SIN,  # (cos_bs, seq_len, head_dim)
    cos_bs: ConstInt,
    seq_len: ConstInt,
    N_Q_HEADS: ConstInt,
    N_K_HEADS: ConstInt,
    HEAD_DIM_HALF: ConstInt,
    sin_sign: ct.Constant[float],
    TILE_HD: ConstInt,
):
    """General gather/scatter-based rope kernel.

    Works for arbitrary head_dim (including odd) and arbitrary n_heads:
    - tile size is padded via _next_power_of_2 but original data is NOT padded
    - ct.gather uses padding_value=0.0 for OOB reads (when TILE_HD > head_dim_half)
    - elements past 2*head_dim_half (tail when head_dim is odd) are preserved unchanged
    """
    pid = ct.bid(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    dim_idx = ct.arange(TILE_HD, dtype=ct.int32)  # [0, TILE_HD)

    cos_row = ct.astype(
        ct.gather(COS, (cos_batch_idx, seq_idx, dim_idx), check_bounds=True, padding_value=0.0),
        ct.float32,
    )
    sin_row = (
        ct.astype(
            ct.gather(SIN, (cos_batch_idx, seq_idx, dim_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        * sin_sign
    )

    valid_mask = dim_idx < HEAD_DIM_HALF

    for h in range(N_Q_HEADS):
        q_r = ct.astype(
            ct.gather(Q, (batch_idx, seq_idx, h, dim_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        q_i = ct.astype(
            ct.gather(Q, (batch_idx, seq_idx, h, dim_idx + HEAD_DIM_HALF), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        new_q_r = q_r * cos_row - q_i * sin_row
        new_q_i = q_i * cos_row + q_r * sin_row
        new_q_r = ct.where(valid_mask, new_q_r, q_r)
        new_q_i = ct.where(valid_mask, new_q_i, q_i)
        ct.scatter(Q, (batch_idx, seq_idx, h, dim_idx), ct.astype(new_q_r, Q.dtype), check_bounds=True)
        ct.scatter(Q, (batch_idx, seq_idx, h, dim_idx + HEAD_DIM_HALF), ct.astype(new_q_i, Q.dtype), check_bounds=True)

    for h in range(N_K_HEADS):
        k_r = ct.astype(
            ct.gather(K, (batch_idx, seq_idx, h, dim_idx), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        k_i = ct.astype(
            ct.gather(K, (batch_idx, seq_idx, h, dim_idx + HEAD_DIM_HALF), check_bounds=True, padding_value=0.0),
            ct.float32,
        )
        new_k_r = k_r * cos_row - k_i * sin_row
        new_k_i = k_i * cos_row + k_r * sin_row
        new_k_r = ct.where(valid_mask, new_k_r, k_r)
        new_k_i = ct.where(valid_mask, new_k_i, k_i)
        ct.scatter(K, (batch_idx, seq_idx, h, dim_idx), ct.astype(new_k_r, K.dtype), check_bounds=True)
        ct.scatter(K, (batch_idx, seq_idx, h, dim_idx + HEAD_DIM_HALF), ct.astype(new_k_i, K.dtype), check_bounds=True)


def rope_forward(q, k, cos, sin):
    bsz, n_q_heads, seq_len, head_dim = q.shape
    n_k_heads = k.shape[1]
    head_dim_half = head_dim // 2
    original_dtype = q.dtype

    TILE_HD = _next_power_of_2(head_dim_half)
    TILE_QH = _next_power_of_2(n_q_heads)
    TILE_KH = _next_power_of_2(n_k_heads)
    # ALIGNED: shapes are all power-of-2 → _rope_4d_kernel_ct's block-indexed ct.load
    # works (its TILE_HD must equal head_dim_half exactly, which must be pow2).
    # Otherwise we fall back to _rope_general_kernel_ct (ct.gather/scatter on element
    # indices). Both are cuTile-native; we cannot reuse the 4D kernel for non-pow2
    # head_dim_half because ct.load requires the load shape to be power-of-2.
    # Note: ct.load handles non-contig tensors natively via strides — no .contiguous()
    # call needed on q/k for the fast path.
    ALIGNED = (TILE_HD == head_dim_half) and (TILE_QH == n_q_heads) and (TILE_KH == n_k_heads)

    cos_3d = cos.contiguous()
    sin_3d = sin.contiguous()
    cos_bs = cos_3d.shape[0]
    grid = (bsz * seq_len,)

    if ALIGNED:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _rope_4d_kernel_ct,
            (
                q,
                k,
                cos_3d,
                sin_3d,
                int(seq_len),
                float(1.0),
                int(TILE_QH),
                int(TILE_KH),
                int(TILE_HD),
            ),
        )
        return q, k, cos_3d, sin_3d, cos_bs, ALIGNED, TILE_QH, TILE_KH, TILE_HD, original_dtype
    else:
        # General path: handles arbitrary head_dim (including odd) and n_heads.
        # Uses gather/scatter directly on the 4D layout — no data padding.
        q_t = q.transpose(1, 2).contiguous()  # (bsz, seq_len, n_q_heads, head_dim)
        k_t = k.transpose(1, 2).contiguous()
        cos_3d = cos.contiguous()
        sin_3d = sin.contiguous()
        cos_bs = cos_3d.shape[0]
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _rope_general_kernel_ct,
            (
                q_t,
                k_t,
                cos_3d,
                sin_3d,
                int(cos_bs),
                int(seq_len),
                int(n_q_heads),
                int(n_k_heads),
                int(head_dim_half),
                float(1.0),
                int(TILE_HD),
            ),
        )
        q_out = q_t.transpose(1, 2).to(original_dtype)
        k_out = k_t.transpose(1, 2).to(original_dtype)
        return q_out, k_out, cos_3d, sin_3d, cos_bs, ALIGNED, TILE_QH, TILE_KH, TILE_HD, original_dtype


def rope_backward(
    dq,
    dk,
    cos,
    sin,
    cos_bs,
    ALIGNED,
    TILE_QH,
    TILE_KH,
    TILE_HD,
    original_dtype,
    bsz,
    seq_len,
    n_q_heads,
    n_k_heads,
    head_dim,
):
    head_dim_half = head_dim // 2
    n_row = bsz * seq_len
    grid = (n_row,)

    if ALIGNED:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _rope_4d_kernel_ct,
            (
                dq,
                dk,
                cos,
                sin,
                int(seq_len),
                float(-1.0),
                int(TILE_QH),
                int(TILE_KH),
                int(TILE_HD),
            ),
        )
        return dq, dk
    else:
        dq_t = dq.transpose(1, 2).contiguous()
        dk_t = dk.transpose(1, 2).contiguous()
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _rope_general_kernel_ct,
            (
                dq_t,
                dk_t,
                cos,
                sin,
                int(cos_bs),
                int(seq_len),
                int(n_q_heads),
                int(n_k_heads),
                int(head_dim_half),
                float(-1.0),
                int(TILE_HD),
            ),
        )
        dq_out = dq_t.transpose(1, 2).to(original_dtype)
        dk_out = dk_t.transpose(1, 2).to(original_dtype)
        return dq_out, dk_out


class LigerRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        bsz, n_q_heads, seq_len, head_dim = q.shape
        n_k_heads = k.shape[1]
        q_out, k_out, saved_cos, saved_sin, cos_bs, ALIGNED, TILE_QH, TILE_KH, TILE_HD, original_dtype = rope_forward(
            q, k, cos, sin
        )
        ctx.save_for_backward(saved_cos, saved_sin)
        ctx.bsz = bsz
        ctx.seq_len = seq_len
        ctx.n_q_heads = n_q_heads
        ctx.n_k_heads = n_k_heads
        ctx.head_dim = head_dim
        ctx.cos_bs = cos_bs
        ctx.original_dtype = original_dtype
        ctx.ALIGNED = ALIGNED
        ctx.TILE_QH = TILE_QH
        ctx.TILE_KH = TILE_KH
        ctx.TILE_HD = TILE_HD
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        saved_cos, saved_sin = ctx.saved_tensors
        dq_out, dk_out = rope_backward(
            dq,
            dk,
            saved_cos,
            saved_sin,
            ctx.cos_bs,
            ctx.ALIGNED,
            ctx.TILE_QH,
            ctx.TILE_KH,
            ctx.TILE_HD,
            ctx.original_dtype,
            ctx.bsz,
            ctx.seq_len,
            ctx.n_q_heads,
            ctx.n_k_heads,
            ctx.head_dim,
        )
        return dq_out, dk_out, None, None, None, None
