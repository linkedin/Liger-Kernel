# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Adapted from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py

"""
Qwen2VL Multimodal Rotary Position Embedding (M-RoPE) kernel (CuTile backend).

Half-split layout: left half of head_dim = real part, right half = imaginary part.
Three RoPE sections: temporal [0, t_end), height [t_end, h_end), width [h_end, hd//2).
cos/sin shape: (3, bsz, seq_len, head_dim).
Grid: (bsz, seq_len) — one program per token (2D grid avoids divmod on pid).
"""

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _qwen2vl_mrope_kernel_ct(
    query,  # 1D flat, len = bsz*sl*n_qh*hd
    key,  # 1D flat, len = bsz*sl*n_kh*hd
    cos,  # 1D flat, len = 3*bsz*sl*hd
    sin,  # 1D flat
    sl,
    BS_SL,  # bsz * sl  (slab stride = BS_SL * HEAD_DIM, computed in-kernel)
    N_QH: ConstInt,
    N_KH: ConstInt,
    MROPE_SECTION_T: ConstInt,
    MROPE_SECTION_H: ConstInt,
    BACKWARD: ct.Constant[bool],
    HEAD_DIM: ConstInt,
    HEAD_DIM_HALF: ConstInt,
    TILE_HD: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    HD_POW2: ct.Constant[bool],
):
    batch_idx = ct.bid(0)
    seq_idx = ct.bid(1)

    t_end = MROPE_SECTION_T
    h_end = t_end + MROPE_SECTION_H

    FLAT = TILE_QH * TILE_HD
    row_1d = ct.arange(TILE_QH, dtype=ct.int32)
    col_1d = ct.arange(TILE_HD, dtype=ct.int32)
    flat_row = ct.broadcast_to(row_1d[:, None], (TILE_QH, TILE_HD)).reshape((FLAT,))
    flat_col = ct.broadcast_to(col_1d[None, :], (TILE_QH, TILE_HD)).reshape((FLAT,))

    # Issue Q+K gathers first to start DRAM fetch, then cos/sin gathers can overlap.
    q_token_off = (batch_idx * sl + seq_idx) * (N_QH * HEAD_DIM)
    q_r_idx = q_token_off + flat_row * HEAD_DIM + flat_col
    q_i_idx = q_r_idx + HEAD_DIM_HALF
    if HD_POW2:
        q_mask = flat_row < N_QH
    else:
        q_mask = (flat_row < N_QH) & (flat_col < HEAD_DIM_HALF)
    q_r = ct.gather(query, q_r_idx, mask=q_mask, check_bounds=False, latency=2)
    q_i = ct.gather(query, q_i_idx, mask=q_mask, check_bounds=False, latency=2)

    # K indices (computed early so K gathers can fire right after Q's)
    FLAT_K = TILE_KH * TILE_HD
    krow_1d = ct.arange(TILE_KH, dtype=ct.int32)
    k_flat_row = ct.broadcast_to(krow_1d[:, None], (TILE_KH, TILE_HD)).reshape((FLAT_K,))
    k_flat_col = ct.broadcast_to(col_1d[None, :], (TILE_KH, TILE_HD)).reshape((FLAT_K,))
    k_token_off = (batch_idx * sl + seq_idx) * (N_KH * HEAD_DIM)
    k_r_idx = k_token_off + k_flat_row * HEAD_DIM + k_flat_col
    k_i_idx = k_r_idx + HEAD_DIM_HALF
    if HD_POW2:
        k_mask = k_flat_row < N_KH
    else:
        k_mask = (k_flat_row < N_KH) & (k_flat_col < HEAD_DIM_HALF)
    k_r = ct.gather(key, k_r_idx, mask=k_mask, check_bounds=False, latency=2)
    k_i = ct.gather(key, k_i_idx, mask=k_mask, check_bounds=False, latency=2)

    token_cs_off = (batch_idx * sl + seq_idx) * HEAD_DIM
    slab_stride = BS_SL * HEAD_DIM
    t_idx = flat_col + token_cs_off
    h_idx = t_idx + slab_stride
    w_idx = h_idx + slab_stride
    t_cos = ct.gather(cos, t_idx, check_bounds=False, latency=2)
    t_sin = ct.gather(sin, t_idx, check_bounds=False, latency=2)
    h_cos = ct.gather(cos, h_idx, check_bounds=False, latency=2)
    h_sin = ct.gather(sin, h_idx, check_bounds=False, latency=2)
    w_cos = ct.gather(cos, w_idx, check_bounds=False, latency=2)
    w_sin = ct.gather(sin, w_idx, check_bounds=False, latency=2)

    in_t = flat_col < t_end
    in_h = flat_col < h_end
    cos_row = ct.where(in_t, t_cos, ct.where(in_h, h_cos, w_cos))
    sin_row = ct.where(in_t, t_sin, ct.where(in_h, h_sin, w_sin))
    if BACKWARD:
        sin_row = -sin_row

    cos_q = cos_row.astype(query.dtype)
    sin_q = sin_row.astype(query.dtype)
    new_q_r = q_r * cos_q - q_i * sin_q
    new_q_i = q_i * cos_q + q_r * sin_q

    # Reuse Q's cos_row when FLAT_K <= FLAT (common case: TILE_KH <= TILE_QH).
    if TILE_KH <= TILE_QH:
        cos_k = ct.extract(cos_row, (0,), shape=(FLAT_K,)).astype(key.dtype)
        sin_k = ct.extract(sin_row, (0,), shape=(FLAT_K,)).astype(key.dtype)
    else:
        t_idx_k = k_flat_col + token_cs_off
        h_idx_k = t_idx_k + slab_stride
        w_idx_k = h_idx_k + slab_stride
        t_cos_k = ct.gather(cos, t_idx_k, check_bounds=False, latency=2)
        t_sin_k = ct.gather(sin, t_idx_k, check_bounds=False, latency=2)
        h_cos_k = ct.gather(cos, h_idx_k, check_bounds=False, latency=2)
        h_sin_k = ct.gather(sin, h_idx_k, check_bounds=False, latency=2)
        w_cos_k = ct.gather(cos, w_idx_k, check_bounds=False, latency=2)
        w_sin_k = ct.gather(sin, w_idx_k, check_bounds=False, latency=2)
        in_t_k = k_flat_col < t_end
        in_h_k = k_flat_col < h_end
        cos_k_raw = ct.where(in_t_k, t_cos_k, ct.where(in_h_k, h_cos_k, w_cos_k))
        sin_k_raw = ct.where(in_t_k, t_sin_k, ct.where(in_h_k, h_sin_k, w_sin_k))
        if BACKWARD:
            sin_k_raw = -sin_k_raw
        cos_k = cos_k_raw.astype(key.dtype)
        sin_k = sin_k_raw.astype(key.dtype)
    new_k_r = k_r * cos_k - k_i * sin_k
    new_k_i = k_i * cos_k + k_r * sin_k
    ct.scatter(query, q_r_idx, new_q_r, mask=q_mask, check_bounds=False, latency=1)
    ct.scatter(query, q_i_idx, new_q_i, mask=q_mask, check_bounds=False, latency=1)
    ct.scatter(key, k_r_idx, new_k_r, mask=k_mask, check_bounds=False, latency=1)
    ct.scatter(key, k_i_idx, new_k_i, mask=k_mask, check_bounds=False, latency=1)


def _qwen2vl_mrope_forward(q, k, cos, sin, mrope_section):
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    head_dim_half = head_dim // 2
    TILE_HD = _next_power_of_2(head_dim_half)
    TILE_QH = _next_power_of_2(n_q_head)
    TILE_KH = _next_power_of_2(n_kv_head)
    bs_sl = batch_size * seq_len

    cos = cos.contiguous()
    sin = sin.contiguous()

    grid = (batch_size, seq_len)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _qwen2vl_mrope_kernel_ct,
        (
            q.view(-1),
            k.view(-1),
            cos.view(-1),
            sin.view(-1),
            int(seq_len),
            int(bs_sl),
            int(n_q_head),
            int(n_kv_head),
            int(mrope_section[0]),
            int(mrope_section[1]),
            False,
            int(head_dim),
            int(head_dim_half),
            int(TILE_HD),
            int(TILE_QH),
            int(TILE_KH),
            bool(TILE_HD == head_dim_half),
        ),
    )

    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def _qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section):
    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    n_kv_head = dk.shape[2]
    head_dim_half = head_dim // 2
    TILE_HD = _next_power_of_2(head_dim_half)
    TILE_QH = _next_power_of_2(n_q_head)
    TILE_KH = _next_power_of_2(n_kv_head)
    bs_sl = batch_size * seq_len

    grid = (batch_size, seq_len)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _qwen2vl_mrope_kernel_ct,
        (
            dq.view(-1),
            dk.view(-1),
            cos.view(-1),
            sin.view(-1),
            int(seq_len),
            int(bs_sl),
            int(n_q_head),
            int(n_kv_head),
            int(mrope_section[0]),
            int(mrope_section[1]),
            True,
            int(head_dim),
            int(head_dim_half),
            int(TILE_HD),
            int(TILE_QH),
            int(TILE_KH),
            bool(TILE_HD == head_dim_half),
        ),
    )

    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerQwen2VLMRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        q, k, cos, sin = _qwen2vl_mrope_forward(q, k, cos, sin, mrope_section)
        ctx.save_for_backward(cos, sin)
        ctx.mrope_section = mrope_section
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        mrope_section = ctx.mrope_section
        dq, dk = _qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section)
        return dq, dk, None, None, None, None
