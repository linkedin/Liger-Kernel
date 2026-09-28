# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Llama4-style Rotary Position Embedding (RoPE) kernel (CuTile backend).

Applies in-place complex multiplication: (q_r + i*q_i) * (f_r + i*f_i).

Grid: (batch_size, seq_len, n_heads_max) — one block per (batch, seq, head).

Interleaved layout: q[b, s, h, 2*d] = real part, q[b, s, h, 2*d+1] = imaginary part.
We construct the stride-2 index pairs using:
  base = ct.arange(BLOCK_SIZE)
  doubled = base + base          # [0, 2, 4, ..., 2*(BLOCK_SIZE-1)]
  real_idx = doubled + d_start*2
  imag_idx = real_idx + 1

q and k are passed as 2D views (B*S*H, head_dim) for simpler indexing.
freqs_cis is passed as (S, head_dim) after view_as_real + reshape.
"""

import cuda.tile as ct
import torch


def _select_block_size(head_dim_half: int) -> int:
    if head_dim_half >= 256:
        return 128
    if head_dim_half >= 96:
        return 128
    if head_dim_half >= 48:
        return 64
    if head_dim_half >= 24:
        return 32
    return 16


@ct.kernel
def _llama4_rope_kernel_ct(
    query,  # (B*S*H_q, head_dim) query — modified in-place
    key,  # (B*S*H_k, head_dim) key — modified in-place
    freqs,  # (S, head_dim) frequencies (view_as_real, flattened last 2 dims)
    seq_len,
    HEAD_DIM_HALF: ct.Constant[int],
    N_Q_HEADS: ct.Constant[int],
    N_K_HEADS: ct.Constant[int],
    imag_sign,
    BLOCK_SIZE: ct.Constant[int],
):
    """
    RoPE kernel.

    Grid: (batch_size, seq_len, n_heads_max).
    One block per (batch, seq, head) position.

    For each d-block, loads BLOCK_SIZE (real, imag) pairs from Q/K and FREQS,
    computes complex multiplication, and stores back in-place.

    Index construction (stride-2 interleaved):
        base    = arange(BLOCK_SIZE)          # [0, 1, ..., BLOCK_SIZE-1]
        doubled = base + base                 # [0, 2, 4, ..., 2*(BLOCK_SIZE-1)]
        real_idx = doubled + d_start*2        # real column indices
        imag_idx = real_idx + 1               # imag column indices
    """
    batch_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    pid_h = ct.bid(2)

    # Number of BLOCK_SIZE blocks over head_dim_half
    n_d_blocks = (HEAD_DIM_HALF + BLOCK_SIZE - 1) // BLOCK_SIZE

    for di in range(n_d_blocks):
        d_start = di * BLOCK_SIZE

        # Build interleaved column indices for real/imag parts
        base = ct.arange(BLOCK_SIZE, dtype=ct.int32)
        doubled = base + base  # [0, 2, 4, ..., 2*(BLOCK_SIZE-1)]
        real_idx = doubled + d_start * 2  # real column indices in (seq, head_dim)
        imag_idx = real_idx + 1  # imag column indices

        # Load frequencies for this seq position and d-block
        f_r = ct.astype(
            ct.gather(freqs, (seq_idx, real_idx), check_bounds=True, padding_value=0.0, latency=3),
            ct.float32,
        )
        f_i = ct.astype(
            ct.gather(freqs, (seq_idx, imag_idx), check_bounds=True, padding_value=0.0, latency=3),
            ct.float32,
        )
        f_i = f_i * imag_sign

        # Process query head
        if pid_h < N_Q_HEADS:
            q_row = batch_idx * seq_len * N_Q_HEADS + seq_idx * N_Q_HEADS + pid_h
            q_r = ct.astype(
                ct.gather(query, (q_row, real_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
            )
            q_i = ct.astype(
                ct.gather(query, (q_row, imag_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
            )

            # Complex multiply: (q_r + i*q_i) * (f_r + i*f_i)
            new_q_r = q_r * f_r - q_i * f_i
            new_q_i = q_r * f_i + q_i * f_r

            ct.scatter(query, (q_row, real_idx), ct.astype(new_q_r, query.dtype), check_bounds=True)
            ct.scatter(query, (q_row, imag_idx), ct.astype(new_q_i, query.dtype), check_bounds=True)

        # Process key head
        if pid_h < N_K_HEADS:
            k_row = batch_idx * seq_len * N_K_HEADS + seq_idx * N_K_HEADS + pid_h
            k_r = ct.astype(
                ct.gather(key, (k_row, real_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
            )
            k_i = ct.astype(
                ct.gather(key, (k_row, imag_idx), check_bounds=True, padding_value=0.0, latency=3), ct.float32
            )

            new_k_r = k_r * f_r - k_i * f_i
            new_k_i = k_r * f_i + k_i * f_r

            ct.scatter(key, (k_row, real_idx), ct.astype(new_k_r, key.dtype), check_bounds=True)
            ct.scatter(key, (k_row, imag_idx), ct.astype(new_k_i, key.dtype), check_bounds=True)


def _llama4_rope_forward_ct(q, k, freqs_cis, BLOCK_SIZE=None, imag_sign=1.0):
    original_dtype = q.dtype

    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape
    head_dim_half = head_dim // 2

    # Normalize freqs_cis to (seq_len, head_dim) real layout
    if freqs_cis.is_complex():
        freqs_cis = freqs_cis.reshape(-1, freqs_cis.shape[-1])
        if freqs_cis.shape[0] > seq_len:
            freqs_cis = freqs_cis[:seq_len]
        freqs_cis = torch.view_as_real(freqs_cis)  # (seq_len, head_dim_half, 2)

    if freqs_cis.ndim == 3:
        # (seq_len, head_dim_half, 2) → (seq_len, head_dim)
        freqs_cis = freqs_cis.reshape(freqs_cis.shape[0], -1)

    compute_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    q = q.to(compute_dtype).contiguous()
    k = k.to(compute_dtype).contiguous()
    freqs_cis = freqs_cis.float().contiguous()

    if BLOCK_SIZE is None:
        BLOCK_SIZE = _select_block_size(head_dim_half)

    # Reshape to 2D for the kernel: (B*S*H, head_dim)
    q_2d = q.reshape(batch_size * seq_len * n_q_heads, head_dim).contiguous()
    k_2d = k.reshape(batch_size * seq_len * n_k_heads, head_dim).contiguous()

    n_heads_max = max(n_q_heads, n_k_heads)
    grid = (batch_size, seq_len, n_heads_max)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _llama4_rope_kernel_ct,
        (
            q_2d,
            k_2d,
            freqs_cis,
            int(seq_len),
            int(head_dim_half),
            int(n_q_heads),
            int(n_k_heads),
            float(imag_sign),
            int(BLOCK_SIZE),
        ),
    )

    q_out = q_2d.reshape(batch_size, seq_len, n_q_heads, head_dim)
    k_out = k_2d.reshape(batch_size, seq_len, n_k_heads, head_dim)

    if q_out.dtype != original_dtype:
        q_out = q_out.to(original_dtype)
    if k_out.dtype != original_dtype:
        k_out = k_out.to(original_dtype)

    return q_out, k_out


class LigerLlama4RopeFunction(torch.autograd.Function):
    """CuTile autograd wrapper for Llama4 RoPE."""

    @staticmethod
    def forward(ctx, q, k, freqs_cis, BLOCK_SIZE=None):
        q_out, k_out = _llama4_rope_forward_ct(q, k, freqs_cis, BLOCK_SIZE, imag_sign=1.0)
        ctx.save_for_backward(freqs_cis.detach() if isinstance(freqs_cis, torch.Tensor) else freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors
        BLOCK_SIZE = getattr(ctx, "BLOCK_SIZE", None)
        dq_out, dk_out = _llama4_rope_forward_ct(dq, dk, freqs_cis, BLOCK_SIZE, imag_sign=-1.0)
        return dq_out, dk_out, None, None
