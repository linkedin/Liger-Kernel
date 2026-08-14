import math

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ops.softmax import _softmax_backward
from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.utils import get_total_gpu_memory

_MAX_GRID_PROGRAMS = 65535


# ---------------------------------------------------------------------------
# Sparse forward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_neighborhood_qk_softmax_kernel(
    Q_ptr,
    K_ptr,
    Attn_ptr,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_neighbor_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute neighborhood QK scores and row softmax, scattering into Attn."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    half_kernel = kernel_size // 2

    num_batch_heads = batch_size * num_heads
    num_m_tiles = tl.cdiv(seq_len, BLOCK_SIZE_M)
    total_tiles = num_batch_heads * num_m_tiles

    for global_tile_id in tl.range(pid, total_tiles, num_programs):
        batch_head_id = global_tile_id // num_m_tiles
        tile_m = global_tile_id % num_m_tiles

        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads

        row_start = tile_m * BLOCK_SIZE_M
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
        safe_row = tl.minimum(row_offsets, seq_len - 1)
        row_valid = row_offsets < seq_len
        neighbor_cols = tl.arange(0, kernel_size)[None, :]
        key_cols_all = row_offsets[:, None] + (neighbor_cols - half_kernel) * dilation
        key_valid_all = row_valid[:, None] & (key_cols_all >= 0) & (key_cols_all < seq_len)

        dot_acc = tl.zeros((BLOCK_SIZE_M, kernel_size), dtype=tl.float32)

        for k_start in range(0, head_dim, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < head_dim
            safe_k = tl.minimum(k_offsets, head_dim - 1)

            q_ptrs = (
                Q_ptr
                + batch_id * q_batch_stride
                + head_id * q_head_stride
                + safe_row[:, None] * q_seq_stride
                + safe_k[None, :] * q_dim_stride
            )
            q_mask = row_valid[:, None] & k_mask[None, :]
            q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)

            for neighbor_idx in range(kernel_size):
                offset = neighbor_idx - half_kernel
                key_cols = row_offsets + offset * dilation
                key_valid = row_valid & (key_cols >= 0) & (key_cols < seq_len)
                safe_key = tl.minimum(tl.maximum(key_cols, 0), seq_len - 1)

                k_ptrs = (
                    K_ptr
                    + batch_id * k_batch_stride
                    + head_id * k_head_stride
                    + safe_key[:, None] * k_seq_stride
                    + safe_k[None, :] * k_dim_stride
                )
                k_mask_2d = key_valid[:, None] & k_mask[None, :]
                k_chunk = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)

                partial = tl.sum(q_chunk * k_chunk, axis=1)
                col_select = neighbor_cols == neighbor_idx
                dot_acc = tl.where(col_select, dot_acc + partial[:, None], dot_acc)

        scores = tl.where(key_valid_all, dot_acc * scale, float("-inf"))
        row_max = tl.max(scores, axis=1)
        exp_scores = tl.exp(scores - row_max[:, None])
        denom = tl.sum(exp_scores, axis=1)
        attn_local = exp_scores / denom[:, None]

        attn_ptrs = (
            Attn_ptr
            + batch_id * attn_batch_stride
            + head_id * attn_head_stride
            + safe_row[:, None] * attn_seq_stride
            + neighbor_cols * attn_neighbor_stride
        )
        tl.store(attn_ptrs, attn_local, mask=key_valid_all)


@triton.jit
def _fused_neighborhood_attention_av_kernel(
    Attn_ptr,
    V_ptr,
    Out_ptr,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_neighbor_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    v_dim_stride,
    out_batch_stride,
    out_head_stride,
    out_seq_stride,
    out_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute Attention @ V."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_batch_heads = batch_size * num_heads
    num_m_tiles = tl.cdiv(seq_len, BLOCK_SIZE_M)
    total_tiles = num_batch_heads * num_m_tiles
    half_kernel = kernel_size // 2

    for global_tile_id in tl.range(pid, total_tiles, num_programs):
        batch_head_id = global_tile_id // num_m_tiles
        tile_m = global_tile_id % num_m_tiles

        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads

        row_start = tile_m * BLOCK_SIZE_M
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
        safe_row = tl.minimum(row_offsets, seq_len - 1)
        row_valid = row_offsets < seq_len

        num_n_tiles = tl.cdiv(head_dim, BLOCK_SIZE_N)

        for tile_n in range(num_n_tiles):
            col_start = tile_n * BLOCK_SIZE_N
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
            safe_col = tl.minimum(col_offsets, head_dim - 1)
            dim_valid = col_offsets[None, :] < head_dim

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for neighbor_idx in range(kernel_size):
                offset = neighbor_idx - half_kernel
                key_cols = row_offsets + offset * dilation
                key_valid = row_valid & (key_cols >= 0) & (key_cols < seq_len)
                safe_key = tl.minimum(tl.maximum(key_cols, 0), seq_len - 1)

                attn_ptrs = (
                    Attn_ptr
                    + batch_id * attn_batch_stride
                    + head_id * attn_head_stride
                    + safe_row * attn_seq_stride
                    + neighbor_idx * attn_neighbor_stride
                )
                attn_vals = tl.load(attn_ptrs, mask=key_valid, other=0.0)

                v_ptrs = (
                    V_ptr
                    + batch_id * v_batch_stride
                    + head_id * v_head_stride
                    + safe_key[:, None] * v_seq_stride
                    + safe_col[None, :] * v_dim_stride
                )
                v_mask = key_valid[:, None] & dim_valid
                v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)
                acc += attn_vals[:, None] * v_chunk

            out_ptrs = (
                Out_ptr
                + batch_id * out_batch_stride
                + head_id * out_head_stride
                + safe_row[:, None] * out_seq_stride
                + safe_col[None, :] * out_dim_stride
            )
            valid_mask = row_valid[:, None] & dim_valid
            tl.store(out_ptrs, acc, mask=valid_mask)


# ---------------------------------------------------------------------------
# Sparse backward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _fused_neighborhood_attention_grad_v_kernel(
    Attn_ptr,
    grad_output_ptr,
    grad_V_ptr,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_neighbor_stride,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    grad_v_batch_stride,
    grad_v_head_stride,
    grad_v_seq_stride,
    grad_v_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute grad_V from sparse attn: grad_V = Attn^T @ grad_output."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)
    half_kernel = kernel_size // 2

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    safe_row = tl.minimum(row_offsets, seq_len - 1)
    safe_col = tl.minimum(col_offsets, head_dim - 1)
    row_valid = row_offsets < seq_len
    dim_valid = col_offsets[None, :] < head_dim

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for neighbor_idx in range(kernel_size):
        offset = neighbor_idx - half_kernel
        query_rows = row_offsets - offset * dilation
        query_valid = row_valid & (query_rows >= 0) & (query_rows < seq_len)
        safe_query = tl.minimum(tl.maximum(query_rows, 0), seq_len - 1)

        attn_ptrs = (
            Attn_ptr
            + batch_id * attn_batch_stride
            + head_id * attn_head_stride
            + safe_query * attn_seq_stride
            + neighbor_idx * attn_neighbor_stride
        )
        attn_vals = tl.load(attn_ptrs, mask=query_valid, other=0.0)

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + safe_query[:, None] * grad_out_seq_stride
            + safe_col[None, :] * grad_out_dim_stride
        )
        grad_out_mask = query_valid[:, None] & dim_valid
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)
        acc += attn_vals[:, None] * grad_out_chunk

    grad_v_ptrs = (
        grad_V_ptr
        + batch_id * grad_v_batch_stride
        + head_id * grad_v_head_stride
        + safe_row[:, None] * grad_v_seq_stride
        + safe_col[None, :] * grad_v_dim_stride
    )
    valid_mask = row_valid[:, None] & dim_valid
    tl.store(grad_v_ptrs, acc, mask=valid_mask)


@triton.jit
def _sparse_neighborhood_grad_attn_softmax_bwd_kernel(
    grad_output_ptr,
    V_ptr,
    attn_ptr,
    grad_qk_ptr,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    v_dim_stride,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_neighbor_stride,
    grad_qk_batch_stride,
    grad_qk_head_stride,
    grad_qk_seq_stride,
    grad_qk_neighbor_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused grad_attn + neighborhood softmax backward -> grad_qk."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    half_kernel = kernel_size // 2

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    safe_row = tl.minimum(row_offsets, seq_len - 1)
    row_valid = row_offsets < seq_len
    neighbor_cols = tl.arange(0, kernel_size)[None, :]
    key_cols_all = row_offsets[:, None] + (neighbor_cols - half_kernel) * dilation
    key_valid_all = row_valid[:, None] & (key_cols_all >= 0) & (key_cols_all < seq_len)

    dy_cols = tl.zeros((BLOCK_SIZE_M, kernel_size), dtype=tl.float32)

    for tile_n in range(tl.cdiv(head_dim, BLOCK_SIZE_N)):
        col_start = tile_n * BLOCK_SIZE_N
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
        safe_col = tl.minimum(col_offsets, head_dim - 1)
        dim_valid = col_offsets[None, :] < head_dim

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + safe_row[:, None] * grad_out_seq_stride
            + safe_col[None, :] * grad_out_dim_stride
        )
        grad_out_mask = row_valid[:, None] & dim_valid
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        for neighbor_idx in range(kernel_size):
            offset = neighbor_idx - half_kernel
            key_cols = row_offsets + offset * dilation
            key_valid = row_valid & (key_cols >= 0) & (key_cols < seq_len)
            safe_key = tl.minimum(tl.maximum(key_cols, 0), seq_len - 1)

            v_ptrs = (
                V_ptr
                + batch_id * v_batch_stride
                + head_id * v_head_stride
                + safe_key[:, None] * v_seq_stride
                + safe_col[None, :] * v_dim_stride
            )
            v_mask = key_valid[:, None] & dim_valid
            v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)

            partial = tl.sum(grad_out_chunk * v_chunk, axis=1)
            col_select = neighbor_cols == neighbor_idx
            dy_cols = tl.where(col_select, dy_cols + partial[:, None], dy_cols)

    attn_ptrs = (
        attn_ptr
        + batch_id * attn_batch_stride
        + head_id * attn_head_stride
        + safe_row[:, None] * attn_seq_stride
        + neighbor_cols * attn_neighbor_stride
    )
    y_cols = tl.load(attn_ptrs, mask=key_valid_all, other=0.0)

    dot = tl.sum(dy_cols * y_cols, axis=1)
    grad_qk_cols = y_cols * (dy_cols - dot[:, None])

    grad_qk_ptrs = (
        grad_qk_ptr
        + batch_id * grad_qk_batch_stride
        + head_id * grad_qk_head_stride
        + safe_row[:, None] * grad_qk_seq_stride
        + neighbor_cols * grad_qk_neighbor_stride
    )
    tl.store(grad_qk_ptrs, grad_qk_cols, mask=key_valid_all)


@triton.jit
def _sparse_neighborhood_grad_q_kernel(
    grad_qk_ptr,
    K_ptr,
    grad_Q_ptr,
    grad_qk_batch_stride,
    grad_qk_head_stride,
    grad_qk_seq_stride,
    grad_qk_neighbor_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    grad_q_batch_stride,
    grad_q_head_stride,
    grad_q_seq_stride,
    grad_q_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """grad_Q from sparse-band grad_qk: sum over neighborhood keys."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)
    half_kernel = kernel_size // 2

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    safe_row = tl.minimum(row_offsets, seq_len - 1)
    safe_col = tl.minimum(col_offsets, head_dim - 1)
    row_valid = row_offsets < seq_len
    dim_valid = col_offsets[None, :] < head_dim
    neighbor_cols = tl.arange(0, kernel_size)[None, :]
    key_cols_all = row_offsets[:, None] + (neighbor_cols - half_kernel) * dilation
    key_valid_all = row_valid[:, None] & (key_cols_all >= 0) & (key_cols_all < seq_len)

    grad_qk_ptrs = (
        grad_qk_ptr
        + batch_id * grad_qk_batch_stride
        + head_id * grad_qk_head_stride
        + safe_row[:, None] * grad_qk_seq_stride
        + neighbor_cols * grad_qk_neighbor_stride
    )
    gq_tile = tl.load(grad_qk_ptrs, mask=key_valid_all, other=0.0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for neighbor_idx in range(kernel_size):
        offset = neighbor_idx - half_kernel
        key_cols = row_offsets + offset * dilation
        key_valid = row_valid & (key_cols >= 0) & (key_cols < seq_len)
        safe_key = tl.minimum(tl.maximum(key_cols, 0), seq_len - 1)

        col_select = neighbor_cols == neighbor_idx
        gq = tl.sum(gq_tile * col_select, axis=1)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + safe_key[:, None] * k_seq_stride
            + safe_col[None, :] * k_dim_stride
        )
        k_mask = key_valid[:, None] & dim_valid
        k_chunk = tl.load(k_ptrs, mask=k_mask, other=0.0)
        acc += gq[:, None] * k_chunk

    acc = acc * scale

    grad_q_ptrs = (
        grad_Q_ptr
        + batch_id * grad_q_batch_stride
        + head_id * grad_q_head_stride
        + safe_row[:, None] * grad_q_seq_stride
        + safe_col[None, :] * grad_q_dim_stride
    )
    valid_mask = row_valid[:, None] & dim_valid
    tl.store(grad_q_ptrs, acc, mask=valid_mask)


@triton.jit
def _sparse_neighborhood_grad_k_kernel(
    grad_qk_ptr,
    Q_ptr,
    grad_K_ptr,
    grad_qk_batch_stride,
    grad_qk_head_stride,
    grad_qk_seq_stride,
    grad_qk_neighbor_stride,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    grad_k_batch_stride,
    grad_k_head_stride,
    grad_k_seq_stride,
    grad_k_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """grad_K from sparse-band grad_qk: accumulate from neighborhood queries."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)
    half_kernel = kernel_size // 2

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    safe_row = tl.minimum(row_offsets, seq_len - 1)
    safe_col = tl.minimum(col_offsets, head_dim - 1)
    row_valid = row_offsets < seq_len
    dim_valid = col_offsets[None, :] < head_dim

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for neighbor_idx in range(kernel_size):
        offset = neighbor_idx - half_kernel
        query_rows = row_offsets - offset * dilation
        query_valid = row_valid & (query_rows >= 0) & (query_rows < seq_len)
        safe_query = tl.minimum(tl.maximum(query_rows, 0), seq_len - 1)

        grad_qk_ptrs = (
            grad_qk_ptr
            + batch_id * grad_qk_batch_stride
            + head_id * grad_qk_head_stride
            + safe_query * grad_qk_seq_stride
            + neighbor_idx * grad_qk_neighbor_stride
        )
        gk = tl.load(grad_qk_ptrs, mask=query_valid, other=0.0)

        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + safe_query[:, None] * q_seq_stride
            + safe_col[None, :] * q_dim_stride
        )
        q_mask = query_valid[:, None] & dim_valid
        q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)
        acc += gk[:, None] * q_chunk

    acc = acc * scale

    grad_k_ptrs = (
        grad_K_ptr
        + batch_id * grad_k_batch_stride
        + head_id * grad_k_head_stride
        + safe_row[:, None] * grad_k_seq_stride
        + safe_col[None, :] * grad_k_dim_stride
    )
    valid_mask = row_valid[:, None] & dim_valid
    tl.store(grad_k_ptrs, acc, mask=valid_mask)


@triton.jit
def _sparse_neighborhood_grad_qkv_fused_kernel(
    grad_qk_ptr,
    attn_ptr,
    Q_ptr,
    K_ptr,
    grad_output_ptr,
    grad_Q_ptr,
    grad_K_ptr,
    grad_V_ptr,
    grad_qk_batch_stride,
    grad_qk_head_stride,
    grad_qk_seq_stride,
    grad_qk_neighbor_stride,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_neighbor_stride,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    grad_q_batch_stride,
    grad_q_head_stride,
    grad_q_seq_stride,
    grad_q_dim_stride,
    grad_k_batch_stride,
    grad_k_head_stride,
    grad_k_seq_stride,
    grad_k_dim_stride,
    grad_v_batch_stride,
    grad_v_head_stride,
    grad_v_seq_stride,
    grad_v_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused sparse-band grad_Q, grad_K, grad_V in one 3D-grid kernel."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)
    half_kernel = kernel_size // 2

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    safe_row = tl.minimum(row_offsets, seq_len - 1)
    safe_col = tl.minimum(col_offsets, head_dim - 1)
    row_valid = row_offsets < seq_len
    dim_valid = col_offsets[None, :] < head_dim
    neighbor_cols = tl.arange(0, kernel_size)[None, :]
    key_cols_all = row_offsets[:, None] + (neighbor_cols - half_kernel) * dilation
    key_valid_all = row_valid[:, None] & (key_cols_all >= 0) & (key_cols_all < seq_len)

    grad_qk_q_ptrs = (
        grad_qk_ptr
        + batch_id * grad_qk_batch_stride
        + head_id * grad_qk_head_stride
        + safe_row[:, None] * grad_qk_seq_stride
        + neighbor_cols * grad_qk_neighbor_stride
    )
    gq_tile = tl.load(grad_qk_q_ptrs, mask=key_valid_all, other=0.0)

    acc_q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for neighbor_idx in range(kernel_size):
        offset = neighbor_idx - half_kernel
        key_cols = row_offsets + offset * dilation
        key_valid = row_valid & (key_cols >= 0) & (key_cols < seq_len)
        safe_key = tl.minimum(tl.maximum(key_cols, 0), seq_len - 1)

        query_rows = row_offsets - offset * dilation
        query_valid = row_valid & (query_rows >= 0) & (query_rows < seq_len)
        safe_query = tl.minimum(tl.maximum(query_rows, 0), seq_len - 1)

        col_select = neighbor_cols == neighbor_idx
        gq = tl.sum(gq_tile * col_select, axis=1)

        grad_qk_k_ptrs = (
            grad_qk_ptr
            + batch_id * grad_qk_batch_stride
            + head_id * grad_qk_head_stride
            + safe_query * grad_qk_seq_stride
            + neighbor_idx * grad_qk_neighbor_stride
        )
        attn_ptrs = (
            attn_ptr
            + batch_id * attn_batch_stride
            + head_id * attn_head_stride
            + safe_query * attn_seq_stride
            + neighbor_idx * attn_neighbor_stride
        )

        gk = tl.load(grad_qk_k_ptrs, mask=query_valid, other=0.0)
        attn_val = tl.load(attn_ptrs, mask=query_valid, other=0.0)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + safe_key[:, None] * k_seq_stride
            + safe_col[None, :] * k_dim_stride
        )
        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + safe_query[:, None] * q_seq_stride
            + safe_col[None, :] * q_dim_stride
        )
        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + safe_query[:, None] * grad_out_seq_stride
            + safe_col[None, :] * grad_out_dim_stride
        )
        gather_mask_k = key_valid[:, None] & dim_valid
        gather_mask_q = query_valid[:, None] & dim_valid

        k_chunk = tl.load(k_ptrs, mask=gather_mask_k, other=0.0)
        q_chunk = tl.load(q_ptrs, mask=gather_mask_q, other=0.0)
        grad_out_chunk = tl.load(grad_out_ptrs, mask=gather_mask_q, other=0.0)

        acc_q += gq[:, None] * k_chunk
        acc_k += gk[:, None] * q_chunk
        acc_v += attn_val[:, None] * grad_out_chunk

    acc_q = acc_q * scale
    acc_k = acc_k * scale

    valid_mask = row_valid[:, None] & dim_valid
    grad_q_ptrs = (
        grad_Q_ptr
        + batch_id * grad_q_batch_stride
        + head_id * grad_q_head_stride
        + safe_row[:, None] * grad_q_seq_stride
        + safe_col[None, :] * grad_q_dim_stride
    )
    grad_k_ptrs = (
        grad_K_ptr
        + batch_id * grad_k_batch_stride
        + head_id * grad_k_head_stride
        + safe_row[:, None] * grad_k_seq_stride
        + safe_col[None, :] * grad_k_dim_stride
    )
    grad_v_ptrs = (
        grad_V_ptr
        + batch_id * grad_v_batch_stride
        + head_id * grad_v_head_stride
        + safe_row[:, None] * grad_v_seq_stride
        + safe_col[None, :] * grad_v_dim_stride
    )
    tl.store(grad_q_ptrs, acc_q, mask=valid_mask)
    tl.store(grad_k_ptrs, acc_k, mask=valid_mask)
    tl.store(grad_v_ptrs, acc_v, mask=valid_mask)


# ---------------------------------------------------------------------------
# Dense backward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_expand_to_dense_kernel(
    sparse_attn_ptr,
    dense_attn_ptr,
    sparse_batch_stride,
    sparse_head_stride,
    sparse_seq_stride,
    sparse_neighbor_stride,
    dense_batch_stride,
    dense_head_stride,
    dense_seq_stride,
    dense_seq2_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Scatter sparse neighborhood attention weights into a dense [S, S] layout."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    half_kernel = kernel_size // 2

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    safe_row = tl.minimum(row_offsets, seq_len - 1)
    row_valid = row_offsets < seq_len
    neighbor_cols = tl.arange(0, kernel_size)[None, :]
    key_cols_all = row_offsets[:, None] + (neighbor_cols - half_kernel) * dilation
    key_valid_all = row_valid[:, None] & (key_cols_all >= 0) & (key_cols_all < seq_len)

    sparse_ptrs = (
        sparse_attn_ptr
        + batch_id * sparse_batch_stride
        + head_id * sparse_head_stride
        + safe_row[:, None] * sparse_seq_stride
        + neighbor_cols * sparse_neighbor_stride
    )
    attn_tile = tl.load(sparse_ptrs, mask=key_valid_all, other=0.0)

    for neighbor_idx in range(kernel_size):
        offset = neighbor_idx - half_kernel
        key_cols = row_offsets + offset * dilation
        key_valid = row_valid & (key_cols >= 0) & (key_cols < seq_len)
        safe_key = tl.minimum(tl.maximum(key_cols, 0), seq_len - 1)

        col_select = neighbor_cols == neighbor_idx
        attn_val = tl.sum(attn_tile * col_select, axis=1)

        dense_ptrs = (
            dense_attn_ptr
            + batch_id * dense_batch_stride
            + head_id * dense_head_stride
            + safe_row * dense_seq_stride
            + safe_key * dense_seq2_stride
        )
        tl.store(dense_ptrs, attn_val, mask=key_valid)


@triton.jit
def _dense_neighborhood_grad_attn_kernel(
    grad_output_ptr,
    V_ptr,
    grad_attn_ptr,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    v_dim_stride,
    grad_attn_batch_stride,
    grad_attn_head_stride,
    grad_attn_seq_stride,
    grad_attn_seq2_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """grad_attn = grad_output @ V^T using tl.dot."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, head_dim, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < head_dim

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + row_offsets[:, None] * grad_out_seq_stride
            + k_offsets[None, :] * grad_out_dim_stride
        )
        grad_out_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0).to(tl.float32)

        v_ptrs = (
            V_ptr
            + batch_id * v_batch_stride
            + head_id * v_head_stride
            + col_offsets[None, :] * v_seq_stride
            + k_offsets[:, None] * v_dim_stride
        )
        v_mask = (col_offsets[None, :] < seq_len) & k_mask[:, None]
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        acc += tl.dot(grad_out_chunk, v_chunk)

    grad_attn_ptrs = (
        grad_attn_ptr
        + batch_id * grad_attn_batch_stride
        + head_id * grad_attn_head_stride
        + row_offsets[:, None] * grad_attn_seq_stride
        + col_offsets[None, :] * grad_attn_seq2_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < seq_len)
    tl.store(grad_attn_ptrs, acc, mask=valid_mask)


@triton.jit
def _dense_neighborhood_grad_qk_fused_kernel(
    grad_qk_ptr,
    K_ptr,
    Q_ptr,
    grad_Q_ptr,
    grad_K_ptr,
    grad_qk_batch_stride,
    grad_qk_head_stride,
    grad_qk_seq_stride,
    grad_qk_seq2_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    grad_q_batch_stride,
    grad_q_head_stride,
    grad_q_seq_stride,
    grad_q_dim_stride,
    grad_k_batch_stride,
    grad_k_head_stride,
    grad_k_seq_stride,
    grad_k_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused grad_Q and grad_K from dense grad_qk in one pass."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    acc_q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < seq_len

        grad_qk_ptrs = (
            grad_qk_ptr
            + batch_id * grad_qk_batch_stride
            + head_id * grad_qk_head_stride
            + row_offsets[:, None] * grad_qk_seq_stride
            + k_offsets[None, :] * grad_qk_seq2_stride
        )
        grad_qk_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_qk_chunk = tl.load(grad_qk_ptrs, mask=grad_qk_mask, other=0.0).to(tl.float32)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + k_offsets[:, None] * k_seq_stride
            + col_offsets[None, :] * k_dim_stride
        )
        k_mask_2d = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        k_chunk = tl.load(k_ptrs, mask=k_mask_2d, other=0.0).to(tl.float32)
        acc_q += tl.dot(grad_qk_chunk, k_chunk)

        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + k_offsets[:, None] * q_seq_stride
            + col_offsets[None, :] * q_dim_stride
        )
        q_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        grad_qk_T_ptrs = (
            grad_qk_ptr
            + batch_id * grad_qk_batch_stride
            + head_id * grad_qk_head_stride
            + row_offsets[:, None] * grad_qk_seq2_stride
            + k_offsets[None, :] * grad_qk_seq_stride
        )
        grad_qk_T_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_qk_T_chunk = tl.load(grad_qk_T_ptrs, mask=grad_qk_T_mask, other=0.0).to(tl.float32)
        acc_k += tl.dot(grad_qk_T_chunk, q_chunk)

    acc_q = acc_q * scale
    acc_k = acc_k * scale

    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    grad_q_ptrs = (
        grad_Q_ptr
        + batch_id * grad_q_batch_stride
        + head_id * grad_q_head_stride
        + row_offsets[:, None] * grad_q_seq_stride
        + col_offsets[None, :] * grad_q_dim_stride
    )
    grad_k_ptrs = (
        grad_K_ptr
        + batch_id * grad_k_batch_stride
        + head_id * grad_k_head_stride
        + row_offsets[:, None] * grad_k_seq_stride
        + col_offsets[None, :] * grad_k_dim_stride
    )
    tl.store(grad_q_ptrs, acc_q, mask=valid_mask)
    tl.store(grad_k_ptrs, acc_k, mask=valid_mask)


@triton.jit
def _dense_neighborhood_grad_qk_fused_persistent_kernel(
    grad_qk_ptr,
    K_ptr,
    Q_ptr,
    grad_Q_ptr,
    grad_K_ptr,
    grad_qk_batch_stride,
    grad_qk_head_stride,
    grad_qk_seq_stride,
    grad_qk_seq2_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    grad_q_batch_stride,
    grad_q_head_stride,
    grad_q_seq_stride,
    grad_q_dim_stride,
    grad_k_batch_stride,
    grad_k_head_stride,
    grad_k_seq_stride,
    grad_k_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_batch_heads = batch_size * num_heads
    num_m_tiles = tl.cdiv(seq_len, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(head_dim, BLOCK_SIZE_N)
    total_tiles = num_batch_heads * num_m_tiles * num_n_tiles

    for global_tile_id in tl.range(pid, total_tiles, num_programs):
        batch_head_id = global_tile_id // (num_m_tiles * num_n_tiles)
        rem = global_tile_id % (num_m_tiles * num_n_tiles)
        tile_m = rem // num_n_tiles
        tile_n = rem % num_n_tiles

        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads

        row_start = tile_m * BLOCK_SIZE_M
        col_start = tile_n * BLOCK_SIZE_N
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

        acc_q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_k = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_start in range(0, seq_len, BLOCK_SIZE_K):
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < seq_len

            grad_qk_ptrs = (
                grad_qk_ptr
                + batch_id * grad_qk_batch_stride
                + head_id * grad_qk_head_stride
                + row_offsets[:, None] * grad_qk_seq_stride
                + k_offsets[None, :] * grad_qk_seq2_stride
            )
            grad_qk_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
            grad_qk_chunk = tl.load(grad_qk_ptrs, mask=grad_qk_mask, other=0.0)

            k_ptrs = (
                K_ptr
                + batch_id * k_batch_stride
                + head_id * k_head_stride
                + k_offsets[:, None] * k_seq_stride
                + col_offsets[None, :] * k_dim_stride
            )
            k_mask_2d = k_mask[:, None] & (col_offsets[None, :] < head_dim)
            k_chunk = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)
            acc_q += tl.dot(grad_qk_chunk, k_chunk)

            q_ptrs = (
                Q_ptr
                + batch_id * q_batch_stride
                + head_id * q_head_stride
                + k_offsets[:, None] * q_seq_stride
                + col_offsets[None, :] * q_dim_stride
            )
            q_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
            q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)

            grad_qk_T_ptrs = (
                grad_qk_ptr
                + batch_id * grad_qk_batch_stride
                + head_id * grad_qk_head_stride
                + row_offsets[:, None] * grad_qk_seq2_stride
                + k_offsets[None, :] * grad_qk_seq_stride
            )
            grad_qk_T_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
            grad_qk_T_chunk = tl.load(grad_qk_T_ptrs, mask=grad_qk_T_mask, other=0.0)
            acc_k += tl.dot(grad_qk_T_chunk, q_chunk)

        acc_q = acc_q * scale
        acc_k = acc_k * scale

        valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
        grad_q_ptrs = (
            grad_Q_ptr
            + batch_id * grad_q_batch_stride
            + head_id * grad_q_head_stride
            + row_offsets[:, None] * grad_q_seq_stride
            + col_offsets[None, :] * grad_q_dim_stride
        )
        grad_k_ptrs = (
            grad_K_ptr
            + batch_id * grad_k_batch_stride
            + head_id * grad_k_head_stride
            + row_offsets[:, None] * grad_k_seq_stride
            + col_offsets[None, :] * grad_k_dim_stride
        )
        tl.store(grad_q_ptrs, acc_q, mask=valid_mask)
        tl.store(grad_k_ptrs, acc_k, mask=valid_mask)


@triton.jit
def _dense_neighborhood_grad_v_kernel(
    Attn_ptr,
    grad_output_ptr,
    grad_V_ptr,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_seq2_stride,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    grad_v_batch_stride,
    grad_v_head_stride,
    grad_v_seq_stride,
    grad_v_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """grad_V = Attn^T @ grad_output using tl.dot."""
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < seq_len

        attn_ptrs = (
            Attn_ptr
            + batch_id * attn_batch_stride
            + head_id * attn_head_stride
            + k_offsets[:, None] * attn_seq_stride
            + row_offsets[None, :] * attn_seq2_stride
        )
        attn_mask = k_mask[:, None] & (row_offsets[None, :] < seq_len)
        attn_chunk = tl.load(attn_ptrs, mask=attn_mask, other=0.0).to(tl.float32)

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + k_offsets[:, None] * grad_out_seq_stride
            + col_offsets[None, :] * grad_out_dim_stride
        )
        grad_out_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0).to(tl.float32)

        acc += tl.dot(tl.trans(attn_chunk), grad_out_chunk)

    grad_v_ptrs = (
        grad_V_ptr
        + batch_id * grad_v_batch_stride
        + head_id * grad_v_head_stride
        + row_offsets[:, None] * grad_v_seq_stride
        + col_offsets[None, :] * grad_v_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(grad_v_ptrs, acc, mask=valid_mask)


# ---------------------------------------------------------------------------
# Tiling and memory helpers
# ---------------------------------------------------------------------------


def _get_dense_backward_block_k(seq_len: int, full_batch: bool = False) -> int:
    block_k = min(128, seq_len, triton.next_power_of_2(seq_len))
    if full_batch and seq_len >= 4096:
        block_k = min(block_k, 64)
    return max(16, block_k)


def _scores_chunk_byte_limit() -> int:
    """Cap one [chunk_b, H, S, S] score tile to a fraction of device memory."""
    total_bytes = get_total_gpu_memory() * (1024**3)
    return max(256 * 1024**2, int(total_bytes * 0.08))


def _attention_batch_chunk_size(batch_size: int, num_heads: int, seq_len: int, kernel_size: int) -> int:
    per_batch_bytes = num_heads * seq_len * kernel_size * 4
    if per_batch_bytes == 0:
        return batch_size
    max_chunk = max(1, _scores_chunk_byte_limit() // per_batch_bytes)
    return min(batch_size, max_chunk)


def _get_attention_block_sizes(seq_len: int, head_dim: int) -> tuple[int, int, int]:
    """UB-aware tile sizes for neighborhood attention kernels."""
    memory_multiplier = 3.0
    block_m = min(64, seq_len, triton.next_power_of_2(seq_len))
    block_n = min(64, head_dim, triton.next_power_of_2(head_dim))
    block_k = min(32, max(16, triton.next_power_of_2(head_dim)))

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.85,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((block_m, block_k), (block_n, block_k)),
        tiling_dims=(0, 0),
    )
    if tile_shapes and len(tile_shapes) == 2:
        block_m = min(block_m, tile_shapes[0][0])
        block_k = min(block_k, tile_shapes[0][1], tile_shapes[1][1])
        block_n = min(block_n, tile_shapes[1][0])

    return max(8, block_m), max(8, block_n), max(16, block_k)


def _sparse_backward_launch_block_n(seq_len: int, block_n: int) -> int:
    """Head-dim tile width for sparse backward q/k/v launches."""
    if seq_len >= 4096:
        return block_n
    if seq_len >= 512:
        return min(block_n, 64)
    return block_n


def _get_sparse_backward_block_sizes(
    seq_len: int, head_dim: int, batch_size: int = 1, num_heads: int = 1
) -> tuple[int, int]:
    """Tile sizes tuned for sparse-band backward on Ascend."""
    block_m, block_n, _ = _get_attention_block_sizes(seq_len, head_dim)
    if seq_len >= 4096:
        block_m = min(block_m, 32)
        block_n = min(128, head_dim, triton.next_power_of_2(head_dim))
    elif seq_len >= 2048:
        block_m = min(block_m, 16)
        block_n = min(128, head_dim, triton.next_power_of_2(head_dim))
    elif seq_len >= 512:
        block_m = min(block_m, 32)
        block_n = min(128, head_dim, triton.next_power_of_2(head_dim))

    launch_block_n = _sparse_backward_launch_block_n(seq_len, block_n)
    total_bh = batch_size * num_heads
    while total_bh * triton.cdiv(seq_len, block_m) * triton.cdiv(head_dim, launch_block_n) > _MAX_GRID_PROGRAMS:
        block_m = min(seq_len, block_m * 2)
        if block_m >= seq_len:
            break

    return max(8, block_m), max(8, block_n)


def _sparse_band_dense_fast_path_available(batch_size: int, num_heads: int, seq_len: int, element_size: int) -> bool:
    """True when sparse-band backward can use the non-chunked dense tl.dot path."""
    if seq_len > 2048:
        return False
    total_bh = batch_size * num_heads
    workspace_bytes = total_bh * seq_len * seq_len * 3 * element_size
    return workspace_bytes <= _dense_backward_byte_limit() and _dense_backward_grid_fits(batch_size, num_heads, seq_len)


def _expand_sparse_attn_to_dense(
    attn_sparse: torch.Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    kernel_size: int,
    dilation: int,
    block_m: int,
) -> torch.Tensor:
    compute_dtype = attn_sparse.dtype
    attn_dense = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=attn_sparse.device, dtype=compute_dtype)
    _sparse_attn_expand_to_dense_kernel[(batch_size * num_heads, triton.cdiv(seq_len, block_m))](
        attn_sparse,
        attn_dense,
        attn_sparse.stride(0),
        attn_sparse.stride(1),
        attn_sparse.stride(2),
        attn_sparse.stride(3),
        attn_dense.stride(0),
        attn_dense.stride(1),
        attn_dense.stride(2),
        attn_dense.stride(3),
        batch_size,
        num_heads,
        seq_len,
        kernel_size,
        dilation,
        block_m,
    )
    return attn_dense


def _dense_backward_byte_limit() -> int:
    total_bytes = get_total_gpu_memory() * (1024**3)
    return max(512 * 1024**2, int(total_bytes * 0.18))


def _dense_backward_batch_head_chunk(
    batch_size: int, num_heads: int, seq_len: int, use_sparse_band: bool = False, grad_qk_elem_size: int = 4
) -> int:
    """Limit dense backward workspace by processing a few batch-heads at a time."""
    per_bh_bytes = seq_len * seq_len * grad_qk_elem_size * (3 if use_sparse_band else 2)
    if per_bh_bytes == 0:
        return batch_size * num_heads
    total_bh = batch_size * num_heads
    max_chunk = max(1, _dense_backward_byte_limit() // per_bh_bytes)
    max_chunk = min(max_chunk, total_bh if use_sparse_band else 16, total_bh)
    while max_chunk > 1:
        block_m = (
            min(64, seq_len, triton.next_power_of_2(seq_len))
            if use_sparse_band
            else _get_dense_block_m(seq_len, max_chunk, 1)
        )
        grid_programs = max_chunk * triton.cdiv(seq_len, block_m)
        if not use_sparse_band:
            grid_programs *= triton.cdiv(seq_len, block_m)
        elif seq_len >= 4096:
            grid_programs *= triton.cdiv(seq_len, block_m)
        if grid_programs <= _MAX_GRID_PROGRAMS:
            break
        max_chunk -= 1
    return max(1, max_chunk)


def _get_dense_block_m(seq_len: int, batch_size: int, num_heads: int) -> int:
    """Pick a row tile size that keeps the dense attention grid within Ascend limits."""
    block_m = min(64, seq_len, triton.next_power_of_2(seq_len))
    batch_heads = batch_size * num_heads
    while batch_heads * triton.cdiv(seq_len, block_m) ** 2 > _MAX_GRID_PROGRAMS:
        block_m = min(seq_len, block_m * 2)
        if block_m >= seq_len:
            break
    return max(8, block_m)


def _dense_backward_grid_fits(batch_size: int, num_heads: int, seq_len: int) -> bool:
    block_m = _get_dense_block_m(seq_len, batch_size, num_heads)
    total_bh = batch_size * num_heads
    return total_bh * triton.cdiv(seq_len, block_m) ** 2 <= _MAX_GRID_PROGRAMS


def _reshape_batch_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten [B, H, ...] into [B*H, ...] for chunked dense backward."""
    batch_size, num_heads = tensor.shape[0], tensor.shape[1]
    return tensor.reshape(batch_size * num_heads, *tensor.shape[2:])


def _get_softmax_bwd_params(seq_len: int) -> tuple[int, int, bool]:
    """Match softmax forward tiling for backward reuse."""
    max_fused_block_size = 8192
    block_size = triton.next_power_of_2(seq_len)
    block_size = min(block_size, max_fused_block_size)
    if seq_len <= block_size:
        rows_per_block = min(max_fused_block_size // block_size, 32)
        rows_per_block = triton.next_power_of_2(rows_per_block)
        return block_size, rows_per_block, False
    return block_size, 1, True


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _launch_sparse_qk_softmax_kernel(
    query,
    key,
    attn_weights,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    kernel_size,
    dilation,
    block_m,
    block_k,
    num_cores,
):
    _sparse_neighborhood_qk_softmax_kernel[(num_cores,)](
        query,
        key,
        attn_weights,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        attn_weights.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        kernel_size,
        dilation,
        block_m,
        block_k,
    )


def _launch_dense_grad_qk_fused_kernel(
    grad_qk,
    key,
    query,
    grad_query,
    grad_key,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    block_m,
    block_n,
    block_k,
    head_stride_q=0,
    head_stride_k=0,
    head_stride_gq=0,
    head_stride_gk=0,
    head_stride_gqk=0,
):
    total_bh = batch_size * num_heads
    grid = (total_bh, triton.cdiv(seq_len, block_m), triton.cdiv(head_dim, block_n))
    kernel_args = (
        grad_qk,
        key,
        query,
        grad_query,
        grad_key,
        grad_qk.stride(0),
        head_stride_gqk if head_stride_gqk else grad_qk.stride(1),
        grad_qk.stride(-2),
        grad_qk.stride(-1),
        key.stride(0),
        head_stride_k if head_stride_k else key.stride(1),
        key.stride(-2),
        key.stride(-1),
        query.stride(0),
        head_stride_q if head_stride_q else query.stride(1),
        query.stride(-2),
        query.stride(-1),
        grad_query.stride(0),
        head_stride_gq if head_stride_gq else grad_query.stride(1),
        grad_query.stride(-2),
        grad_query.stride(-1),
        grad_key.stride(0),
        head_stride_gk if head_stride_gk else grad_key.stride(1),
        grad_key.stride(-2),
        grad_key.stride(-1),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        block_m,
        block_n,
        block_k,
    )
    use_persistent = seq_len >= 1024 and grad_qk.dtype in (torch.float32, torch.float64)
    if use_persistent:
        num_cores = get_npu_core_count()
        _dense_neighborhood_grad_qk_fused_persistent_kernel[(num_cores,)](*kernel_args)
    else:
        _dense_neighborhood_grad_qk_fused_kernel[grid](*kernel_args)


def _launch_sparse_grad_qk_kernels(
    grad_qk,
    query,
    key,
    grad_query,
    grad_key,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    kernel_size,
    dilation,
    block_m,
    block_n,
):
    launch_block_n = _sparse_backward_launch_block_n(seq_len, block_n)
    grid = (
        batch_size * num_heads,
        triton.cdiv(seq_len, block_m),
        triton.cdiv(head_dim, launch_block_n),
    )
    qk_common = (
        grad_qk.stride(0),
        grad_qk.stride(1),
        grad_qk.stride(2),
        grad_qk.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        kernel_size,
        block_m,
        launch_block_n,
    )
    _sparse_neighborhood_grad_q_kernel[grid](
        grad_qk,
        key,
        grad_query,
        *qk_common[:4],
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        grad_query.stride(0),
        grad_query.stride(1),
        grad_query.stride(2),
        grad_query.stride(3),
        *qk_common[4:],
        dilation,
    )
    _sparse_neighborhood_grad_k_kernel[grid](
        grad_qk,
        query,
        grad_key,
        *qk_common[:4],
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        grad_key.stride(0),
        grad_key.stride(1),
        grad_key.stride(2),
        grad_key.stride(3),
        *qk_common[4:],
        dilation,
    )


def _launch_sparse_grad_attn_softmax_bwd_kernel(
    grad_output,
    value,
    attn_weights,
    grad_qk,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    kernel_size,
    dilation,
    block_m,
    block_n,
):
    grid = (
        batch_size * num_heads,
        triton.cdiv(seq_len, block_m),
    )
    _sparse_neighborhood_grad_attn_softmax_bwd_kernel[grid](
        grad_output,
        value,
        attn_weights,
        grad_qk,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        attn_weights.stride(3),
        grad_qk.stride(0),
        grad_qk.stride(1),
        grad_qk.stride(2),
        grad_qk.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        kernel_size,
        dilation,
        block_m,
        block_n,
    )


def _launch_sparse_band_grad_v_kernel(
    attn_weights,
    grad_output,
    grad_value,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    kernel_size,
    dilation,
    block_m,
    block_n,
):
    grid = (
        batch_size * num_heads,
        triton.cdiv(seq_len, block_m),
        triton.cdiv(head_dim, block_n),
    )
    _fused_neighborhood_attention_grad_v_kernel[grid](
        attn_weights,
        grad_output,
        grad_value,
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        attn_weights.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_value.stride(0),
        grad_value.stride(1),
        grad_value.stride(2),
        grad_value.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        kernel_size,
        dilation,
        block_m,
        block_n,
    )


def _launch_sparse_grad_qkv_kernels(
    grad_qk,
    attn_weights,
    query,
    key,
    grad_output,
    grad_query,
    grad_key,
    grad_value,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
    kernel_size,
    dilation,
    block_m,
    block_n,
):
    if seq_len >= 512:
        launch_block_n = _sparse_backward_launch_block_n(seq_len, block_n)
        _launch_sparse_grad_qk_kernels(
            grad_qk,
            query,
            key,
            grad_query,
            grad_key,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            kernel_size,
            dilation,
            block_m,
            block_n,
        )
        _launch_sparse_band_grad_v_kernel(
            attn_weights,
            grad_output,
            grad_value,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            kernel_size,
            dilation,
            block_m,
            launch_block_n,
        )
        return

    grid = (
        batch_size * num_heads,
        triton.cdiv(seq_len, block_m),
        triton.cdiv(head_dim, block_n),
    )
    kernel_args = (
        grad_qk,
        attn_weights,
        query,
        key,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        grad_qk.stride(0),
        grad_qk.stride(1),
        grad_qk.stride(2),
        grad_qk.stride(3),
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        attn_weights.stride(3),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_query.stride(0),
        grad_query.stride(1),
        grad_query.stride(2),
        grad_query.stride(3),
        grad_key.stride(0),
        grad_key.stride(1),
        grad_key.stride(2),
        grad_key.stride(3),
        grad_value.stride(0),
        grad_value.stride(1),
        grad_value.stride(2),
        grad_value.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        kernel_size,
        dilation,
        block_m,
        block_n,
    )
    _sparse_neighborhood_grad_qkv_fused_kernel[grid](*kernel_args)


# ---------------------------------------------------------------------------
# Backward orchestration
# ---------------------------------------------------------------------------


def _dense_neighborhood_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_weights: torch.Tensor,
    scale: float,
    kernel_size: int,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = query.shape
    compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype

    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    use_sparse_band = attn_weights.shape[-1] == kernel_size
    grad_qk_dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else compute_dtype
    grad_qk_elem_size = query.element_size() if use_sparse_band else 4
    workspace_multiplier = 3 * grad_qk_elem_size if use_sparse_band else 8

    total_bh = batch_size * num_heads
    if (
        total_bh * seq_len * seq_len * workspace_multiplier <= _dense_backward_byte_limit()
        and _dense_backward_grid_fits(batch_size, num_heads, seq_len)
    ):
        block_m = _get_dense_block_m(seq_len, batch_size, num_heads)
        block_n = min(64, head_dim, triton.next_power_of_2(head_dim))
        block_k = _get_dense_backward_block_k(seq_len, full_batch=True)

        if attn_weights.shape[-1] == seq_len and not use_sparse_band:
            attn_dense = attn_weights
        else:
            attn_dense = _expand_sparse_attn_to_dense(
                attn_weights,
                batch_size,
                num_heads,
                seq_len,
                kernel_size,
                dilation,
                block_m,
            )

        if use_sparse_band:
            grid_qkv = (total_bh, triton.cdiv(seq_len, block_m), triton.cdiv(head_dim, block_n))
            _dense_neighborhood_grad_v_kernel[grid_qkv](
                attn_dense,
                grad_output,
                grad_value,
                attn_dense.stride(0),
                attn_dense.stride(1),
                attn_dense.stride(2),
                attn_dense.stride(3),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                grad_output.stride(3),
                grad_value.stride(0),
                grad_value.stride(1),
                grad_value.stride(2),
                grad_value.stride(3),
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                block_m,
                block_n,
                block_k,
            )

        grad_attn = torch.empty_like(attn_dense)
        grid_attn = (total_bh, triton.cdiv(seq_len, block_m), triton.cdiv(seq_len, block_m))
        _dense_neighborhood_grad_attn_kernel[grid_attn](
            grad_output,
            value,
            grad_attn,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            grad_attn.stride(0),
            grad_attn.stride(1),
            grad_attn.stride(2),
            grad_attn.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            block_m,
            block_m,
            block_k,
        )
        block_size, rows_per_block, multi_block_launch = _get_softmax_bwd_params(seq_len)
        grad_qk = _softmax_backward(
            grad_attn.view(total_bh * seq_len, seq_len),
            attn_dense.view(total_bh * seq_len, seq_len),
            block_size,
            rows_per_block,
            multi_block_launch,
        ).view(batch_size, num_heads, seq_len, seq_len)

        _launch_dense_grad_qk_fused_kernel(
            grad_qk,
            key,
            query,
            grad_query,
            grad_key,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            block_m,
            block_n,
            block_k,
        )
        if not use_sparse_band:
            grid_qkv = (total_bh, triton.cdiv(seq_len, block_m), triton.cdiv(head_dim, block_n))
            _dense_neighborhood_grad_v_kernel[grid_qkv](
                attn_dense,
                grad_output,
                grad_value,
                attn_dense.stride(0),
                attn_dense.stride(1),
                attn_dense.stride(2),
                attn_dense.stride(3),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                grad_output.stride(3),
                grad_value.stride(0),
                grad_value.stride(1),
                grad_value.stride(2),
                grad_value.stride(3),
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                block_m,
                block_n,
                block_k,
            )
        return grad_query, grad_key, grad_value

    bh_chunk = _dense_backward_batch_head_chunk(batch_size, num_heads, seq_len, use_sparse_band, grad_qk_elem_size)
    block_m = _get_dense_block_m(seq_len, bh_chunk, 1)
    if use_sparse_band and seq_len >= 4096:
        block_m = min(64, seq_len, triton.next_power_of_2(seq_len))
    block_n = min(64, head_dim, triton.next_power_of_2(head_dim))
    block_k = _get_dense_backward_block_k(seq_len)

    attn_buf_dtype = attn_weights.dtype if use_sparse_band else compute_dtype
    attn_buf = torch.zeros(1, seq_len, seq_len, device=query.device, dtype=attn_buf_dtype)
    grad_qk_buf = torch.empty(1, seq_len, seq_len, device=query.device, dtype=grad_qk_dtype)
    grad_attn_buf = torch.zeros_like(attn_buf)

    block_size, rows_per_block, multi_block_launch = _get_softmax_bwd_params(seq_len)

    query_flat = _reshape_batch_heads(query)
    key_flat = _reshape_batch_heads(key)
    value_flat = _reshape_batch_heads(value)
    grad_output_flat = _reshape_batch_heads(grad_output)
    grad_query_flat = _reshape_batch_heads(grad_query)
    grad_key_flat = _reshape_batch_heads(grad_key)
    grad_value_flat = _reshape_batch_heads(grad_value)
    attn_sparse_flat = _reshape_batch_heads(attn_weights)
    attn_flat = attn_sparse_flat if attn_weights.shape[-1] == seq_len else None

    for bh_start in range(0, total_bh, bh_chunk):
        bh_end = min(bh_start + bh_chunk, total_bh)
        chunk_bh = bh_end - bh_start

        grad_out_rows = grad_output_flat[bh_start:bh_end]
        key_rows = key_flat[bh_start:bh_end]
        query_rows = query_flat[bh_start:bh_end]
        value_rows = value_flat[bh_start:bh_end]
        grad_q_rows = grad_query_flat[bh_start:bh_end]
        grad_k_rows = grad_key_flat[bh_start:bh_end]
        if attn_flat is not None:
            attn_rows = attn_flat[bh_start:bh_end]
        else:
            attn_rows = attn_sparse_flat[bh_start:bh_end]

        if grad_qk_buf.shape[0] < chunk_bh:
            grad_qk_buf = torch.zeros(chunk_bh, seq_len, seq_len, device=query.device, dtype=grad_qk_dtype)
        else:
            grad_qk_buf = grad_qk_buf[:chunk_bh]

        if use_sparse_band:
            if attn_buf.shape[0] < chunk_bh:
                attn_buf = torch.zeros(chunk_bh, seq_len, seq_len, device=query.device, dtype=attn_weights.dtype)
                grad_attn_buf = torch.empty_like(attn_buf)
            else:
                attn_buf = attn_buf[:chunk_bh]
                grad_attn_buf = grad_attn_buf[:chunk_bh]
            attn_buf.zero_()
            expand_grid = (chunk_bh, triton.cdiv(seq_len, block_m))
            _sparse_attn_expand_to_dense_kernel[expand_grid](
                attn_rows,
                attn_buf,
                attn_rows.stride(0),
                0,
                attn_rows.stride(1),
                attn_rows.stride(2),
                attn_buf.stride(0),
                0,
                attn_buf.stride(1),
                attn_buf.stride(2),
                chunk_bh,
                1,
                seq_len,
                kernel_size,
                dilation,
                block_m,
            )
        else:
            if attn_buf.shape[0] < chunk_bh:
                attn_buf = torch.zeros(chunk_bh, seq_len, seq_len, device=query.device, dtype=compute_dtype)
                grad_attn_buf = torch.empty_like(attn_buf)
            else:
                attn_buf = attn_buf[:chunk_bh]
                grad_attn_buf = grad_attn_buf[:chunk_bh]
            attn_buf.zero_()

            if attn_weights.shape[-1] == seq_len:
                attn_buf.copy_(attn_rows)
            else:
                expand_grid = (chunk_bh, triton.cdiv(seq_len, block_m))
                _sparse_attn_expand_to_dense_kernel[expand_grid](
                    attn_rows,
                    attn_buf,
                    attn_rows.stride(0),
                    0,
                    attn_rows.stride(1),
                    attn_rows.stride(2),
                    attn_buf.stride(0),
                    0,
                    attn_buf.stride(1),
                    attn_buf.stride(2),
                    chunk_bh,
                    1,
                    seq_len,
                    kernel_size,
                    dilation,
                    block_m,
                )

        grid_attn = (chunk_bh, triton.cdiv(seq_len, block_m), triton.cdiv(seq_len, block_m))
        _dense_neighborhood_grad_attn_kernel[grid_attn](
            grad_out_rows,
            value_rows,
            grad_attn_buf,
            grad_out_rows.stride(0),
            0,
            grad_out_rows.stride(1),
            grad_out_rows.stride(2),
            value_rows.stride(0),
            0,
            value_rows.stride(1),
            value_rows.stride(2),
            grad_attn_buf.stride(0),
            0,
            grad_attn_buf.stride(1),
            grad_attn_buf.stride(2),
            chunk_bh,
            1,
            seq_len,
            head_dim,
            block_m,
            block_m,
            block_k,
        )
        grad_qk_buf.copy_(
            _softmax_backward(
                grad_attn_buf.reshape(chunk_bh * seq_len, seq_len),
                attn_buf.reshape(chunk_bh * seq_len, seq_len),
                block_size,
                rows_per_block,
                multi_block_launch,
            ).reshape(chunk_bh, seq_len, seq_len)
        )

        _launch_dense_grad_qk_fused_kernel(
            grad_qk_buf,
            key_rows,
            query_rows,
            grad_q_rows,
            grad_k_rows,
            chunk_bh,
            1,
            seq_len,
            head_dim,
            scale,
            block_m,
            block_n,
            block_k,
            head_stride_q=0,
            head_stride_k=0,
            head_stride_gq=0,
            head_stride_gk=0,
            head_stride_gqk=0,
        )
        grid_qkv = (chunk_bh, triton.cdiv(seq_len, block_m), triton.cdiv(head_dim, block_n))
        _dense_neighborhood_grad_v_kernel[grid_qkv](
            attn_buf,
            grad_out_rows,
            grad_value_flat[bh_start:bh_end],
            attn_buf.stride(0),
            0,
            attn_buf.stride(1),
            attn_buf.stride(2),
            grad_out_rows.stride(0),
            0,
            grad_out_rows.stride(1),
            grad_out_rows.stride(2),
            grad_value_flat[bh_start:bh_end].stride(0),
            0,
            grad_value_flat[bh_start:bh_end].stride(1),
            grad_value_flat[bh_start:bh_end].stride(2),
            chunk_bh,
            1,
            seq_len,
            head_dim,
            block_m,
            block_n,
            block_k,
        )

    return grad_query, grad_key, grad_value


def _sparse_neighborhood_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_weights: torch.Tensor,
    scale: float,
    kernel_size: int,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = query.shape

    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_value = torch.zeros_like(value)

    block_m, block_n = _get_sparse_backward_block_sizes(seq_len, head_dim, batch_size, num_heads)
    compute_dtype = attn_weights.dtype

    chunk_batch = _attention_batch_chunk_size(batch_size, num_heads, seq_len, kernel_size)

    def _backward_chunk(batch_start: int, chunk_size: int, grad_qk_chunk: torch.Tensor) -> None:
        grad_out_chunk = grad_output[batch_start : batch_start + chunk_size]
        attn_chunk = attn_weights[batch_start : batch_start + chunk_size]
        value_chunk = value[batch_start : batch_start + chunk_size]
        query_chunk = query[batch_start : batch_start + chunk_size]
        key_chunk = key[batch_start : batch_start + chunk_size]

        _launch_sparse_grad_attn_softmax_bwd_kernel(
            grad_out_chunk,
            value_chunk,
            attn_chunk,
            grad_qk_chunk,
            chunk_size,
            num_heads,
            seq_len,
            head_dim,
            kernel_size,
            dilation,
            block_m,
            block_n,
        )
        _launch_sparse_grad_qkv_kernels(
            grad_qk_chunk,
            attn_chunk,
            query_chunk,
            key_chunk,
            grad_out_chunk,
            grad_query[batch_start : batch_start + chunk_size],
            grad_key[batch_start : batch_start + chunk_size],
            grad_value[batch_start : batch_start + chunk_size],
            chunk_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            kernel_size,
            dilation,
            block_m,
            block_n,
        )

    if chunk_batch >= batch_size:
        grad_qk_full = torch.empty(
            batch_size, num_heads, seq_len, kernel_size, device=query.device, dtype=compute_dtype
        )
        _backward_chunk(0, batch_size, grad_qk_full)
    else:
        grad_qk_buf = torch.empty(
            chunk_batch, num_heads, seq_len, kernel_size, device=query.device, dtype=compute_dtype
        )
        for batch_start in range(0, batch_size, chunk_batch):
            chunk_size = min(chunk_batch, batch_size - batch_start)
            _backward_chunk(batch_start, chunk_size, grad_qk_buf[:chunk_size])

    return grad_query, grad_key, grad_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_neighborhood_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
    return_lse: bool = False,
) -> tuple:
    batch_size, num_heads, seq_len, head_dim = query.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    output = torch.empty_like(query)
    compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype

    block_m, block_n, block_k = _get_attention_block_sizes(seq_len, head_dim)
    num_cores = get_npu_core_count()

    chunk_batch = _attention_batch_chunk_size(batch_size, num_heads, seq_len, kernel_size)

    if chunk_batch >= batch_size:
        attn_weights = torch.zeros(
            batch_size, num_heads, seq_len, kernel_size, device=query.device, dtype=compute_dtype
        )
        _launch_sparse_qk_softmax_kernel(
            query,
            key,
            attn_weights,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            kernel_size,
            dilation,
            block_m,
            block_k,
            num_cores,
        )
    else:
        attn_buf = torch.zeros(chunk_batch, num_heads, seq_len, kernel_size, device=query.device, dtype=compute_dtype)
        attn_weights = torch.zeros(
            batch_size, num_heads, seq_len, kernel_size, device=query.device, dtype=compute_dtype
        )
        for batch_start in range(0, batch_size, chunk_batch):
            batch_end = min(batch_start + chunk_batch, batch_size)
            chunk_size = batch_end - batch_start
            attn_buf.zero_()
            _launch_sparse_qk_softmax_kernel(
                query[batch_start:batch_end],
                key[batch_start:batch_end],
                attn_buf[:chunk_size],
                chunk_size,
                num_heads,
                seq_len,
                head_dim,
                scale,
                kernel_size,
                dilation,
                block_m,
                block_k,
                num_cores,
            )
            attn_weights[batch_start:batch_end] = attn_buf[:chunk_size]

    _fused_neighborhood_attention_av_kernel[(num_cores,)](
        attn_weights,
        value,
        output,
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        attn_weights.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        kernel_size,
        dilation,
        block_m,
        block_n,
    )

    if return_lse:
        raise NotImplementedError("return_lse=True is not supported yet.")

    attn_for_backward = attn_weights
    if attn_for_backward.dtype != query.dtype:
        attn_for_backward = attn_for_backward.to(query.dtype)

    return output, attn_for_backward


def fused_neighborhood_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_weights: torch.Tensor,
    scale: float,
    kernel_size: int = 7,
    dilation: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    grad_output = grad_output.contiguous()
    attn_weights = attn_weights.contiguous()

    batch_size, num_heads, seq_len, head_dim = query.shape
    if attn_weights.shape[-1] == kernel_size:
        elem_size = query.element_size() if query.dtype in (torch.float16, torch.bfloat16) else 4
        if _sparse_band_dense_fast_path_available(batch_size, num_heads, seq_len, elem_size):
            return _dense_neighborhood_attention_backward(
                grad_output, query, key, value, attn_weights, scale, kernel_size, dilation
            )
        return _sparse_neighborhood_attention_backward(
            grad_output, query, key, value, attn_weights, scale, kernel_size, dilation
        )

    return _dense_neighborhood_attention_backward(
        grad_output, query, key, value, attn_weights, scale, kernel_size, dilation
    )


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


class LigerFusedNeighborhoodAttentionFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, query, key, value, kernel_size=7, dilation=1, scale=None):
        output, attn_weights = fused_neighborhood_attention_forward(query, key, value, kernel_size, dilation, scale)
        ctx.save_for_backward(query, key, value, attn_weights)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, attn_weights = ctx.saved_tensors
        _, num_heads, _, head_dim = query.shape
        scale = ctx.scale if ctx.scale is not None else 1.0 / math.sqrt(head_dim)

        grad_cache_key = (
            grad_output.untyped_storage().data_ptr(),
            grad_output._version,
            tuple(grad_output.shape),
            tuple(grad_output.stride()),
            grad_output.dtype,
            grad_output.device,
        )
        backward_cache = getattr(ctx, "_backward_cache", None)
        if backward_cache is not None and grad_cache_key in backward_cache:
            grad_query, grad_key, grad_value = backward_cache[grad_cache_key]
            return grad_query, grad_key, grad_value, None, None, None

        grad_output = grad_output.contiguous()
        grad_query, grad_key, grad_value = fused_neighborhood_attention_backward(
            grad_output,
            query,
            key,
            value,
            attn_weights,
            scale,
            ctx.kernel_size,
            ctx.dilation,
        )

        if backward_cache is None:
            backward_cache = {}
            ctx._backward_cache = backward_cache
        if len(backward_cache) >= 4:
            backward_cache.pop(next(iter(backward_cache)))
        backward_cache[grad_cache_key] = (grad_query, grad_key, grad_value)
        return grad_query, grad_key, grad_value, None, None, None
