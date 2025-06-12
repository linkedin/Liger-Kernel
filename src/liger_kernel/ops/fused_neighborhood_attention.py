import math

import torch
import triton
import triton.language as tl

from liger_kernel.ops.softmax import _softmax_backward
from liger_kernel.ops.softmax import _softmax_forward
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _neighborhood_mask_kernel(
    mask_ptr,
    seq_len: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Generate a neighborhood attention mask for a given sequence.

    This kernel creates a binary mask that defines which positions in a sequence
    can attend to each other based on a neighborhood window with optional dilation.
    Each row of the mask corresponds to a query position, and each column indicates
    whether that key position is within the allowed neighborhood.

    The neighborhood is defined as positions within kernel_size//2 * dilation distance
    from the center position. When dilation > 1, only positions at multiples of the
    dilation factor are included in the neighborhood.

    Args:
        mask_ptr: Pointer to the output mask tensor [seq_len, seq_len]
        seq_len: Length of the input sequence
        kernel_size: Size of the neighborhood window (must be odd)
        dilation: Dilation factor for the neighborhood pattern
        BLOCK_SIZE: Block size for processing (compile-time constant)
        num_stages: Number of pipeline stages (compile-time constant)
        num_warps: Number of warps (compile-time constant)

    Grid: (seq_len,)
    Each program processes one row of the mask matrix.
    """
    row_id = tl.program_id(0)

    center = row_id
    half_kernel = kernel_size // 2

    start = tl.maximum(0, center - half_kernel * dilation)
    end = tl.minimum(seq_len, center + half_kernel * dilation + 1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len

    valid_neighbors = (col_offsets >= start) & (col_offsets < end)
    if dilation > 1:
        relative_pos = col_offsets - center
        valid_dilation = (relative_pos % dilation) == 0
        valid_neighbors = valid_neighbors & valid_dilation

    mask_values = tl.where(valid_neighbors & mask, 1.0, 0.0)

    base_offset = row_id * seq_len
    tl.store(mask_ptr + base_offset + col_offsets, mask_values, mask=mask)


@triton.jit
def _fused_neighborhood_attention_qk_kernel(
    Q_ptr,
    K_ptr,
    QK_ptr,
    mask_ptr,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    qk_batch_stride,
    qk_head_stride,
    qk_seq_stride,
    qk_seq2_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute Q @ K^T with neighborhood masking and scaling.

    This kernel performs the first stage of neighborhood attention by computing
    the attention scores between queries and keys, applying scaling, and masking
    positions outside the neighborhood window. The result is a matrix of attention
    scores ready for softmax normalization.

    The computation is tiled across sequence dimensions for memory efficiency.
    Each tile computes a block of the attention score matrix by iterating over
    the head dimension and accumulating dot products.

    Args:
        Q_ptr: Pointer to query tensor [batch_size, num_heads, seq_len, head_dim]
        K_ptr: Pointer to key tensor [batch_size, num_heads, seq_len, head_dim]
        QK_ptr: Pointer to output tensor [batch_size, num_heads, seq_len, seq_len]
        mask_ptr: Pointer to neighborhood mask [seq_len, seq_len]
        q_*_stride: Strides for query tensor
        k_*_stride: Strides for key tensor
        qk_*_stride: Strides for output tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        scale: Scaling factor for attention scores (typically 1/sqrt(head_dim))
        kernel_size: Size of the neighborhood window
        dilation: Dilation factor for the neighborhood
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for sequence dimension (cols)
        BLOCK_SIZE_K: Block size for head dimension
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(seq_len, BLOCK_SIZE_N))
    Each program computes a tile of the attention score matrix.
    """
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

        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + row_offsets[:, None] * q_seq_stride
            + k_offsets[None, :] * q_dim_stride
        )
        q_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + col_offsets[:, None] * k_seq_stride
            + k_offsets[None, :] * k_dim_stride
        )
        k_mask = (col_offsets[:, None] < seq_len) & k_mask[None, :]
        k_chunk = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc += tl.dot(q_chunk, tl.trans(k_chunk))

    acc = acc * scale

    mask_ptrs = mask_ptr + row_offsets[:, None] * seq_len + col_offsets[None, :]
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < seq_len)
    neighborhood_mask = tl.load(mask_ptrs, mask=valid_mask, other=0.0)

    acc = tl.where(neighborhood_mask > 0.0, acc, float("-inf"))

    qk_ptrs = (
        QK_ptr
        + batch_id * qk_batch_stride
        + head_id * qk_head_stride
        + row_offsets[:, None] * qk_seq_stride
        + col_offsets[None, :] * qk_seq2_stride
    )
    tl.store(qk_ptrs, acc, mask=valid_mask)


@triton.jit
def _fused_neighborhood_attention_av_kernel(
    Attn_ptr,
    V_ptr,
    Out_ptr,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_seq2_stride,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute Attention @ V to produce the final output.

    This kernel performs the second stage of neighborhood attention by multiplying
    the normalized attention weights with the value matrix. The computation is
    tiled for memory efficiency, with each tile computing a block of the output.

    Args:
        Attn_ptr: Pointer to attention weights [batch_size, num_heads, seq_len, seq_len]
        V_ptr: Pointer to value tensor [batch_size, num_heads, seq_len, head_dim]
        Out_ptr: Pointer to output tensor [batch_size, num_heads, seq_len, head_dim]
        attn_*_stride: Strides for attention weights tensor
        v_*_stride: Strides for value tensor
        out_*_stride: Strides for output tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for head dimension (cols)
        BLOCK_SIZE_K: Block size for sequence dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(head_dim, BLOCK_SIZE_N))
    Each program computes a tile of the output matrix.
    """
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
            + row_offsets[:, None] * attn_seq_stride
            + k_offsets[None, :] * attn_seq2_stride
        )
        attn_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        attn_chunk = tl.load(attn_ptrs, mask=attn_mask, other=0.0)

        v_ptrs = (
            V_ptr
            + batch_id * v_batch_stride
            + head_id * v_head_stride
            + k_offsets[:, None] * v_seq_stride
            + col_offsets[None, :] * v_dim_stride
        )
        v_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(attn_chunk, v_chunk)

    out_ptrs = (
        Out_ptr
        + batch_id * out_batch_stride
        + head_id * out_head_stride
        + row_offsets[:, None] * out_seq_stride
        + col_offsets[None, :] * out_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(out_ptrs, acc, mask=valid_mask)


@triton.jit
def _fused_neighborhood_attention_grad_qk_kernel(
    grad_attn_ptr,
    K_ptr,
    grad_Q_ptr,
    grad_attn_batch_stride,
    grad_attn_head_stride,
    grad_attn_seq_stride,
    grad_attn_seq2_stride,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute gradient with respect to queries: grad_Q = grad_attn @ K * scale.

    This kernel computes the gradient of the loss with respect to the query tensor
    by multiplying the gradient of attention weights with the key tensor. The
    computation follows the chain rule for the attention mechanism.

    Args:
        grad_attn_ptr: Pointer to gradient of attention weights [batch_size, num_heads, seq_len, seq_len]
        K_ptr: Pointer to key tensor [batch_size, num_heads, seq_len, head_dim]
        grad_Q_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, head_dim]
        grad_attn_*_stride: Strides for gradient attention tensor
        k_*_stride: Strides for key tensor
        grad_q_*_stride: Strides for gradient query tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        scale: Scaling factor applied to attention scores
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for head dimension (cols)
        BLOCK_SIZE_K: Block size for sequence dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(head_dim, BLOCK_SIZE_N))
    Each program computes a tile of the query gradient matrix.
    """
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

        grad_attn_ptrs = (
            grad_attn_ptr
            + batch_id * grad_attn_batch_stride
            + head_id * grad_attn_head_stride
            + row_offsets[:, None] * grad_attn_seq_stride
            + k_offsets[None, :] * grad_attn_seq2_stride
        )
        grad_attn_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_attn_chunk = tl.load(grad_attn_ptrs, mask=grad_attn_mask, other=0.0)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + k_offsets[:, None] * k_seq_stride
            + col_offsets[None, :] * k_dim_stride
        )
        k_mask_2d = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        k_chunk = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)

        acc += tl.dot(grad_attn_chunk, k_chunk)

    acc = acc * scale

    grad_q_ptrs = (
        grad_Q_ptr
        + batch_id * grad_q_batch_stride
        + head_id * grad_q_head_stride
        + row_offsets[:, None] * grad_q_seq_stride
        + col_offsets[None, :] * grad_q_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(grad_q_ptrs, acc, mask=valid_mask)


@triton.jit
def _fused_neighborhood_attention_grad_k_kernel(
    grad_attn_ptr,
    Q_ptr,
    grad_K_ptr,
    grad_attn_batch_stride,
    grad_attn_head_stride,
    grad_attn_seq_stride,
    grad_attn_seq2_stride,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute gradient with respect to keys: grad_K = grad_attn^T @ Q * scale.

    This kernel computes the gradient of the loss with respect to the key tensor
    by multiplying the transpose of the gradient of attention weights with the
    query tensor. The computation follows the chain rule for the attention mechanism.

    Args:
        grad_attn_ptr: Pointer to gradient of attention weights [batch_size, num_heads, seq_len, seq_len]
        Q_ptr: Pointer to query tensor [batch_size, num_heads, seq_len, head_dim]
        grad_K_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, head_dim]
        grad_attn_*_stride: Strides for gradient attention tensor
        q_*_stride: Strides for query tensor
        grad_k_*_stride: Strides for gradient key tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        scale: Scaling factor applied to attention scores
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for head dimension (cols)
        BLOCK_SIZE_K: Block size for sequence dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(head_dim, BLOCK_SIZE_N))
    Each program computes a tile of the key gradient matrix.
    """
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

        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + k_offsets[:, None] * q_seq_stride
            + col_offsets[None, :] * q_dim_stride
        )
        q_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)

        grad_attn_T_ptrs = (
            grad_attn_ptr
            + batch_id * grad_attn_batch_stride
            + head_id * grad_attn_head_stride
            + row_offsets[:, None] * grad_attn_seq2_stride
            + k_offsets[None, :] * grad_attn_seq_stride
        )
        grad_attn_T_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_attn_T_chunk = tl.load(grad_attn_T_ptrs, mask=grad_attn_T_mask, other=0.0)

        acc += tl.dot(grad_attn_T_chunk, q_chunk)

    acc = acc * scale

    grad_k_ptrs = (
        grad_K_ptr
        + batch_id * grad_k_batch_stride
        + head_id * grad_k_head_stride
        + row_offsets[:, None] * grad_k_seq_stride
        + col_offsets[None, :] * grad_k_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(grad_k_ptrs, acc, mask=valid_mask)


@triton.jit
def _fused_neighborhood_attention_grad_v_kernel(
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
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute gradient with respect to values: grad_V = Attn^T @ grad_output.

    This kernel computes the gradient of the loss with respect to the value tensor
    by multiplying the transpose of the attention weights with the gradient of the
    output. The computation follows the chain rule for the attention mechanism.

    Args:
        Attn_ptr: Pointer to attention weights [batch_size, num_heads, seq_len, seq_len]
        grad_output_ptr: Pointer to gradient of output [batch_size, num_heads, seq_len, head_dim]
        grad_V_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, head_dim]
        attn_*_stride: Strides for attention weights tensor
        grad_out_*_stride: Strides for gradient output tensor
        grad_v_*_stride: Strides for gradient value tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for head dimension (cols)
        BLOCK_SIZE_K: Block size for sequence dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(head_dim, BLOCK_SIZE_N))
    Each program computes a tile of the value gradient matrix.
    """
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
        attn_chunk = tl.load(attn_ptrs, mask=attn_mask, other=0.0)

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + k_offsets[:, None] * grad_out_seq_stride
            + col_offsets[None, :] * grad_out_dim_stride
        )
        grad_out_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

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


@triton.jit
def _fused_neighborhood_attention_grad_attn_kernel(
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
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute gradient with respect to attention weights: grad_attn = grad_output @ V^T.

    This kernel computes the gradient of the loss with respect to the attention
    weights by multiplying the gradient of the output with the transpose of the
    value tensor. This gradient will later be passed through the softmax backward
    pass to compute gradients for the attention scores.

    Args:
        grad_output_ptr: Pointer to gradient of output [batch_size, num_heads, seq_len, head_dim]
        V_ptr: Pointer to value tensor [batch_size, num_heads, seq_len, head_dim]
        grad_attn_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, seq_len]
        grad_out_*_stride: Strides for gradient output tensor
        v_*_stride: Strides for value tensor
        grad_attn_*_stride: Strides for gradient attention tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for sequence dimension (cols)
        BLOCK_SIZE_K: Block size for head dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(seq_len, BLOCK_SIZE_N))
    Each program computes a tile of the attention gradient matrix.
    """
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
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        v_ptrs = (
            V_ptr
            + batch_id * v_batch_stride
            + head_id * v_head_stride
            + col_offsets[None, :] * v_seq_stride
            + k_offsets[:, None] * v_dim_stride
        )
        v_mask = (col_offsets[None, :] < seq_len) & k_mask[:, None]
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)

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


def fused_neighborhood_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
    return_lse: bool = False,
) -> tuple:
    """
    Fused neighborhood attention forward pass.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        kernel_size: Size of the neighborhood window
        dilation: Dilation factor for the neighborhood
        scale: Scaling factor for attention scores (default: rsqrt(head_dim))
        return_lse: Whether to return log-sum-exp values

    Returns:
        Tuple of (output tensor, softmax parameters for backward)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    output = torch.empty_like(query)
    qk_scores = torch.empty(batch_size, num_heads, seq_len, seq_len, device=query.device, dtype=query.dtype)

    mask = torch.zeros(seq_len, seq_len, device=query.device, dtype=torch.float32)

    BLOCK_SIZE, num_warps = calculate_settings(seq_len)
    BLOCK_SIZE_M = min(64, triton.next_power_of_2(seq_len))
    BLOCK_SIZE_N = min(64, triton.next_power_of_2(seq_len))
    BLOCK_SIZE_K = max(16, triton.next_power_of_2(head_dim))

    num_stages = 4 if seq_len >= 512 else 2

    grid_mask = (seq_len,)
    _neighborhood_mask_kernel[grid_mask](
        mask,
        seq_len,
        kernel_size,
        dilation,
        BLOCK_SIZE,
        num_stages,
        num_warps,
    )

    grid_qk = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(seq_len, BLOCK_SIZE_N))
    _fused_neighborhood_attention_qk_kernel[grid_qk](
        query,
        key,
        qk_scores,
        mask,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        qk_scores.stride(0),
        qk_scores.stride(1),
        qk_scores.stride(2),
        qk_scores.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        kernel_size,
        dilation,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_stages,
        num_warps,
    )

    qk_reshaped = qk_scores.view(batch_size * num_heads * seq_len, seq_len)
    attn_reshaped, BLOCK_SIZE_softmax, num_warps_softmax, multi_block_launch = _softmax_forward(qk_reshaped)
    attn_weights = attn_reshaped.view(batch_size, num_heads, seq_len, seq_len)

    grid_av = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(head_dim, BLOCK_SIZE_N))
    _fused_neighborhood_attention_av_kernel[grid_av](
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
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_stages,
        num_warps,
    )

    if return_lse:
        raise NotImplementedError("return_lse=True is not supported yet.")

    softmax_params = (BLOCK_SIZE_softmax, num_warps_softmax, multi_block_launch)
    return output, attn_weights, softmax_params


class LigerFusedNeighborhoodAttentionFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, query, key, value, kernel_size=7, dilation=1, scale=None):
        output, attn_weights, softmax_params = fused_neighborhood_attention_forward(
            query, key, value, kernel_size, dilation, scale
        )
        ctx.save_for_backward(query, key, value, attn_weights)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.scale = scale
        ctx.softmax_params = softmax_params
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        query, key, value, attn_weights = ctx.saved_tensors
        BLOCK_SIZE_softmax, num_warps_softmax, multi_block_launch = ctx.softmax_params

        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = ctx.scale if ctx.scale is not None else 1.0 / math.sqrt(head_dim)

        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        grad_attn_weights = torch.zeros_like(attn_weights)

        BLOCK_SIZE_M = min(64, triton.next_power_of_2(seq_len))
        BLOCK_SIZE_N = min(64, triton.next_power_of_2(seq_len))
        BLOCK_SIZE_K = min(64, triton.next_power_of_2(head_dim))
        num_stages = 4 if seq_len >= 512 else 2
        _, num_warps = calculate_settings(seq_len)

        grid_grad_attn = (
            batch_size * num_heads,
            triton.cdiv(seq_len, BLOCK_SIZE_M),
            triton.cdiv(seq_len, BLOCK_SIZE_N),
        )
        _fused_neighborhood_attention_grad_attn_kernel[grid_grad_attn](
            grad_output,
            value,
            grad_attn_weights,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            grad_attn_weights.stride(0),
            grad_attn_weights.stride(1),
            grad_attn_weights.stride(2),
            grad_attn_weights.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_stages,
            num_warps,
        )

        grad_attn_reshaped = grad_attn_weights.view(batch_size * num_heads * seq_len, seq_len)
        attn_reshaped = attn_weights.view(batch_size * num_heads * seq_len, seq_len)

        grad_qk_reshaped = _softmax_backward(
            grad_attn_reshaped, attn_reshaped, BLOCK_SIZE_softmax, num_warps_softmax, multi_block_launch
        )
        grad_qk_scores = grad_qk_reshaped.view(batch_size, num_heads, seq_len, seq_len)

        grid_grad_q = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(head_dim, BLOCK_SIZE_N))
        _fused_neighborhood_attention_grad_qk_kernel[grid_grad_q](
            grad_qk_scores,
            key,
            grad_query,
            grad_qk_scores.stride(0),
            grad_qk_scores.stride(1),
            grad_qk_scores.stride(2),
            grad_qk_scores.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            grad_query.stride(0),
            grad_query.stride(1),
            grad_query.stride(2),
            grad_query.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_stages,
            num_warps,
        )

        grid_grad_k = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(head_dim, BLOCK_SIZE_N))
        _fused_neighborhood_attention_grad_k_kernel[grid_grad_k](
            grad_qk_scores,
            query,
            grad_key,
            grad_qk_scores.stride(0),
            grad_qk_scores.stride(1),
            grad_qk_scores.stride(2),
            grad_qk_scores.stride(3),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            grad_key.stride(0),
            grad_key.stride(1),
            grad_key.stride(2),
            grad_key.stride(3),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_stages,
            num_warps,
        )

        grid_grad_v = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(head_dim, BLOCK_SIZE_N))
        _fused_neighborhood_attention_grad_v_kernel[grid_grad_v](
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
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_stages,
            num_warps,
        )

        return grad_query, grad_key, grad_value, None, None, None
