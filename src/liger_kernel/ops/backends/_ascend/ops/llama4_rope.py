import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy


def _prepare_freqs(freqs_cis: torch.Tensor, seq_len: int, head_dim_half: int):
    """
    Canonicalize freqs to (seq_len, head_dim_half) real/imag tensors.

    Supports:
    - complex freqs: (..., head_dim_half) complex -> real/imag
    - packed freqs: (..., 2*head_dim_half) real -> split into real/imag
    """
    if freqs_cis.is_complex():
        freqs_real = freqs_cis.real
        freqs_imag = freqs_cis.imag
    else:
        if freqs_cis.shape[-1] == 2 * head_dim_half:
            freqs_real = freqs_cis[..., :head_dim_half]
            freqs_imag = freqs_cis[..., head_dim_half:]
        else:
            raise ValueError(
                f"Unexpected freqs_cis shape for non-complex input: {freqs_cis.shape}, "
                f"expected last dim = {2 * head_dim_half}"
            )

    if freqs_real.shape[-1] != head_dim_half:
        raise ValueError(f"Unexpected last dim for freqs: {freqs_real.shape[-1]} (expected {head_dim_half})")

    # Flatten leading dims -> (N, head_dim_half)
    freqs_real = freqs_real.reshape(-1, head_dim_half)
    freqs_imag = freqs_imag.reshape(-1, head_dim_half)

    # Broadcast/slice to (seq_len, head_dim_half)
    if freqs_real.shape[0] < seq_len:
        if freqs_real.shape[0] == 1:
            freqs_real = freqs_real.expand(seq_len, -1)
            freqs_imag = freqs_imag.expand(seq_len, -1)
        else:
            raise ValueError(f"Insufficient rows in freqs: {freqs_real.shape[0]} < seq_len={seq_len}")
    elif freqs_real.shape[0] > seq_len:
        freqs_real = freqs_real[:seq_len]
        freqs_imag = freqs_imag[:seq_len]

    return freqs_real, freqs_imag


def _cast_and_contiguous(q, k, freqs_real, freqs_imag):
    # Align dtype: fp32 only when q is fp32; otherwise keep q dtype for perf
    compute_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype

    if k.dtype != q.dtype:
        k = k.to(q.dtype)

    q = q.to(compute_dtype).contiguous()
    k = k.to(compute_dtype).contiguous()
    freqs_real = freqs_real.to(compute_dtype).contiguous()
    freqs_imag = freqs_imag.to(compute_dtype).contiguous()
    return q, k, freqs_real, freqs_imag, compute_dtype


@triton.jit
def _triton_llama4_rope_npu(
    q_ptr,
    k_ptr,
    freqs_real_ptr,
    freqs_imag_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    freqs_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    imag_sign: tl.constexpr,
):
    """
    Llama4 RoPE on Ascend NPU for interleaved complex layout:
    - q/k shape: (bs, sl, n_heads, hd)
    - last dim layout: [real0, imag0, real1, imag1, ...]
    - freqs_real/imag: (sl, hd//2)
    """
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // sl
    seq_idx = pid % sl

    if batch_idx >= bs:
        return

    q_base = q_ptr + pid * q_row_stride
    k_base = k_ptr + pid * k_row_stride

    freq_base = seq_idx * freqs_row_stride
    hd_idx = tl.arange(0, hd)
    hd_mask = hd_idx < (hd)

    freq_idx = tl.arange(0, hd // 2)
    freq_mask = freq_idx < (hd // 2)

    freqs_real = tl.load(freqs_real_ptr + freq_base + freq_idx, mask=freq_mask, other=0.0)
    freqs_imag = tl.load(freqs_imag_ptr + freq_base + freq_idx, mask=freq_mask, other=0.0) * imag_sign

    # Q heads (chunked for UB)
    for qh_block in range(0, n_qh, BLOCK_Q):
        qh_idx = tl.arange(0, BLOCK_Q) + qh_block
        qh_mask = qh_idx < n_qh
        block_mask = qh_mask[:, None] & hd_mask[None, :]

        head_ptr = q_base + qh_idx[:, None] * q_head_stride

        q_pair = tl.load(
            head_ptr + hd_idx[None, :],
            mask=block_mask,
            other=0.0,
        )
        q_pair = q_pair.reshape(BLOCK_Q, hd // 2, 2, can_reorder=True)
        q_real, q_imag = tl.split(q_pair)

        new_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
        new_imag = tl.math.fma(q_real, freqs_imag, q_imag * freqs_real)
        new_q_pair = tl.interleave(new_real, new_imag)

        tl.store(head_ptr + hd_idx[None, :], new_q_pair, mask=block_mask)

    # K heads (chunked for UB)
    for kh_block in range(0, n_kh, BLOCK_K):
        kh_idx = tl.arange(0, BLOCK_K) + kh_block
        kh_mask = kh_idx < n_kh
        block_mask = kh_mask[:, None] & hd_mask[None, :]

        head_ptr = k_base + kh_idx[:, None] * k_head_stride

        k_pair = tl.load(
            head_ptr + hd_idx[None, :],
            mask=block_mask,
            other=0.0,
        )

        k_pair = k_pair.reshape(BLOCK_K, hd // 2, 2, can_reorder=True)
        k_real, k_imag = tl.split(k_pair)

        new_real = tl.math.fma(k_real, freqs_real, -(k_imag * freqs_imag))
        new_imag = tl.math.fma(k_real, freqs_imag, k_imag * freqs_real)
        new_k_pair = tl.interleave(new_real, new_imag)

        tl.store(head_ptr + hd_idx[None, :], new_k_pair, mask=block_mask)


def llama4_rope_forward(q, k, freqs_cis):
    """
    Ascend NPU implementation of Llama4 RoPE.

    q/k: (bs, sl, n_heads, hd) with interleaved complex last-dim layout.
    freqs_cis: complex (..., hd//2) OR packed (..., 2*(hd//2)).
    """
    original_dtype = q.dtype

    bs, sl, n_qh, hd = q.shape
    _, _, n_kh, _ = k.shape
    if hd % 2 != 0:
        raise ValueError(f"head_dim must be even for interleaved complex layout, got {hd}")
    hd_half = hd // 2

    freqs_real, freqs_imag = _prepare_freqs(freqs_cis, sl, hd_half)
    q, k, freqs_real, freqs_imag, compute_dtype = _cast_and_contiguous(q, k, freqs_real, freqs_imag)

    # UB tiling strategy: tile heads dimension only
    dtype_size = q.element_size()
    shapes = ((n_qh, hd), (n_kh, hd))
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.90,
        dtype_size=dtype_size,
        memory_multiplier=12.0,
        shapes=shapes,
        tiling_dims=(0, 0),
    )

    if tile_shapes is not None and len(tile_shapes) == len(shapes):
        q_tile_shape, k_tile_shape = tile_shapes
        BLOCK_Q, _ = q_tile_shape
        BLOCK_K, _ = k_tile_shape
    else:
        BLOCK_Q = triton.next_power_of_2(n_qh)
        BLOCK_K = triton.next_power_of_2(n_kh)

    n_row = bs * sl

    _triton_llama4_rope_npu[(n_row,)](
        q,
        k,
        freqs_real,
        freqs_imag,
        q.stride(1),
        k.stride(1),
        q.stride(2),
        k.stride(2),
        freqs_real.stride(0),
        sl,
        bs,
        n_qh,
        n_kh,
        hd,
        BLOCK_Q,
        BLOCK_K,
        imag_sign=1.0,
    )

    if compute_dtype != original_dtype:
        q = q.to(original_dtype)
        k = k.to(original_dtype)
    return q, k


def llama4_rope_backward(dq, dk, freqs_cis):
    """
    Ascend NPU implementation of Llama4 RoPE.

    q/k: (bs, sl, n_heads, hd) with interleaved complex last-dim layout.
    freqs_cis: complex (..., hd//2) OR packed (..., 2*(hd//2)).
    """
    original_dtype = dq.dtype

    bs, sl, n_qh, hd = dq.shape
    _, _, n_kh, _ = dk.shape
    if hd % 2 != 0:
        raise ValueError(f"head_dim must be even for interleaved complex layout, got {hd}")
    hd_half = hd // 2

    freqs_real, freqs_imag = _prepare_freqs(freqs_cis, sl, hd_half)
    dq, dk, freqs_real, freqs_imag, compute_dtype = _cast_and_contiguous(dq, dk, freqs_real, freqs_imag)

    # UB tiling strategy: tile heads dimension only
    dtype_size = dq.element_size()
    shapes = ((n_qh, hd), (n_kh, hd))
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.90,
        dtype_size=dtype_size,
        memory_multiplier=12.0,
        shapes=shapes,
        tiling_dims=(0, 0),
    )

    if tile_shapes is not None and len(tile_shapes) == len(shapes):
        q_tile_shape, k_tile_shape = tile_shapes
        BLOCK_Q, _ = q_tile_shape
        BLOCK_K, _ = k_tile_shape
    else:
        BLOCK_Q = triton.next_power_of_2(n_qh)
        BLOCK_K = triton.next_power_of_2(n_kh)

    n_row = bs * sl

    _triton_llama4_rope_npu[(n_row,)](
        dq,
        dk,
        freqs_real,
        freqs_imag,
        dq.stride(1),
        dk.stride(1),
        dq.stride(2),
        dk.stride(2),
        freqs_real.stride(0),
        sl,
        bs,
        n_qh,
        n_kh,
        hd,
        BLOCK_Q,
        BLOCK_K,
        imag_sign=-1.0,
    )

    if compute_dtype != original_dtype:
        dq = dq.to(original_dtype)
        dk = dk.to(original_dtype)
    return dq, dk


class LigerLlama4RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis, BLOCK_SIZE: int = None):
        # BLOCK_SIZE is ignored for Ascend (we auto-tile heads by UB), kept for API compatibility
        q_out, k_out = llama4_rope_forward(q, k, freqs_cis)
        ctx.save_for_backward(freqs_cis.detach() if isinstance(freqs_cis, torch.Tensor) else freqs_cis)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors
        dq_out, dk_out = llama4_rope_backward(dq, dk, freqs_cis)
        return dq_out, dk_out, None, None
