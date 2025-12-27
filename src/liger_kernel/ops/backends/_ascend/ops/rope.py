import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy


@triton.jit
def _triton_rope_npu(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // sl
    cos_row_idx = pid % sl

    cos = cos + tl.where(
        cos_bs == 1,
        cos_row_idx * cos_row_stride,
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
    )
    sin = sin + tl.where(
        cos_bs == 1,
        cos_row_idx * sin_row_stride,
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
    )

    q_base = q_ptr + pid * q_row_stride
    k_base = k_ptr + pid * k_row_stride

    # Pre-compute d_idx and cos/sin values outside loops (they don't depend on heads)
    d_idx = tl.arange(0, hd // 2)
    d_mask = d_idx < (hd // 2)  # Always True, but kept for clarity
    cos_vals = tl.load(cos + d_idx, mask=d_mask, other=0)
    sin_vals = tl.load(sin + d_idx, mask=d_mask, other=0)

    # Process q heads in chunks to prevent UB overflow
    for qh_block in range(0, n_qh, BLOCK_Q):
        qh_idx = tl.arange(0, BLOCK_Q) + qh_block
        qh_mask = qh_idx < n_qh

        # block_mask: qh_mask broadcasted over d_idx dimension
        block_mask = qh_mask[:, None]

        offsets = qh_idx[:, None] * hd + d_idx[None, :]

        q_left = tl.load(q_base + offsets, mask=block_mask, other=0)
        q_right = tl.load(q_base + offsets + (hd // 2), mask=block_mask, other=0)

        if not BACKWARD_PASS:
            new_left = q_left * cos_vals - q_right * sin_vals
            new_right = q_right * cos_vals + q_left * sin_vals
        else:
            new_left = q_left * cos_vals + q_right * sin_vals
            new_right = q_right * cos_vals - q_left * sin_vals

        tl.store(q_base + offsets, new_left, mask=block_mask)
        tl.store(q_base + offsets + (hd // 2), new_right, mask=block_mask)

    # Process k heads in chunks to prevent UB overflow
    for kh_block in range(0, n_kh, BLOCK_K):
        kh_idx = tl.arange(0, BLOCK_K) + kh_block
        kh_mask = kh_idx < n_kh

        # block_mask: kh_mask broadcasted over d_idx dimension
        block_mask = kh_mask[:, None]

        offsets = kh_idx[:, None] * hd + d_idx[None, :]

        k_left = tl.load(k_base + offsets, mask=block_mask, other=0)
        k_right = tl.load(k_base + offsets + (hd // 2), mask=block_mask, other=0)

        if not BACKWARD_PASS:
            new_left = k_left * cos_vals - k_right * sin_vals
            new_right = k_right * cos_vals + k_left * sin_vals
        else:
            new_left = k_left * cos_vals + k_right * sin_vals
            new_right = k_right * cos_vals - k_left * sin_vals

        tl.store(k_base + offsets, new_left, mask=block_mask)
        tl.store(k_base + offsets + (hd // 2), new_right, mask=block_mask)


def rope_forward(q, k, cos, sin):
    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_batch_size = cos.shape[0]

    # Compute tiling strategy based on UB capacity
    dtype_size = q.element_size()
    # ROPE forward tiling strategy (based on optimized ROPE kernel):
    # - cos_vals and sin_vals are loaded once outside loops (shared): pad_hd // 2 elements each
    # - In q heads loop (peak memory):
    #   * q_left: BLOCK_Q * (pad_hd // 2) elements
    #   * q_right: BLOCK_Q * (pad_hd // 2) elements
    #   * new_left: BLOCK_Q * (pad_hd // 2) elements (intermediate result)
    #   * new_right: BLOCK_Q * (pad_hd // 2) elements (intermediate result)
    #   * Total: 4 * BLOCK_Q * (pad_hd // 2) = 2 * BLOCK_Q * pad_hd elements
    # - In k heads loop (peak memory):
    #   * k_left: BLOCK_K * (pad_hd // 2) elements
    #   * k_right: BLOCK_K * (pad_hd // 2) elements
    #   * new_left: BLOCK_K * (pad_hd // 2) elements (intermediate result)
    #   * new_right: BLOCK_K * (pad_hd // 2) elements (intermediate result)
    #   * Total: 4 * BLOCK_K * (pad_hd // 2) = 2 * BLOCK_K * pad_hd elements
    # - Since q and k are processed separately, peak memory is max(BLOCK_Q, BLOCK_K) case
    # - Plus shared cos/sin: 2 * (pad_hd // 2) = pad_hd elements
    # - Conservative estimate: (2 * BLOCK_SIZE * pad_hd + pad_hd) * dtype_size * 8 bits
    # - Simplified: (2 * BLOCK_SIZE + 1) * pad_hd * dtype_size * 8 bits
    # - For safety, use: memory_multiplier=3.0 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits
    # - shapes: ((pad_n_q_head, pad_hd), (pad_n_kv_head, pad_hd))
    # - tiling_dims: (0, 0) means first dimension of each shape can be tiled
    # - Returns: ((block_size_q, pad_hd), (block_size_kv, pad_hd))
    shapes = ((pad_n_q_head, pad_hd), (pad_n_kv_head, pad_hd))
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.90,
        dtype_size=dtype_size,
        memory_multiplier=3.0,
        shapes=shapes,
        tiling_dims=(0, 0),
    )

    if tile_shapes is not None and len(tile_shapes) == len(shapes):
        # Strategy returns ((block_size_q, pad_hd), (block_size_kv, pad_hd))
        q_tile_shape, k_tile_shape = tile_shapes
        BLOCK_Q, _ = q_tile_shape
        BLOCK_K, _ = k_tile_shape
    else:
        # Fallback to conservative defaults
        BLOCK_Q = triton.next_power_of_2(pad_n_q_head)
        BLOCK_K = triton.next_power_of_2(pad_n_kv_head)

    _triton_rope_npu[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        BLOCK_Q,
        BLOCK_K,
        BACKWARD_PASS=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    cos_batch_size = cos.shape[0]
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    # ensure dq and dk are contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()

    # Compute tiling strategy based on UB capacity
    dtype_size = dq.element_size()
    # ROPE backward tiling strategy (based on optimized ROPE kernel):
    # - cos_vals and sin_vals are loaded once outside loops (shared): pad_hd // 2 elements each
    # - In q heads loop (peak memory):
    #   * q_left: BLOCK_Q * (pad_hd // 2) elements
    #   * q_right: BLOCK_Q * (pad_hd // 2) elements
    #   * new_left: BLOCK_Q * (pad_hd // 2) elements (intermediate result)
    #   * new_right: BLOCK_Q * (pad_hd // 2) elements (intermediate result)
    #   * Total: 4 * BLOCK_Q * (pad_hd // 2) = 2 * BLOCK_Q * pad_hd elements
    # - In k heads loop (peak memory):
    #   * k_left: BLOCK_K * (pad_hd // 2) elements
    #   * k_right: BLOCK_K * (pad_hd // 2) elements
    #   * new_left: BLOCK_K * (pad_hd // 2) elements (intermediate result)
    #   * new_right: BLOCK_K * (pad_hd // 2) elements (intermediate result)
    #   * Total: 4 * BLOCK_K * (pad_hd // 2) = 2 * BLOCK_K * pad_hd elements
    # - Since q and k are processed separately, peak memory is max(BLOCK_Q, BLOCK_K) case
    # - Plus shared cos/sin: 2 * (pad_hd // 2) = pad_hd elements
    # - Conservative estimate: (2 * BLOCK_SIZE * pad_hd + pad_hd) * dtype_size * 8 bits
    # - Simplified: (2 * BLOCK_SIZE + 1) * pad_hd * dtype_size * 8 bits
    # - For safety, use: memory_multiplier=3.0 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits
    # - shapes: ((pad_n_q_head, pad_hd), (pad_n_kv_head, pad_hd))
    # - tiling_dims: (0, 0) means first dimension of each shape can be tiled
    # - Returns: ((block_size_q, pad_hd), (block_size_kv, pad_hd))
    shapes = ((pad_n_q_head, pad_hd), (pad_n_kv_head, pad_hd))
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.90,
        dtype_size=dtype_size,
        memory_multiplier=3.0,
        shapes=shapes,
        tiling_dims=(0, 0),
    )

    if tile_shapes is not None and len(tile_shapes) == len(shapes):
        # Strategy returns ((block_size_q, pad_hd), (block_size_kv, pad_hd))
        q_tile_shape, k_tile_shape = tile_shapes
        BLOCK_Q, _ = q_tile_shape
        BLOCK_K, _ = k_tile_shape
    else:
        # Fallback to conservative defaults
        BLOCK_Q = min(32, triton.next_power_of_2(pad_n_q_head))
        BLOCK_K = min(32, triton.next_power_of_2(pad_n_kv_head))

    _triton_rope_npu[(n_row,)](
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        BLOCK_Q,
        BLOCK_K,
        BACKWARD_PASS=True,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """

        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None
