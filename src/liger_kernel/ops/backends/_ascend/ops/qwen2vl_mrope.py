import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def _triton_qwen2vl_mrope_npu(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    sin,
    sl,
    bs: tl.constexpr,
    total_rows: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_per_program = (total_rows + num_programs - 1) // num_programs
    start_row = program_id * rows_per_program
    actual_rows = tl.minimum(rows_per_program, total_rows - start_row)

    for row_offset in tl.range(0, actual_rows, num_stages=NUM_STAGES):
        pid = start_row + row_offset

        t_end = mrope_section_t
        h_end = t_end + mrope_section_h

        t_cos = cos + pid * hd
        h_cos = t_cos + bs * sl * hd
        w_cos = h_cos + bs * sl * hd
        t_sin = sin + pid * hd
        h_sin = t_sin + bs * sl * hd
        w_sin = h_sin + bs * sl * hd

        q_base = q_ptr + pid * q_row_stride
        k_base = k_ptr + pid * k_row_stride

        d_idx = tl.arange(0, hd // 2)
        d_mask = d_idx < (hd // 2)

        pos_mask_t = d_idx < t_end
        pos_mask_h = (d_idx >= t_end) & (d_idx < h_end)

        text_cos_vals = tl.load(t_cos + d_idx, mask=d_mask, other=0)
        text_sin_vals = tl.load(t_sin + d_idx, mask=d_mask, other=0)
        height_cos_vals = tl.load(h_cos + d_idx, mask=d_mask, other=0)
        height_sin_vals = tl.load(h_sin + d_idx, mask=d_mask, other=0)
        width_cos_vals = tl.load(w_cos + d_idx, mask=d_mask, other=0)
        width_sin_vals = tl.load(w_sin + d_idx, mask=d_mask, other=0)

        cos_vals = tl.where(pos_mask_t, text_cos_vals, tl.where(pos_mask_h, height_cos_vals, width_cos_vals))
        sin_vals = tl.where(pos_mask_t, text_sin_vals, tl.where(pos_mask_h, height_sin_vals, width_sin_vals))

        # Process q heads in chunks to prevent UB overflow
        for qh_block in range(0, n_qh, BLOCK_Q):
            qh_idx = tl.arange(0, BLOCK_Q) + qh_block
            qh_mask = qh_idx < n_qh

            block_mask = qh_mask[:, None] & d_mask[None, :]
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

            block_mask = kh_mask[:, None] & d_mask[None, :]
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


def get_optimal_block_size_mrope(pad_n_q_head, pad_n_kv_head, pad_hd, dtype_size):
    # MROPE forward tiling strategy:
    # - cos_vals and sin_vals (include text, height and width) are loaded once outside loops (shared): (pad_hd // 2) * 6 = 3 * pad_hd elements each
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
    # - Plus shared cos/sin: 6 * (pad_hd // 2) = 3 * pad_hd elements
    # - Conservative estimate: (2 * BLOCK_SIZE * pad_hd + 3 * pad_hd) * dtype_size * 8 bits
    # - Simplified: (2 * BLOCK_SIZE + 3) * pad_hd * dtype_size * 8 bits
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
        BLOCK_Q = 2048
        BLOCK_K = 2048

    return BLOCK_Q, BLOCK_K


def qwen2vl_mrope_forward(q, k, cos, sin, mrope_section):
    # transpose it back to the physical shape because Triton looks at the physical storage
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    dtype_size = q.element_size()
    BLOCK_Q, BLOCK_K = get_optimal_block_size_mrope(pad_n_q_head, pad_n_kv_head, pad_hd, dtype_size)

    num_cores = get_npu_core_count()
    grid_size = min(num_cores, n_row)

    _triton_qwen2vl_mrope_npu[(grid_size,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        sin,
        seq_len,
        batch_size,
        n_row,
        n_q_head,
        n_kv_head,
        head_dim,
        mrope_section[0],
        mrope_section[1],
        BLOCK_Q,
        BLOCK_K,
        NUM_STAGES=3,
        BACKWARD_PASS=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    # ensure dq and dk are contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()

    dtype_size = dq.element_size()
    BLOCK_Q, BLOCK_K = get_optimal_block_size_mrope(pad_n_q_head, pad_n_kv_head, pad_hd, dtype_size)

    num_cores = get_npu_core_count()
    grid_size = min(num_cores, n_row)

    _triton_qwen2vl_mrope_npu[(grid_size,)](
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        cos,
        sin,
        seq_len,
        batch_size,
        n_row,
        n_q_head,
        n_kv_head,
        head_dim,
        mrope_section[0],
        mrope_section[1],
        BLOCK_Q,
        BLOCK_K,
        NUM_STAGES=3,
        BACKWARD_PASS=True,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerQwen2VLMRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (3, bsz, seq_len, head_dim)
        sin size: (3, bsz, seq_len, head_dim)
        """
        q, k, cos, sin = qwen2vl_mrope_forward(q, k, cos, sin, mrope_section)
        ctx.save_for_backward(cos, sin)
        ctx.mrope_section = mrope_section
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (3, bsz, seq_len, head_dim)
        sin size: (3, bsz, seq_len, head_dim)
        """
        cos, sin = ctx.saved_tensors
        mrope_section = ctx.mrope_section
        dq, dk = qwen2vl_mrope_backward(dq, dk, cos, sin, mrope_section)
        return dq, dk, None, None, None, None
