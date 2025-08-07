import torch
import triton
import triton.language as tl


def _prepare_freqs(freqs_cis: torch.Tensor, seq_len: int, head_dim_half: int):
    # Split or unpack complex frequencies into real and imag parts
    if freqs_cis.is_complex():
        freqs_real = freqs_cis.real
        freqs_imag = freqs_cis.imag
    else:
        # Already split: last dim should be 2*head_dim_half
        if freqs_cis.shape[-1] == 2 * head_dim_half:
            freqs_real = freqs_cis[..., :head_dim_half]
            freqs_imag = freqs_cis[..., head_dim_half:]
        else:
            raise ValueError(
                f"Unexpected freqs_cis shape for non-complex input: {freqs_cis.shape}, expected last dim = {2 * head_dim_half}"
            )

    # Canonicalize to shape (seq_len, head_dim_half):
    # 1) Ensure the last dimension is head_dim_half
    if freqs_real.shape[-1] != head_dim_half:
        raise ValueError(f"Unexpected last dim for freqs: {freqs_real.shape[-1]} (expected {head_dim_half})")
    # 2) Flatten all leading dims to a single row dimension
    freqs_real = freqs_real.reshape(-1, head_dim_half)
    freqs_imag = freqs_imag.reshape(-1, head_dim_half)
    # 3) If we have fewer rows than seq_len, allow broadcasting when single row
    if freqs_real.shape[0] < seq_len:
        if freqs_real.shape[0] == 1:
            freqs_real = freqs_real.expand(seq_len, -1)
            freqs_imag = freqs_imag.expand(seq_len, -1)
        else:
            raise ValueError(f"Insufficient rows in freqs: {freqs_real.shape[0]} < seq_len={seq_len}")
    # 4) If we have more rows than seq_len (e.g., batch present), take the first seq_len rows
    elif freqs_real.shape[0] > seq_len:
        freqs_real = freqs_real[:seq_len]
        freqs_imag = freqs_imag[:seq_len]

    return freqs_real, freqs_imag


def _maybe_to_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t if t.dtype == dtype else t.to(dtype)


def _maybe_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _cast_and_contiguous(q, k, freqs_real, freqs_imag):
    # Choose compute dtype: use fp32 only when inputs are fp32; otherwise keep input dtype for performance
    compute_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype

    # Make sure q/k share the same dtype before casting to compute dtype
    if k.dtype != q.dtype:
        k = k.to(q.dtype)

    q = _maybe_contiguous(_maybe_to_dtype(q, compute_dtype))
    k = _maybe_contiguous(_maybe_to_dtype(k, compute_dtype))
    freqs_real = _maybe_contiguous(_maybe_to_dtype(freqs_real, compute_dtype))
    freqs_imag = _maybe_contiguous(_maybe_to_dtype(freqs_imag, compute_dtype))
    return q, k, freqs_real, freqs_imag


@triton.jit
def _llama4_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_real_ptr,
    freqs_imag_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    freqs_row_stride,
    seq_len,
    batch_size,
    imag_sign,
    head_dim_half: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_k_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    H100-optimized RoPE kernel with improved parallelization across heads and dimensions.
    Grid: (batch*seq, head)
    """
    # 2D grid
    pid_bs = tl.program_id(0)  # over batch*seq
    pid_h = tl.program_id(1)  # over heads

    batch_idx = pid_bs // seq_len
    seq_idx = pid_bs % seq_len

    # Bounds check
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Base pointers for this (batch, seq) position
    base_offset = batch_idx * seq_len + seq_idx
    q_base = q_ptr + base_offset * q_row_stride
    k_base = k_ptr + base_offset * k_row_stride

    # Tiling over dim/2
    for d_start in tl.static_range(0, head_dim_half, BLOCK_SIZE):
        d_indices = d_start + tl.arange(0, BLOCK_SIZE)
        mask_d = d_indices < head_dim_half

        # Load frequencies once per tile (freqs layout: [seq_len, head_dim_half])
        freq_idx = d_indices
        freqs_real = tl.load(freqs_real_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask_d, other=0.0)
        freqs_imag = tl.load(freqs_imag_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask_d, other=0.0)
        freqs_imag = freqs_imag * imag_sign

        # Process one query head per program in pid_h
        if pid_h < n_q_heads:
            q_head_ptr = q_base + pid_h * q_head_stride
            q_real = tl.load(q_head_ptr + d_indices * 2, mask=mask_d, other=0.0)
            q_imag = tl.load(q_head_ptr + d_indices * 2 + 1, mask=mask_d, other=0.0)

            # Complex multiply with FMAs: (a+ib)*(c+i d) = (a*c - b*d) + i(a*d + b*c)
            new_q_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
            new_q_imag = tl.math.fma(q_real, freqs_imag, q_imag * freqs_real)

            tl.store(q_head_ptr + d_indices * 2, new_q_real, mask=mask_d)
            tl.store(q_head_ptr + d_indices * 2 + 1, new_q_imag, mask=mask_d)

        # Process one key head per program in pid_h
        if pid_h < n_k_heads:
            k_head_ptr = k_base + pid_h * k_head_stride
            k_real = tl.load(k_head_ptr + d_indices * 2, mask=mask_d, other=0.0)
            k_imag = tl.load(k_head_ptr + d_indices * 2 + 1, mask=mask_d, other=0.0)

            new_k_real = tl.math.fma(k_real, freqs_real, -(k_imag * freqs_imag))
            new_k_imag = tl.math.fma(k_real, freqs_imag, k_imag * freqs_real)

            tl.store(k_head_ptr + d_indices * 2, new_k_real, mask=mask_d)
            tl.store(k_head_ptr + d_indices * 2 + 1, new_k_imag, mask=mask_d)


def _select_kernel_meta(head_dim_half: int):
    # Heuristic tuning for block size and num_warps
    if head_dim_half >= 256:
        return 128, 8
    if head_dim_half >= 96:
        return 128, 4
    if head_dim_half >= 48:
        return 64, 4
    if head_dim_half >= 24:
        return 32, 2
    return 16, 2


def llama4_rope_forward(q, k, freqs_cis, BLOCK_SIZE: int = None, imag_sign: float = 1.0):
    # Save original dtype for casting back
    original_dtype = q.dtype

    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape
    head_dim_half = head_dim // 2

    # Prepare frequencies
    freqs_real, freqs_imag = _prepare_freqs(freqs_cis, seq_len, head_dim_half)

    # Cast to appropriate dtype and make contiguous only when needed
    q, k, freqs_real, freqs_imag = _cast_and_contiguous(q, k, freqs_real, freqs_imag)

    # H100-optimized meta-params
    if BLOCK_SIZE is None:
        BLOCK_SIZE, num_warps = _select_kernel_meta(head_dim_half)
    else:
        # Provide a default num_warps if caller pins BLOCK_SIZE
        _, num_warps = _select_kernel_meta(head_dim_half)

    # 2D grid: one program per (batch, seq, head)
    n_heads_max = max(n_q_heads, n_k_heads)
    grid = (batch_size * seq_len, n_heads_max)

    # Launch kernel
    _llama4_rope_kernel[grid](
        q,
        k,
        freqs_real,
        freqs_imag,
        q.stride(1),
        k.stride(1),
        q.stride(2),
        k.stride(2),
        freqs_real.stride(0),
        seq_len,
        batch_size,
        imag_sign,
        head_dim_half,
        n_q_heads,
        n_k_heads,
        BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )

    # Cast back to original dtype only if it differs from compute dtype
    if q.dtype != original_dtype:
        q = q.to(original_dtype)
    if k.dtype != original_dtype:
        k = k.to(original_dtype)

    return q, k


class LigerLlama4RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis, BLOCK_SIZE: int = None):
        q_out, k_out = llama4_rope_forward(q, k, freqs_cis, BLOCK_SIZE, imag_sign=1.0)
        ctx.save_for_backward(freqs_cis.detach() if isinstance(freqs_cis, torch.Tensor) else freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors
        BLOCK_SIZE = getattr(ctx, "BLOCK_SIZE", None)
        # Use imag_sign=-1.0 for conjugate without materializing a new tensor
        dq_out, dk_out = llama4_rope_forward(dq, dk, freqs_cis, BLOCK_SIZE, imag_sign=-1.0)
        return dq_out, dk_out, None
