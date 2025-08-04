"""
Triton implementation of Llama4 Rotary Position Embedding (RoPE).
Supports both text (complex polar) and vision (2D spatial) variants.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _llama4_rope_kernel(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    freqs_real_ptr,
    freqs_imag_ptr,
    freqs_row_stride,
    seq_len,
    batch_size: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply Llama4 RoPE using the exact same complex arithmetic as original HuggingFace.
    """
    pid = tl.program_id(0)

    # Decompose to get batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Bounds check
    if batch_idx >= batch_size:
        return
    if seq_idx >= seq_len:
        return

    # Calculate base pointers for this (batch, seq) position
    row_idx = batch_idx * seq_len + seq_idx
    q_ptr = q_ptr + row_idx * q_row_stride
    k_ptr = k_ptr + row_idx * k_row_stride

    # Process each query head
    for h in range(n_q_heads):
        q_head_ptr = q_ptr + h * head_dim

        # Process dimensions in blocks
        for d_start in tl.static_range(0, head_dim // 2, BLOCK_SIZE):
            # Load indices for this block
            d_indices = d_start + tl.arange(0, BLOCK_SIZE)
            mask = d_indices < (head_dim // 2)

            # Load frequencies (real and imaginary parts) with proper indexing
            # freqs_real and freqs_imag are (seq_len, head_dim//2)
            # We need to access freqs_real[seq_idx, freq_idx] and freqs_imag[seq_idx, freq_idx]
            freq_idx = d_indices
            freqs_real = tl.load(freqs_real_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask, other=0.0)
            freqs_imag = tl.load(freqs_imag_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask, other=0.0)

            # Load consecutive pairs as real/imaginary parts
            # For input [1,2,3,4], we want q_real = [1,3] and q_imag = [2,4]
            # This matches torch.view_as_complex behavior
            q_real = tl.load(q_head_ptr + d_indices * 2, mask=mask, other=0.0)
            q_imag = tl.load(q_head_ptr + d_indices * 2 + 1, mask=mask, other=0.0)

            # Cast to float32 for high-precision computation
            q_real = q_real.to(tl.float32)
            q_imag = q_imag.to(tl.float32)
            freqs_real = freqs_real.to(tl.float32)
            freqs_imag = freqs_imag.to(tl.float32)

            # Complex multiplication: (q_real + i*q_imag) * (freqs_real + i*freqs_imag)
            # Real part: q_real*freqs_real - q_imag*freqs_imag
            # Imag part: q_real*freqs_imag + q_imag*freqs_real
            new_q_real = q_real * freqs_real - q_imag * freqs_imag
            new_q_imag = q_real * freqs_imag + q_imag * freqs_real

            # Store results back to consecutive pairs
            tl.store(q_head_ptr + d_indices * 2, new_q_real, mask=mask)
            tl.store(q_head_ptr + d_indices * 2 + 1, new_q_imag, mask=mask)

    # Process each key head
    for h in range(n_k_heads):
        k_head_ptr = k_ptr + h * head_dim

        # Process dimensions in blocks
        for d_start in tl.static_range(0, head_dim // 2, BLOCK_SIZE):
            # Load indices for this block
            d_indices = d_start + tl.arange(0, BLOCK_SIZE)
            mask = d_indices < (head_dim // 2)

            # Load frequencies (real and imaginary parts) with proper indexing
            # freqs_real and freqs_imag are (seq_len, head_dim//2)
            # We need to access freqs_real[seq_idx, freq_idx] and freqs_imag[seq_idx, freq_idx]
            freq_idx = d_indices
            freqs_real = tl.load(freqs_real_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask, other=0.0)
            freqs_imag = tl.load(freqs_imag_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask, other=0.0)

            # Load consecutive pairs as real/imaginary parts
            # For input [5,6,7,8], we want k_real = [5,7] and k_imag = [6,8]
            # This matches torch.view_as_complex behavior
            k_real = tl.load(k_head_ptr + d_indices * 2, mask=mask, other=0.0)
            k_imag = tl.load(k_head_ptr + d_indices * 2 + 1, mask=mask, other=0.0)

            # Cast to float32 for high-precision computation
            k_real = k_real.to(tl.float32)
            k_imag = k_imag.to(tl.float32)
            freqs_real = freqs_real.to(tl.float32)
            freqs_imag = freqs_imag.to(tl.float32)

            # Complex multiplication: (k_real + i*k_imag) * (freqs_real + i*freqs_imag)
            new_k_real = k_real * freqs_real - k_imag * freqs_imag
            new_k_imag = k_real * freqs_imag + k_imag * freqs_real

            # Store results back to consecutive pairs
            tl.store(k_head_ptr + d_indices * 2, new_k_real, mask=mask)
            tl.store(k_head_ptr + d_indices * 2 + 1, new_k_imag, mask=mask)


def llama4_rope_forward(q, k, freqs_cis):
    """
    Apply Llama4 RoPE to query and key tensors using fused Triton kernel.

    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        freqs_cis: Complex frequency tensor

    Returns:
        Tuple of (q, k) with RoPE applied
    """
    # Input tensors are already in (batch, seq, num_heads, head_dim) format
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    _, _, num_k_heads, _ = k.shape

    # Make contiguous - this is critical for Triton!
    q = q.contiguous()
    k = k.contiguous()

    # Store original dtype to cast back later
    original_dtype = q.dtype

    # Cast to float32 for high-precision computation to match original HF implementation
    q = q.float()
    k = k.float()

    # Handle complex frequencies
    if freqs_cis.is_complex():
        freqs_real = freqs_cis.real
        freqs_imag = freqs_cis.imag
    else:
        # If not complex, assume it's already split
        freq_dim = freqs_cis.shape[-1] // 2
        freqs_real = freqs_cis[..., :freq_dim]
        freqs_imag = freqs_cis[..., freq_dim:]

    # Ensure frequencies are in float32 precision to match original HF implementation
    freqs_real = freqs_real.float()
    freqs_imag = freqs_imag.float()

    # Handle frequency tensor shape
    if freqs_real.dim() == 3:
        if freqs_real.shape[0] == 1:
            freqs_real = freqs_real[0]
            freqs_imag = freqs_imag[0]
        elif freqs_real.shape[0] == batch_size:
            freqs_real = freqs_real[0]
            freqs_imag = freqs_imag[0]

    # Ensure 2D shape (seq_len, freq_dim)
    if freqs_real.dim() == 1:
        freqs_real = freqs_real.unsqueeze(0).expand(seq_len, -1)
        freqs_imag = freqs_imag.unsqueeze(0).expand(seq_len, -1)

    # Truncate if needed
    if freqs_real.shape[0] > seq_len:
        freqs_real = freqs_real[:seq_len]
        freqs_imag = freqs_imag[:seq_len]
    elif freqs_real.shape[0] < seq_len:
        pad_len = seq_len - freqs_real.shape[0]
        freqs_real = torch.nn.functional.pad(freqs_real, (0, 0, 0, pad_len))
        freqs_imag = torch.nn.functional.pad(freqs_imag, (0, 0, 0, pad_len))

    # Handle frequency dimension
    freq_dim = freqs_real.shape[1]
    if freq_dim == head_dim // 2:
        pass
    elif freq_dim == head_dim // 4:
        freqs_real = freqs_real.repeat_interleave(2, dim=1)
        freqs_imag = freqs_imag.repeat_interleave(2, dim=1)
    elif freq_dim == 5:  # Vision RoPE specific pattern
        # For vision RoPE, we need to repeat the 5 frequencies to match head_dim//2
        repeat_factor = (head_dim // 2) // freq_dim
        if repeat_factor * freq_dim == head_dim // 2:
            freqs_real = freqs_real.repeat_interleave(repeat_factor, dim=1)
            freqs_imag = freqs_imag.repeat_interleave(repeat_factor, dim=1)
        else:
            # If exact division doesn't work, pad to the required size
            target_dim = head_dim // 2
            if freq_dim < target_dim:
                # Repeat and pad
                repeat_times = target_dim // freq_dim
                remainder = target_dim % freq_dim
                freqs_real = freqs_real.repeat_interleave(repeat_times, dim=1)
                freqs_imag = freqs_imag.repeat_interleave(repeat_times, dim=1)
                if remainder > 0:
                    freqs_real = torch.cat([freqs_real, freqs_real[:, :remainder]], dim=1)
                    freqs_imag = torch.cat([freqs_imag, freqs_imag[:, :remainder]], dim=1)
            else:
                # Truncate if too large
                freqs_real = freqs_real[:, :target_dim]
                freqs_imag = freqs_imag[:, :target_dim]
    else:
        repeat_factor = (head_dim // 2) // freq_dim
        if repeat_factor * freq_dim == head_dim // 2:
            freqs_real = freqs_real.repeat_interleave(repeat_factor, dim=1)
            freqs_imag = freqs_imag.repeat_interleave(repeat_factor, dim=1)
        else:
            # Try to handle any other frequency dimension
            target_dim = head_dim // 2
            if freq_dim < target_dim:
                # Repeat and pad
                repeat_times = target_dim // freq_dim
                remainder = target_dim % freq_dim
                freqs_real = freqs_real.repeat_interleave(repeat_times, dim=1)
                freqs_imag = freqs_imag.repeat_interleave(repeat_times, dim=1)
                if remainder > 0:
                    freqs_real = torch.cat([freqs_real, freqs_real[:, :remainder]], dim=1)
                    freqs_imag = torch.cat([freqs_imag, freqs_imag[:, :remainder]], dim=1)
            else:
                # Truncate if too large
                freqs_real = freqs_real[:, :target_dim]
                freqs_imag = freqs_imag[:, :target_dim]

    # Ensure contiguous and correct dtype
    freqs_real = freqs_real.contiguous()
    freqs_imag = freqs_imag.contiguous()

    # Grid: one program per (batch, seq) position
    grid = (batch_size * seq_len,)

    # Block size for dimension processing
    BLOCK_SIZE = min(64, triton.next_power_of_2(head_dim // 2))

    # Launch kernel
    _llama4_rope_kernel[grid](
        q,
        q.stride(1),  # stride for seq dimension (num_heads * head_dim)
        k,
        k.stride(1),  # stride for seq dimension (num_heads * head_dim)
        freqs_real,
        freqs_imag,
        freqs_real.stride(0) if freqs_real.dim() >= 2 else 0,  # seq stride in freqs
        seq_len,
        batch_size,
        num_q_heads,
        num_k_heads,
        head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Cast back to original dtype to match original HF implementation
    q = q.to(original_dtype)
    k = k.to(original_dtype)

    return q, k


class LigerLlama4RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis):
        # Clone inputs to avoid modifying originals
        q_out = q.clone()
        k_out = k.clone()

        # Apply RoPE
        q_out, k_out = llama4_rope_forward(q_out, k_out, freqs_cis)

        # Save for backward
        ctx.save_for_backward(freqs_cis)

        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors

        # Clone gradients
        dq_out = dq.clone()
        dk_out = dk.clone()

        # For backward pass, use conjugate of frequencies
        if freqs_cis.is_complex():
            freqs_cis_conj = torch.conj(freqs_cis)
        else:
            # If split into real/imag, negate the imaginary part
            freq_dim = freqs_cis.shape[-1] // 2
            freqs_cis_conj = freqs_cis.clone()
            freqs_cis_conj[..., freq_dim:] = -freqs_cis_conj[..., freq_dim:]

        # Apply RoPE with conjugate frequencies
        dq_out, dk_out = llama4_rope_forward(dq_out, dk_out, freqs_cis_conj)

        return dq_out, dk_out, None
