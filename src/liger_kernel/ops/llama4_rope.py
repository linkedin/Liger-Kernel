import torch
import triton
import triton.language as tl


def _prepare_freqs(freqs_cis: torch.Tensor, seq_len: int, head_dim_half: int):
    # Split or unpack complex frequencies into real and imag parts
    if freqs_cis.is_complex():
        freqs_real = freqs_cis.real
        freqs_imag = freqs_cis.imag
    else:
        # Already split: [seq_len, 2*head_dim_half]
        freqs_real = freqs_cis[..., :head_dim_half]
        freqs_imag = freqs_cis[..., head_dim_half:]
    # Handle batch dimension
    if freqs_real.dim() == 3 and freqs_real.shape[0] == 1:
        freqs_real = freqs_real.squeeze(0)
        freqs_imag = freqs_imag.squeeze(0)
    # Expand 1D to 2D
    if freqs_real.dim() == 1:
        freqs_real = freqs_real.unsqueeze(0).expand(seq_len, -1)
        freqs_imag = freqs_imag.unsqueeze(0).expand(seq_len, -1)
    return freqs_real, freqs_imag


def _cast_and_contiguous(q, k, freqs_real, freqs_imag):
    # Cast to float32 and ensure contiguous layout
    return q.float().contiguous(), k.float().contiguous(), \
           freqs_real.float().contiguous(), freqs_imag.float().contiguous()

@triton.jit
def _llama4_rope_kernel(
    q_ptr, k_ptr,
    freqs_real_ptr, freqs_imag_ptr,
    q_row_stride, k_row_stride,
    freqs_row_stride,
    seq_len, batch_size,
    head_dim_half: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_k_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    H100-optimized RoPE kernel with better memory access patterns and warp utilization.
    """
    # 1D grid: one program per (batch, seq) position
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Bounds check
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Base pointers for this (batch, seq) position
    base_offset = batch_idx * seq_len + seq_idx
    q_base = q_ptr + base_offset * q_row_stride
    k_base = k_ptr + base_offset * k_row_stride
    
    # Process dimensions in blocks for better memory access
    for d_start in tl.static_range(0, head_dim_half, BLOCK_SIZE):
        # Load indices for this block
        d_indices = d_start + tl.arange(0, BLOCK_SIZE)
        mask = d_indices < head_dim_half
        
        # Load frequencies ONCE per block (shared across all heads)
        freq_idx = d_indices
        freqs_real = tl.load(freqs_real_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask, other=0.0)
        freqs_imag = tl.load(freqs_imag_ptr + seq_idx * freqs_row_stride + freq_idx, mask=mask, other=0.0)
        
        # Process ALL query heads in this dimension block
        for h in range(n_q_heads):
            q_head_ptr = q_base + h * head_dim_half * 2
            
            # Load consecutive pairs as real/imaginary parts
            q_real = tl.load(q_head_ptr + d_indices * 2, mask=mask, other=0.0)
            q_imag = tl.load(q_head_ptr + d_indices * 2 + 1, mask=mask, other=0.0)
            
            # Complex multiplication
            new_q_real = q_real * freqs_real - q_imag * freqs_imag
            new_q_imag = q_real * freqs_imag + q_imag * freqs_real
            
            # Store results back
            tl.store(q_head_ptr + d_indices * 2, new_q_real, mask=mask)
            tl.store(q_head_ptr + d_indices * 2 + 1, new_q_imag, mask=mask)
        
        # Process ALL key heads in this dimension block
        for h in range(n_k_heads):
            k_head_ptr = k_base + h * head_dim_half * 2
            
            # Load consecutive pairs as real/imaginary parts
            k_real = tl.load(k_head_ptr + d_indices * 2, mask=mask, other=0.0)
            k_imag = tl.load(k_head_ptr + d_indices * 2 + 1, mask=mask, other=0.0)
            
            # Complex multiplication
            new_k_real = k_real * freqs_real - k_imag * freqs_imag
            new_k_imag = k_real * freqs_imag + k_imag * freqs_real
            
            # Store results back
            tl.store(k_head_ptr + d_indices * 2, new_k_real, mask=mask)
            tl.store(k_head_ptr + d_indices * 2 + 1, new_k_imag, mask=mask)


def llama4_rope_forward(q, k, freqs_cis, BLOCK_SIZE: int = None):
    # Save original dtype for casting back
    original_dtype = q.dtype
    
    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape
    head_dim_half = head_dim // 2
    
    # Prepare frequencies
    freqs_real, freqs_imag = _prepare_freqs(freqs_cis, seq_len, head_dim_half)
    
    # Cast to float32 and make contiguous
    q, k, freqs_real, freqs_imag = _cast_and_contiguous(q, k, freqs_real, freqs_imag)
    
    # H100-optimized block size: align with warp size (128)
    if BLOCK_SIZE is None:
        BLOCK_SIZE = min(128, triton.next_power_of_2(head_dim_half))
    
    # 1D grid: one program per (batch, seq) position
    grid = (batch_size * seq_len,)
    
    # Launch kernel
    _llama4_rope_kernel[grid](
        q, k,
        freqs_real, freqs_imag,
        q.stride(1), k.stride(1),
        freqs_real.stride(0),
        seq_len, batch_size,
        head_dim_half, n_q_heads, n_k_heads, BLOCK_SIZE
    )
    
    # Cast back to original dtype
    if original_dtype != torch.float32:
        q = q.to(original_dtype)
        k = k.to(original_dtype)
    
    return q, k

class LigerLlama4RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis, BLOCK_SIZE: int = None):
        q_out, k_out = llama4_rope_forward(q, k, freqs_cis, BLOCK_SIZE)
        ctx.save_for_backward(freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return q_out, k_out
    @staticmethod
    def backward(ctx, dq, dk):
        freqs_cis, = ctx.saved_tensors
        BLOCK_SIZE = getattr(ctx, 'BLOCK_SIZE', None)
        if freqs_cis.is_complex():
            freqs_conj = torch.conj(freqs_cis)
        else:
            dim = freqs_cis.shape[-1] // 2
            freqs_conj = freqs_cis.clone()
            freqs_conj[..., dim:] = -freqs_conj[..., dim:]
        dq_out, dk_out = llama4_rope_forward(dq, dk, freqs_conj, BLOCK_SIZE)
        return dq_out, dk_out, None
