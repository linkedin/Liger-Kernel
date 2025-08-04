#!/usr/bin/env python3
"""
Debug script to compare Llama4 RoPE implementations for float32 precision.
"""

import torch
import time

# Import our implementation
from src.liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb

def original_apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Original HuggingFace implementation for comparison."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Properly broadcast freqs_cis to match the expected shape
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def debug_fp32_rope():
    """Debug float32 RoPE implementation."""
    print("Debugging Float32 Llama4 RoPE Implementation")
    print("=" * 50)
    
    # Test configuration
    batch_size = 1
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    
    print(f"Configuration: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
    
    # Generate test data with float32 precision
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float32)
    freqs = torch.randn(seq_len, head_dim // 2, device='cuda', dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    print(f"\nInput shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  Freqs_cis: {freqs_cis.shape}")
    print(f"  Q dtype: {q.dtype}")
    print(f"  K dtype: {k.dtype}")
    print(f"  Freqs_cis dtype: {freqs_cis.dtype}")
    
    # Test original implementation
    print(f"\nTesting Original HuggingFace...")
    q_orig, k_orig = original_apply_rotary_emb(q.clone(), k.clone(), freqs_cis)
    
    # Test our implementation
    print(f"Testing Liger Implementation...")
    q_liger, k_liger = liger_llama4_text_rotary_pos_emb(q.clone(), k.clone(), freqs_cis)
    
    # Compare results
    q_diff = torch.abs(q_orig - q_liger).max().item()
    k_diff = torch.abs(k_orig - k_liger).max().item()
    
    print(f"\nResults:")
    print(f"  Q max difference: {q_diff:.2e}")
    print(f"  K max difference: {k_diff:.2e}")
    
    if q_diff < 1e-5 and k_diff < 1e-5:
        print("  ✅ PASSED - Float32 numerical accuracy verified")
    else:
        print("  ❌ FAILED - Float32 numerical differences detected")
    
    # Simple test with known values
    print(f"\nSimple Float32 RoPE Test:")
    q_simple = torch.tensor([[[[1., 2., 3., 4.]]]], device='cuda', dtype=torch.float32)
    k_simple = torch.tensor([[[[5., 6., 7., 8.]]]], device='cuda', dtype=torch.float32)
    freqs_simple = torch.tensor([[0.5+0.5j, 0.3+0.7j]], device='cuda', dtype=torch.complex64)
    
    q_orig_simple, k_orig_simple = original_apply_rotary_emb(
        q_simple.clone(), k_simple.clone(), freqs_simple
    )
    q_liger_simple, k_liger_simple = liger_llama4_text_rotary_pos_emb(
        q_simple.clone(), k_simple.clone(), freqs_simple
    )
    
    print(f"  Original Q: {q_orig_simple[0,0,0]}")
    print(f"  Liger Q: {q_liger_simple[0,0,0]}")
    print(f"  Original K: {k_orig_simple[0,0,0]}")
    print(f"  Liger K: {k_liger_simple[0,0,0]}")
    
    # Check intermediate values
    print(f"\nIntermediate Value Comparison:")
    
    # Original approach
    q_orig_intermediate = torch.view_as_complex(q_simple.float().reshape(*q_simple.shape[:-1], -1, 2))
    freqs_orig_broadcast = freqs_simple.unsqueeze(0).unsqueeze(2)
    q_orig_complex = q_orig_intermediate * freqs_orig_broadcast
    q_orig_final = torch.view_as_real(q_orig_complex).flatten(3)
    
    # Our approach - let's trace through the conversion
    print(f"  Original complex Q shape: {q_orig_intermediate.shape}")
    print(f"  Original freqs broadcast shape: {freqs_orig_broadcast.shape}")
    print(f"  Original complex result shape: {q_orig_complex.shape}")
    print(f"  Original final shape: {q_orig_final.shape}")
    
    # Let's also check what our conversion produces
    cos = freqs_simple.real
    sin = freqs_simple.imag
    print(f"  Our cos shape: {cos.shape}")
    print(f"  Our sin shape: {sin.shape}")
    print(f"  Our cos values: {cos}")
    print(f"  Our sin values: {sin}")

if __name__ == "__main__":
    debug_fp32_rope() 