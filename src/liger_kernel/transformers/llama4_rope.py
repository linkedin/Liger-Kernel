"""
Liger Kernel implementation of Llama4 Rotary Position Embedding (RoPE).
Supports both text and vision RoPE variants with fused operations for optimal performance.
"""

import torch

from liger_kernel.ops.llama4_rope import LigerLlama4RopeFunction


def liger_llama4_text_rotary_pos_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Liger-optimized implementation of Llama4 text rotary position embedding.

    This implementation uses a fused Triton kernel for complex multiplication,
    providing significant performance improvements over the original PyTorch implementation.

    Args:
        xq (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        xk (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        freqs_cis (torch.Tensor): Complex frequency tensor from Llama4TextRotaryEmbedding

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors
    """
    # Use fused Triton kernel for complex RoPE
    return LigerLlama4RopeFunction.apply(xq, xk, freqs_cis)


def liger_llama4_vision_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Liger-optimized implementation of Llama4 vision rotary position embedding.

    This implementation uses the same fused Triton kernel as text RoPE,
    providing performance improvements for vision transformer attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        freqs_ci (torch.Tensor): Complex frequency tensor for 2D positions

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors
    """
    # Handle broadcasting for vision RoPE
    if freqs_ci.dim() == 3:  # (seq_len, head_dim//2)
        # Expand to match (batch, seq_len, head_dim//2)
        freqs_ci = freqs_ci.unsqueeze(0).expand(query.shape[0], -1, -1)

    # Use the same fused kernel as text RoPE
    return LigerLlama4RopeFunction.apply(query, key, freqs_ci)


# Note: We only patch the functions, not the classes
# The original Llama4TextRotaryEmbedding and Llama4VisionRotaryEmbedding classes remain unchanged


# Convenience functions for monkey patching
def apply_liger_llama4_rope_full(modeling_module):
    """
    Apply Liger optimizations to Llama4 RoPE functions.

    Args:
        modeling_module: The transformers modeling module to patch
    """
    # Replace the text RoPE function
    modeling_module.apply_rotary_emb = liger_llama4_text_rotary_pos_emb

    # Replace the vision RoPE function
    modeling_module.vision_apply_rotary_emb = liger_llama4_vision_rotary_pos_emb
