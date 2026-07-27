"""
Liger Kernel implementation of Llama4 Rotary Position Embedding (RoPE).
Supports both text and vision RoPE variants with fused operations for optimal performance.
"""

import torch

from liger_kernel.ops import LigerLlama4RopeFunction


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


def _llama4_vision_rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference (non-fused) Llama4 vision RoPE, matching HF ``vision_apply_rotary_emb``.

    Used as a fallback when ``freqs_ci`` is not a complex tensor (see
    ``liger_llama4_vision_rotary_pos_emb``). ``query``/``key`` are interpreted as
    interleaved real/imag pairs; multiplying by ``freqs_ci`` (which may itself be
    real, e.g. a complex buffer whose imaginary part was dropped by a dtype cast)
    reproduces exactly what the HF implementation computes.
    """
    query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    ndim = query_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query_.shape)]
    freqs_ci = freqs_ci.view(*shape).to(query_.device)
    query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
    key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)
    return query_out.type_as(query), key_out.type_as(key)


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
    # The vision RoPE frequencies are stored on the model as a complex64 buffer
    # (``Llama4VisionRotaryEmbedding.freqs_ci``). Casting the model to a real dtype
    # (e.g. ``model.to(torch.bfloat16)``) silently casts that buffer to a real
    # tensor, discarding the imaginary part. The fused complex kernel below assumes
    # ``freqs_ci`` is complex (or its real/imag interleaving); a real tensor makes it
    # index out of bounds and emit NaNs. Fall back to the reference implementation,
    # which matches HF's ``vision_apply_rotary_emb`` for both complex and real freqs.
    if not torch.is_complex(freqs_ci):
        return _llama4_vision_rope_reference(query, key, freqs_ci)

    # Handle broadcasting for vision RoPE
    if freqs_ci.dim() == 3:
        try:
            # Try the regular 3D expansion
            freqs_ci = freqs_ci.unsqueeze(0).expand(query.shape[0], -1, -1)
        except RuntimeError as e:
            if "expand" in str(e) and "4" in str(e):
                # The tensor is actually 4D internally, handle it differently
                freqs_ci = freqs_ci.squeeze(1)  # Remove the middle dimension
                freqs_ci = freqs_ci.unsqueeze(0).expand(query.shape[0], -1, -1)
            else:
                raise e
    elif freqs_ci.dim() == 4:  # (1, seq_len, 1, head_dim//2) - already properly shaped
        # Squeeze the middle dimension to get (1, seq_len, head_dim//2)
        freqs_ci = freqs_ci.squeeze(2)
    elif freqs_ci.dim() == 2:  # (seq_len, head_dim//2) - needs expansion
        freqs_ci = freqs_ci.unsqueeze(0).expand(query.shape[0], -1, -1)
    else:
        raise ValueError(f"Unexpected freqs_ci shape: {freqs_ci.shape}")

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
