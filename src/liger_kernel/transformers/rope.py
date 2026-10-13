from typing import Tuple

import torch

from liger_kernel.backends import dispatch


def liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, rotary_dim=None):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.
        rotary_dim (int, optional): How many leading entries of each head are rotated. Defaults to None, which
            infers it from the width of cos/sin. Pass it when cos/sin are not duplicated, i.e. have width
            rotary_dim / 2 as in GPT-OSS.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """

    if rotary_dim is None:
        return dispatch("rope", q, k, cos, sin, position_ids, unsqueeze_dim)
    return dispatch("rope", q, k, cos, sin, position_ids, unsqueeze_dim, rotary_dim=rotary_dim)


def liger_rotary_pos_emb_gpt_oss(q, k, cos, sin):
    """
    liger_rotary_pos_emb for GPT-OSS's apply_rotary_pos_emb.

    GPT-OSS rotates the whole head, but its rotary embedding returns cos/sin of width head_dim / 2
    (`emb = freqs`, not `cat(freqs, freqs)`), so the rotary width cannot be inferred from cos/sin
    and is passed explicitly.
    Reference: https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py
    """
    return liger_rotary_pos_emb(q, k, cos, sin, rotary_dim=q.shape[-1])


def liger_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modified version of liger_rotary_pos_emb for qwen3_vl's apply_rotary_pos_emb_vision function.
    Manually tranposed the input and output to match the expected shape for liger_rotary_pos_emb.
    Reference: https://https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L116

    Args:
        q (torch.Tensor): The query tensor of shape (seq_length, num_heads, head_dim),
        with stride (num_heads * head_dim, head_dim, 1).
        k (torch.Tensor): The query tensor of shape (seq_length, num_heads, head_dim),
        with stride (num_heads * head_dim, head_dim, 1). Same as q.
        cos (torch.Tensor): The cosine tensor of shape (seq_length, head_dim).
        sin (torch.Tensor): The sine tensor of shape (seq_length, head_dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with the same shape and stride as inputs.
    """
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype

    # tranpose to (1, num_heads, seq_length, head_dim) and cast to float32 to match liger_rotary_pos_emb input shape
    # also unsqueeze for batch dim
    q32 = q.to(torch.float32).unsqueeze(0).transpose(1, 2)
    k32 = k.to(torch.float32).unsqueeze(0).transpose(1, 2)
    cos32 = cos.to(torch.float32)
    sin32 = sin.to(torch.float32)

    q_out, k_out = liger_rotary_pos_emb(q32, k32, cos32, sin32)

    # transpose back to (seq_length, num_heads, head_dim) and cast back to original dtype
    # also squeeze out batch dim
    q_out = q_out.transpose(1, 2).squeeze(0).to(orig_q_dtype)
    k_out = k_out.transpose(1, 2).squeeze(0).to(orig_k_dtype)
    return q_out, k_out
