from typing import Tuple

import torch

from liger_kernel.ops.rope import LigerRopeFunction


def liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """

    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


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
    """
    print(f"{q.shape=}, {k.shape=}")
    print(f"{cos.shape=}, {sin.shape=}")
    print(f"{q.stride()=}, {k.stride()=}")
    print(f"{cos.stride()=}, {sin.stride()=}")
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype

    q32 = q.to(torch.float32).unsqueeze(0).transpose(1, 2)  # (1, num_heads, seq_length, head_dim)
    k32 = k.to(torch.float32).unsqueeze(0).transpose(1, 2)  # (1, num_heads, seq_length, head_dim)
    cos32 = cos.to(torch.float32)
    sin32 = sin.to(torch.float32)

    q_out, k_out = liger_rotary_pos_emb(q32, k32, cos32, sin32)
    q_out = q_out.transpose(1, 2).squeeze(0).to(orig_q_dtype)
    k_out = k_out.transpose(1, 2).squeeze(0).to(orig_q_dtype)
    return q_out, k_out
