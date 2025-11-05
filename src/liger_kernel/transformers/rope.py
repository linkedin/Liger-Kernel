from typing import Optional
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


def liger_rotary_pos_emb_with_cast(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype

    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    cos32 = cos.to(torch.float32)
    sin32 = sin.to(torch.float32)

    q_out, k_out = liger_rotary_pos_emb(q32, k32, cos32, sin32, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    return q_out.to(orig_q_dtype), k_out.to(orig_k_dtype)


def liger_rotary_pos_emb_with_cast_and_leading_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype

    q32 = q.to(torch.float32).unsqueeze(0)
    k32 = k.to(torch.float32).unsqueeze(0)
    cos32 = cos.to(torch.float32).unsqueeze(0)
    sin32 = sin.to(torch.float32).unsqueeze(0)

    q_out, k_out = liger_rotary_pos_emb(q32, k32, cos32, sin32, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    return q_out.to(orig_q_dtype).squeeze(0), k_out.to(orig_k_dtype).squeeze(0)
