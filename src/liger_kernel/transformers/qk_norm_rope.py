from typing import Tuple

import torch

from liger_kernel.ops.qk_norm_rope import LigerQkNormRopeFunction


def liger_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused per-head RMSNorm(Q/K) followed by rotary positional embedding.

    This fuses the ``q_norm``/``k_norm`` + ``apply_rotary_pos_emb`` sequence used by
    models such as Qwen3.  ``q`` and ``k`` are expected in the *pre-transpose*
    projection layout, i.e. the output of ``proj(x).view(bsz, seq_len, n_head,
    head_dim)`` before ``.transpose(1, 2)``.  The returned tensors are already
    transposed to ``(bsz, n_head, seq_len, head_dim)`` so they can be fed directly
    into the attention interface.

    Args:
        q: query states, shape ``(bsz, seq_len, n_q_head, head_dim)``.
        k: key states, shape ``(bsz, seq_len, n_kv_head, head_dim)``.
        q_weight: RMSNorm weight for the query, shape ``(head_dim,)``.
        k_weight: RMSNorm weight for the key, shape ``(head_dim,)``.
        cos: cosine table, shape ``(1, seq_len, head_dim)`` or ``(bsz, seq_len, head_dim)``.
        sin: sine table, same shape as ``cos``.
        eps: RMSNorm epsilon.

    Returns:
        Tuple of query and key tensors, each ``(bsz, n_head, seq_len, head_dim)``.
    """
    return LigerQkNormRopeFunction.apply(q, k, q_weight, k_weight, cos, sin, eps)
