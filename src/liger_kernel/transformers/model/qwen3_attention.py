from typing import Callable
from typing import Optional

import torch

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

from liger_kernel.transformers.qk_norm_rope import liger_qk_norm_rope


def qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in replacement for ``Qwen3Attention.forward`` using the fused
    QK-Norm + RoPE Triton kernel.

    The reference implementation does (modeling_qwen3.py):

        query_states = self.q_norm(self.q_proj(h).view(hidden_shape)).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(h).view(hidden_shape)).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    We fuse ``q_norm``/``k_norm`` + ``apply_rotary_pos_emb`` into a single kernel.
    The RMSNorm weights live on ``self.q_norm``/``self.k_norm`` and act on the
    ``head_dim`` axis, which is exactly the layout the fused kernel expects.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # pre-transpose layout: (bsz, seq_len, n_head, head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = liger_qk_norm_rope(
        query_states,
        key_states,
        self.q_norm.weight,
        self.k_norm.weight,
        cos,
        sin,
        self.q_norm.variance_epsilon,
    )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
