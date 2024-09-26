from typing import Optional, Tuple

from transformers.cache_utils import Cache
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, logger, LlamaSdpaAttention

from liger_kernel.ops.flash_attention.wrapper import flash_attn_func


# Adapted from LlamaSdpaAttention.forward
def liger_general_sdpa_forward(
    self: LlamaSdpaAttention,  # Might not always be this module in particular, but is a good general placholder
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    if output_attentions:
        raise NotImplementedError("Output attentions")  # TODO: support this?

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj.forward(hidden_states)
    key_states = self.k_proj.forward(hidden_states)
    value_states = self.v_proj.forward(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # key_states = repeat_kv(key_states, self.num_key_value_groups)  not needed as we support GQA
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        attn_bias = attention_mask[:, :, :, : key_states.shape[-2]]
    else:
        attn_bias = None

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if attention_mask is None and q_len > 1 else False

    attn_output = flash_attn_func(
        q=query_states,
        k=key_states,
        v=value_states,
        attention_mask=None,
        attention_bias=attn_bias,
        dropout_p=(self.attention_dropout if self.training else 0.0),
        causal=is_causal,
        softmax_scale=None,
        dropout_seed=None,
    )

    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
