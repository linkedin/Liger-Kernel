import torch

from liger_kernel.ops import LigerLfm2ShortConvFunction


def liger_lfm2_short_conv_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
):
    """Fused full-sequence training forward for LFM2 short convolution."""
    if past_key_values is not None:
        return self.slow_forward(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

    bcx = self.in_proj(hidden_states)
    hidden_states = LigerLfm2ShortConvFunction.apply(bcx, self.conv.weight, self.conv.bias)
    return self.out_proj(hidden_states)
