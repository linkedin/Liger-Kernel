import inspect

import torch

from liger_kernel.ops import LigerLfm2ShortConvFunction


def liger_lfm2_short_conv_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
    seq_idx=None,
):
    """Fused full-sequence training forward for LFM2 short convolution."""
    if past_key_values is not None or seq_idx is not None:
        original_forward = getattr(self, "_liger_original_forward", None)
        if original_forward is None:
            original_forward = getattr(self, "slow_forward", None)
        if original_forward is None:
            raise RuntimeError("The original LFM2 short-convolution forward is unavailable for cached execution.")

        parameters = getattr(self, "_liger_original_forward_parameters", None)
        if parameters is None:
            parameters = frozenset(inspect.signature(original_forward).parameters)
            self._liger_original_forward_parameters = parameters
        original_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
        if "cache_position" in parameters:
            original_kwargs["cache_position"] = cache_position
        if "seq_idx" in parameters:
            original_kwargs["seq_idx"] = seq_idx
        return original_forward(hidden_states, **original_kwargs)

    if attention_mask is not None:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

    bcx = self.in_proj(hidden_states)
    hidden_states = LigerLfm2ShortConvFunction.apply(bcx, self.conv.weight, self.conv.bias)
    return self.out_proj(hidden_states)
