from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.cache_utils import Cache
from transformers.utils import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerLfm2VlCausalLMOutputWithPast


@can_return_tuple
def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    spatial_shapes: Optional[torch.Tensor] = None,
    pixel_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, LigerLfm2VlCausalLMOutputWithPast]:
    """LFM2-VL forward with fused linear cross entropy during training."""
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        spatial_shapes=spatial_shapes,
        pixel_attention_mask=pixel_attention_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]
    shift_labels = kwargs.pop("shift_labels", None)

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")
    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None
    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    return LigerLfm2VlCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
