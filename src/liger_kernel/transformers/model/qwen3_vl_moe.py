from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import load_balancing_loss_func
from transformers.utils import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerQwen3VLMoeCausalLMOutputWithPast


@can_return_tuple
def lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, LigerQwen3VLMoeCausalLMOutputWithPast]:
    """
    Qwen3-VL-MoE forward with fused linear cross entropy support mirroring Qwen3-VL behaviour.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    shift_labels = kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

    # Compute auxiliary load-balancing loss for MoE when requested
    aux_loss = None
    if kwargs.get("output_router_logits", False):
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
        )
        # If we computed training loss, add the scaled aux loss to it
        if loss is not None and aux_loss is not None:
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = (loss,) + output if loss is not None else output
        output = output + (aux_loss,) if aux_loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        return output

    return LigerQwen3VLMoeCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        aux_loss=aux_loss,
        token_accuracy=token_accuracy,
    )
