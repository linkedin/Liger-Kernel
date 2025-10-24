from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModelOutputWithPast
from transformers.utils import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss


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
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, Qwen3VLMoeCausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.text_config.output_attentions
    config = self.config
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", getattr(config, "hidden_size", None))
    vocab_size = getattr(text_config, "vocab_size", getattr(config, "vocab_size", None))
    num_experts = getattr(text_config, "num_experts", getattr(config, "num_experts", None))
    num_experts_per_tok = getattr(text_config, "num_experts_per_tok", getattr(config, "num_experts_per_tok", None))
    router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", getattr(config, "router_aux_loss_coef", 0.0))

    output_attentions = output_attentions if output_attentions is not None else getattr(config, "output_attentions", False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else getattr(config, "output_hidden_states", False)
    )
    output_router_logits = (
        output_router_logits if output_router_logits is not None else getattr(text_config, "output_router_logits", False)
    )
    return_dict = return_dict if return_dict is not None else getattr(config, "use_return_dict", True)

    shift_labels = kwargs.pop("shift_labels", None)

    outputs: Qwen3VLMoeModelOutputWithPast = self.model(
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
        output_router_logits=output_router_logits,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    logits = None
    loss = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        loss = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=hidden_size,
            **kwargs,
        )
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=vocab_size,
                **kwargs,
            )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            num_experts,
            num_experts_per_tok,
            attention_mask,
        )
        if loss is not None:
            loss = loss + router_aux_loss_coef * aux_loss.to(loss.device)

    return Qwen3VLMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        rope_deltas=getattr(outputs, "rope_deltas", None),
    )
