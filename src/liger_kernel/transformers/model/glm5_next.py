import torch

from transformers.modeling_outputs import MoeModelOutputWithPast

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast


def lce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    output_router_logits: bool | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    skip_logits: bool | None = None,
    **kwargs,
) -> LigerMoeCausalLMOutputWithPast:
    r"""
    Glm5NextForConditionalGeneration.forward with the lm_head projection fused into the loss.

    Newer transformers versions add arguments such as `mm_encoder_outputs`; they reach `self.model`
    through `**kwargs`.
    """
    text_config = self.config.text_config
    output_router_logits = (
        output_router_logits if output_router_logits is not None else text_config.output_router_logits
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=text_config.hidden_size,
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
                vocab_size=text_config.vocab_size,
                **kwargs,
            )

    aux_loss = None
    if output_router_logits:
        from transformers.models.glm5_next.modeling_glm5_next import load_balancing_loss_func

        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None and loss is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    return LigerMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
