from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling
from transformers.utils.deprecation import deprecate_kwarg

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


def lce_forward_deprecated(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

            image_features = self.multi_modal_projector(selected_image_feature)
            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

    # TODO: @raushan retain only the new behavior after v4.47
    elif image_features is not None:
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    outputs = self.language_model.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]

    loss = None
    logits = None

    if self.training and (labels is not None):
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(hidden_states.shape[1] - 1) :].to(hidden_states.device)
            shift_hidden_states = hidden_states[..., :-1, :][
                shift_attention_mask.to(hidden_states.device) != 0
            ].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.language_model.lm_head.weight, shift_hidden_states, shift_labels)

    if not return_dict:
        # NOTE: This part has not been tested.
        output = outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
def lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: torch.Tensor = None,
    skip_logits: Optional[bool] = None,
    **lm_kwargs,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).


    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
        )

        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_tokens = (input_ids == self.config.image_token_index).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    outputs = self.language_model.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )
    hidden_states = outputs[0]

    loss = None
    logits = None

    # Overwrite skip_logits, since llava never materializes logits
    skip_logits = labels is not None

    if skip_logits:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(hidden_states.shape[1] - 1) :].to(hidden_states.device)
            shift_hidden_states = hidden_states[..., :-1, :][
                shift_attention_mask.to(hidden_states.device) != 0
            ].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(
            self.language_model.lm_head.weight,
            shift_hidden_states.view(-1, shift_hidden_states.size(-1)),
            shift_labels.view(-1).to(shift_hidden_states.device),
        )

    if not return_dict:
        # NOTE: This part has not been tested.
        output = outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )
