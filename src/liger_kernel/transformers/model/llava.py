from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from transformers.models.llava.modeling_llava import _CONFIG_FOR_DOC
from transformers.models.llava.modeling_llava import LLAVA_INPUTS_DOCSTRING
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.models.llava.modeling_llava import logger
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.utils import replace_return_docstrings

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


@add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
    num_logits_to_keep: int = 0,
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

    if num_logits_to_keep != 0:
        logger.warning_once("llava liger-kernel does not support num_logits_to_keep. This argument is ignored.")

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
        )

        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
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
        # num_logits_to_keep=num_logits_to_keep,
    )

    hidden_states = outputs[0]

    logits = None
    loss = None
    if self.training and (labels is not None):
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
    else:
        logits = self.language_model.lm_head(hidden_states)
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )
