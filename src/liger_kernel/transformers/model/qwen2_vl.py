from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    _CONFIG_FOR_DOC,
    QWEN2_VL_INPUTS_DOCSTRING,
    Qwen2VLCausalLMOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)


@add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
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
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    r"""
    Copy paste Qwen2VL's forward but replace torch cross entropy with liger fused linear cross entropy

    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(
                inputs_embeds.device
            )
            image_mask = input_ids == self.config.image_token_id
            if self.training:
                inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = image_embeds
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(
                inputs_embeds.device
            )
            video_mask = input_ids == self.config.video_token_id
            inputs_embeds[video_mask] = video_embeds
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    # The code is copied from https://github.com/huggingface/transformers/pull/33487
    if position_ids is None and input_ids is not None:
        position_ids, _ = self.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
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
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
    else:
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )
