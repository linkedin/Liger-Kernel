from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.utils import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerInternVLCausalLMOutputWithPast


# Copied from https://github.com/huggingface/transformers/blob/d888bd435d0c0eaabaabad5b33d52af518c7187c/src/transformers/models/internvl/modeling_internvl.py#L862
@can_return_tuple
def lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, List[int]]] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: Optional[torch.Tensor] = None,
    skip_logits: Optional[bool] = None,  # Added argument for liger-kernel
    **lm_kwargs,  # renamed from kwargs
) -> Union[Tuple, LigerInternVLCausalLMOutputWithPast]:
    r"""
    Example:

    ```python
    >>> import torch
    >>> from transformers import AutoProcessor, AutoModelForImageTextToText

    >>> torch_device = "cuda"
    >>> processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
    >>> model = AutoModelForImageTextToText.from_pretrained(
    ...     "OpenGVLab/InternVL3-1B-hf", dtype=torch.bfloat16, device_map=torch_device
    ... )

    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {
    ...                 "type": "image",
    ...                 "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    ...             },
    ...             {
    ...                 "type": "image",
    ...                 "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
    ...             },
    ...             {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
    ...         ],
    ...     },
    ... ]

    >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
    >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
    >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
    The images depict the Statue of Liberty and the Golden Gate Bridge.
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

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        image_sizes=image_sizes,
        **lm_kwargs,
    )

    # Copied from llava.py
    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = lm_kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        # By default, if in training mode, don't materialize logits
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **lm_kwargs,
        )
        loss, _, token_accuracy = unpack_cross_entropy_result(result)

    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **lm_kwargs
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = (loss,) + output if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        return output

    # Return custom output class with token_accuracy field
    return LigerInternVLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        token_accuracy=token_accuracy,
    )
