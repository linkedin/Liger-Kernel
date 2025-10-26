from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

import torch

from transformers.models.smolvlm.modeling_smolvlm import SmolVLMCausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils.generic import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

if TYPE_CHECKING:
    from transformers.cache_utils import Cache
    from transformers.utils.generic import TransformersKwargs


# Forward adapted to enable fused Linear + CE without materializing logits.
# Mirrors the pattern used for other multimodal models (e.g., InternVL, LLaVA).
@can_return_tuple
def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional["Cache"] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_attention_mask: Optional[torch.BoolTensor] = None,
    image_hidden_states: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,  # Added argument for liger-kernel
    **lm_kwargs: Unpack["TransformersKwargs"],  # renamed from kwargs
) -> Union[tuple, SmolVLMCausalLMOutputWithPast]:
    r"""
    pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
        Mask to avoid performing attention on padding pixel indices.
    image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
        The hidden states of the image encoder after modality projection.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
        ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> import requests
    >>> import torch
    >>> from PIL import Image
    >>> from io import BytesIO

    >>> from transformers import AutoProcessor, AutoModelForImageTextToText
    >>> from transformers.image_utils import load_image

    >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
    >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
    >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

    >>> processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    >>> model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=torch.bfloat16, device_map="auto")

    >>> # Create inputs
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "video", "path": path/to/video},
    ...             {"type": "text", "text": "What is happening in this video?"},
    ...         ]
    ...     }
    ... ]

    >>> inputs = processor.apply_chat_template([messages], add_generation_prompt=True)

    >>> # Generate
    >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
    >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    >>> print(generated_texts)
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        image_hidden_states=image_hidden_states,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        return_dict=True,
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

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        # By default, if in training mode, don't materialize logits
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        loss = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **lm_kwargs,
        )

    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **lm_kwargs
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return SmolVLMCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )
