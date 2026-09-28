from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.cache_utils import Cache
from transformers.utils import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerMuseGlimmerCausalLMOutputWithPast


@can_return_tuple
def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, LigerMuseGlimmerCausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
        ignored (masked), the loss is only computed for the tokens with labels in
        `[0, ..., config.text_config.vocab_size]`.
    skip_logits (`bool`, *optional*):
        Whether to skip materializing the logits and use Liger's fused linear cross entropy instead. Defaults to
        `True` during training when labels are provided.

    Example:

    ```python
    >>> from transformers import AutoProcessor, MuseGlimmerForConditionalGeneration

    >>> model = MuseGlimmerForConditionalGeneration.from_pretrained(MUSE_GLIMMER_CHECKPOINT)
    >>> processor = AutoProcessor.from_pretrained(MUSE_GLIMMER_CHECKPOINT)

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://example.com/image.jpeg"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]

    >>> inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )

    >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
    >>> output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ```
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    text_config = self.config.text_config
    output_multiplier = text_config.output_multiplier
    final_logit_softcapping = text_config.final_logit_softcapping

    shift_labels = kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        # MuseGlimmer computes `T * tanh(logits * m / T)` where `m = output_multiplier` and
        # `T = final_logit_softcapping`, but Liger's fused linear cross entropy softcap only
        # implements `T * tanh(logits / T)`. Fold `m` into the hidden states instead, since
        # `(m * h) @ W.T == m * (h @ W.T)`. Scaling the hidden states rather than the lm_head
        # weight keeps the extra tensor at `[batch, seq, hidden]` instead of `[vocab, hidden]`.
        # Make kept_hidden_states contiguous for LigerForCausalLMLoss after slicing
        kept_hidden_states = kept_hidden_states * output_multiplier

        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=text_config.hidden_size,
            final_logit_softcapping=final_logit_softcapping,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)

        logits = logits * output_multiplier
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping

        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=text_config.vocab_size,
                **kwargs,
            )

    return LigerMuseGlimmerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
