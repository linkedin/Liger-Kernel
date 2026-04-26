from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.cache_utils import Cache
from transformers.utils import logging

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast

try:
    from liger_kernel.transformers.model.output_classes import LigerGemma4CausalLMOutputWithPast
except ImportError:
    # Older transformers without gemma4 — multimodal_forward is then unreachable
    # because monkey_patch.apply_liger_kernel_to_gemma4 imports gemma4 modules
    # behind the same try/except.
    LigerGemma4CausalLMOutputWithPast = None

logger = logging.get_logger(__name__)


def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **loss_kwargs,
) -> Union[Tuple, LigerCausalLMOutputWithPast]:
    r"""
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

    Fused-linear-cross-entropy forward for Gemma4ForCausalLM. Mirrors liger's
    gemma3 causal_forward. Gemma 4 31B uses final_logit_softcapping=30.0, so
    the softcap branch is exercised on the non-fused path.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma4ForCausalLM

    >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-31b")  # illustrative slug
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma4 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with "
            "`AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **loss_kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]
    shift_labels = loss_kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        # final_logit_softcapping via getattr: some future Gemma 4 variants may omit the attribute entirely.
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            final_logit_softcapping=getattr(self.config, "final_logit_softcapping", None),
            **loss_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **loss_kwargs,
            )

    if not return_dict:
        output_tuple = (logits,) + outputs[1:]
        output_tuple = (loss,) + output_tuple if loss is not None else output_tuple
        output_tuple = output_tuple + (token_accuracy,) if token_accuracy is not None else output_tuple
        output_tuple = output_tuple + (predicted_tokens,) if predicted_tokens is not None else output_tuple
        return output_tuple

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def multimodal_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_position_ids: Optional[torch.LongTensor] = None,
    video_position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **lm_kwargs,
):
    r"""Fused-linear-cross-entropy forward for ``Gemma4ForConditionalGeneration``.

    Mirrors :func:`liger_kernel.transformers.model.gemma3.multimodal_forward`
    with Gemma 4-specific kwargs (``pixel_values_videos``, ``input_features``,
    ``image_position_ids``, ``video_position_ids``, ``mm_token_type_ids``) and
    output fields (``image_hidden_states``, ``audio_hidden_states``).

    The win on Gemma 4 multimodal is large: vocab=262,144 means the (B, T, V)
    fp32 logits tensor is ~17 GB at T=8192 in bf16 (and another ~34 GB once the
    loss path upcasts), OOMing 96 GB cards. Routing loss through
    ``LigerForCausalLMLoss`` materializes only the loss scalar.

    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either
        be in `[0, ..., config.text_config.vocab_size]` or -100 (see `input_ids`
        docstring). Tokens with indices set to `-100` are ignored.

    logits_to_keep (`int` or `torch.Tensor`, *optional*):
        If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`,
        calculate logits for all `input_ids` (special case). If a `torch.Tensor`,
        must be 1D corresponding to the indices to keep in the sequence-length
        dimension.
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
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    text_cfg = self.config.get_text_config()
    softcap = getattr(text_cfg, "final_logit_softcapping", None)
    shift_labels = lm_kwargs.pop("shift_labels", None)

    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=text_cfg.hidden_size,
            final_logit_softcapping=softcap,
            **lm_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=text_cfg.vocab_size,
                **lm_kwargs,
            )

    if not return_dict:
        output_tuple = (logits,) + outputs[1:]
        output_tuple = (loss,) + output_tuple if loss is not None else output_tuple
        output_tuple = output_tuple + (token_accuracy,) if token_accuracy is not None else output_tuple
        output_tuple = output_tuple + (predicted_tokens,) if predicted_tokens is not None else output_tuple
        return output_tuple

    return LigerGemma4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=getattr(outputs, "image_hidden_states", None),
        audio_hidden_states=getattr(outputs, "audio_hidden_states", None),
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
