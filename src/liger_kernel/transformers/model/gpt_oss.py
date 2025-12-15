from typing import List
from typing import Optional
from typing import Union

import torch

from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast


def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> LigerMoeCausalLMOutputWithPast:
    r"""
        Forward pass for causal language modeling with Mixture of Experts (MoE) architecture using Liger Kernel optimizations.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using tokenizers.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`List[torch.FloatTensor]` or `Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up
            sequential decoding. See `past_key_values` input for more details.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the router logits of all MoE layers. See `router_logits` under returned tensors
            for more detail.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).
        skip_logits (`bool`, *optional*):
            Whether to skip logit computation and directly compute loss. If `None`, defaults to `True` during training
            when labels are provided (to save memory), and `False` during inference.

    Returns:
        `LigerMoeCausalLMOutputWithPast`: An output object containing:
            - loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                Language modeling loss (for next-token prediction), including the auxiliary load balancing loss.
            - aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
                Auxiliary load balancing loss for the sparse MoE modules.
            - logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
                Note: logits are `None` during training when `skip_logits=True` to save memory.
            - past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed):
                Cached key and value projection states for faster sequential decoding.
            - hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for each layer) of shape
                `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer.
            - attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`. Attentions weights after the attention softmax.
            - router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
                Router logits of the MoE layers, useful to compute the auxiliary loss and z_loss.
            - token_accuracy (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
                Token-level prediction accuracy.

    Example:

    ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM
        >>> from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss

        >>> # Apply Liger Kernel patches for optimized performance
        >>> apply_liger_kernel_to_gpt_oss()

        >>> model = GptOssForCausalLM.from_pretrained("openai/gpt-oss-20b")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Inference: Forward pass returns logits
        >>> outputs = model(**inputs)
        >>> outputs.logits.shape
        torch.Size([1, 12, 201088])

        >>> # Get next token prediction
        >>> next_token_logits = outputs.logits[:, -1, :]
        >>> predicted_token_id = next_token_logits.argmax(dim=-1)

        >>> # Training: Forward pass with labels returns loss
        >>> labels = inputs.input_ids.clone()
        >>> outputs = model(**inputs, labels=labels)
        >>> outputs.loss
        tensor(2.6454)
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy = unpack_cross_entropy_result(result)
    else:  # if in inference model materialize logits
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **kwargs,
            )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    return LigerMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        token_accuracy=token_accuracy,
    )
