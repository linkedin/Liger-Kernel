import types

from typing import List
from typing import Optional

import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="/shared/public/models/Meta-Llama-3-8B",
        **kwargs,
    ):
        super().__init__(**kwargs)
        model.medusa_num_heads = medusa_num_heads
        model.medusa_num_layers = medusa_num_layers
        model.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(model, hidden_size):
        super().__init__()
        model.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(model.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        model.act = nn.SiLU()

    def forward(model, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + model.act(model.linear(x))


def calculate_loss_contribution(
    loss_i,
    i,
    medusa_only_heads,
    medusa_decay_coefficient,
    medusa_heads_coefficient,
    medusa_scheduler_coefficient,
):
    if i == 0:
        return loss_i if not medusa_only_heads else 0
    else:
        return loss_i * medusa_decay_coefficient**i * medusa_heads_coefficient * medusa_scheduler_coefficient


def add_medusa_heads(
    model,
    medusa_num_heads=4,
    medusa_num_layers=0,
    medusa_return: bool = False,
    medusa_only_heads: bool = False,
    with_liger=True,
):
    """
    Args:
        model (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): The number of additional tokens to predict. Defaults to 3.
        medusa_num_layers (int, optional): The number of ResBlock layers for each Medusa head. Defaults to 0.
        medusa_return (bool, optional): If True, returns the Medusa logits; otherwise, the forward pass will use the `lm_head`. Defaults to False.
        medusa_only_heads (bool, optional): If True, only the Medusa head weights will be updated during fine-tuning; otherwise, the entire model's weights will be updated. Defaults to False.
        with_liger (bool, optional): If True, applies Liger loss. Defaults to True.
    """
    hidden_size = model.lm_head.weight.shape[-1]
    vocab_size = model.lm_head.weight.shape[0]
    model.config.medusa_num_layers = medusa_num_layers
    model.config.medusa_num_heads = medusa_num_heads
    model.medusa_num_heads = medusa_num_heads
    # Create a list of Medusa heads
    model.medusa_head = nn.ModuleList(
        [
            nn.Sequential(
                *([ResBlock(hidden_size) for _ in range(medusa_num_layers)]),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(medusa_num_heads)
        ]
    )

    # Ensure medusa_head's dtype and device align with the base_model
    model.medusa_head.to(model.dtype).to(model.device)

    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's weights
        model.medusa_head[i][-1].weight.data[:] = model.lm_head.weight.data[:]
    # logging the model summary
    print(model)
    model.old_forward = model.forward

    def forward(
        model,
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
    ):
        """Forward pass of the MedusaModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        loss = 0
        medusa_logits = None
        # LOG.debug("medusa_return: %s", medusa_return)
        if not medusa_return:
            return model.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        if medusa_only_heads:
            with torch.no_grad():
                outputs = model.model(
                    input_ids=input_ids,
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
                # The lm_head will be frozen as well, so it's within the context of torch.no_grad()
                if not with_liger:
                    medusa_logits = [model.lm_head(hidden_states)]
        else:
            outputs = model.model(
                input_ids=input_ids,
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
            if not with_liger:
                medusa_logits = [model.lm_head(hidden_states)]

        if not with_liger:
            for i in range(model.medusa_num_heads):
                medusa_logits.append(model.medusa_head[i](hidden_states))
            medusa_logits = torch.stack(medusa_logits, dim=0)

        if model.training:
            # Fix all the coefficients to 1 for now
            medusa_scheduler_coefficient = 1
            medusa_heads_coefficient = 1
            medusa_decay_coefficient = 1
            loss = 0

            if with_liger:
                lce = LigerFusedLinearCrossEntropyLoss()
                for i in range(model.medusa_num_heads + 1):
                    shift_hidden_states = (
                        hidden_states[..., : -(1 + i), :].contiguous().view(-1, model.config.hidden_size)
                    )
                    shift_labels = labels[..., (1 + i) :].contiguous().view(-1)

                    weight = model.lm_head.weight if i == 0 else model.medusa_head[i - 1][-1].weight
                    loss_i = lce(weight, shift_hidden_states, shift_labels)

                    loss += calculate_loss_contribution(
                        loss_i,
                        i,
                        medusa_only_heads,
                        medusa_decay_coefficient,
                        medusa_heads_coefficient,
                        medusa_scheduler_coefficient,
                    )
            else:
                loss_fct = CrossEntropyLoss()
                for i in range(model.medusa_num_heads + 1):
                    medusa_logits_i = medusa_logits[i, :, : -(1 + i)].contiguous().view(-1, medusa_logits.shape[-1])
                    medusa_logits_i = medusa_logits_i.float()
                    medusa_labels = labels[..., (1 + i) :].contiguous().view(-1).to(medusa_logits_i.device)

                    loss_i = loss_fct(medusa_logits_i, medusa_labels)

                    loss += calculate_loss_contribution(
                        loss_i,
                        i,
                        medusa_only_heads,
                        medusa_decay_coefficient,
                        medusa_heads_coefficient,
                        medusa_scheduler_coefficient,
                    )
        else:
            if model.config.pretraining_tp > 1:
                raise NotImplementedError
            else:
                medusa_logits = [model.lm_head(hidden_states)]
                for i in range(model.medusa_num_heads):
                    medusa_logits.append(model.medusa_head[i](hidden_states))

        return_dict = return_dict if return_dict is not None else model.config.use_return_dict

        if not return_dict:
            output = (medusa_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=medusa_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward, model)
