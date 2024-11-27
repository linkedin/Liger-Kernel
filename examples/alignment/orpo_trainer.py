from trl.trainer import ORPOTrainer
from typing import Dict, Union, List, Literal, Tuple, Any, Callable
from torch.distributed.fsdp import FullyShardedDataParallel
import torch
import torch.nn as nn
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss


class _FSDPForwardRedirection:
    """
    Modified based on
    https://github.com/Lightning-AI/pytorch-lightning/blob/d3f9c83d6efa4f1def36aa6c199600946cdb9117/src/lightning/pytorch/strategies/strategy.py#L601-L648
    Redirect a method call through FullyShardedDataParallel.forward so that the FSDP module's root pre-forward and
    post-forward can be properly executed around the method call.
    This is needed in cases where we call a submodule of a FSDP module. For instance, when we want to call only
    the `LlamaModel` part out of a FSDP-wrapped `LlamaForCausalLM` to get the hidden states without involving
    GPU-memory-heavy `lm_head` and cross entropy computation, doing this directly (i.e. `model.model.forward()`)
    will not work because the first `nn.Emebedding` layer is not independently wrapped as a FSDP module (because of
    the transformer-based wrapping policy), and not calling it through FSDP root module forward will not all-gather
    its parameter, thus resulting in "RuntimeError: 'weight' must be 2-D" error. Similarly, if we want to call just
    the `lm_head` part of a model, we need this trick too to properly get its params all-gathered.
    """
    def __call__(
        self,
        wrapper_module: FullyShardedDataParallel,
        method: Callable,
        *args: Any,
        **kwargs: Any
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.
        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
        """
        assert isinstance(wrapper_module, FullyShardedDataParallel)
        original_module = wrapper_module._fsdp_wrapped_module
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            return out
        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]
        wrapper_output = wrapper_module(*args, **kwargs)
        return wrapper_output


class LigerORPOTrainer(ORPOTrainer):
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        model_kwargs = (
            {
                "decoder_input_ids": self._shift_right(concatenated_batch["concatenated_labels"]),
            }
            if self.is_encoder_decoder
            else {}
        )

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if isinstance(model, FullyShardedDataParallel):
            outputs = _FSDPForwardRedirection()(
                model,
                model._fsdp_wrapped_module.model,
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                **model_kwargs,
            )
        else:
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            outputs = model.model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                **model_kwargs,
            )

        orpo_loss_fn = LigerFusedLinearORPOLoss(ignore_index=self.label_pad_token_id, beta=self.beta)

        def orpo_partial(lm_head, last_hidden_state, concatenated_labels):
            return orpo_loss_fn(lm_head.weight, last_hidden_state, concatenated_labels, lm_head.bias)

        orpo_loss = _FSDPForwardRedirection()(
            model,
            orpo_partial,
            model.lm_head,
            outputs.last_hidden_state,
            concatenated_batch["concatenated_labels"],
        )
        # orpo_loss = LigerFusedLinearORPOLoss(ignore_index=self.label_pad_token_id, beta=self.beta)(model.lm_head.weight, outputs.last_hidden_state, concatenated_batch["concatenated_labels"], model.lm_head.bias)
        return orpo_loss
        # all_logits = outputs.logits

        # def cross_entropy_loss(logits, labels):
        #     if not self.is_encoder_decoder:
        #         # Shift so that tokens < n predict n
        #         logits = logits[..., :-1, :].contiguous()
        #         labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     logits = logits.view(-1, logits.shape[-1])
        #     labels = labels.view(-1)
        #     # Enable model parallelism
        #     labels = labels.to(logits.device)
        #     loss = loss_fct(logits, labels)
        #     return loss

        # if self.is_encoder_decoder:
        #     labels = concatenated_batch["concatenated_labels"].clone()
        # else:
        #     labels = concatenated_batch["concatenated_input_ids"].clone()
        #     attention_mask = concatenated_batch["concatenated_attention_mask"]
        #     labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)

        # chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        # all_logps = self.get_batch_logps(
        #     all_logits,
        #     concatenated_batch["concatenated_labels"],
        #     average_log_prob=True,
        #     is_encoder_decoder=self.is_encoder_decoder,
        #     label_pad_token_id=self.label_pad_token_id,
        # )

        # chosen_logps = all_logps[:len_chosen]
        # rejected_logps = all_logps[len_chosen:]

        # chosen_logits = all_logits[:len_chosen]
        # rejected_logits = all_logits[len_chosen:]

        # if self.aux_loss_enabled:
        #     return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss, outputs.aux_loss)

        # return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        loss = self.concatenated_forward(model, batch)
        return loss, metrics
        # (
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     policy_chosen_logits,
        #     policy_rejected_logits,
        #     policy_nll_loss,
        # ) = forward_output[:5]
        # if self.aux_loss_enabled:
        #     aux_loss = forward_output[5]

        # losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
        #     policy_chosen_logps, policy_rejected_logps
        # )
        # # full ORPO loss
        # loss = policy_nll_loss - losses.mean()

        # reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
        # metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        # metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean()
        # metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean()
        # metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean()
        # metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean()
        # metrics[f"{prefix}log_odds_ratio"] = log_odds_ratio
        # metrics[f"{prefix}log_odds_chosen"] = log_odds_chosen
        # for k, v in metrics.items():
        #     metrics[k] = v.item()
        # if self.aux_loss_enabled:
        #     loss += self.aux_loss_coef * aux_loss

        # return loss, metrics
