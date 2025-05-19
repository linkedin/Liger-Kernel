from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel
from trl.trainer import ORPOTrainer

from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
from liger_kernel.transformers.fsdp import _FSDPForwardRedirection


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

        if self.is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"].clone()
        else:
            labels = concatenated_batch["concatenated_input_ids"].clone()
            attention_mask = concatenated_batch["concatenated_attention_mask"]
            labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)

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

        def orpo_partial(lm_head, last_hidden_state, concatenated_labels, nll_target):
            return orpo_loss_fn(
                lm_head.weight, last_hidden_state, concatenated_labels, lm_head.bias, nll_target=nll_target
            )

        orpo_loss, aux_outputs = _FSDPForwardRedirection()(
            model,
            orpo_partial,
            model.lm_head,
            outputs.last_hidden_state[:, :-1] if not self.is_encoder_decoder else outputs.last_hidden_state,
            concatenated_batch["concatenated_labels"][:, 1:]
            if not self.is_encoder_decoder
            else concatenated_batch["concatenated_labels"],
            labels[:, 1:] if not self.is_encoder_decoder else labels,
        )
        # if aux_loss_enabled, add the aux_loss to the orpo_loss
        if self.aux_loss_enabled:
            orpo_loss += self.aux_loss_coef * outputs.aux_loss

        return orpo_loss, aux_outputs

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        loss, aux_outputs = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = aux_outputs[:5]

        # return loss, metrics
        chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = aux_outputs[5:]

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean()
        metrics[f"{prefix}log_odds_ratio"] = log_odds_ratio
        metrics[f"{prefix}log_odds_chosen"] = log_odds_chosen
        for k, v in metrics.items():
            metrics[k] = v.item()

        return loss, metrics
