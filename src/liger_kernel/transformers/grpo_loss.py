import torch

from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase
from liger_kernel.ops import GrpoLossFunction


def triton_grpo_loss(
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask=None,
    temperature=0.9,
    beta=0.04,
    eps_low=0.2,
    eps_high=0.4,
    inplace=True,
    loss_type="dapo",
    max_completion_length=None,
    importance_sampling_level="token",
    reduce=False,
):
    assert logits is not None and completion_ids is not None and advantages is not None, (
        "must provide logits、completion_ids and advantages"
    )
    if importance_sampling_level != "token":
        raise ValueError(
            f"Triton GRPO loss only supports token-level importance sampling. Got {importance_sampling_level}."
        )

    per_token_loss, per_token_kl, is_clipped = GrpoLossFunction.apply(
        logits,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace,
    )
    if not reduce:
        return per_token_loss, per_token_kl, is_clipped

    loss = _reduce_grpo_loss(
        per_token_loss,
        completion_mask,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
    )

    metrics = []
    if beta != 0.0 and per_token_kl is not None:
        metrics.append(_masked_mean(per_token_kl, completion_mask))
    metrics.append(_masked_mean(is_clipped.float(), completion_mask))
    return loss, metrics


def _reduce_grpo_loss(per_token_loss, completion_mask, loss_type, max_completion_length):
    mask = completion_mask
    if mask is None:
        mask = torch.ones_like(per_token_loss, dtype=per_token_loss.dtype, device=per_token_loss.device)
    mask = mask.to(per_token_loss.dtype)

    if loss_type == "grpo":
        per_seq = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return per_seq.mean()
    if loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("max_completion_length must be provided when using loss_type='dr_grpo'")
        batch = per_token_loss.shape[0]
        return (per_token_loss * mask).sum() / (batch * max_completion_length)
    if loss_type == "dapo":
        normalizer = LigerFusedLinearPPOBase._compute_dapo_normalizer(mask)
        return (per_token_loss * mask).sum() / normalizer
    raise ValueError(f"Unsupported loss_type '{loss_type}' for Triton GRPO loss.")


def _masked_mean(values, mask):
    if mask is None:
        mask = torch.ones_like(values, dtype=values.dtype, device=values.device)
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


# This is a demo how to use grpo_loss in GRPOTrainer. The Trl version must be 0.16
"""
import torch
import trl
assert trl.__version__.startswith("0.16"), "please pip install trl==0.16"
from trl.extras.profiling import profiling_decorator

@profiling_decorator
def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    return fused_selective_log_softmax(logits, input_ids, self.temperature, mask=attention_mask)

@profiling_decorator
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    # Compute the per-token log probabilities for the model

    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits

    ref_per_token_logps = inputs["ref_per_token_logps"]
    advantages = inputs["advantages"]
    old_per_token_logps = inputs["old_per_token_logps"]
    

    per_token_loss, per_token_kl, is_clipped = triton_grpo_loss(logits, 
                                                                old_per_token_logps,
                                                                ref_per_token_logps,
                                                                completion_ids,
                                                                advantages,
                                                                completion_mask,
                                                                self.temperature,
                                                                self.beta,
                                                                self.epsilon_low,
                                                                self.epsilon_high,)
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

    # Log the metrics
    mode = "eval" if self.control.should_evaluate else "train"

    if self.beta != 0.0:
        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
    self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
    return loss

trl.GRPOTrainer._get_per_token_logps = _get_per_token_logps
trl.GRPOTrainer.compute_loss = compute_loss
trigger = None
"""

# add this line at the first line of grpo.py in open-r1
"""
from liger_kernel.transformers.grpo_loss import trigger
"""
