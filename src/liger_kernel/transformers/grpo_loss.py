from liger_kernel.ops.grpo_loss import GrpoLossFunction


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
):
    assert logits is not None and completion_ids is not None and advantages is not None, (
        "must provide logits、completion_ids and advantages"
    )

    return GrpoLossFunction.apply(
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
