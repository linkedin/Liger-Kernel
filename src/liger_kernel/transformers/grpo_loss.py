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
    vllm_is_ratio=None,
):
    """
    Triton-optimized GRPO loss function.

    Args:
        logits: Model logits (B, L+1, V)
        old_logp: Old policy log probabilities (B, L) or None
        ref_logp: Reference model log probabilities (B, L) or None (required if beta != 0)
        completion_ids: Token IDs for completions (B, L)
        advantages: Per-sequence advantages (B,)
        completion_mask: Mask for valid tokens (B, L) or None
        temperature: Temperature for log softmax
        beta: KL penalty coefficient
        eps_low: Lower clipping bound for importance ratio
        eps_high: Upper clipping bound for importance ratio
        inplace: Whether to modify logits in-place during backward
        loss_type: Loss reduction type ("grpo", "bnpo", "dr_grpo", "dapo")
        max_completion_length: Max completion length for dr_grpo loss type
        importance_sampling_level: "token" or "sequence" importance sampling
        reduce: If True, return reduced loss; if False, return per-token loss
        vllm_is_ratio: vLLM importance sampling ratio (B, L) or (B, 1) or None.
            Used to correct for distribution mismatch when using vLLM for generation.
            Applied to PPO loss BEFORE adding KL penalty.

    Returns:
        If reduce=True: (loss, metrics) where metrics = [kl_mean, clip_ratio] or [clip_ratio]
        If reduce=False: (per_token_loss, per_token_kl, is_clipped)
    """
    assert logits is not None and completion_ids is not None and advantages is not None, (
        "must provide logits, completion_ids and advantages"
    )
    assert importance_sampling_level in ("token", "sequence"), (
        f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}"
    )

    result = GrpoLossFunction.apply(
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
        loss_type,
        max_completion_length,
        reduce,
        importance_sampling_level,
        vllm_is_ratio,
    )

    if not reduce:
        # Returns (per_token_loss, per_token_kl, is_clipped) - all (B, L) tensors
        return result

    # reduce=True: Returns (reduced_loss, kl_mean, clip_ratio) - all scalars
    reduced_loss, kl_mean, clip_ratio = result
    metrics = []
    if beta != 0.0 and kl_mean is not None:
        metrics.append(kl_mean)
    metrics.append(clip_ratio)
    return reduced_loss, metrics


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


# This is a demo how to use grpo_loss in GRPOTrainer. The Trl version must be 0.26.2+
"""
import torch
import trl
from packaging.version import Version
assert Version(trl.__version__) >= Version("0.26.2"), "please pip install trl>=0.26.2"
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

    # Get vLLM importance sampling ratio if using vLLM with importance sampling correction
    vllm_is_ratio = inputs.get("importance_sampling_ratio", None)

    per_token_loss, per_token_kl, is_clipped = triton_grpo_loss(
        logits,
        old_per_token_logps,
        ref_per_token_logps,
        completion_ids,
        advantages,
        completion_mask,
        temperature=self.temperature,
        beta=self.beta,
        eps_low=self.epsilon_low,
        eps_high=self.epsilon_high,
        importance_sampling_level=self.importance_sampling_level,  # "token" or "sequence"
        vllm_is_ratio=vllm_is_ratio,  # vLLM distribution correction
    )
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
