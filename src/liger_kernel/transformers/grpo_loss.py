import torch

from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase
from liger_kernel.chunked_loss.grpo_loss import get_gamma_weights
from liger_kernel.ops import GrpoLossFunction
from liger_kernel.ops.grpo_loss import fused_selective_log_softmax


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
    sapo_temperature_pos=1.0,
    sapo_temperature_neg=1.05,
    vllm_is_ratio=None,
    delta=None,
    use_bias_correction_kl=False,
    num_items_in_batch=None,
    vespo_k_pos=2.0,
    vespo_lambda_pos=3.0,
    vespo_k_neg=3.0,
    vespo_lambda_neg=2.0,
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
        loss_type: Loss reduction type ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo")
        max_completion_length: Max completion length for dr_grpo loss type; defaults to sequence length if None
        importance_sampling_level: "token" or "sequence" importance sampling
        reduce: If True, return reduced loss; if False, return per-token loss
        vllm_is_ratio: vLLM importance sampling ratio (B, L) or (B, 1) or None.
            Used to correct for distribution mismatch when using vLLM for generation.
            Applied to PPO loss BEFORE adding KL penalty.
        delta: Upper clamp for two-sided clipping (INTELLECT-2). When set, coef_1 is clamped
            to max=delta before computing the PPO loss. Only supported for standard PPO loss
            types (grpo, bnpo, dr_grpo, dapo, luspo). None means disabled.
        use_bias_correction_kl: If True, multiply KL divergence by coef_1 (importance sampling
            ratio) for bias-corrected KL estimation (DeepSeek-V3.2). Default False.
        num_items_in_batch: Optional total active tokens across the entire generation batch
            (all gradient-accumulation micro-batches × all processes). When provided, dapo /
            cispo / vespo normalization uses ``num_items_in_batch / num_processes`` to match
            TRL's ``compute_loss``. When None, falls back to the current micro-batch's mask
            sum.
        vespo_k_pos, vespo_lambda_pos, vespo_k_neg, vespo_lambda_neg: VESPO gamma weighting
            hyperparameters (k for shape, lambda for rate; ``_pos`` for non-negative
            advantages, ``_neg`` for negative). Only used when ``loss_type='vespo'``.

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

    # VESPO: pre-compute phi_seq (detached, sequence-level gamma weighting). The vllm
    # importance-sampling correction is folded into phi_seq via log_is_ratio rather than
    # multiplied onto per_token_loss, so we drop vllm_is_ratio for the kernel call.
    phi_seq = None
    if loss_type == "vespo":
        if importance_sampling_level == "sequence":
            raise ValueError("loss_type='vespo' requires importance_sampling_level='token'.")
        # Need per-token logp for log_ratio. fused_selective_log_softmax is no-grad —
        # phi_seq is detached anyway, so this is fine.
        per_token_logps = fused_selective_log_softmax(logits, completion_ids, temperature, completion_mask)
        if old_logp is None:
            log_ratio = torch.zeros_like(per_token_logps)
        else:
            log_ratio = per_token_logps - old_logp
        mask = (
            completion_mask
            if completion_mask is not None
            else torch.ones_like(per_token_logps, dtype=per_token_logps.dtype)
        )
        # Normalize vllm_is_ratio shape to (B, T) for get_gamma_weights' sum-over-time.
        vllm_for_phi = vllm_is_ratio
        if vllm_for_phi is not None:
            if vllm_for_phi.dim() == 1:
                vllm_for_phi = vllm_for_phi.unsqueeze(-1).expand_as(per_token_logps)
            elif vllm_for_phi.dim() == 2 and vllm_for_phi.shape[1] == 1:
                vllm_for_phi = vllm_for_phi.expand_as(per_token_logps)
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=mask,
            importance_sampling_ratio=vllm_for_phi,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )  # (B, 1)
        # vllm correction is folded into phi_seq; do not pass it to the kernel separately.
        vllm_is_ratio = None

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
        sapo_temperature_pos,
        sapo_temperature_neg,
        vllm_is_ratio,
        delta,
        use_bias_correction_kl,
        num_items_in_batch,
        phi_seq,
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


def _reduce_grpo_loss(per_token_loss, completion_mask, loss_type, max_completion_length, num_items_in_batch=None):
    mask = completion_mask
    if mask is None:
        mask = torch.ones_like(per_token_loss, dtype=per_token_loss.dtype, device=per_token_loss.device)
    mask = mask.to(per_token_loss.dtype)

    if loss_type == "grpo" or loss_type == "sapo":
        # SAPO uses the same normalization as GRPO (per-sequence average)
        per_seq = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return per_seq.mean()
    if loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_type == "dr_grpo":
        batch = per_token_loss.shape[0]
        max_len = max_completion_length if max_completion_length is not None else per_token_loss.shape[1]
        return (per_token_loss * mask).sum() / (batch * max_len)
    if loss_type == "dapo" or loss_type == "cispo" or loss_type == "vespo":
        # CISPO and VESPO use the same normalization as DAPO
        normalizer = LigerFusedLinearPPOBase._compute_dapo_normalizer(mask, num_items_in_batch=num_items_in_batch)
        return (per_token_loss * mask).sum() / normalizer
    if loss_type == "luspo":
        # LUSPO: scale each sequence's loss by its valid token count, then average across sequences
        return (per_token_loss * mask.sum(-1, keepdim=True)).mean()
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
