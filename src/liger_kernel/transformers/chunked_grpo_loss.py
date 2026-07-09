"""Chunked Triton GRPO loss: fused lm_head + GRPO objective without materializing logits.

Same objective math as :func:`liger_kernel.transformers.grpo_loss.triton_grpo_loss`
but takes (hidden_states, lm_head_weight) instead of logits, so the (N, V)
logits tensor never exists. The heavy work (selective log-softmax through the
lm_head, forward and backward) runs in the fused Triton kernels of
:mod:`liger_kernel.ops.chunked_grpo_loss`; the GRPO objective itself operates
on tiny (B, L) tensors in torch and is taken verbatim from the reference
implementations (TRL's torch path / the Triton kernel's loss math).
"""

import torch

from liger_kernel.ops.chunked_grpo_loss import chunked_selective_log_softmax
from liger_kernel.transformers.grpo_loss import _masked_mean
from liger_kernel.transformers.grpo_loss import _reduce_grpo_loss

_PPO_CLIP_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo", "luspo")
_SUPPORTED_LOSS_TYPES = _PPO_CLIP_LOSS_TYPES + ("cispo", "sapo")


def chunked_triton_grpo_loss(
    hidden,
    weight,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask=None,
    temperature=0.9,
    beta=0.04,
    eps_low=0.2,
    eps_high=0.4,
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
):
    """Chunked Triton GRPO loss (fused linear, logits never materialized).

    Args:
        hidden: Last hidden states (B, L, H) aligned to predict completion_ids
            (i.e. already sliced to the L completion positions, unlike the
            (B, L+1, V) logits that triton_grpo_loss takes).
        weight: lm_head weight (V, H).
        old_logp: Old policy log probabilities (B, L) or None.
        ref_logp: Reference policy log probabilities (B, L) or None (required if beta != 0).
        completion_ids: Token IDs for completions (B, L).
        advantages: Per-sequence advantages (B,).
        completion_mask: Mask for valid tokens (B, L) or None.

    Remaining arguments and the return convention match triton_grpo_loss:
        If reduce=True: (loss, metrics) where metrics = [kl_mean, clip_ratio] or [clip_ratio]
        If reduce=False: (per_token_loss, per_token_kl, is_clipped), all (B, L)

    loss_type "vespo" is not supported.
    """
    if loss_type not in _SUPPORTED_LOSS_TYPES:
        raise ValueError(f"Unsupported loss_type '{loss_type}' for chunked Triton GRPO loss.")
    if importance_sampling_level not in ("token", "sequence"):
        raise ValueError(f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}")
    if delta is not None and loss_type not in _PPO_CLIP_LOSS_TYPES:
        raise ValueError(f"delta (two-sided clipping) is not supported for loss_type='{loss_type}'.")
    if beta != 0.0 and ref_logp is None:
        raise ValueError("ref_logp is required when beta != 0.")

    b, seq_len, h = hidden.shape
    logp = chunked_selective_log_softmax(
        hidden.reshape(-1, h),
        weight,
        completion_ids.reshape(-1),
        temperature,
    ).view(b, seq_len)

    mask = completion_mask
    if mask is None:
        mask = torch.ones_like(logp)
    mask = mask.to(logp.dtype)

    old = old_logp.to(torch.float32) if old_logp is not None else logp.detach()
    log_ratio = logp - old
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    else:
        log_importance_weights = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)

    coef_1 = torch.exp(log_importance_weights)
    adv = advantages.unsqueeze(1).to(torch.float32)

    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=eps_high).detach()
        per_token_loss = -clamped_ratios * adv * logp  # logp keeps this per-token
        is_clipped = ((coef_1 > eps_high) & (adv > 0)).to(logp.dtype)
    elif loss_type == "sapo":
        temperatures = torch.where(adv > 0, sapo_temperature_pos, sapo_temperature_neg)
        soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
        per_token_loss = -soft_coef_1 * adv
        is_clipped = torch.zeros_like(coef_1)
    else:  # standard PPO clipping: grpo / bnpo / dr_grpo / dapo / luspo
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
        is_clipped = (((coef_1 < 1 - eps_low) & (adv < 0)) | ((coef_1 > 1 + eps_high) & (adv > 0))).to(logp.dtype)
        if delta is not None:
            coef_1 = torch.clamp(coef_1, max=delta)
        per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)

    if vllm_is_ratio is not None:
        per_token_loss = per_token_loss * vllm_is_ratio

    per_token_kl = None
    if beta != 0.0:
        ref = ref_logp.to(torch.float32)
        per_token_kl = torch.exp(ref - logp) - (ref - logp) - 1
        if use_bias_correction_kl:
            per_token_kl = per_token_kl * coef_1
        per_token_loss = per_token_loss + beta * per_token_kl

    # Sequence-level tensors are (B, 1); expand to (B, L) to match the Triton
    # kernel's reduce=False convention (same value for all tokens in a sequence).
    if per_token_loss.shape[1] == 1:
        per_token_loss = per_token_loss.expand(b, seq_len)
    if is_clipped.shape[1] == 1:
        is_clipped = is_clipped.expand(b, seq_len)

    if not reduce:
        return per_token_loss, per_token_kl, is_clipped

    loss = _reduce_grpo_loss(per_token_loss, mask, loss_type, max_completion_length, num_items_in_batch)
    metrics = []
    if beta != 0.0:
        metrics.append(_masked_mean(per_token_kl, mask))
    metrics.append(_masked_mean(is_clipped, mask))
    return loss, metrics
