"""Memory-bounded Ascend fused-linear GRPO adapter.

The registered Ascend GRPO kernel consumes transformer-style logits with one
extra row (``[B, T + 1, V]``), while Liger's fused-linear API starts from
hidden states (``[B, T, H]``). This adapter owns the projection and processes
a bounded number of batch rows at a time, so each retained autograd chunk
contains only its local vocabulary logits.
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_kernel.ops.backends._ascend.ops.grpo_loss import GrpoLossFunction
from liger_kernel.ops.backends._ascend.ops.grpo_loss import _compute_dapo_normalizer
from liger_kernel.ops.backends._ascend.ops.grpo_loss import fused_selective_log_softmax


def _project_and_pad(hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], batch: int, seq: int):
    if hidden.ndim == 2:
        hidden = hidden.reshape(batch, seq, hidden.shape[-1])
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden with shape [B,T,H] or [B*T,H], got {tuple(hidden.shape)}")
    logits = hidden @ weight.t()
    if bias is not None:
        logits = logits + bias
    pad = torch.zeros(
        (batch, 1, logits.shape[-1]),
        dtype=logits.dtype,
        device=logits.device,
    )
    return torch.cat((logits, pad), dim=1).contiguous()


def _validate_contract(
    hidden,
    selected_token_ids,
    attention_mask,
    ref_input,
    ref_weight,
    delta,
    loss_type,
    max_completion_length,
    importance_sampling_level,
):
    if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo", "vespo"):
        raise ValueError(
            f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'. "
            "Use importance_sampling_level='token' instead."
        )
    if delta is not None and loss_type in ("cispo", "sapo", "vespo"):
        raise ValueError(f"delta (two-sided clipping) is not supported for loss_type='{loss_type}'.")
    if loss_type == "dr_grpo" and max_completion_length is None:
        raise ValueError("max_completion_length must be provided for loss_type 'dr_grpo'")
    if hidden.ndim == 2:
        if selected_token_ids.ndim != 2:
            raise ValueError("2D hidden requires selected_token_ids with shape [B,T]")
        batch, seq = selected_token_ids.shape
    elif hidden.ndim == 3:
        batch, seq = hidden.shape[:2]
    else:
        raise ValueError(f"expected hidden with shape [B,T,H] or [B*T,H], got {tuple(hidden.shape)}")
    if selected_token_ids.shape != (batch, seq):
        raise ValueError(f"selected_token_ids must have shape {(batch, seq)}, got {tuple(selected_token_ids.shape)}")
    if attention_mask.shape != (batch, seq):
        raise ValueError(f"attention_mask must have shape {(batch, seq)}, got {tuple(attention_mask.shape)}")
    if ref_input is not None and ref_weight is None:
        raise ValueError("ref_weight is required when ref_input is provided")
    return batch, seq


def _loss_normalizer(mask, loss_type, batch, seq, max_completion_length, num_items):
    if loss_type in ("grpo", "sapo"):
        return None
    if loss_type == "bnpo":
        return mask.sum().clamp(min=1.0)
    if loss_type == "dr_grpo":
        return mask.new_tensor(float(batch * max_completion_length))
    if loss_type in ("dapo", "cispo", "vespo"):
        return _compute_dapo_normalizer(mask, num_items_in_batch=num_items)
    if loss_type == "luspo":
        return mask.new_tensor(float(batch * seq))
    raise ValueError(
        f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo, vespo"
    )


def _chunk_reduced_loss(per_token_loss, mask, loss_type, batch, seq, normalizer, importance_sampling_level):
    # Sequence-level LUSPO follows TRL's reduction exactly: the sequence
    # coefficient is broadcast over all positions, then weighted by the
    # valid-token count before the final B*T mean.
    if loss_type == "luspo":
        weighted = per_token_loss * mask.sum(-1, keepdim=True)
        return weighted.sum() / (batch * seq)
    if loss_type in ("grpo", "sapo"):
        return (per_token_loss * mask).sum(-1).div(mask.sum(-1).clamp(min=1.0)).sum() / batch
    return (per_token_loss * mask).sum() / normalizer


def _fused_linear_grpo_native(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    selected_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ref_per_token_logps: Optional[torch.Tensor] = None,
    old_per_token_logps: Optional[torch.Tensor] = None,
    ref_input: Optional[torch.Tensor] = None,
    ref_weight: Optional[torch.Tensor] = None,
    ref_bias: Optional[torch.Tensor] = None,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    loss_type: str = "dapo",
    max_completion_length: Optional[int] = None,
    importance_sampling_level: str = "token",
    sapo_temperature_pos: float = 1.0,
    sapo_temperature_neg: float = 1.05,
    temperature: float = 1.0,
    vllm_is_ratio: Optional[torch.Tensor] = None,
    delta: Optional[float] = None,
    use_bias_correction_kl: bool = False,
    num_items_in_batch: Optional[int] = None,
    vespo_k_pos: float = 2.0,
    vespo_lambda_pos: float = 3.0,
    vespo_k_neg: float = 3.0,
    vespo_lambda_neg: float = 2.0,
    chunk_size: int = 1,
):
    """Build the fused-linear loss graph without doing backward work in forward."""
    batch, seq = _validate_contract(
        hidden,
        selected_token_ids,
        attention_mask,
        ref_input,
        ref_weight,
        delta,
        loss_type,
        max_completion_length,
        importance_sampling_level,
    )
    full_mask = attention_mask.to(torch.float32)
    normalizer = _loss_normalizer(full_mask, loss_type, batch, seq, max_completion_length, num_items_in_batch)
    num_chunks = max(1, batch // max(1, int(chunk_size)))
    batch_chunk = max(1, (batch + num_chunks - 1) // num_chunks)
    if int(chunk_size) == 1 and batch > 1:
        # The public default historically selected one chunk per batch row.
        # On Ascend, keep explicit chunk_size behavior but coalesce rows when
        # the retained logits fit a conservative 2 GiB activation budget. This
        # removes per-row launch overhead for large vocabularies while keeping
        # the chunked fallback for larger batches/models.
        bytes_per_row = (seq + 1) * weight.shape[0] * hidden.element_size()
        max_rows = max(1, (2 * 1024 * 1024 * 1024) // max(1, bytes_per_row))
        batch_chunk = min(batch, max(batch_chunk, max_rows))
    loss_value = hidden.new_zeros((), dtype=torch.float32)
    kl_value = hidden.new_zeros((), dtype=torch.float32) if beta != 0.0 else None
    clip_value = hidden.new_zeros((), dtype=torch.float32)
    full_active = full_mask.sum().clamp(min=1.0)

    for start in range(0, batch, batch_chunk):
        end = min(start + batch_chunk, batch)
        hidden_part = hidden[start:end] if hidden.ndim == 3 else hidden.reshape(batch, seq, -1)[start:end]
        ids_part = selected_token_ids[start:end].contiguous()
        mask_part = attention_mask[start:end].contiguous()
        adv_part = advantages[start:end]
        logits = _project_and_pad(hidden_part, weight, bias, end - start, seq)

        ref_logp_part = None if ref_per_token_logps is None else ref_per_token_logps[start:end].contiguous()
        if beta != 0.0 and ref_logp_part is None and ref_input is not None:
            ref_part = ref_input[start:end] if ref_input.ndim == 3 else ref_input.reshape(batch, seq, -1)[start:end]
            with torch.no_grad():
                ref_logits = _project_and_pad(ref_part, ref_weight, ref_bias, end - start, seq)
                ref_mask = None if loss_type == "luspo" and importance_sampling_level == "sequence" else mask_part
                ref_logp_part = fused_selective_log_softmax(ref_logits, ids_part, temperature, ref_mask)
        if beta != 0.0 and ref_logp_part is None:
            ref_mask = None if loss_type == "luspo" and importance_sampling_level == "sequence" else mask_part
            ref_logp_part = fused_selective_log_softmax(logits.detach(), ids_part, temperature, ref_mask)

        ratio_part = None if vllm_is_ratio is None else vllm_is_ratio[start:end].contiguous()
        phi_part = None
        if loss_type == "vespo":
            from liger_kernel.chunked_loss.grpo_loss import get_gamma_weights

            per_token_logps = fused_selective_log_softmax(logits.detach(), ids_part, temperature, mask_part)
            old_part = None if old_per_token_logps is None else old_per_token_logps[start:end]
            log_ratio = torch.zeros_like(per_token_logps) if old_part is None else per_token_logps - old_part
            vllm_for_phi = ratio_part
            if vllm_for_phi is not None:
                if vllm_for_phi.ndim == 1:
                    vllm_for_phi = vllm_for_phi[:, None].expand_as(per_token_logps)
                elif vllm_for_phi.ndim == 2 and vllm_for_phi.shape[1] == 1:
                    vllm_for_phi = vllm_for_phi.expand_as(per_token_logps)
            phi_part = get_gamma_weights(
                advantages=adv_part,
                log_ratio_per_token=log_ratio,
                mask=mask_part.to(per_token_logps.dtype),
                importance_sampling_ratio=vllm_for_phi,
                k_pos=vespo_k_pos,
                lambda_pos=vespo_lambda_pos,
                k_neg=vespo_k_neg,
                lambda_neg=vespo_lambda_neg,
            )
            ratio_part = None

        loss_part, kl_part, clip_part = GrpoLossFunction.apply(
            logits,
            old_per_token_logps[start:end].contiguous() if old_per_token_logps is not None else None,
            ref_logp_part,
            ids_part,
            adv_part,
            mask_part,
            temperature,
            beta,
            epsilon_low,
            epsilon_high,
            False,
            loss_type,
            max_completion_length,
            False,
            importance_sampling_level,
            sapo_temperature_pos,
            sapo_temperature_neg,
            ratio_part,
            delta,
            use_bias_correction_kl,
            num_items_in_batch,
            phi_part,
        )
        local_loss = _chunk_reduced_loss(
            loss_part,
            mask_part.to(torch.float32),
            loss_type,
            end - start,
            seq,
            normalizer,
            importance_sampling_level,
        )
        if loss_type in ("grpo", "sapo", "luspo"):
            local_loss = local_loss * ((end - start) / batch)
        loss_value = loss_value + local_loss

        if kl_value is not None:
            kl_value = kl_value + (kl_part.detach().to(torch.float32) * full_mask[start:end]).sum()
        clip_value = clip_value + (clip_part.detach().to(torch.float32) * full_mask[start:end]).sum()

    if kl_value is not None:
        kl_value = kl_value / full_active
    clip_value = clip_value / full_active
    return loss_value, kl_value, clip_value


def fused_linear_grpo_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    selected_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ref_per_token_logps: Optional[torch.Tensor] = None,
    old_per_token_logps: Optional[torch.Tensor] = None,
    ref_input: Optional[torch.Tensor] = None,
    ref_weight: Optional[torch.Tensor] = None,
    ref_bias: Optional[torch.Tensor] = None,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    loss_type: str = "dapo",
    max_completion_length: Optional[int] = None,
    importance_sampling_level: str = "token",
    sapo_temperature_pos: float = 1.0,
    sapo_temperature_neg: float = 1.05,
    temperature: float = 1.0,
    vllm_is_ratio: Optional[torch.Tensor] = None,
    delta: Optional[float] = None,
    use_bias_correction_kl: bool = False,
    num_items_in_batch: Optional[int] = None,
    vespo_k_pos: float = 2.0,
    vespo_lambda_pos: float = 3.0,
    vespo_k_neg: float = 3.0,
    vespo_lambda_neg: float = 2.0,
    chunk_size: int = 1,
):
    return _fused_linear_grpo_native(
        hidden,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias,
        ref_per_token_logps,
        old_per_token_logps,
        ref_input,
        ref_weight,
        ref_bias,
        beta,
        epsilon_low,
        epsilon_high,
        loss_type,
        max_completion_length,
        importance_sampling_level,
        sapo_temperature_pos,
        sapo_temperature_neg,
        temperature,
        vllm_is_ratio,
        delta,
        use_bias_correction_kl,
        num_items_in_batch,
        vespo_k_pos,
        vespo_lambda_pos,
        vespo_k_neg,
        vespo_lambda_neg,
        chunk_size,
    )
