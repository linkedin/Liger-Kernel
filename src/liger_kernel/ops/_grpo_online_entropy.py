# BSD 2-Clause License
# Copyright (c) 2024-2026 LinkedIn Corporation. All rights reserved.
"""Single-pass Triton online selective log-softmax + token entropy and GRPO quantile masking.

Addresses the Triton-path gap tracked in `linkedin/Liger-Kernel#1082`:
1. Extends `_selective_log_softmax_kernel` in `src/liger_kernel/ops/grpo_loss.py` with a
   shift-invariant, non-negative running entropy accumulator `s_i` alongside `(m_i, l_i)`:
       m_new = max(m_old, m_block)
       alpha = exp(m_old - m_new)
       s_new = alpha * s_old + (alpha * l_old) * (m_new - m_old)
               + sum_{v in block} exp(-(m_new - u_v)) * (m_new - u_v)
       H = log(l_N) + s_N / l_N
   Unlike naive `sum(exp(u_v - m) * u_v)`, every summand in `s_i` is >= 0 and strictly
   invariant to arbitrary shared logit offsets (`u_v + C`), preventing FP32 catastrophic
   cancellation in near-deterministic late-RL regimes (cf. `Liger-Kernel#1535`) and
   avoiding `0 * (-inf) = NaN` when vocabulary blocks are masked (cf. `Liger-Kernel#1653`).
2. Provides `_selective_log_softmax_bwd_kernel`, fusing gradients from both `grad_log_probs`
   and optional differentiable `grad_entropy` (`-(g_H / T) * p_v * (log_p_v + H)`) in a
   single vocabulary scan while respecting arbitrary row/column tensor strides.
3. Implements `compute_high_entropy_mask` and `triton_grpo_loss_with_entropy` matching
   Hugging Face TRL's exact `_compute_loss` ordering (`policy -> off_policy_mask ->
   entropy_mask -> vllm_is_ratio -> + beta * kl - entropy_coef * H`) and `-1e9` sentinel
   cross-rank padding (`huggingface/trl#3933`) so empty micro-batch masks never deadlock.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import triton
    import triton.language as tl

    HAS_TORCH_TRITON = True
except ImportError:  # Allows CPU/NumPy mathematical & AST verification when GPU stack is absent
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TORCH_TRITON = False

if TYPE_CHECKING:
    import torch as _torch


# ==============================================================================
# 1. Triton Forward & Backward Kernels (for `src/liger_kernel/ops/grpo_loss.py`)
# ==============================================================================

if HAS_TORCH_TRITON:

    def _pick_vocab_block_config(vocab_size: int) -> tuple[int, int]:
        """Select (BLOCK_SIZE, num_warps) following Liger-Kernel's online reduction heuristic."""
        if vocab_size >= 65536:
            return 4096, 16
        if vocab_size >= 16384:
            return 2048, 8
        if vocab_size >= 4096:
            return 1024, 8
        return max(256, triton.next_power_of_2(max(vocab_size, 16))), 4

    @triton.jit
    def _selective_log_softmax_kernel(
        LOGITS_PTR,
        INDEX_PTR,
        MASK_PTR,
        LOG_P_PTR,
        ENTROPY_PTR,
        LSE_PTR,
        stride_logits_row: tl.int64,
        stride_logits_col: tl.int64,
        stride_index_row: tl.int64,
        stride_mask_row: tl.int64,
        stride_logp_row: tl.int64,
        stride_entropy_row: tl.int64,
        stride_lse_row: tl.int64,
        inv_temperature: tl.float32,
        vocab_size: tl.int64,
        HAS_MASK: tl.constexpr,
        RETURN_ENTROPY: tl.constexpr,
        SAVE_LSE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Single-pass streaming selective log-softmax + shift-invariant Shannon entropy.

        Grid: `(num_rows,)` where `num_rows = batch_size * seq_len`.
        All pointer offsets are computed in `tl.int64` to prevent int32 wraparound past 2^31
        elements on large-vocabulary long-context batches (e.g. `B*T = 16384, V = 151936`).
        """
        row_idx = tl.program_id(0).to(tl.int64)

        if HAS_MASK:
            row_active = tl.load(MASK_PTR + row_idx * stride_mask_row) != 0
            if not row_active:
                tl.store(LOG_P_PTR + row_idx * stride_logp_row, 0.0)
                if RETURN_ENTROPY:
                    tl.store(ENTROPY_PTR + row_idx * stride_entropy_row, 0.0)
                if SAVE_LSE:
                    tl.store(LSE_PTR + row_idx * stride_lse_row, 0.0)
                return

        target_idx = tl.load(INDEX_PTR + row_idx * stride_index_row).to(tl.int64)
        row_logits_ptr = LOGITS_PTR + row_idx * stride_logits_row

        # Guard out-of-range / negative padding indices (-100) safely
        valid_target = (target_idx >= 0) & (target_idx < vocab_size)
        safe_target_idx = tl.where(valid_target, target_idx, 0)
        target_logit = tl.load(row_logits_ptr + safe_target_idx * stride_logits_col).to(tl.float32) * inv_temperature

        m_i = -float("inf")
        l_i = 0.0
        s_i = 0.0

        for start_v in range(0, vocab_size, BLOCK_SIZE):
            offs_v = start_v + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            v_mask = offs_v < vocab_size
            u_block = (
                tl.load(row_logits_ptr + offs_v * stride_logits_col, mask=v_mask, other=-float("inf")).to(tl.float32)
                * inv_temperature
            )
            u_block = tl.where(v_mask, u_block, -float("inf"))
            m_block = tl.max(u_block, axis=0)

            # Finite guard when an entire block is masked or contains -inf (#1653)
            block_has_finite = m_block > -float("inf")
            m_new = tl.where(block_has_finite, tl.maximum(m_i, m_block), m_i)
            has_old = m_i > -float("inf")
            alpha = tl.where(has_old & block_has_finite, tl.exp(m_i - m_new), tl.where(has_old, 1.0, 0.0))
            delta_m = tl.where(has_old & block_has_finite, m_new - m_i, 0.0)

            diff = tl.where(v_mask & block_has_finite, m_new - u_block, 0.0)
            p_unnorm = tl.where(v_mask & block_has_finite, tl.exp(-diff), 0.0)

            if RETURN_ENTROPY:
                s_i = alpha * s_i + (alpha * l_i) * delta_m + tl.sum(p_unnorm * diff, axis=0)

            l_i = alpha * l_i + tl.sum(p_unnorm, axis=0)
            m_i = m_new

        has_any_valid = l_i > 0.0
        safe_l = tl.where(has_any_valid, l_i, 1.0)
        safe_m = tl.where(has_any_valid, m_i, 0.0)
        lse = safe_m + tl.log(safe_l)

        log_p = tl.where(valid_target & has_any_valid, target_logit - lse, 0.0)
        tl.store(LOG_P_PTR + row_idx * stride_logp_row, log_p)

        if RETURN_ENTROPY:
            entropy = tl.where(has_any_valid, tl.log(safe_l) + s_i / safe_l, 0.0)
            tl.store(ENTROPY_PTR + row_idx * stride_entropy_row, entropy)

        if SAVE_LSE:
            tl.store(LSE_PTR + row_idx * stride_lse_row, lse)

    @triton.jit
    def _selective_log_softmax_bwd_kernel(
        LOGITS_PTR,
        INDEX_PTR,
        MASK_PTR,
        LSE_PTR,
        ENTROPY_PTR,
        GRAD_LOGP_PTR,
        GRAD_ENTROPY_PTR,
        GRAD_LOGITS_PTR,
        stride_logits_row: tl.int64,
        stride_logits_col: tl.int64,
        stride_index_row: tl.int64,
        stride_mask_row: tl.int64,
        stride_lse_row: tl.int64,
        stride_entropy_row: tl.int64,
        stride_glogp_row: tl.int64,
        stride_gent_row: tl.int64,
        stride_glogits_row: tl.int64,
        stride_glogits_col: tl.int64,
        inv_temperature: tl.float32,
        vocab_size: tl.int64,
        HAS_MASK: tl.constexpr,
        HAS_GRAD_ENTROPY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused single-pass backward kernel for selective log-softmax + optional entropy gradient.

        Analytical Jacobian w.r.t. raw logits `z_v` (`u_v = z_v * inv_temperature`):
            dL / dz_v = inv_temperature * [
                g_logp * (1_{v == target} - p_v)
                - g_ent * p_v * (log_p_v + H)
            ]
        """
        row_idx = tl.program_id(0).to(tl.int64)
        row_glogits_ptr = GRAD_LOGITS_PTR + row_idx * stride_glogits_row

        if HAS_MASK:
            row_active = tl.load(MASK_PTR + row_idx * stride_mask_row) != 0
            if not row_active:
                for start_v in range(0, vocab_size, BLOCK_SIZE):
                    offs_v = start_v + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                    v_mask = offs_v < vocab_size
                    tl.store(row_glogits_ptr + offs_v * stride_glogits_col, 0.0, mask=v_mask)
                return

        g_logp = tl.load(GRAD_LOGP_PTR + row_idx * stride_glogp_row).to(tl.float32)
        g_ent = tl.load(GRAD_ENTROPY_PTR + row_idx * stride_gent_row).to(tl.float32) if HAS_GRAD_ENTROPY else 0.0

        # Fast-path zero-fill when upstream token receives zero gradient (e.g. masked out by top_entropy_quantile)
        if (g_logp == 0.0) & (g_ent == 0.0):
            for start_v in range(0, vocab_size, BLOCK_SIZE):
                offs_v = start_v + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                v_mask = offs_v < vocab_size
                tl.store(row_glogits_ptr + offs_v * stride_glogits_col, 0.0, mask=v_mask)
            return

        target_idx = tl.load(INDEX_PTR + row_idx * stride_index_row).to(tl.int64)
        valid_target = (target_idx >= 0) & (target_idx < vocab_size)
        eff_g_logp = tl.where(valid_target, g_logp, 0.0)
        lse = tl.load(LSE_PTR + row_idx * stride_lse_row).to(tl.float32)
        ent = tl.load(ENTROPY_PTR + row_idx * stride_entropy_row).to(tl.float32) if HAS_GRAD_ENTROPY else 0.0
        row_logits_ptr = LOGITS_PTR + row_idx * stride_logits_row

        for start_v in range(0, vocab_size, BLOCK_SIZE):
            offs_v = start_v + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            v_mask = offs_v < vocab_size
            u_block = (
                tl.load(row_logits_ptr + offs_v * stride_logits_col, mask=v_mask, other=-float("inf")).to(tl.float32)
                * inv_temperature
            )
            log_p = u_block - lse
            p = tl.where(v_mask, tl.exp(log_p), 0.0)
            is_target = tl.where(valid_target & (offs_v == target_idx), 1.0, 0.0)

            grad_u = eff_g_logp * (is_target - p)
            if HAS_GRAD_ENTROPY:
                grad_u = grad_u - g_ent * p * (log_p + ent)

            grad_z = grad_u * inv_temperature
            tl.store(row_glogits_ptr + offs_v * stride_glogits_col, grad_z, mask=v_mask)

    class LigerSelectiveLogSoftmaxEntropyFunction(torch.autograd.Function):
        """Autograd wrapper around single-pass Triton selective log-softmax + token entropy."""

        @staticmethod
        def forward(
            ctx,
            logits: _torch.Tensor,
            index: _torch.Tensor,
            temperature: float = 1.0,
            mask: _torch.Tensor | None = None,
            return_entropy: bool = True,
            inplace_backward: bool = False,
        ) -> tuple[_torch.Tensor, _torch.Tensor | None]:
            if temperature <= 0.0:
                raise ValueError(f"temperature must be > 0, got {temperature}")
            ctx.set_materialize_grads(False)

            orig_shape = index.shape
            vocab_size = logits.shape[-1]
            num_rows = index.numel()
            inv_temp = 1.0 / float(temperature)

            # Reshape leading dimensions while preserving non-contiguous row/col strides when possible
            logits_2d = logits.reshape(num_rows, vocab_size)
            index_1d = index.reshape(num_rows)
            mask_1d = mask.reshape(num_rows) if mask is not None else None

            needs_grad = logits.requires_grad and torch.is_grad_enabled()
            log_p = torch.empty((num_rows,), device=logits.device, dtype=torch.float32)
            entropy = (
                torch.empty((num_rows,), device=logits.device, dtype=torch.float32)
                if return_entropy
                else torch.empty((0,), device=logits.device, dtype=torch.float32)
            )
            lse = (
                torch.empty((num_rows,), device=logits.device, dtype=torch.float32)
                if needs_grad
                else torch.empty((0,), device=logits.device, dtype=torch.float32)
            )

            if num_rows > 0:
                block_size, num_warps = _pick_vocab_block_config(vocab_size)
                _selective_log_softmax_kernel[(num_rows,)](
                    logits_2d,
                    index_1d,
                    mask_1d if mask_1d is not None else index_1d,
                    log_p,
                    entropy if return_entropy else log_p,
                    lse if needs_grad else log_p,
                    logits_2d.stride(0),
                    logits_2d.stride(1),
                    index_1d.stride(0),
                    mask_1d.stride(0) if mask_1d is not None else 0,
                    log_p.stride(0),
                    entropy.stride(0) if return_entropy else 0,
                    lse.stride(0) if needs_grad else 0,
                    inv_temp,
                    vocab_size,
                    HAS_MASK=mask_1d is not None,
                    RETURN_ENTROPY=return_entropy,
                    SAVE_LSE=needs_grad,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )

            if needs_grad:
                ctx.save_for_backward(
                    logits_2d,
                    index_1d,
                    mask_1d if mask_1d is not None else torch.empty((0,), device=logits.device, dtype=torch.bool),
                    lse,
                    entropy,
                )
                ctx.has_mask = mask_1d is not None
                ctx.return_entropy = return_entropy
                ctx.inv_temp = inv_temp
                ctx.orig_logits_shape = logits.shape
                ctx.inplace_backward = inplace_backward and logits_2d.is_contiguous()

            log_p_out = log_p.reshape(orig_shape)
            entropy_out = entropy.reshape(orig_shape) if return_entropy else None
            return log_p_out, entropy_out

        @staticmethod
        def backward(ctx, grad_log_p: _torch.Tensor | None, grad_entropy: _torch.Tensor | None):
            if grad_log_p is None and grad_entropy is None:
                return None, None, None, None, None, None

            logits_2d, index_1d, mask_1d, lse, entropy = ctx.saved_tensors
            num_rows, vocab_size = logits_2d.shape

            if grad_log_p is None:
                g_logp_1d = torch.zeros((num_rows,), device=logits_2d.device, dtype=torch.float32)
            else:
                g_logp_1d = grad_log_p.reshape(num_rows)

            has_grad_ent = ctx.return_entropy and (grad_entropy is not None)
            g_ent_1d = grad_entropy.reshape(num_rows) if has_grad_ent else g_logp_1d

            grad_logits_2d = logits_2d if ctx.inplace_backward else torch.empty_like(logits_2d)

            if num_rows > 0:
                block_size, num_warps = _pick_vocab_block_config(vocab_size)
                _selective_log_softmax_bwd_kernel[(num_rows,)](
                    logits_2d,
                    index_1d,
                    mask_1d if ctx.has_mask else index_1d,
                    lse,
                    entropy if has_grad_ent else lse,
                    g_logp_1d,
                    g_ent_1d,
                    grad_logits_2d,
                    logits_2d.stride(0),
                    logits_2d.stride(1),
                    index_1d.stride(0),
                    mask_1d.stride(0) if ctx.has_mask else 0,
                    lse.stride(0),
                    entropy.stride(0) if has_grad_ent else 0,
                    g_logp_1d.stride(0),
                    g_ent_1d.stride(0) if has_grad_ent else 0,
                    grad_logits_2d.stride(0),
                    grad_logits_2d.stride(1),
                    ctx.inv_temp,
                    vocab_size,
                    HAS_MASK=ctx.has_mask,
                    HAS_GRAD_ENTROPY=has_grad_ent,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )

            return grad_logits_2d.reshape(ctx.orig_logits_shape), None, None, None, None, None

    def selective_log_softmax_with_entropy(
        logits: _torch.Tensor,
        index: _torch.Tensor,
        temperature: float = 1.0,
        mask: _torch.Tensor | None = None,
        return_entropy: bool = True,
        entropy_requires_grad: bool = True,
        inplace_backward: bool = False,
    ) -> tuple[_torch.Tensor, _torch.Tensor | None]:
        """Public Triton entrypoint computing selected token log-probs and exact FP32 entropy in 1 pass."""
        log_p, entropy = LigerSelectiveLogSoftmaxEntropyFunction.apply(
            logits, index, temperature, mask, return_entropy, inplace_backward
        )
        if entropy is not None and not entropy_requires_grad:
            entropy = entropy.detach()
        return log_p, entropy

    def compute_high_entropy_mask(
        entropies: _torch.Tensor,
        mask: _torch.Tensor,
        top_entropy_quantile: float,
        process_group: object | None = None,
    ) -> _torch.Tensor:
        """Compute DAPO/TRL high-entropy token mask (`q = 1.0 - top_entropy_quantile`) safely across ranks.

        Uses `-1e9` sentinel padding across distributed ranks (`huggingface/trl#3933`) so ranks
        with zero valid tokens still participate in collective communication without deadlocking.
        """
        if not (0.0 < top_entropy_quantile <= 1.0):
            raise ValueError(f"top_entropy_quantile must be in (0.0, 1.0], got {top_entropy_quantile}")
        mask_bool = mask.bool()
        if top_entropy_quantile >= 1.0:
            return mask_bool

        local_valid = entropies.detach()[mask_bool].float()
        gathered_valid = local_valid

        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(group=process_group)
            if world_size > 1:
                sentinel = -1e9
                local_len = torch.tensor([local_valid.numel()], device=entropies.device, dtype=torch.int64)
                max_len_tensor = local_len.clone()
                dist.all_reduce(max_len_tensor, op=dist.ReduceOp.MAX, group=process_group)
                max_len = int(max_len_tensor.item())
                if max_len == 0:
                    return torch.zeros_like(mask_bool)
                padded = torch.full((max_len,), sentinel, device=entropies.device, dtype=torch.float32)
                if local_valid.numel() > 0:
                    padded[: local_valid.numel()] = local_valid
                gather_list = [torch.empty_like(padded) for _ in range(world_size)]
                dist.all_gather(gather_list, padded, group=process_group)
                all_padded = torch.cat(gather_list, dim=0)
                gathered_valid = all_padded[all_padded != sentinel]

        if gathered_valid.numel() == 0:
            return torch.zeros_like(mask_bool)

        quantile_q = 1.0 - float(top_entropy_quantile)
        threshold = torch.quantile(gathered_valid, quantile_q)
        masked_entropies = entropies.detach().float() * mask_bool.float()
        return (masked_entropies >= threshold) & mask_bool

    def triton_grpo_loss_with_entropy(
        logits: _torch.Tensor,
        selected_token_ids: _torch.Tensor,
        attention_mask: _torch.Tensor,
        advantages: _torch.Tensor,
        old_per_token_logps: _torch.Tensor | None = None,
        ref_per_token_logps: _torch.Tensor | None = None,
        vllm_is_ratio: _torch.Tensor | None = None,
        sampling_per_token_logps: _torch.Tensor | None = None,
        temperature: float = 1.0,
        beta: float = 0.04,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        delta: float | None = None,
        loss_type: str = "grpo",
        max_completion_length: int | None = None,
        num_items_in_batch: float | _torch.Tensor | None = None,
        importance_sampling_level: str = "token",
        use_bias_correction_kl: bool = False,
        off_policy_mask_threshold: float | None = None,
        top_entropy_quantile: float = 1.0,
        entropy_coef: float = 0.0,
        inplace_backward: bool = False,
        process_group: object | None = None,
    ) -> tuple[_torch.Tensor, dict[str, _torch.Tensor]]:
        """Full Triton-backed GRPO/DAPO/BNPO/Dr.GRPO/CISPO loss with 1-pass entropy & quantile masking.

        Follows Hugging Face TRL `_compute_loss` operation ordering strictly:
            1. Single-pass Triton `_selective_log_softmax_kernel` -> `(per_token_logps, entropies)`
            2. Policy objective (`grpo` / `dapo` / `bnpo` / `dr_grpo` / `cispo`)
            3. `off_policy_mask` (OPSM, when `off_policy_mask_threshold` is set)
            4. `entropy_mask` (`top_entropy_quantile < 1.0`)
            5. `vllm_is_ratio` importance sampling scaling
            6. `+ beta * per_token_kl`
            7. Loss-type reduction (`grpo`, `bnpo`, `dr_grpo`, `dapo`/`cispo`)
            8. `- entropy_coef * masked_mean_entropy` (over `effective_mask = mask * entropy_mask`)
        """
        need_entropy = True  # Always zero-extra-HBM in single-pass Triton kernel for metrics/mask/bonus
        entropy_grad = entropy_coef != 0.0
        mask = attention_mask.to(dtype=torch.float32)

        per_token_logps, entropies = selective_log_softmax_with_entropy(
            logits=logits,
            index=selected_token_ids,
            temperature=temperature,
            mask=attention_mask,
            return_entropy=need_entropy,
            entropy_requires_grad=entropy_grad,
            inplace_backward=inplace_backward,
        )
        assert entropies is not None

        entropy_mask = (
            compute_high_entropy_mask(entropies, attention_mask, top_entropy_quantile, process_group=process_group)
            if top_entropy_quantile < 1.0
            else None
        )

        adv = advantages.float()
        if adv.dim() == 1:
            adv = adv.unsqueeze(1)

        old_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps.float()
        log_ratio = per_token_logps - old_logps

        if importance_sampling_level == "token":
            log_iw = log_ratio
        elif importance_sampling_level == "sequence":
            log_iw = ((log_ratio * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported importance_sampling_level: {importance_sampling_level!r}")

        coef_1 = torch.exp(log_iw)

        if loss_type == "cispo":
            clamped = torch.clamp(coef_1, max=epsilon_high).detach()
            per_token_loss = -clamped * adv * per_token_logps
        elif loss_type in ("grpo", "bnpo", "dr_grpo", "dapo"):
            coef_2 = torch.clamp(coef_1, 1.0 - epsilon_low, 1.0 + epsilon_high)
            coef_1_eff = torch.clamp(coef_1, max=delta) if delta is not None else coef_1
            per_token_loss = -torch.minimum(coef_1_eff * adv, coef_2 * adv)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type!r}")

        if off_policy_mask_threshold is not None:
            sampler_logps = old_logps if sampling_per_token_logps is None else sampling_per_token_logps.float()
            kl_drift = torch.nan_to_num(sampler_logps - per_token_logps.detach(), nan=0.0)
            avg_seq_kl = (kl_drift * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            opsm = ((adv >= 0.0) | (avg_seq_kl <= off_policy_mask_threshold)).to(dtype=per_token_loss.dtype)
            per_token_loss = per_token_loss * opsm

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask.to(dtype=per_token_loss.dtype)

        if vllm_is_ratio is not None:
            per_token_loss = per_token_loss * vllm_is_ratio.float()

        per_token_kl = None
        if beta != 0.0:
            if ref_per_token_logps is None:
                raise ValueError("ref_per_token_logps must be provided when beta != 0.0")
            ref_logps = ref_per_token_logps.float()
            diff_ref = ref_logps - per_token_logps
            per_token_kl = torch.exp(diff_ref) - diff_ref - 1.0
            if use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1
            per_token_loss = per_token_loss + beta * per_token_kl

        if loss_type == "grpo":
            loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)).mean()
        elif loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            denom_len = max_completion_length if max_completion_length is not None else per_token_loss.shape[1]
            loss = (per_token_loss * mask).sum() / float(per_token_loss.shape[0] * denom_len)
        elif loss_type in ("dapo", "cispo"):
            if num_items_in_batch is not None:
                denom = (
                    num_items_in_batch.float().clamp(min=1.0)
                    if isinstance(num_items_in_batch, torch.Tensor)
                    else max(float(num_items_in_batch), 1.0)
                )
            else:
                denom = mask.sum().clamp(min=1.0)
            loss = (per_token_loss * mask).sum() / denom

        policy_loss_detached = loss.detach()
        effective_mask = mask if entropy_mask is None else mask * entropy_mask.to(dtype=mask.dtype)
        mean_effective_entropy = (entropies * effective_mask).sum() / effective_mask.sum().clamp(min=1.0)

        if entropy_coef != 0.0:
            loss = loss - entropy_coef * mean_effective_entropy

        metrics: dict[str, _torch.Tensor] = {
            "policy_loss": policy_loss_detached,
            "entropy": (entropies.detach() * mask).sum() / mask.sum().clamp(min=1.0),
            "per_token_entropy": entropies.detach(),
        }
        if per_token_kl is not None:
            metrics["kl"] = (per_token_kl.detach() * mask).sum() / mask.sum().clamp(min=1.0)
        if entropy_mask is not None:
            metrics["entropy_masked_fraction"] = 1.0 - (effective_mask.sum() / mask.sum().clamp(min=1.0)).detach()
        return loss, metrics


# ==============================================================================
# 2. Pure-NumPy Tiled Triton Emulator (for CPU / Sandbox Exact Math Validation)
# ==============================================================================


def simulate_bf16_round_np(x: np.ndarray) -> np.ndarray:
    """Round FP32/FP64 array to exact IEEE-754 BFloat16 precision (round-to-nearest-even)."""
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    rounded = np.where(np.isnan(x32), bits, (bits + rounding_bias) & np.uint32(0xFFFF0000))
    return rounded.view(np.float32)


def numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd(
    logits: np.ndarray,
    index: np.ndarray,
    temperature: float = 1.0,
    mask: np.ndarray | None = None,
    grad_log_p: np.ndarray | None = None,
    grad_entropy: np.ndarray | None = None,
    block_size: int = 256,
    simulate_bf16_input: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile-by-tile FP32 emulation of `_selective_log_softmax_kernel` & `_selective_log_softmax_bwd_kernel`."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    orig_shape = index.shape
    vocab_size = logits.shape[-1]
    num_rows = int(np.prod(orig_shape))
    inv_temp = np.float32(1.0 / float(temperature))

    z_2d = simulate_bf16_round_np(logits) if simulate_bf16_input else np.asarray(logits, dtype=np.float32)
    z_2d = z_2d.reshape(num_rows, vocab_size)
    idx_1d = np.asarray(index, dtype=np.int64).reshape(num_rows)
    mask_1d = np.ones((num_rows,), dtype=bool) if mask is None else np.asarray(mask).reshape(num_rows).astype(bool)

    log_p_out = np.zeros((num_rows,), dtype=np.float32)
    ent_out = np.zeros((num_rows,), dtype=np.float32)
    lse_saved = np.zeros((num_rows,), dtype=np.float32)

    for r in range(num_rows):
        if not mask_1d[r]:
            continue
        m_i = np.float32(-np.inf)
        l_i = np.float32(0.0)
        s_i = np.float32(0.0)

        for start_v in range(0, vocab_size, block_size):
            end_v = min(start_v + block_size, vocab_size)
            u_blk = (z_2d[r, start_v:end_v].astype(np.float32)) * inv_temp
            finite_mask = np.isfinite(u_blk)
            if not np.any(finite_mask):
                continue
            m_blk = np.float32(np.max(np.where(finite_mask, u_blk, np.float32(-np.inf))))
            has_old = np.isfinite(m_i)
            m_new = np.float32(max(float(m_i), float(m_blk))) if has_old else m_blk
            alpha = np.float32(math.exp(float(m_i - m_new))) if has_old else np.float32(0.0)
            delta_m = np.float32(m_new - m_i) if has_old else np.float32(0.0)

            diff = np.where(finite_mask, m_new - u_blk, np.float32(0.0)).astype(np.float32)
            p_unnorm = np.where(finite_mask, np.exp(-diff, dtype=np.float32), np.float32(0.0))

            s_i = np.float32(alpha * s_i + (alpha * l_i) * delta_m + np.sum(p_unnorm * diff, dtype=np.float32))
            l_i = np.float32(alpha * l_i + np.sum(p_unnorm, dtype=np.float32))
            m_i = m_new

        if l_i > 0.0:
            lse = np.float32(m_i + np.float32(math.log(float(l_i))))
            lse_saved[r] = lse
            ent_out[r] = np.float32(math.log(float(l_i)) + float(s_i / l_i))
            t_idx = int(idx_1d[r])
            if 0 <= t_idx < vocab_size:
                u_target = np.float32(z_2d[r, t_idx]) * inv_temp
                log_p_out[r] = np.float32(u_target - lse)

    grad_z = np.zeros_like(z_2d, dtype=np.float32)
    g_lp = (
        np.zeros((num_rows,), dtype=np.float32)
        if grad_log_p is None
        else np.asarray(grad_log_p, dtype=np.float32).reshape(num_rows)
    )
    g_h = (
        np.zeros((num_rows,), dtype=np.float32)
        if grad_entropy is None
        else np.asarray(grad_entropy, dtype=np.float32).reshape(num_rows)
    )

    for r in range(num_rows):
        if not mask_1d[r] or (g_lp[r] == 0.0 and g_h[r] == 0.0):
            continue
        t_idx = int(idx_1d[r])
        eff_glp = g_lp[r] if (0 <= t_idx < vocab_size) else np.float32(0.0)
        lse = lse_saved[r]
        ent = ent_out[r]

        for start_v in range(0, vocab_size, block_size):
            end_v = min(start_v + block_size, vocab_size)
            u_blk = z_2d[r, start_v:end_v].astype(np.float32) * inv_temp
            finite_mask = np.isfinite(u_blk)
            log_p = np.where(finite_mask, u_blk - lse, np.float32(-np.inf))
            p = np.where(finite_mask, np.exp(log_p, dtype=np.float32), np.float32(0.0))
            is_tgt = np.zeros_like(p)
            if start_v <= t_idx < end_v:
                is_tgt[t_idx - start_v] = np.float32(1.0)
            grad_u = eff_glp * (is_tgt - p) - g_h[r] * p * np.where(finite_mask, log_p + ent, np.float32(0.0))
            grad_z[r, start_v:end_v] = grad_u * inv_temp

    return log_p_out.reshape(orig_shape), ent_out.reshape(orig_shape), grad_z.reshape(logits.shape)
