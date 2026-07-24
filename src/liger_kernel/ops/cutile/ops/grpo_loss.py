# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
GRPO loss (cuTile backend).

Token-level GRPO-family policy-gradient loss. Each grid block (b, l) computes the per-token
loss for one (batch, completion-token) position: logsumexp over the vocab, target logp, the
importance ratio coef_1 = exp(logp - old_logp), then the loss for the selected loss_type
(GRPO/DAPO/BNPO/DR-GRPO/LUSPO share the PPO-clipped form; CISPO; SAPO), optional dual-clip
(delta), optional vLLM IS-ratio reweighting, and an optional KL penalty. Backward recomputes
logp from the cached lse and writes dlogits.

Full feature parity with the Triton GrpoLossFunction: all loss types (GRPO/DAPO/BNPO/DR-GRPO/
LUSPO/CISPO/SAPO/VESPO), token- and sequence-level (GSPO) importance sampling, dual-clip,
vLLM IS-ratio, KL penalty, and num_items_in_batch normalization.
"""

import math
import os

from types import SimpleNamespace

import cuda.tile as ct
import torch

from cuda.tile import ByTarget
from cuda.tile.tune import exhaustive_search

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2

LOG2E = 1.4426950408889634

_LOSS_TYPE_GRPO = 0
_LOSS_TYPE_CISPO = 1
_LOSS_TYPE_SAPO = 2
_LOSS_TYPE_VESPO = 3

_str_to_loss_type = {
    "grpo": _LOSS_TYPE_GRPO,
    "dapo": _LOSS_TYPE_GRPO,
    "bnpo": _LOSS_TYPE_GRPO,
    "dr_grpo": _LOSS_TYPE_GRPO,
    "luspo": _LOSS_TYPE_GRPO,  # LUSPO uses the same per-token PPO clipping as GRPO
    "cispo": _LOSS_TYPE_CISPO,
    "sapo": _LOSS_TYPE_SAPO,
    "vespo": _LOSS_TYPE_VESPO,
}


@ct.kernel
def _grpo_loss_fwd_ct(
    logits_input,  # (B*(L+1), N) logits (2D view of (B, L+1, N))
    old_logp_input,  # (B, L) float32 or dummy (1,)
    ref_logp_input,  # (B, L) float32 or dummy (1,)
    input_ids,  # (B, L) int64 completion token ids
    completion_mask,  # (B, L) int32 or dummy (1,)
    advantages,  # (B,) float32
    vllm_is_ratio,  # (B, L) or dummy (1,) float32
    phi_seq_input,  # (B,) VESPO per-sequence gamma weight (or dummy (1,))
    loss_output,  # (B, L) float32 output
    lse_cache,  # (B, L) float32 cached log-sum-exp
    kl_output,  # (B, L) float32 or dummy (1,)
    is_clipped_output,  # (B, L) float32
    B: ct.Constant[int],
    L: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    temperature,
    beta: ct.Constant[float],
    eps_low,
    eps_high,
    loss_type: ct.Constant[int],
    sapo_temp_pos,
    sapo_temp_neg,
    delta,
    use_bias_correction_kl: ct.Constant[int],
    HAS_COMPLETION_MASK: ct.Constant[int],
    HAS_OLD_LOGP: ct.Constant[int],
    HAS_VLLM_IS_RATIO: ct.Constant[int],
    vllm_is_ratio_stride,
):
    """
    GRPO forward.

    Grid: (B, L). Each block (off_b, off_l) computes the loss for one
    (batch, completion-token) position.
    """
    off_b = ct.bid(0)
    off_l = ct.bid(1)

    # Optional completion mask: skip masked tokens
    if HAS_COMPLETION_MASK:
        mask_val = ct.astype(ct.load(completion_mask, (off_b, off_l), shape=()), ct.int32)
        if mask_val == 0:
            return

    # Logits row for this (b, l): off_b*(L+1) + off_l
    logits_row = ct.add(ct.mul(off_b, L + 1), off_l)
    n_chunks = (N + BLOCK_N - 1) // BLOCK_N
    inv_temperature = 1.0 / temperature

    # ---- Compute logsumexp via online algorithm (fold trick) ----
    m_i = ct.full((), -math.inf, dtype=ct.float32)
    l_i = ct.full((), 0.0, dtype=ct.float32)

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
        logits = ct.astype(
            ct.gather(logits_input, (logits_row, col_idx), check_bounds=True, padding_value=-math.inf, latency=3),
            ct.float32,
        )
        logits_scaled = logits * inv_temperature

        chunk_max = ct.max(logits_scaled, 0, keepdims=False)
        new_m = ct.maximum(m_i, chunk_max)
        alpha = ct.exp2(ct.mul(m_i - new_m, LOG2E))
        l_i = ct.add(ct.mul(l_i, alpha), ct.sum(ct.exp2(ct.mul(logits_scaled - new_m, LOG2E)), 0, keepdims=False))
        m_i = new_m

    lse = m_i + ct.log(l_i)

    # ---- Load logit at target token ----
    idx_raw = ct.load(input_ids, (off_b, off_l), shape=())
    idx = ct.astype(idx_raw, ct.int32)
    idx_tile = ct.add(ct.arange(1, dtype=ct.int32), idx)
    x_tile = ct.astype(ct.gather(logits_input, (logits_row, idx_tile), check_bounds=False), ct.float32)
    x = ct.sum(x_tile, 0, keepdims=False) * inv_temperature
    logp = x - lse

    # ---- Load old_logp ----
    if HAS_OLD_LOGP:
        old_logp = ct.astype(ct.load(old_logp_input, (off_b, off_l), shape=()), ct.float32)
    else:
        old_logp = logp
    coef_1 = ct.exp(logp - old_logp)
    advantage = ct.astype(ct.load(advantages, off_b, shape=()), ct.float32)

    # ---- Compute per-token loss based on loss_type ----
    if loss_type == 0:  # GRPO: standard PPO clipping
        coef_2_low = ct.maximum(coef_1, ct.full((), 1.0 - eps_low, dtype=ct.float32))
        coef_2_high = ct.minimum(coef_2_low, ct.full((), 1.0 + eps_high, dtype=ct.float32))
        is_low_clipped = ct.astype(coef_1 < (1.0 - eps_low), ct.float32) * ct.astype(advantage < 0.0, ct.float32)
        is_high_clipped = ct.astype(coef_1 > (1.0 + eps_high), ct.float32) * ct.astype(advantage > 0.0, ct.float32)
        is_clipped = ct.minimum(is_low_clipped + is_high_clipped, ct.full((), 1.0, dtype=ct.float32))
        # Apply delta upper-clip on importance ratio (dual-clip extension)
        if delta != 0.0:
            coef_1_for_loss = ct.minimum(coef_1, ct.full((), delta, dtype=ct.float32))
        else:
            coef_1_for_loss = coef_1
        per_token_loss1 = coef_1_for_loss * advantage
        per_token_loss2 = coef_2_high * advantage
        per_token_loss = -ct.minimum(per_token_loss1, per_token_loss2)
    elif loss_type == 1:  # CISPO
        coef_2 = ct.minimum(coef_1, ct.full((), eps_high, dtype=ct.float32))
        per_token_loss = -coef_2 * advantage * logp
        is_clipped = ct.astype(coef_1 > eps_high, ct.float32) * ct.astype(advantage > 0.0, ct.float32)
    elif loss_type == 2:  # SAPO
        temp_sapo = ct.maximum(
            ct.full((), sapo_temp_pos, dtype=ct.float32),
            ct.full((), sapo_temp_neg, dtype=ct.float32),
        )
        if advantage > 0.0:
            temp_sapo = ct.full((), sapo_temp_pos, dtype=ct.float32)
        else:
            temp_sapo = ct.full((), sapo_temp_neg, dtype=ct.float32)
        sigmoid_input = temp_sapo * (coef_1 - 1.0)
        # sigmoid(x) = 1 / (1 + exp(-x)) — cuda.tile has no ct.sigmoid.
        sig = ct.truediv(1.0, 1.0 + ct.exp(0.0 - sigmoid_input))
        sapo_coef = sig * 4.0 / temp_sapo
        per_token_loss = -sapo_coef * advantage
        is_clipped = ct.full((), 0.0, dtype=ct.float32)
    else:  # loss_type == 3: VESPO — detached per-sequence gamma weight on logp
        phi_seq = ct.astype(ct.load(phi_seq_input, off_b, shape=()), ct.float32)
        per_token_loss = -phi_seq * advantage * logp
        is_clipped = ct.full((), 0.0, dtype=ct.float32)

    # ---- Apply vLLM IS ratio (optional) ----
    if HAS_VLLM_IS_RATIO:
        vllm_col = off_l % vllm_is_ratio_stride
        vllm_row_base = off_b * vllm_is_ratio_stride
        # scalar gather: 1-element tile
        vllm_tile = ct.gather(
            vllm_is_ratio, ct.arange(1, dtype=ct.int32) + vllm_row_base + vllm_col, check_bounds=False
        )
        vllm_ratio = ct.astype(ct.sum(ct.astype(vllm_tile, ct.float32), 0, keepdims=False), ct.float32)
        per_token_loss = per_token_loss * vllm_ratio

    # ---- KL penalty (optional, beta != 0 is compile-time) ----
    if beta != 0.0:
        ref_logp = ct.astype(ct.load(ref_logp_input, (off_b, off_l), shape=()), ct.float32)
        kl = ct.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        if use_bias_correction_kl:
            # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= coef_1
            kl = kl * coef_1
        per_token_loss = per_token_loss + beta * kl
        ct.scatter(kl_output, (off_b, off_l), ct.astype(kl, kl_output.dtype))

    # ---- Store outputs ----
    ct.scatter(loss_output, (off_b, off_l), ct.astype(per_token_loss, loss_output.dtype))
    ct.scatter(lse_cache, (off_b, off_l), ct.astype(lse, lse_cache.dtype))
    ct.scatter(is_clipped_output, (off_b, off_l), ct.astype(is_clipped, is_clipped_output.dtype))


@ct.kernel
def _grpo_loss_bwd_ct(
    dloss_input,  # (B, L) float32 upstream gradient
    dlogits_output,  # (B*(L+1), N) gradient output
    logits_input,  # (B*(L+1), N) saved logits
    old_logp_input,  # (B, L) or dummy (1,)
    ref_logp_input,  # (B, L) or dummy (1,)
    input_ids,  # (B, L) int64
    advantages,  # (B,) float32
    completion_mask,  # (B, L) int32 or dummy (1,)
    lse_cache,  # (B, L) float32 saved lse
    vllm_is_ratio,  # (B, L) or dummy (1,)
    phi_seq_input,  # (B,) VESPO per-sequence gamma weight (or dummy (1,))
    B: ct.Constant[int],
    L: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    temperature,
    beta: ct.Constant[float],
    eps_low,
    eps_high,
    loss_type: ct.Constant[int],
    sapo_temp_pos,
    sapo_temp_neg,
    delta,
    use_bias_correction_kl: ct.Constant[int],
    HAS_COMPLETION_MASK: ct.Constant[int],
    HAS_OLD_LOGP: ct.Constant[int],
    HAS_VLLM_IS_RATIO: ct.Constant[int],
    vllm_is_ratio_stride,
):
    """GRPO backward. Grid: (B, L)."""
    off_b = ct.bid(0)
    off_l = ct.bid(1)

    logits_row = ct.add(ct.mul(off_b, L + 1), off_l)
    n_chunks = (N + BLOCK_N - 1) // BLOCK_N
    inv_temperature = 1.0 / temperature

    if HAS_COMPLETION_MASK:
        mask_val = ct.astype(ct.load(completion_mask, (off_b, off_l), shape=()), ct.int32)
        if mask_val == 0:
            for ci in range(n_chunks):
                col_idx = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
                zero_tile = ct.full((BLOCK_N,), 0.0, dtype=dlogits_output.dtype)
                ct.scatter(dlogits_output, (logits_row, col_idx), zero_tile, check_bounds=True)
            return

    dloss = ct.astype(ct.load(dloss_input, (off_b, off_l), shape=()), ct.float32)
    lse = ct.astype(ct.load(lse_cache, (off_b, off_l), shape=()), ct.float32)

    idx_raw = ct.load(input_ids, (off_b, off_l), shape=())
    idx = ct.astype(idx_raw, ct.int32)
    idx_tile = ct.add(ct.arange(1, dtype=ct.int32), idx)
    x_tile = ct.astype(ct.gather(logits_input, (logits_row, idx_tile), check_bounds=False), ct.float32)
    x = ct.sum(x_tile, 0, keepdims=False) * inv_temperature
    logp = x - lse

    if HAS_OLD_LOGP:
        old_logp = ct.astype(ct.load(old_logp_input, (off_b, off_l), shape=()), ct.float32)
    else:
        old_logp = logp
    coef_1 = ct.exp(logp - old_logp)
    advantage = ct.astype(ct.load(advantages, off_b, shape=()), ct.float32)

    if loss_type == 0:  # GRPO
        coef_2_low = ct.maximum(coef_1, ct.full((), 1.0 - eps_low, dtype=ct.float32))
        coef_2_high = ct.minimum(coef_2_low, ct.full((), 1.0 + eps_high, dtype=ct.float32))
        if delta != 0.0:
            coef_1_for_loss = ct.minimum(coef_1, ct.full((), delta, dtype=ct.float32))
        else:
            coef_1_for_loss = coef_1
        per_token_loss1 = coef_1_for_loss * advantage
        per_token_loss2 = coef_2_high * advantage
        # gradient flows only when unclipped (per_token_loss2 >= per_token_loss1)
        grad_mask = ct.astype(per_token_loss2 >= per_token_loss1, ct.float32)
        # Gradient uses original coef_1; zero when delta-clamped (constant → no gradient)
        dlogp = -coef_1 * advantage * grad_mask
        if delta != 0.0:
            dlogp = dlogp * ct.astype(coef_1 <= ct.full((), delta, dtype=ct.float32), ct.float32)
    elif loss_type == 1:  # CISPO
        coef_2 = ct.minimum(coef_1, ct.full((), eps_high, dtype=ct.float32))
        dlogp = -coef_2 * advantage
    elif loss_type == 2:  # SAPO
        if advantage > 0.0:
            temp_sapo = ct.full((), sapo_temp_pos, dtype=ct.float32)
        else:
            temp_sapo = ct.full((), sapo_temp_neg, dtype=ct.float32)
        sigmoid_input = temp_sapo * (coef_1 - 1.0)
        sigmoid_val = ct.truediv(1.0, 1.0 + ct.exp(0.0 - sigmoid_input))  # sigmoid via 1/(1+exp(-x))
        d_sapo_d_coef1 = 4.0 * sigmoid_val * (1.0 - sigmoid_val)
        dlogp = -advantage * d_sapo_d_coef1 * coef_1
    else:  # loss_type == 3: VESPO — loss = -phi_seq*advantage*logp, phi_seq detached
        phi_seq = ct.astype(ct.load(phi_seq_input, off_b, shape=()), ct.float32)
        dlogp = -phi_seq * advantage

    if HAS_VLLM_IS_RATIO:
        vllm_col = off_l % vllm_is_ratio_stride
        vllm_tile = ct.gather(
            vllm_is_ratio,
            ct.add(ct.arange(1, dtype=ct.int32), ct.mul(off_b, vllm_is_ratio_stride) + vllm_col),
            check_bounds=False,
        )
        vllm_ratio = ct.astype(ct.sum(ct.astype(vllm_tile, ct.float32), 0, keepdims=False), ct.float32)
        dlogp = dlogp * vllm_ratio

    if beta != 0.0:
        ref_logp = ct.astype(ct.load(ref_logp_input, (off_b, off_l), shape=()), ct.float32)
        if use_bias_correction_kl:
            # d(kl * coef_1)/d(logp) = coef_1 * (logp - ref_logp), where coef_1 = exp(logp - old_logp)
            dlogp = dlogp + beta * coef_1 * (logp - ref_logp)
        else:
            dlogp = dlogp + beta * (1.0 - ct.exp(ref_logp - logp))

    dlogp_scaled = dlogp * dloss * inv_temperature

    # Compute and store dlogits for all vocab positions
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
        logits = ct.astype(
            ct.gather(logits_input, (logits_row, col_idx), check_bounds=True, padding_value=-math.inf, latency=10),
            ct.float32,
        )
        probs = ct.exp(logits * inv_temperature - lse)

        # dlogits[j] = (indicator(j==idx) - prob[j]) * dlogp
        idx_tile_b = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
        is_target = ct.astype(idx_tile_b == idx, ct.float32)
        dlogits = (is_target - probs) * dlogp_scaled

        ct.scatter(
            dlogits_output,
            (logits_row, col_idx),
            ct.astype(dlogits, dlogits_output.dtype),
            check_bounds=True,
            latency=10,
        )


# --- bwd occupancy: static per-batch-size selection (zero lookup overhead) ---
_bwd_occ_small = _grpo_loss_bwd_ct.replace_hints(occupancy=ByTarget(sm_100=4, default=4))
_bwd_occ_large = _grpo_loss_bwd_ct.replace_hints(occupancy=ByTarget(sm_100=12, default=12))


# --- fwd occupancy autotune (per launch shape) --------------------------------
_FWD_OCC_CONFIGS = [
    SimpleNamespace(occupancy=1),
    SimpleNamespace(occupancy=2),
    SimpleNamespace(occupancy=4),
    SimpleNamespace(occupancy=12),
]
_FWD_FALLBACK_OCC = 1
_fwd_autotune_cache: dict = {}


def _tuned_fwd_kernel(stream, cache_key, grid, fwd_args):
    if os.environ.get("DISABLE_AUTOTUNE") == "1":
        return _grpo_loss_fwd_ct.replace_hints(occupancy=ByTarget(sm_100=_FWD_FALLBACK_OCC, default=_FWD_FALLBACK_OCC))
    if cache_key not in _fwd_autotune_cache:
        result = exhaustive_search(
            _FWD_OCC_CONFIGS,
            stream,
            lambda cfg: grid,
            _grpo_loss_fwd_ct,
            lambda cfg: fwd_args,
            lambda cfg: {"occupancy": ByTarget(sm_100=cfg.occupancy, default=cfg.occupancy)},
            quiet=True,
        )
        best_occ = result.best.config.occupancy
        _fwd_autotune_cache[cache_key] = _grpo_loss_fwd_ct.replace_hints(
            occupancy=ByTarget(sm_100=best_occ, default=best_occ)
        )
    return _fwd_autotune_cache[cache_key]


def _grpo_loss_forward_ct(
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
    loss_type_int,
    sapo_temperature_pos,
    sapo_temperature_neg,
    delta,
    use_bias_correction_kl,
    vllm_is_ratio,
    vllm_is_ratio_stride,
    phi_seq,
):
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    BLOCK_N = min(8192, _next_power_of_2(N))

    logits_2d = logits.reshape(B * L_ADD_1, N).contiguous()

    loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    lse = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    is_clipped = torch.zeros(B, L, device=logits.device, dtype=torch.float32)

    has_beta = float(beta) != 0.0
    kl = torch.zeros(B, L, device=logits.device, dtype=torch.float32) if has_beta else None

    dummy_f = torch.zeros(1, device=logits.device, dtype=torch.float32)
    dummy_i = torch.zeros(1, device=logits.device, dtype=torch.int32)

    old_logp_arg = old_logp.contiguous() if old_logp is not None else dummy_f
    ref_logp_arg = ref_logp.contiguous() if ref_logp is not None else dummy_f
    mask_arg = completion_mask.to(torch.int32).contiguous() if completion_mask is not None else dummy_i
    kl_arg = kl if kl is not None else dummy_f
    # Flatten to 1-D: the kernel gathers with a flat row*stride+col index, so the
    # array must be rank-1 (strict tileiras rejects a rank-2 array with a 1-tuple index).
    vllm_arg = vllm_is_ratio.contiguous().view(-1) if vllm_is_ratio is not None else dummy_f
    phi_seq_arg = phi_seq.contiguous() if phi_seq is not None else dummy_f

    has_mask = int(completion_mask is not None)
    has_old_logp = int(old_logp is not None)
    # VESPO (loss_type 3) folds the vLLM correction into phi_seq, so the kernel skips vllm.
    has_vllm = int(vllm_is_ratio is not None and int(loss_type_int) != 3)

    grid = (B, L, 1)
    stream = torch.cuda.current_stream()
    fwd_args = (
        logits_2d,
        old_logp_arg,
        ref_logp_arg,
        completion_ids.contiguous(),
        mask_arg,
        advantages.contiguous(),
        vllm_arg,
        phi_seq_arg,
        loss,
        lse,
        kl_arg,
        is_clipped,
        int(B),
        int(L),
        int(N),
        int(BLOCK_N),
        float(temperature),
        float(beta),
        float(eps_low),
        float(eps_high),
        int(loss_type_int),
        float(sapo_temperature_pos),
        float(sapo_temperature_neg),
        float(delta),
        int(use_bias_correction_kl),
        int(has_mask),
        int(has_old_logp),
        int(has_vllm),
        int(vllm_is_ratio_stride),
    )
    cache_key = (
        int(B),
        int(L),
        int(N),
        int(BLOCK_N),
        int(loss_type_int),
        int(has_mask),
        int(has_old_logp),
        int(has_vllm),
        str(logits_2d.device),
    )
    kernel = _tuned_fwd_kernel(stream, cache_key, grid, fwd_args)
    ct.launch(stream, grid, kernel, fwd_args)

    return loss, lse, is_clipped, kl


def _grpo_loss_backward_ct(
    dloss,
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask,
    lse,
    temperature,
    beta,
    eps_low,
    eps_high,
    inplace,
    loss_type_int,
    sapo_temperature_pos,
    sapo_temperature_neg,
    delta,
    use_bias_correction_kl,
    vllm_is_ratio,
    vllm_is_ratio_stride,
    phi_seq,
):
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    BLOCK_N = min(4096, _next_power_of_2(N))

    logits_2d = logits.reshape(B * L_ADD_1, N).contiguous()
    dlogits_2d = logits.data.reshape(B * L_ADD_1, N) if inplace else torch.empty_like(logits_2d)

    dummy_f = torch.zeros(1, device=logits.device, dtype=torch.float32)
    dummy_i = torch.zeros(1, device=logits.device, dtype=torch.int32)

    old_logp_arg = old_logp.contiguous() if old_logp is not None else dummy_f
    ref_logp_arg = ref_logp.contiguous() if ref_logp is not None else dummy_f
    mask_arg = completion_mask.to(torch.int32).contiguous() if completion_mask is not None else dummy_i
    # Flatten to 1-D: the kernel gathers with a flat row*stride+col index, so the
    # array must be rank-1 (strict tileiras rejects a rank-2 array with a 1-tuple index).
    vllm_arg = vllm_is_ratio.contiguous().view(-1) if vllm_is_ratio is not None else dummy_f
    phi_seq_arg = phi_seq.contiguous() if phi_seq is not None else dummy_f

    has_mask = int(completion_mask is not None)
    has_old_logp = int(old_logp is not None)
    has_vllm = int(vllm_is_ratio is not None and int(loss_type_int) != 3)

    grid = (B, L, 1)
    bwd_kernel = _bwd_occ_small if B <= 2 else _bwd_occ_large
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        bwd_kernel,
        (
            dloss.contiguous(),
            dlogits_2d,
            logits_2d,
            old_logp_arg,
            ref_logp_arg,
            completion_ids.contiguous(),
            advantages.contiguous(),
            mask_arg,
            lse,
            vllm_arg,
            phi_seq_arg,
            int(B),
            int(L),
            int(N),
            int(BLOCK_N),
            float(temperature),
            float(beta),
            float(eps_low),
            float(eps_high),
            int(loss_type_int),
            float(sapo_temperature_pos),
            float(sapo_temperature_neg),
            float(delta),
            int(use_bias_correction_kl),
            int(has_mask),
            int(has_old_logp),
            int(has_vllm),
            int(vllm_is_ratio_stride),
        ),
    )

    dlogits = dlogits_2d.reshape(B, L_ADD_1, N)
    dlogits[:, -1, :] = 0
    return dlogits


# ---------------------------------------------------------------------------
# Sequence-level importance sampling (GSPO): per-sequence coef_1 precomputed on host.
# ---------------------------------------------------------------------------


@ct.kernel
def _grpo_loss_fwd_seq_ct(
    logits_input,  # (B*(L+1), N)
    ref_logp_input,  # (B, L) or dummy (1,)
    input_ids,  # (B, L) int64
    completion_mask,  # (B, L) int32 or dummy (1,)
    advantages,  # (B,) float32
    coef_1,  # (B,) per-sequence importance weight, post delta-clamp
    coef_1_raw,  # (B,) per-sequence importance weight, pre delta-clamp (for bias-corrected KL)
    coef_2,  # (B,) clipped coef
    is_clipped_seq,  # (B,) clipping indicator
    vllm_is_ratio,  # (B, L)/(B, 1) or dummy (1,)
    loss_output,  # (B, L)
    lse_cache,  # (B, L)
    kl_output,  # (B, L) or dummy (1,)
    is_clipped_output,  # (B, L)
    B: ct.Constant[int],
    L: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    temperature,
    beta: ct.Constant[float],
    use_bias_correction_kl: ct.Constant[int],
    HAS_COMPLETION_MASK: ct.Constant[int],
    HAS_VLLM_IS_RATIO: ct.Constant[int],
    vllm_is_ratio_stride,
):
    """Sequence-level GRPO forward. Grid: (B, L). Uses precomputed per-sequence coefficients."""
    off_b = ct.bid(0)
    off_l = ct.bid(1)

    if HAS_COMPLETION_MASK:
        mask_val = ct.astype(ct.load(completion_mask, (off_b, off_l), shape=()), ct.int32)
        if mask_val == 0:
            return

    logits_row = ct.add(ct.mul(off_b, L + 1), off_l)
    n_chunks = (N + BLOCK_N - 1) // BLOCK_N
    inv_temperature = 1.0 / temperature

    m_i = ct.full((), -math.inf, dtype=ct.float32)
    l_i = ct.full((), 0.0, dtype=ct.float32)
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
        logits = ct.astype(
            ct.gather(logits_input, (logits_row, col_idx), check_bounds=True, padding_value=-math.inf, latency=3),
            ct.float32,
        )
        logits_scaled = logits * inv_temperature
        chunk_max = ct.max(logits_scaled, 0, keepdims=False)
        new_m = ct.maximum(m_i, chunk_max)
        alpha = ct.exp2(ct.mul(m_i - new_m, LOG2E))
        l_i = ct.add(ct.mul(l_i, alpha), ct.sum(ct.exp2(ct.mul(logits_scaled - new_m, LOG2E)), 0, keepdims=False))
        m_i = new_m
    lse = m_i + ct.log(l_i)

    idx = ct.astype(ct.load(input_ids, (off_b, off_l), shape=()), ct.int32)
    idx_tile = ct.add(ct.arange(1, dtype=ct.int32), idx)
    x = (
        ct.sum(
            ct.astype(ct.gather(logits_input, (logits_row, idx_tile), check_bounds=False), ct.float32),
            0,
            keepdims=False,
        )
        * inv_temperature
    )
    logp = x - lse

    coef_1_v = ct.astype(ct.load(coef_1, off_b, shape=()), ct.float32)
    coef_2_v = ct.astype(ct.load(coef_2, off_b, shape=()), ct.float32)
    is_clip_v = ct.astype(ct.load(is_clipped_seq, off_b, shape=()), ct.float32)
    advantage = ct.astype(ct.load(advantages, off_b, shape=()), ct.float32)

    per_token_loss = -ct.minimum(coef_1_v * advantage, coef_2_v * advantage)

    if HAS_VLLM_IS_RATIO:
        vllm_col = off_l % vllm_is_ratio_stride
        vllm_tile = ct.gather(
            vllm_is_ratio,
            ct.add(ct.arange(1, dtype=ct.int32), ct.mul(off_b, vllm_is_ratio_stride) + vllm_col),
            check_bounds=False,
        )
        vllm_ratio = ct.astype(ct.sum(ct.astype(vllm_tile, ct.float32), 0, keepdims=False), ct.float32)
        per_token_loss = per_token_loss * vllm_ratio

    if beta != 0.0:
        ref_logp = ct.astype(ct.load(ref_logp_input, (off_b, off_l), shape=()), ct.float32)
        kl = ct.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        if use_bias_correction_kl:
            kl = kl * ct.astype(ct.load(coef_1_raw, off_b, shape=()), ct.float32)
        per_token_loss = per_token_loss + beta * kl
        ct.scatter(kl_output, (off_b, off_l), ct.astype(kl, kl_output.dtype))

    ct.scatter(loss_output, (off_b, off_l), ct.astype(per_token_loss, loss_output.dtype))
    ct.scatter(lse_cache, (off_b, off_l), ct.astype(lse, lse_cache.dtype))
    ct.scatter(is_clipped_output, (off_b, off_l), ct.astype(is_clip_v, is_clipped_output.dtype))


@ct.kernel
def _grpo_loss_bwd_seq_ct(
    dloss_input,  # (B, L) per-token upstream grad (for KL term)
    dloss_sum_input,  # (B,) per-sequence sum of dloss (for policy grad)
    dlogits_output,  # (B*(L+1), N)
    logits_input,  # (B*(L+1), N)
    ref_logp_input,  # (B, L) or dummy (1,)
    input_ids,  # (B, L) int64
    advantages,  # (B,) float32
    completion_mask,  # (B, L) int32 or dummy (1,)
    lse_cache,  # (B, L)
    coef_1,  # (B,) per-sequence importance weight (pre delta-clamp)
    seq_len,  # (B,) valid tokens per sequence
    B: ct.Constant[int],
    L: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    temperature,
    beta: ct.Constant[float],
    use_bias_correction_kl: ct.Constant[int],
    eps_low,
    eps_high,
    delta,
    HAS_COMPLETION_MASK: ct.Constant[int],
):
    """Sequence-level GRPO backward. Grid: (B, L)."""
    off_b = ct.bid(0)
    off_l = ct.bid(1)
    logits_row = ct.add(ct.mul(off_b, L + 1), off_l)
    n_chunks = (N + BLOCK_N - 1) // BLOCK_N
    inv_temperature = 1.0 / temperature

    if HAS_COMPLETION_MASK:
        mask_val = ct.astype(ct.load(completion_mask, (off_b, off_l), shape=()), ct.int32)
        if mask_val == 0:
            for ci in range(n_chunks):
                col_idx = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
                ct.scatter(
                    dlogits_output,
                    (logits_row, col_idx),
                    ct.full((BLOCK_N,), 0.0, dtype=dlogits_output.dtype),
                    check_bounds=True,
                )
            return

    dloss = ct.astype(ct.load(dloss_input, (off_b, off_l), shape=()), ct.float32)
    dloss_sum = ct.astype(ct.load(dloss_sum_input, off_b, shape=()), ct.float32)
    lse = ct.astype(ct.load(lse_cache, (off_b, off_l), shape=()), ct.float32)
    coef_1_v = ct.astype(ct.load(coef_1, off_b, shape=()), ct.float32)
    seq_len_v = ct.astype(ct.load(seq_len, off_b, shape=()), ct.float32)

    idx = ct.astype(ct.load(input_ids, (off_b, off_l), shape=()), ct.int32)
    idx_tile = ct.add(ct.arange(1, dtype=ct.int32), idx)
    x = (
        ct.sum(
            ct.astype(ct.gather(logits_input, (logits_row, idx_tile), check_bounds=False), ct.float32),
            0,
            keepdims=False,
        )
        * inv_temperature
    )
    logp = x - lse
    advantage = ct.astype(ct.load(advantages, off_b, shape=()), ct.float32)

    coef_2 = ct.minimum(ct.maximum(coef_1_v, 1.0 - eps_low), 1.0 + eps_high)
    if delta != 0.0:
        coef_1_for_loss = ct.minimum(coef_1_v, ct.full((), delta, dtype=ct.float32))
    else:
        coef_1_for_loss = coef_1_v
    is_unclipped = ct.astype((coef_2 * advantage) >= (coef_1_for_loss * advantage), ct.float32)

    dlogp = -coef_1_v * advantage / seq_len_v * is_unclipped * dloss_sum
    if delta != 0.0:
        dlogp = dlogp * ct.astype(coef_1_v <= ct.full((), delta, dtype=ct.float32), ct.float32)

    if beta != 0.0:
        ref_logp = ct.astype(ct.load(ref_logp_input, (off_b, off_l), shape=()), ct.float32)
        if use_bias_correction_kl:
            dlogp = dlogp + beta * coef_1_v * (logp - ref_logp) * dloss
        else:
            dlogp = dlogp + beta * (1.0 - ct.exp(ref_logp - logp)) * dloss

    dlogp_scaled = dlogp * inv_temperature
    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_N, dtype=ct.int32), ci * BLOCK_N)
        logits = ct.astype(
            ct.gather(logits_input, (logits_row, col_idx), check_bounds=True, padding_value=-math.inf, latency=10),
            ct.float32,
        )
        probs = ct.exp(logits * inv_temperature - lse)
        is_target = ct.astype(col_idx == idx, ct.float32)
        dlogits = (is_target - probs) * dlogp_scaled
        ct.scatter(
            dlogits_output,
            (logits_row, col_idx),
            ct.astype(dlogits, dlogits_output.dtype),
            check_bounds=True,
            latency=10,
        )


_bwd_seq_occ_small = _grpo_loss_bwd_seq_ct.replace_hints(occupancy=ByTarget(sm_100=4, default=4))
_bwd_seq_occ_large = _grpo_loss_bwd_seq_ct.replace_hints(occupancy=ByTarget(sm_100=12, default=12))


def _compute_dapo_normalizer(completion_mask, num_items_in_batch=None):
    """Per-process normalizer for DAPO/CISPO/VESPO (matches the Triton path)."""
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()

    if num_items_in_batch is not None:
        if isinstance(num_items_in_batch, torch.Tensor):
            normalizer = num_items_in_batch.to(device=completion_mask.device, dtype=torch.float32)
        else:
            normalizer = torch.as_tensor(float(num_items_in_batch), device=completion_mask.device, dtype=torch.float32)
        normalizer = normalizer / world_size
        return torch.clamp(normalizer, min=1.0)

    normalizer = completion_mask.to(torch.float32).sum()
    if world_size > 1:
        normalizer = normalizer.clone()
        torch.distributed.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
        normalizer = normalizer / world_size
    return torch.clamp(normalizer, min=1.0)


def _reduce_loss(per_token_loss, mask, loss_type, max_completion_length, B, L, num_items_in_batch=None):
    """Apply loss reduction based on loss_type (matches the Triton path)."""
    if loss_type == "grpo" or loss_type == "sapo":
        return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        max_len = max_completion_length if max_completion_length is not None else L
        return (per_token_loss * mask).sum() / (B * max_len)
    elif loss_type == "dapo" or loss_type == "cispo" or loss_type == "vespo":
        return (per_token_loss * mask).sum() / _compute_dapo_normalizer(mask, num_items_in_batch=num_items_in_batch)
    elif loss_type == "luspo":
        return (per_token_loss * mask.sum(-1, keepdim=True)).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo")


def _grpo_loss_forward_seq_ct(
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
    delta_val,
    use_bias_correction_kl,
    vllm_is_ratio,
    vllm_is_ratio_stride,
):
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    BLOCK_N = min(4096, _next_power_of_2(N))
    device = logits.device
    mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=device)

    # Per-token log-probs (host) → per-sequence importance weights (GSPO).
    lg = logits[:, :L, :].float() / temperature
    lse_pt = torch.logsumexp(lg, dim=-1)
    tgt = torch.gather(lg, -1, completion_ids.long().unsqueeze(-1)).squeeze(-1)
    per_token_logps = tgt - lse_pt
    log_ratio = torch.zeros_like(per_token_logps) if old_logp is None else (per_token_logps - old_logp)
    seq_lens = mask.sum(-1).clamp(min=1.0)  # (B,)
    coef_1 = torch.exp((log_ratio * mask).sum(-1) / seq_lens)  # (B,)
    coef_2 = torch.clamp(coef_1, 1.0 - eps_low, 1.0 + eps_high)
    is_clipped_seq = (
        ((coef_1 < 1.0 - eps_low) & (advantages < 0)) | ((coef_1 > 1.0 + eps_high) & (advantages > 0))
    ).float()
    coef_1_for_loss = torch.clamp(coef_1, max=delta_val) if delta_val != 0.0 else coef_1

    logits_2d = logits.reshape(B * L_ADD_1, N).contiguous()
    loss = torch.zeros(B, L, device=device, dtype=torch.float32)
    lse = torch.zeros(B, L, device=device, dtype=torch.float32)
    is_clipped = torch.zeros(B, L, device=device, dtype=torch.float32)
    has_beta = float(beta) != 0.0
    kl = torch.zeros(B, L, device=device, dtype=torch.float32) if has_beta else None

    dummy_f = torch.zeros(1, device=device, dtype=torch.float32)
    dummy_i = torch.zeros(1, device=device, dtype=torch.int32)
    ref_arg = ref_logp.contiguous() if ref_logp is not None else dummy_f
    mask_arg = completion_mask.to(torch.int32).contiguous() if completion_mask is not None else dummy_i
    kl_arg = kl if kl is not None else dummy_f
    # Flatten to 1-D: the kernel gathers with a flat row*stride+col index, so the
    # array must be rank-1 (strict tileiras rejects a rank-2 array with a 1-tuple index).
    vllm_arg = vllm_is_ratio.contiguous().view(-1) if vllm_is_ratio is not None else dummy_f
    has_mask = int(completion_mask is not None)
    has_vllm = int(vllm_is_ratio is not None)

    ct.launch(
        torch.cuda.current_stream(),
        (B, L, 1),
        _grpo_loss_fwd_seq_ct,
        (
            logits_2d,
            ref_arg,
            completion_ids.contiguous(),
            mask_arg,
            advantages.contiguous(),
            coef_1_for_loss.contiguous(),
            coef_1.contiguous(),
            coef_2.contiguous(),
            is_clipped_seq.contiguous(),
            vllm_arg,
            loss,
            lse,
            kl_arg,
            is_clipped,
            int(B),
            int(L),
            int(N),
            int(BLOCK_N),
            float(temperature),
            float(beta),
            int(use_bias_correction_kl),
            has_mask,
            has_vllm,
            int(vllm_is_ratio_stride),
        ),
    )
    return loss, lse, is_clipped, kl, coef_1, seq_lens


def _grpo_loss_backward_seq_ct(
    dloss,
    logits,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask,
    lse,
    coef_1,
    seq_lens,
    temperature,
    beta,
    eps_low,
    eps_high,
    delta_val,
    use_bias_correction_kl,
    inplace,
    vllm_is_ratio,
):
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    BLOCK_N = min(4096, _next_power_of_2(N))
    if vllm_is_ratio is None:
        dloss_sum = dloss.sum(-1).contiguous()
    else:
        ratio = vllm_is_ratio.unsqueeze(-1) if vllm_is_ratio.dim() == 1 else vllm_is_ratio
        dloss_sum = (dloss * ratio).sum(-1).contiguous()

    logits_2d = logits.reshape(B * L_ADD_1, N).contiguous()
    dlogits_2d = logits.data.reshape(B * L_ADD_1, N) if inplace else torch.empty_like(logits_2d)
    dummy_f = torch.zeros(1, device=logits.device, dtype=torch.float32)
    dummy_i = torch.zeros(1, device=logits.device, dtype=torch.int32)
    ref_arg = ref_logp.contiguous() if ref_logp is not None else dummy_f
    mask_arg = completion_mask.to(torch.int32).contiguous() if completion_mask is not None else dummy_i
    has_mask = int(completion_mask is not None)

    bwd_kernel = _bwd_seq_occ_small if B <= 2 else _bwd_seq_occ_large
    ct.launch(
        torch.cuda.current_stream(),
        (B, L, 1),
        bwd_kernel,
        (
            dloss.contiguous(),
            dloss_sum,
            dlogits_2d,
            logits_2d,
            ref_arg,
            completion_ids.contiguous(),
            advantages.contiguous(),
            mask_arg,
            lse,
            coef_1.contiguous(),
            seq_lens.contiguous(),
            int(B),
            int(L),
            int(N),
            int(BLOCK_N),
            float(temperature),
            float(beta),
            int(use_bias_correction_kl),
            float(eps_low),
            float(eps_high),
            float(delta_val),
            has_mask,
        ),
    )
    dlogits = dlogits_2d.reshape(B, L_ADD_1, N)
    dlogits[:, -1, :] = 0
    return dlogits


class GrpoLossFunction(torch.autograd.Function):
    """CuTile autograd wrapper for GRPO loss (token-level).

    Full feature parity with the Triton GrpoLossFunction (all loss types, token/sequence-level IS).
    """

    @staticmethod
    def forward(
        ctx,
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
        loss_type="grpo",
        max_completion_length=None,
        reduce=True,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
        num_items_in_batch=None,
        phi_seq=None,
    ):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        if loss_type not in _str_to_loss_type:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Supported: {list(_str_to_loss_type.keys())}")
        assert importance_sampling_level in ("token", "sequence"), (
            f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}"
        )
        if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo", "vespo"):
            raise ValueError(
                f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'. "
                f"Use importance_sampling_level='token' instead."
            )
        if delta is not None and loss_type in ("cispo", "sapo", "vespo"):
            raise ValueError(f"delta (two-sided clipping) is not supported for loss_type='{loss_type}'.")
        if loss_type == "sapo":
            if sapo_temperature_pos <= 0 or sapo_temperature_neg <= 0:
                raise ValueError("sapo_temperature_pos/neg must be positive.")

        loss_type_int = _str_to_loss_type[loss_type]
        delta_val = 0.0 if delta is None else float(delta)

        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        # VESPO requires a caller-precomputed per-sequence gamma weight phi_seq (B,).
        if loss_type == "vespo":
            if phi_seq is None:
                raise ValueError("loss_type='vespo' requires phi_seq precomputed by the caller (B,) or (B, 1).")
            assert phi_seq.shape in ((B,), (B, 1)), f"phi_seq must be (B,) or (B, 1), got {tuple(phi_seq.shape)}"
            phi_seq = phi_seq.reshape(-1).contiguous()
        else:
            phi_seq = None

        vllm_is_ratio_stride = L
        if vllm_is_ratio is not None:
            assert vllm_is_ratio.dim() in (1, 2)
            if vllm_is_ratio.dim() == 2:
                assert vllm_is_ratio.shape[0] == B and vllm_is_ratio.shape[1] in (1, L)
            else:
                assert vllm_is_ratio.shape[0] == B
            vllm_is_ratio = vllm_is_ratio.contiguous()
            vllm_is_ratio_stride = vllm_is_ratio.shape[1] if vllm_is_ratio.dim() > 1 else 1

        if importance_sampling_level == "sequence":
            loss, lse, is_clipped, kl, coef_1, seq_lens = _grpo_loss_forward_seq_ct(
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
                delta_val,
                int(use_bias_correction_kl),
                vllm_is_ratio,
                vllm_is_ratio_stride,
            )
            ctx.save_for_backward(logits, ref_logp, completion_ids, advantages, completion_mask, lse, coef_1, seq_lens)
        else:
            loss, lse, is_clipped, kl = _grpo_loss_forward_ct(
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
                loss_type_int,
                sapo_temperature_pos,
                sapo_temperature_neg,
                delta_val,
                int(use_bias_correction_kl),
                vllm_is_ratio,
                vllm_is_ratio_stride,
                phi_seq,
            )
            ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse)

        ctx.importance_sampling_level = importance_sampling_level
        ctx.vllm_is_ratio = vllm_is_ratio
        ctx.phi_seq = phi_seq
        ctx.infos = (
            temperature,
            beta,
            eps_low,
            eps_high,
            inplace,
            loss_type,
            loss_type_int,
            sapo_temperature_pos,
            sapo_temperature_neg,
            max_completion_length,
            reduce,
            delta_val,
            use_bias_correction_kl,
            vllm_is_ratio_stride,
            num_items_in_batch,
        )

        mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=logits.device)
        mask_sum = mask.sum().clamp(min=1.0)
        kl_mean = (kl * mask).sum() / mask_sum if kl is not None else None
        clip_ratio = (is_clipped.float() * mask).sum() / mask_sum

        if not reduce:
            loss_out = loss * mask
            kl_out = kl * mask if kl is not None else None
            is_clipped_out = is_clipped * mask
            return loss_out, kl_out, is_clipped_out

        reduced_loss = _reduce_loss(
            loss, mask, loss_type, max_completion_length, B, L, num_items_in_batch=num_items_in_batch
        )
        return reduced_loss, kl_mean, clip_ratio

    @staticmethod
    def backward(ctx, *args):
        dloss_input = args[0]
        level = ctx.importance_sampling_level
        if level == "sequence":
            logits, ref_logp, completion_ids, advantages, completion_mask, lse, coef_1, seq_lens = ctx.saved_tensors
            old_logp = None
        else:
            logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse = ctx.saved_tensors
        (
            temperature,
            beta,
            eps_low,
            eps_high,
            inplace,
            loss_type,
            loss_type_int,
            sapo_temp_pos,
            sapo_temp_neg,
            max_completion_length,
            reduce,
            delta_val,
            use_bias_correction_kl,
            vllm_is_ratio_stride,
            num_items_in_batch,
        ) = ctx.infos
        vllm_is_ratio = ctx.vllm_is_ratio
        phi_seq = ctx.phi_seq

        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=logits.device)

        if not reduce:
            dloss = dloss_input
        elif loss_type == "grpo" or loss_type == "sapo":
            seq_lens_bwd = mask.sum(-1, keepdim=True).clamp(min=1.0)
            dloss = dloss_input * mask / (seq_lens_bwd * B)
        elif loss_type == "bnpo":
            dloss = dloss_input * mask / mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            max_len = max_completion_length if max_completion_length is not None else L
            dloss = dloss_input * mask / (B * max_len)
        elif loss_type == "dapo" or loss_type == "cispo" or loss_type == "vespo":
            dloss = dloss_input * mask / _compute_dapo_normalizer(mask, num_items_in_batch=num_items_in_batch)
        elif loss_type == "luspo":
            seq_lens_bwd = mask.sum(-1, keepdim=True).clamp(min=1.0)
            # d(loss)/d(per_token_loss[b,l]) = seq_len[b] / (B*L), constant within a sequence.
            # Broadcast the (B, 1) scale to (B, L). NOTE: this intentionally diverges from the
            # Triton path, which passes the (B, 1) tensor with stride (1, 1) and reads
            # dloss[off_b + off_l] — out of bounds (wrong) for B < L. The cuTile result is the
            # mathematically correct per-sequence gradient.
            dloss = (dloss_input * seq_lens_bwd / (B * L)).expand(B, L)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        if level == "sequence":
            dlogits = _grpo_loss_backward_seq_ct(
                dloss,
                logits,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                coef_1,
                seq_lens,
                temperature,
                beta,
                eps_low,
                eps_high,
                delta_val,
                use_bias_correction_kl,
                inplace,
                vllm_is_ratio,
            )
        else:
            dlogits = _grpo_loss_backward_ct(
                dloss,
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                temperature,
                beta,
                eps_low,
                eps_high,
                inplace,
                loss_type_int,
                sapo_temp_pos,
                sapo_temp_neg,
                delta_val,
                use_bias_correction_kl,
                vllm_is_ratio,
                vllm_is_ratio_stride,
                phi_seq,
            )
        # 22 forward inputs -> dlogits + 21 None
        return (dlogits, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None)  # fmt: skip
