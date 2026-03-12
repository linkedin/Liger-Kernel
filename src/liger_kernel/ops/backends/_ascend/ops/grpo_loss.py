import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import get_npu_core_count

# Loss type constants for Triton constexpr branching
# GRPO/DAPO/BNPO/DR_GRPO all use the same per-token loss computation (standard PPO clipping)
_LOSS_TYPE_GRPO: tl.constexpr = tl.constexpr(0)
_LOSS_TYPE_CISPO: tl.constexpr = tl.constexpr(1)
_LOSS_TYPE_SAPO: tl.constexpr = tl.constexpr(2)

_str_to_loss_type = {
    "grpo": _LOSS_TYPE_GRPO.value,
    "dapo": _LOSS_TYPE_GRPO.value,
    "bnpo": _LOSS_TYPE_GRPO.value,
    "dr_grpo": _LOSS_TYPE_GRPO.value,
    "luspo": _LOSS_TYPE_GRPO.value,
    "cispo": _LOSS_TYPE_CISPO.value,
    "sapo": _LOSS_TYPE_SAPO.value,
}


# -----------------------------------------------------------------------------
# Helper: Calculate optimal Block Size using compute_default_tiling_strategy
# -----------------------------------------------------------------------------


def get_optimal_grid_size_2d(B, L, num_cores):
    """
    Calculate the optimal two-dimensional grid configuration

    """
    grid_b = B

    cores_per_batch = min(L, num_cores // B)
    cores_per_batch = max(1, cores_per_batch)

    grid_l = cores_per_batch

    total = grid_b * grid_l
    if total > num_cores:
        grid_l = max(1, num_cores // grid_b)

    return (grid_b, grid_l)


def get_optimal_block_size_selective_log_softmax(N):
    """
    Calculate optimal Block Size for selective_log_softmax kernel.
    Processes L tokens per batch element, needs to iterate over N vocab dimension.
    """
    # Memory multiplier for selective log softmax (lighter operation)
    multiplier = 6.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((N,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_n = tile_shapes[0][0]
        return block_n
    else:
        return 2048


def get_optimal_block_size_grpo_loss_fwd(N):
    """
    Calculate optimal Block Size for GRPO loss forward kernel.
    """
    # Forward requires storing loss, lse, kl, is_clipped - heavier memory usage
    multiplier = 10.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((N,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_n = tile_shapes[0][0]
        return block_n
    else:
        return 2048


def get_optimal_block_size_grpo_loss_bwd(N):
    """
    Calculate optimal Block Size for GRPO loss backward kernel.
    """
    # Backward requires storing gradients and intermediate values
    multiplier = 12.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((N,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_n = tile_shapes[0][0]
        return block_n
    else:
        return 2048


# -----------------------------------------------------------------------------
# Kernels (NPU-friendly Grid-Stride Loop Implementation)
# -----------------------------------------------------------------------------


@triton.jit
def _selective_log_softmax_kernel(
    LOGITS,
    INPUT_IDS,
    LOG_P,
    MASK,
    TEMPERATURE,
    stride_input_ids_b,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    num_progs_l = tl.num_programs(1)

    batch_start = pid_b * L
    batch_end = batch_start + L
    start_token = batch_start + pid_l
    stride = num_progs_l

    for token_idx in tl.range(start_token, batch_end, stride):
        off_b = token_idx // L
        off_l = token_idx % L

        # Check mask first
        should_process = 1
        if MASK is not None:
            MASK_local = MASK + off_b * stride_input_ids_b + off_l
            not_skip = tl.load(MASK_local)
            should_process = not_skip

        if should_process != 0:
            LOGITS_local = LOGITS + off_b * (L + 1) * N + off_l * N
            INPUT_IDS_local = INPUT_IDS + off_b * stride_input_ids_b + off_l
            LOG_P_local = LOG_P + token_idx

            m_i = float("-inf")
            l_i = 0.0
            for start in range(0, N, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
                new_m_i = tl.maximum(m_i, tl.max(logits))
                alpha = tl.exp(m_i - new_m_i)
                l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
                m_i = new_m_i
            lse = m_i + tl.log(l_i)

            ids = tl.load(INPUT_IDS_local)
            x = tl.load(LOGITS_local + ids).to(tl.float32) / TEMPERATURE
            logp = x - lse
            tl.store(LOG_P_local, logp)


@triton.jit
def _grpo_loss_fwd_kernel(
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    COMPLETION_MASK,
    ADVANTAGES,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
    LOSS,
    LSE,
    KL,
    IS_CLIPPED,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
    LOSS_TYPE: tl.constexpr,
    SAPO_TEMP_POS,
    SAPO_TEMP_NEG,
    DELTA,
    USE_BIAS_CORRECTION_KL: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    num_progs_l = tl.num_programs(1)

    batch_start = pid_b * L
    batch_end = batch_start + L
    start_token = batch_start + pid_l
    stride = num_progs_l

    for token_idx in tl.range(start_token, batch_end, stride):
        off_b = token_idx // L
        off_l = token_idx % L

        # Check completion mask first
        should_process = 1
        if COMPLETION_MASK is not None:
            COMPLETION_MASK_local = COMPLETION_MASK + off_b * L + off_l
            not_skip = tl.load(COMPLETION_MASK_local)
            should_process = not_skip

        if should_process != 0:
            LOGITS_local = LOGITS + off_b * (L + 1) * N + off_l * N
            INPUT_IDS_local = INPUT_IDS + off_b * L + off_l
            ADVANTAGES_local = ADVANTAGES + off_b
            LOSS_local = LOSS + token_idx
            LSE_local = LSE + token_idx
            IS_CLIPPED_local = IS_CLIPPED + token_idx

            m_i = float("-inf")
            l_i = 0.0
            for start in range(0, N, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
                new_m_i = tl.maximum(m_i, tl.max(logits))
                alpha = tl.exp(m_i - new_m_i)
                l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
                m_i = new_m_i
            lse = m_i + tl.log(l_i)

            idx = tl.load(INPUT_IDS_local)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse
            if OLD_LOGP is None:
                old_logp = logp
            else:
                OLD_LOGP_local = OLD_LOGP + token_idx
                old_logp = tl.load(OLD_LOGP_local).to(tl.float32)
            coef_1 = tl.exp(logp - old_logp)
            advantage = tl.load(ADVANTAGES_local).to(tl.float32)

            # Branch based on loss type
            if LOSS_TYPE == 0:  # GRPO/DAPO/BNPO/DR_GRPO: standard PPO clipping
                coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
                is_low_clipped = (coef_1 < 1 - EPS_LOW) & (advantage < 0)
                is_high_clipped = (coef_1 > 1 + EPS_HIGH) & (advantage > 0)
                is_clipped = is_low_clipped | is_high_clipped
                # Apply delta (two-sided clipping from INTELLECT-2) to coef_1
                if DELTA != 0.0:
                    coef_1 = tl.minimum(coef_1, DELTA)
                per_token_loss1 = coef_1 * advantage
                per_token_loss2 = coef_2 * advantage
                per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)

            elif LOSS_TYPE == 1:  # CISPO: upper-bound only clipping, detached, multiply by logp
                # Reference: MiniMax-M1 technical report
                # https://github.com/huggingface/trl/blob/035c3ff151b953ca72cdfe0ee966bc1469a26fde/trl/trainer/grpo_trainer.py#L2030
                coef_2 = tl.minimum(coef_1, EPS_HIGH)  # upper-bound only (EPS_HIGH is the raw bound for CISPO)
                per_token_loss = -coef_2 * advantage * logp  # includes logp term
                is_clipped = (coef_1 > EPS_HIGH) & (advantage > 0)

            elif LOSS_TYPE == 2:  # SAPO: soft adaptive policy optimization with sigmoid gating
                # Reference: https://huggingface.co/papers/2511.20347
                # Formula: sigmoid(τ * (ρ - 1)) * 4 / τ
                temperature = tl.where(advantage > 0, SAPO_TEMP_POS, SAPO_TEMP_NEG)
                sigmoid_input = temperature * (coef_1 - 1.0)
                sapo_coef = tl.sigmoid(sigmoid_input) * 4.0 / temperature
                per_token_loss = -sapo_coef * advantage
                is_clipped = 0.0  # SAPO has no clipping concept

            # Apply vLLM importance sampling correction BEFORE adding KL penalty
            if VLLM_IS_RATIO is not None:
                # Use modulo to support both (B, L) per-token and (B, 1) per-sequence shapes
                vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
                    tl.float32
                )
                per_token_loss = per_token_loss * vllm_is_ratio

            if BETA != 0.0:
                REF_LOGP_local = REF_LOGP + token_idx
                KL_local = KL + token_idx
                ref_logp = tl.load(REF_LOGP_local).to(tl.float32)
                kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
                if USE_BIAS_CORRECTION_KL:
                    # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= coef_1
                    kl = kl * tl.exp(logp - old_logp)
                per_token_loss += BETA * kl
                tl.store(KL_local, kl)

            tl.store(LOSS_local, per_token_loss)
            tl.store(LSE_local, lse)
            tl.store(IS_CLIPPED_local, is_clipped)


# Sequence-level forward kernel: uses pre-computed coef_1 per sequence
@triton.jit
def _grpo_loss_fwd_kernel_seq(
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    COMPLETION_MASK,
    ADVANTAGES,
    COEF_1,  # Pre-computed sequence-level importance weight (B,)
    COEF_2,  # Pre-computed clipped coef (B,)
    IS_CLIPPED_SEQ,  # Pre-computed clipping indicator (B,)
    VLLM_IS_RATIO,  # vLLM importance sampling ratio (B, L) or (B, 1) or None
    VLLM_IS_RATIO_STRIDE,  # stride for VLLM_IS_RATIO (L for per-token, 1 for per-sequence)
    LOSS,
    LSE,
    KL,
    IS_CLIPPED,
    TEMPERATURE,
    BETA: tl.constexpr,
    USE_BIAS_CORRECTION_KL: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    num_progs_l = tl.num_programs(1)

    batch_start = pid_b * L
    batch_end = batch_start + L
    start_token = batch_start + pid_l
    stride = num_progs_l

    for token_idx in tl.range(start_token, batch_end, stride):
        off_b = token_idx // L
        off_l = token_idx % L

        # Check completion mask first
        should_process = 1
        if COMPLETION_MASK is not None:
            COMPLETION_MASK_local = COMPLETION_MASK + off_b * L + off_l
            not_skip = tl.load(COMPLETION_MASK_local)
            should_process = not_skip

        if should_process != 0:
            LOGITS_local = LOGITS + off_b * (L + 1) * N + off_l * N
            INPUT_IDS_local = INPUT_IDS + off_b * L + off_l
            ADVANTAGES_local = ADVANTAGES + off_b
            COEF_1_local = COEF_1 + off_b
            COEF_2_local = COEF_2 + off_b
            IS_CLIPPED_SEQ_local = IS_CLIPPED_SEQ + off_b
            LOSS_local = LOSS + token_idx
            LSE_local = LSE + token_idx
            IS_CLIPPED_local = IS_CLIPPED + token_idx

            # Compute log softmax
            m_i = float("-inf")
            l_i = 0.0
            for start in range(0, N, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
                new_m_i = tl.maximum(m_i, tl.max(logits))
                alpha = tl.exp(m_i - new_m_i)
                l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
                m_i = new_m_i
            lse = m_i + tl.log(l_i)

            idx = tl.load(INPUT_IDS_local)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse

            # Load pre-computed sequence-level coefficients
            coef_1 = tl.load(COEF_1_local).to(tl.float32)
            coef_2 = tl.load(COEF_2_local).to(tl.float32)
            is_clipped_seq = tl.load(IS_CLIPPED_SEQ_local)

            advantage = tl.load(ADVANTAGES_local).to(tl.float32)
            per_token_loss1 = coef_1 * advantage
            per_token_loss2 = coef_2 * advantage
            per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)

            # Apply vLLM importance sampling correction BEFORE adding KL
            if VLLM_IS_RATIO is not None:
                vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
                    tl.float32
                )
                per_token_loss = per_token_loss * vllm_is_ratio

            if BETA != 0.0:
                REF_LOGP_local = REF_LOGP + token_idx
                KL_local = KL + token_idx
                ref_logp = tl.load(REF_LOGP_local).to(tl.float32)
                kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
                if USE_BIAS_CORRECTION_KL:
                    # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= token-level coef_1
                    if OLD_LOGP is None:
                        old_logp = logp
                    else:
                        old_logp = tl.load(OLD_LOGP + token_idx).to(tl.float32)
                    kl = kl * tl.exp(logp - old_logp)
                per_token_loss += BETA * kl
                tl.store(KL_local, kl)

            tl.store(LOSS_local, per_token_loss)
            tl.store(LSE_local, lse)
            tl.store(IS_CLIPPED_local, is_clipped_seq)  # Same for all tokens in sequence


# Sequence-level backward kernel
@triton.jit
def _grpo_loss_bwd_kernel_seq(
    DLOSS,
    DLOSS_SUM,
    DLOGITS,
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    ADVANTAGES,
    COMPLETION_MASK,
    LSE,
    COEF_1,  # Pre-computed sequence-level importance weight (B,)
    SEQ_LEN,  # Number of valid tokens per sequence (B,)
    TEMPERATURE,
    BETA: tl.constexpr,
    USE_BIAS_CORRECTION_KL: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
    DELTA,
    loss_stride0,
    loss_stride1,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    num_progs_l = tl.num_programs(1)

    batch_start = pid_b * L
    batch_end = batch_start + L
    start_token = batch_start + pid_l
    stride = num_progs_l

    for token_idx in tl.range(start_token, batch_end, stride):
        off_b = token_idx // L
        off_l = token_idx % L

        DLOGITS_local = DLOGITS + off_b * (L + 1) * N + off_l * N

        # Check completion mask first
        should_process = 1
        if COMPLETION_MASK is not None:
            COMPLETION_MASK_local = COMPLETION_MASK + off_b * L + off_l
            not_skip = tl.load(COMPLETION_MASK_local)
            should_process = not_skip

        if should_process == 0:
            # Zero out gradients for masked tokens
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS_local + cols, 0.0, mask=cols < N)
        else:
            LOGITS_local = LOGITS + off_b * (L + 1) * N + off_l * N
            DLOSS_local = DLOSS + off_b * loss_stride0 + off_l * loss_stride1
            DLOSS_SUM_local = DLOSS_SUM + off_b
            INPUT_IDS_local = INPUT_IDS + off_b * L + off_l
            ADVANTAGES_local = ADVANTAGES + off_b
            LSE_local = LSE + token_idx
            COEF_1_local = COEF_1 + off_b
            SEQ_LEN_local = SEQ_LEN + off_b

            dloss = tl.load(DLOSS_local).to(tl.float32)
            dloss_sum = tl.load(DLOSS_SUM_local).to(tl.float32)
            lse = tl.load(LSE_local).to(tl.float32)
            coef_1 = tl.load(COEF_1_local).to(tl.float32)
            seq_len = tl.load(SEQ_LEN_local).to(tl.float32)

            idx = tl.load(INPUT_IDS_local)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse

            advantage = tl.load(ADVANTAGES_local).to(tl.float32)
            coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
            if DELTA != 0.0:
                coef_1_for_loss = tl.minimum(coef_1, DELTA)
            else:
                coef_1_for_loss = coef_1
            per_token_loss1 = coef_1_for_loss * advantage
            per_token_loss2 = coef_2 * advantage
            is_unclipped = per_token_loss2 >= per_token_loss1

            # For sequence-level: gradient flows through mean, so scale by coef_1/seq_len
            # d(loss)/d(logp) = -advantage * coef_1 / seq_len (when unclipped and not delta-clamped)
            dlogp = -coef_1 * advantage / seq_len * is_unclipped * dloss_sum
            if DELTA != 0.0:
                dlogp = dlogp * (coef_1 <= DELTA)

            if BETA != 0.0:
                REF_LOGP_local = REF_LOGP + token_idx
                ref_logp = tl.load(REF_LOGP_local).to(tl.float32)
                if USE_BIAS_CORRECTION_KL:
                    # d(kl * coef_1)/d(logp) = coef_1 * (logp - ref_logp), where coef_1 = exp(logp - old_logp)
                    if OLD_LOGP is None:
                        old_logp = logp
                    else:
                        old_logp = tl.load(OLD_LOGP + token_idx).to(tl.float32)
                    token_coef_1 = tl.exp(logp - old_logp)
                    dlogp += BETA * token_coef_1 * (logp - ref_logp) * dloss
                else:
                    dlogp += BETA * (1 - tl.exp(ref_logp - logp)) * dloss

            dlogp = dlogp / TEMPERATURE
            tl.debug_barrier()
            for start_n in tl.range(0, N, BLOCK_N):
                cols = start_n + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
                probs = tl.exp(logits - lse)
                dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
                tl.store(DLOGITS_local + cols, dlogits, mask=cols < N)


@triton.jit
def _grpo_loss_bwd_kernel(
    DLOSS,
    DLOGITS,
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    ADVANTAGES,
    COMPLETION_MASK,
    LSE,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
    LOSS_TYPE: tl.constexpr,
    SAPO_TEMP_POS,
    SAPO_TEMP_NEG,
    DELTA,
    USE_BIAS_CORRECTION_KL: tl.constexpr,
    loss_stride0,
    loss_stride1,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    num_progs_l = tl.num_programs(1)

    batch_start = pid_b * L
    batch_end = batch_start + L
    start_token = batch_start + pid_l
    stride = num_progs_l

    for token_idx in tl.range(start_token, batch_end, stride):
        off_b = token_idx // L
        off_l = token_idx % L

        DLOGITS_local = DLOGITS + off_b * (L + 1) * N + off_l * N

        # Check completion mask first
        should_process = 1
        if COMPLETION_MASK is not None:
            COMPLETION_MASK_local = COMPLETION_MASK + off_b * L + off_l
            not_skip = tl.load(COMPLETION_MASK_local)
            should_process = not_skip

        if should_process == 0:
            # Zero out gradients for masked tokens
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS_local + cols, 0.0, mask=cols < N)
        else:
            LOGITS_local = LOGITS + off_b * (L + 1) * N + off_l * N
            DLOSS_local = DLOSS + off_b * loss_stride0 + off_l * loss_stride1
            INPUT_IDS_local = INPUT_IDS + off_b * L + off_l
            ADVANTAGES_local = ADVANTAGES + off_b
            LSE_local = LSE + token_idx

            dloss = tl.load(DLOSS_local).to(tl.float32)
            lse = tl.load(LSE_local).to(tl.float32)

            idx = tl.load(INPUT_IDS_local)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse
            if OLD_LOGP is None:
                old_logp = logp
            else:
                OLD_LOGP_local = OLD_LOGP + token_idx
                old_logp = tl.load(OLD_LOGP_local).to(tl.float32)
            coef_1 = tl.exp(logp - old_logp)
            advantage = tl.load(ADVANTAGES_local).to(tl.float32)

            # Branch based on loss type for gradient computation
            if LOSS_TYPE == 0:  # GRPO/DAPO/BNPO/DR_GRPO: standard PPO clipping
                coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
                if DELTA != 0.0:
                    coef_1_for_loss = tl.minimum(coef_1, DELTA)
                else:
                    coef_1_for_loss = coef_1
                per_token_loss1 = coef_1_for_loss * advantage
                per_token_loss2 = coef_2 * advantage
                mask = per_token_loss2 >= per_token_loss1
                # Gradient uses original coef_1; zero when delta-clamped (constant → no gradient)
                dlogp = -coef_1 * advantage * mask
                if DELTA != 0.0:
                    dlogp = dlogp * (coef_1 <= DELTA)

            elif LOSS_TYPE == 1:  # CISPO: coef_2 is DETACHED, so gradient only flows through logp
                # loss = -coef_2 * advantage * logp, where coef_2 = clamp(coef_1, max=eps_high).detach()
                # d(loss)/d(logp) = -coef_2 * advantage (coef_2 treated as constant due to detach)
                coef_2 = tl.minimum(coef_1, EPS_HIGH)
                dlogp = -coef_2 * advantage

            elif LOSS_TYPE == 2:  # SAPO: gradient through sigmoid gating
                # loss = -sapo_coef * advantage, where sapo_coef = sigmoid(τ*(ρ-1)) * 4/τ
                # d(loss)/d(logp) = -advantage * d(sapo_coef)/d(coef_1) * d(coef_1)/d(logp)
                # d(coef_1)/d(logp) = coef_1 (since coef_1 = exp(logp - old_logp))
                # d(sapo_coef)/d(coef_1) = d/d(coef_1)[sigmoid(τ*(coef_1-1)) * 4/τ]
                #                       = τ * sigmoid' * 4/τ = 4 * sigmoid * (1 - sigmoid)
                # (the τ factors cancel out in the derivative)
                temperature = tl.where(advantage > 0, SAPO_TEMP_POS, SAPO_TEMP_NEG)
                sigmoid_input = temperature * (coef_1 - 1.0)
                sigmoid_val = tl.sigmoid(sigmoid_input)
                d_sapo_d_coef1 = 4.0 * sigmoid_val * (1.0 - sigmoid_val)
                dlogp = -advantage * d_sapo_d_coef1 * coef_1

            # Apply vLLM IS ratio to PPO gradient (before KL gradient)
            if VLLM_IS_RATIO is not None:
                # Use modulo to support both (B, L) per-token and (B, 1) per-sequence shapes
                vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
                    tl.float32
                )
                dlogp = dlogp * vllm_is_ratio

            if BETA != 0.0:
                REF_LOGP_local = REF_LOGP + token_idx
                ref_logp = tl.load(REF_LOGP_local).to(tl.float32)
                if USE_BIAS_CORRECTION_KL:
                    # d(kl * coef_1)/d(logp) = coef_1 * (logp - ref_logp), where coef_1 = exp(logp - old_logp)
                    dlogp += BETA * coef_1 * (logp - ref_logp)
                else:
                    dlogp += BETA * (1 - tl.exp(ref_logp - logp))

            dlogp = dlogp * dloss / TEMPERATURE
            tl.debug_barrier()
            for start_n in tl.range(0, N, BLOCK_N):
                cols = start_n + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
                probs = tl.exp(logits - lse)
                dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
                tl.store(DLOGITS_local + cols, dlogits, mask=cols < N)


# -----------------------------------------------------------------------------
# High-level API functions
# -----------------------------------------------------------------------------


# compue old_logp and ref_logp, it reduce 10G peak Memory. it does not requires grad
@torch.no_grad
def fused_selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 0.9, mask=None):
    assert logits.is_contiguous()
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    input_ids = input_ids[:, -L:]
    if mask is not None:
        mask = mask[:, -L:]
    log_p = torch.zeros(B, L, dtype=torch.float32, device=logits.device)

    # Calculate optimal block size
    block_n = get_optimal_block_size_selective_log_softmax(N)

    # Use NPU core count for grid size
    num_cores = get_npu_core_count()
    grid = get_optimal_grid_size_2d(B, L, num_cores)
    _selective_log_softmax_kernel[grid](
        logits,
        input_ids,
        log_p,
        mask,
        temperature,
        input_ids.stride(0),
        L,
        N,
        BLOCK_N=block_n,
    )
    return log_p


def _compute_dapo_normalizer(completion_mask):
    """Global active tokens averaged per process (for distributed DAPO loss)."""
    normalizer = completion_mask.to(torch.float32).sum()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        normalizer = normalizer.clone()
        torch.distributed.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    normalizer = normalizer / world_size
    return torch.clamp(normalizer, min=1.0)


def _reduce_loss(per_token_loss, mask, loss_type, max_completion_length, B, L):
    """Apply loss reduction based on loss_type."""
    if loss_type == "grpo" or loss_type == "sapo":
        return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        max_len = max_completion_length if max_completion_length is not None else L
        return (per_token_loss * mask).sum() / (B * max_len)
    elif loss_type == "dapo" or loss_type == "cispo":
        return (per_token_loss * mask).sum() / _compute_dapo_normalizer(mask)
    elif loss_type == "luspo":
        return (per_token_loss * mask.sum(-1, keepdim=True)).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo")


class GrpoLossFunction(torch.autograd.Function):
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
    ):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
        assert importance_sampling_level in ("token", "sequence"), (
            f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}"
        )

        # Validate loss_type
        if loss_type not in _str_to_loss_type:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Supported types: {list(_str_to_loss_type.keys())}")

        # Validate delta + loss_type combinations
        if delta is not None and loss_type in ("cispo", "sapo"):
            raise ValueError(f"delta (two-sided clipping) is not supported for loss_type='{loss_type}'.")

        # Map delta to float for Triton (Triton can't handle None)
        delta_val = 0.0 if delta is None else float(delta)

        # Validate sequence-level + loss_type combinations
        if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo"):
            raise ValueError(
                f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'. "
                f"Use importance_sampling_level='token' instead."
            )

        # Validate SAPO temperatures to prevent division by zero or numerical instability
        if loss_type == "sapo":
            if sapo_temperature_pos <= 0:
                raise ValueError(f"sapo_temperature_pos must be positive, got {sapo_temperature_pos}")
            if sapo_temperature_neg <= 0:
                raise ValueError(f"sapo_temperature_neg must be positive, got {sapo_temperature_neg}")

        # Convert loss_type string to integer for Triton constexpr
        loss_type_int = _str_to_loss_type[loss_type]

        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        if completion_mask is not None:
            assert completion_mask.is_contiguous()

        mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=logits.device)

        # Handle vLLM IS ratio
        vllm_is_ratio_ptr = None
        vllm_is_ratio_stride = L  # default to per-token (unused when ptr is None)
        if vllm_is_ratio is not None:
            assert vllm_is_ratio.dim() in (1, 2), (
                f"vllm_is_ratio must be 1D (B,) or 2D (B, L) / (B, 1), got {vllm_is_ratio.dim()}D"
            )
            if vllm_is_ratio.dim() == 2:
                assert vllm_is_ratio.shape[0] == B and vllm_is_ratio.shape[1] in (1, L), (
                    f"vllm_is_ratio shape must be ({B}, 1) or ({B}, {L}), got {tuple(vllm_is_ratio.shape)}"
                )
            else:
                assert vllm_is_ratio.shape[0] == B, (
                    f"vllm_is_ratio shape must be ({B},), got {tuple(vllm_is_ratio.shape)}"
                )
            vllm_is_ratio = vllm_is_ratio.contiguous()
            vllm_is_ratio_ptr = vllm_is_ratio
            vllm_is_ratio_stride = vllm_is_ratio.shape[1] if vllm_is_ratio.dim() > 1 else 1

        # Allocate outputs
        loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
        lse = torch.zeros_like(loss)
        is_clipped = torch.zeros_like(loss)
        kl = torch.zeros_like(loss) if beta != 0.0 else None

        # Calculate optimal block size and grid size
        block_n = get_optimal_block_size_grpo_loss_fwd(N)
        num_cores = get_npu_core_count()
        grid = get_optimal_grid_size_2d(B, L, num_cores)

        if importance_sampling_level == "sequence":
            # Sequence-level: pre-compute sequence importance weights, then use Triton kernel
            # Step 1: Get per-token log probs using existing Triton kernel
            per_token_logps = fused_selective_log_softmax(logits, completion_ids, temperature, completion_mask)

            # Step 2: Compute sequence-level importance weights
            if old_logp is None:
                log_ratio = torch.zeros_like(per_token_logps)
            else:
                log_ratio = per_token_logps - old_logp

            seq_lens = mask.sum(-1).clamp(min=1.0)  # (B,)
            seq_log_importance = (log_ratio * mask).sum(-1) / seq_lens  # (B,)
            coef_1 = torch.exp(seq_log_importance)  # (B,)
            coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)  # (B,)

            # Compute is_clipped at sequence level (using original coef_1)
            is_clipped_seq = ((coef_1 < 1 - eps_low) & (advantages < 0)) | ((coef_1 > 1 + eps_high) & (advantages > 0))
            is_clipped_seq = is_clipped_seq.float()  # (B,)

            # Apply delta clamp for loss computation (keep original coef_1 for backward)
            if delta is not None:
                coef_1_for_loss = torch.clamp(coef_1, max=delta)
            else:
                coef_1_for_loss = coef_1

            # Step 3: Run Triton kernel with pre-computed coefficients
            _grpo_loss_fwd_kernel_seq[grid](
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                completion_mask,
                advantages,
                coef_1_for_loss.contiguous(),
                coef_2.contiguous(),
                is_clipped_seq.contiguous(),
                vllm_is_ratio_ptr,
                vllm_is_ratio_stride,
                loss,
                lse,
                kl,
                is_clipped,
                temperature,
                beta,
                use_bias_correction_kl,
                L,
                N,
                BLOCK_N=block_n,
            )

            # Save extra tensors for backward
            ctx.save_for_backward(
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                mask,
                coef_1,
                seq_lens,
                vllm_is_ratio_ptr,
            )
        else:
            # Token-level: use optimized Triton kernel with LOSS_TYPE branching
            _grpo_loss_fwd_kernel[grid](
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                completion_mask,
                advantages,
                vllm_is_ratio_ptr,
                vllm_is_ratio_stride,
                loss,
                lse,
                kl,
                is_clipped,
                temperature,
                beta,
                eps_low,
                eps_high,
                loss_type_int,
                sapo_temperature_pos,
                sapo_temperature_neg,
                delta_val,
                use_bias_correction_kl,
                L,
                N,
                BLOCK_N=block_n,
            )
            ctx.save_for_backward(
                logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse, mask, vllm_is_ratio_ptr
            )

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
            B,
            L,
            importance_sampling_level,
            vllm_is_ratio_stride,
            reduce,
            delta_val,
            use_bias_correction_kl,
        )

        # Compute metrics before reduction
        mask_sum = mask.sum().clamp(min=1.0)
        kl_mean = (kl * mask).sum() / mask_sum if kl is not None else None
        clip_ratio = (is_clipped.float() * mask).sum() / mask_sum

        if not reduce:
            loss_out = loss * mask
            kl_out = kl * mask if kl is not None else None
            is_clipped_out = is_clipped * mask
            return loss_out, kl_out, is_clipped_out

        reduced_loss = _reduce_loss(loss, mask, loss_type, max_completion_length, B, L)
        return reduced_loss, kl_mean, clip_ratio

    @staticmethod
    def backward(ctx, *args):
        dloss_input = args[0]
        saved_tensors = ctx.saved_tensors
        (
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
            B,
            L,
            importance_sampling_level,
            vllm_is_ratio_stride,
            reduce,
            delta_val,
            use_bias_correction_kl,
        ) = ctx.infos

        if importance_sampling_level == "sequence":
            (
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                mask,
                coef_1,
                seq_lens,
                vllm_is_ratio,
            ) = saved_tensors
        else:
            (logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse, mask, vllm_is_ratio) = (
                saved_tensors
            )

        _, L_ADD_1, N = logits.shape

        # Compute per-token gradient scaling based on loss_type
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
        elif loss_type == "dapo" or loss_type == "cispo":
            dloss = dloss_input * mask / _compute_dapo_normalizer(mask)
        elif loss_type == "luspo":
            # loss = mean(per_token_loss * seq_lens), mean divides by B*L
            seq_lens_bwd = mask.sum(-1, keepdim=True).clamp(min=1.0)
            dloss = dloss_input * seq_lens_bwd / (B * L)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        dlogits = logits.data if inplace else torch.empty_like(logits)

        # Calculate optimal block size and grid size for backward
        block_n = get_optimal_block_size_grpo_loss_bwd(N)
        num_cores = get_npu_core_count()
        grid = get_optimal_grid_size_2d(B, L, num_cores)

        if importance_sampling_level == "sequence":
            if vllm_is_ratio is None:
                dloss_sum = dloss.sum(-1).contiguous()
            else:
                if vllm_is_ratio.dim() == 1:
                    ratio = vllm_is_ratio.unsqueeze(-1)
                else:
                    ratio = vllm_is_ratio
                dloss_sum = (dloss * ratio).sum(-1).contiguous()
            # Sequence-level backward kernel
            _grpo_loss_bwd_kernel_seq[grid](
                dloss,
                dloss_sum,
                dlogits,
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                coef_1,
                seq_lens,
                temperature,
                beta,
                use_bias_correction_kl,
                eps_low,
                eps_high,
                delta_val,
                *dloss.stride(),
                L,
                N,
                BLOCK_N=block_n,
            )
        else:
            # Token-level backward kernel with LOSS_TYPE branching
            _grpo_loss_bwd_kernel[grid](
                dloss,
                dlogits,
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                vllm_is_ratio,
                vllm_is_ratio_stride,
                temperature,
                beta,
                eps_low,
                eps_high,
                loss_type_int,
                sapo_temperature_pos,
                sapo_temperature_neg,
                delta_val,
                use_bias_correction_kl,
                *dloss.stride(),
                L,
                N,
                BLOCK_N=block_n,
            )

        dlogits[:, -1, :] = 0
        # Return gradients for all forward inputs: dlogits + 19 None for non-differentiable params
        return (
            dlogits,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
