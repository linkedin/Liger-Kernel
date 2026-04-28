import torch
import triton
import triton.language as tl

# Loss type constants for Triton constexpr branching
# GRPO/DAPO/BNPO/DR_GRPO all use the same per-token loss computation (standard PPO clipping)
_LOSS_TYPE_GRPO: tl.constexpr = tl.constexpr(0)
_LOSS_TYPE_CISPO: tl.constexpr = tl.constexpr(1)
_LOSS_TYPE_SAPO: tl.constexpr = tl.constexpr(2)
_LOSS_TYPE_VESPO: tl.constexpr = tl.constexpr(3)

_str_to_loss_type = {
    "grpo": _LOSS_TYPE_GRPO.value,
    "dapo": _LOSS_TYPE_GRPO.value,
    "bnpo": _LOSS_TYPE_GRPO.value,
    "dr_grpo": _LOSS_TYPE_GRPO.value,
    "luspo": _LOSS_TYPE_GRPO.value,
    "cispo": _LOSS_TYPE_CISPO.value,
    "sapo": _LOSS_TYPE_SAPO.value,
    "vespo": _LOSS_TYPE_VESPO.value,
}


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
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * stride_input_ids_b + off_l
    LOG_P += off_b * L + off_l

    if MASK is not None:
        MASK += off_b * stride_input_ids_b + off_l
        not_skip = tl.load(MASK)
        if not_skip == 0:
            return

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    ids = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + ids).to(tl.float32) / TEMPERATURE
    logp = x - lse
    tl.store(LOG_P, logp)


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
    kwargs = {"BLOCK_N": 2048, "num_stages": 4, "num_warps": 1}
    _selective_log_softmax_kernel[(B, L)](
        logits, input_ids, log_p, mask, temperature, input_ids.stride(0), L, N, **kwargs
    )
    return log_p


# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])
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
    PHI_SEQ,  # VESPO gamma weight per sequence (B,) or None
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
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

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

    elif LOSS_TYPE == 3:  # VESPO: detached gamma weighting phi(w) on per-token logp
        # Reference: TRL grpo_trainer.get_gamma_weights / chunked_loss/grpo_loss.py
        # phi_seq is precomputed per-sequence, vllm correction is folded into phi_seq.
        phi_seq = tl.load(PHI_SEQ + off_b).to(tl.float32)
        per_token_loss = -phi_seq * advantage * logp
        is_clipped = 0.0  # VESPO has no clipping concept

    # Apply vLLM importance sampling correction BEFORE adding KL penalty
    # VESPO folds this into phi_seq (in log space), so we skip it here.
    if VLLM_IS_RATIO is not None and LOSS_TYPE != 3:
        # Use modulo to support both (B, L) per-token and (B, 1) per-sequence shapes
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
            tl.float32
        )
        per_token_loss = per_token_loss * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        if USE_BIAS_CORRECTION_KL:
            # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= coef_1
            kl = kl * tl.exp(logp - old_logp)
        per_token_loss += BETA * kl
        tl.store(KL, kl)

    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)


# Sequence-level forward kernel: uses pre-computed coef_1 per sequence
@triton.jit
def _grpo_loss_fwd_kernel_seq(
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    COMPLETION_MASK,
    ADVANTAGES,
    COEF_1,  # Pre-computed sequence-level importance weight, post delta-clamp (B,)
    COEF_1_RAW,  # Pre-computed sequence-level importance weight, pre delta-clamp (B,)
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
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    COEF_1 += off_b
    COEF_1_RAW += off_b
    COEF_2 += off_b
    IS_CLIPPED_SEQ += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l

    # Compute log softmax
    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse

    # Load pre-computed sequence-level coefficients
    coef_1 = tl.load(COEF_1).to(tl.float32)
    coef_2 = tl.load(COEF_2).to(tl.float32)
    is_clipped_seq = tl.load(IS_CLIPPED_SEQ)

    advantage = tl.load(ADVANTAGES).to(tl.float32)
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
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        if USE_BIAS_CORRECTION_KL:
            # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= coef_1.
            # Use the pre-clamp sequence-level coef_1 to match TRL — the same coef_1
            # that was passed to ``per_token_kl * coef_1`` upstream of the delta clamp.
            kl = kl * tl.load(COEF_1_RAW).to(tl.float32)
        per_token_loss += BETA * kl
        tl.store(KL, kl)

    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped_seq)  # Same for all tokens in sequence


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
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    DLOGITS += off_b * (L + 1) * N + off_l * N
    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS + cols, 0.0, mask=cols < N)
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    DLOSS_SUM += off_b
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l
    COEF_1 += off_b
    SEQ_LEN += off_b

    dloss = tl.load(DLOSS).to(tl.float32)
    dloss_sum = tl.load(DLOSS_SUM).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)
    coef_1 = tl.load(COEF_1).to(tl.float32)
    seq_len = tl.load(SEQ_LEN).to(tl.float32)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse

    advantage = tl.load(ADVANTAGES).to(tl.float32)
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
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        if USE_BIAS_CORRECTION_KL:
            # d(kl * coef_1)/d(logp) ≈ coef_1 * (logp - ref_logp), with coef_1 detached.
            # Use the loaded sequence-level coef_1 (pre delta-clamp) to match TRL's
            # ``per_token_kl * coef_1`` for importance_sampling_level == "sequence".
            dlogp += BETA * coef_1 * (logp - ref_logp) * dloss
        else:
            dlogp += BETA * (1 - tl.exp(ref_logp - logp)) * dloss

    dlogp = dlogp / TEMPERATURE
    tl.debug_barrier()
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)


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
    PHI_SEQ,  # VESPO gamma weight per sequence (B,) or None
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
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    DLOGITS += off_b * (L + 1) * N + off_l * N
    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS + cols, 0.0, mask=cols < N)
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l

    dloss = tl.load(DLOSS).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

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

    elif LOSS_TYPE == 3:  # VESPO: detached gamma weighting on per-token logp
        # loss = -phi_seq * advantage * logp; phi_seq is detached → ∂loss/∂logp = -phi_seq * advantage
        phi_seq = tl.load(PHI_SEQ + off_b).to(tl.float32)
        dlogp = -phi_seq * advantage

    # Apply vLLM IS ratio to PPO gradient (before KL gradient)
    # VESPO folds vllm correction into phi_seq, so we skip it here.
    if VLLM_IS_RATIO is not None and LOSS_TYPE != 3:
        # Use modulo to support both (B, L) per-token and (B, 1) per-sequence shapes
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
            tl.float32
        )
        dlogp = dlogp * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        if USE_BIAS_CORRECTION_KL:
            # d(kl * coef_1)/d(logp) = coef_1 * (logp - ref_logp), where coef_1 = exp(logp - old_logp)
            dlogp += BETA * coef_1 * (logp - ref_logp)
        else:
            dlogp += BETA * (1 - tl.exp(ref_logp - logp))

    dlogp = dlogp * dloss / TEMPERATURE
    tl.debug_barrier()
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)


def _compute_dapo_normalizer(completion_mask, num_items_in_batch=None):
    """Per-process normalizer for DAPO/CISPO.

    When ``num_items_in_batch`` is provided it is used directly, matching TRL's
    ``num_items_in_batch / num_processes`` (total active tokens across the entire
    generation batch including all gradient-accumulation micro-batches × all
    processes). Falling back to the current micro-batch's mask biases per-token
    weights by micro-batch size when grad-accum micro-batches have unequal
    completion lengths.
    """
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        normalizer = normalizer.clone()
        torch.distributed.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
    normalizer = normalizer / world_size
    return torch.clamp(normalizer, min=1.0)


def _reduce_loss(per_token_loss, mask, loss_type, max_completion_length, B, L, num_items_in_batch=None):
    """Apply loss reduction based on loss_type."""
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
    raise ValueError(
        f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo, vespo"
    )


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
        num_items_in_batch=None,
        phi_seq=None,
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
        if delta is not None and loss_type in ("cispo", "sapo", "vespo"):
            raise ValueError(f"delta (two-sided clipping) is not supported for loss_type='{loss_type}'.")

        # Map delta to float for Triton (Triton can't handle None)
        delta_val = 0.0 if delta is None else float(delta)

        # Validate sequence-level + loss_type combinations
        if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo", "vespo"):
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

        # VESPO requires phi_seq pre-computed by the caller (uses get_gamma_weights).
        if loss_type == "vespo":
            if phi_seq is None:
                raise ValueError("loss_type='vespo' requires phi_seq pre-computed (use the triton_grpo_loss wrapper).")
            assert phi_seq.shape in ((B,), (B, 1)), f"phi_seq must be (B,) or (B, 1), got {tuple(phi_seq.shape)}"
            phi_seq = phi_seq.reshape(-1).contiguous()
        else:
            phi_seq = None

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
            kwargs = {"BLOCK_N": 2048, "num_stages": 2, "num_warps": 1}
            _grpo_loss_fwd_kernel_seq[(B, L)](
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                completion_mask,
                advantages,
                coef_1_for_loss.contiguous(),
                coef_1.contiguous(),  # COEF_1_RAW: pre delta-clamp, for bias-corrected KL
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
                **kwargs,
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
            kwargs = {"BLOCK_N": 2048, "num_stages": 2, "num_warps": 1}
            _grpo_loss_fwd_kernel[(B, L)](
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                completion_mask,
                advantages,
                vllm_is_ratio_ptr,
                vllm_is_ratio_stride,
                phi_seq,  # PHI_SEQ (B,) for VESPO, None otherwise
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
                **kwargs,
            )
            ctx.save_for_backward(
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                mask,
                vllm_is_ratio_ptr,
                phi_seq,
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
            num_items_in_batch,
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

        reduced_loss = _reduce_loss(
            loss, mask, loss_type, max_completion_length, B, L, num_items_in_batch=num_items_in_batch
        )
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
            num_items_in_batch,
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
            phi_seq = None
        else:
            (
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                lse,
                mask,
                vllm_is_ratio,
                phi_seq,
            ) = saved_tensors

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
        elif loss_type == "dapo" or loss_type == "cispo" or loss_type == "vespo":
            dloss = dloss_input * mask / _compute_dapo_normalizer(mask, num_items_in_batch=num_items_in_batch)
        elif loss_type == "luspo":
            # loss = mean(per_token_loss * seq_lens), mean divides by B*L
            seq_lens_bwd = mask.sum(-1, keepdim=True).clamp(min=1.0)
            dloss = dloss_input * seq_lens_bwd / (B * L)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        dlogits = logits.data if inplace else torch.empty_like(logits)
        kwargs = {"BLOCK_N": 4096, "num_stages": 1, "num_warps": 16}

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
            _grpo_loss_bwd_kernel_seq[(B, L)](
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
                **kwargs,
            )
        else:
            # Token-level backward kernel with LOSS_TYPE branching
            _grpo_loss_bwd_kernel[(B, L)](
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
                phi_seq,
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
                **kwargs,
            )

        dlogits[:, -1, :] = 0
        # Return gradients for all forward inputs: dlogits + 21 None for non-differentiable params
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
            None,  # num_items_in_batch
            None,  # phi_seq
        )
