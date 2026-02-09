import torch
import triton
import triton.language as tl

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
    "cispo": _LOSS_TYPE_CISPO.value,
    "sapo": _LOSS_TYPE_SAPO.value,
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
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)
        is_low_clipped = (coef_1 < 1 - EPS_LOW) & (advantage < 0)
        is_high_clipped = (coef_1 > 1 + EPS_HIGH) & (advantage > 0)
        is_clipped = is_low_clipped | is_high_clipped

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
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l).to(tl.float32)
        per_token_loss = per_token_loss * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        per_token_loss += BETA * kl
        tl.store(KL, kl)

    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)


# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])
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
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        mask = per_token_loss2 >= per_token_loss1
        dlogp = -per_token_loss1 * mask

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
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l).to(tl.float32)
        dlogp = dlogp * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        dlogp += BETA * (1 - tl.exp(ref_logp - logp))

    dlogp = dlogp * dloss / TEMPERATURE
    tl.debug_barrier()
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)


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
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vllm_is_ratio=None,
    ):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True

        # Validate loss_type
        if loss_type not in _str_to_loss_type:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Supported types: {list(_str_to_loss_type.keys())}")

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

        # Handle vLLM IS ratio
        vllm_is_ratio_ptr = None
        vllm_is_ratio_stride = L  # default to per-token
        if vllm_is_ratio is not None:
            vllm_is_ratio = vllm_is_ratio.contiguous()
            vllm_is_ratio_ptr = vllm_is_ratio
            vllm_is_ratio_stride = vllm_is_ratio.shape[1] if vllm_is_ratio.dim() > 1 else 1

        loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
        lse = torch.zeros_like(loss)
        is_clipped = torch.zeros_like(loss)
        kl = torch.zeros_like(loss) if beta != 0.0 else None
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
            L,
            N,
            **kwargs,
        )
        ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse)
        ctx.vllm_is_ratio = vllm_is_ratio_ptr
        ctx.vllm_is_ratio_stride = vllm_is_ratio_stride
        ctx.infos = (
            temperature,
            beta,
            eps_low,
            eps_high,
            inplace,
            loss_type_int,
            sapo_temperature_pos,
            sapo_temperature_neg,
        )
        return loss, kl, is_clipped

    @staticmethod
    def backward(ctx, *args):
        dloss = args[0]
        logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse = ctx.saved_tensors
        temperature, beta, eps_low, eps_high, inplace, loss_type_int, sapo_temperature_pos, sapo_temperature_neg = (
            ctx.infos
        )
        vllm_is_ratio = ctx.vllm_is_ratio
        vllm_is_ratio_stride = ctx.vllm_is_ratio_stride
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        dlogits = logits.data if inplace else torch.empty_like(logits)
        kwargs = {"BLOCK_N": 4096, "num_stages": 1, "num_warps": 16}
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
            temperature,
            beta,
            eps_low,
            eps_high,
            loss_type_int,
            sapo_temperature_pos,
            sapo_temperature_neg,
            *dloss.stride(),
            L,
            N,
            **kwargs,
        )
        dlogits[:, -1, :] = 0
        # Return None for: old_logp, ref_logp, completion_ids, advantages, completion_mask,
        # temperature, beta, eps_low, eps_high, inplace, loss_type, sapo_temperature_pos, sapo_temperature_neg,
        # vllm_is_ratio
        return dlogits, None, None, None, None, None, None, None, None, None, None, None, None, None, None
