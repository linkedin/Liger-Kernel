import torch
import triton
import triton.language as tl


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
    VLLM_IS_RATIO,  # vLLM importance sampling ratio (B, L) or None
    VLLM_IS_RATIO_STRIDE,  # stride for VLLM_IS_RATIO (L for per-token, 1 for per-sequence)
    LOSS,
    LSE,
    KL,
    IS_CLIPPED,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
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
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)
    is_low_clipped = (coef_1 < 1 - EPS_LOW) & (advantage < 0)
    is_high_clipped = (coef_1 > 1 + EPS_HIGH) & (advantage > 0)
    is_clipped = is_low_clipped | is_high_clipped

    # Apply vLLM importance sampling correction BEFORE adding KL
    if VLLM_IS_RATIO is not None:
        # VLLM_IS_RATIO_STRIDE is L for per-token, 1 for per-sequence
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
            tl.float32
        )
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


# Sequence-level forward kernel: uses pre-computed coef_1 per sequence
@triton.jit
def _grpo_loss_fwd_kernel_seq(
    LOGITS,
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
    REF_LOGP,
    INPUT_IDS,
    ADVANTAGES,
    COMPLETION_MASK,
    LSE,
    COEF_1,  # Pre-computed sequence-level importance weight (B,)
    SEQ_LEN,  # Number of valid tokens per sequence (B,)
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
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
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    is_unclipped = per_token_loss2 >= per_token_loss1

    # For sequence-level: gradient flows through mean, so scale by coef_1/seq_len
    # d(loss)/d(logp) = -advantage * coef_1 / seq_len (when unclipped)
    dlogp = -per_token_loss1 / seq_len * is_unclipped * dloss_sum

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
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
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW,
    EPS_HIGH,
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
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    mask = per_token_loss2 >= per_token_loss1

    dlogp = -per_token_loss1 * mask

    # Apply vLLM IS ratio to PPO gradient (before KL gradient)
    if VLLM_IS_RATIO is not None:
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
            tl.float32
        )
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
    if loss_type == "grpo":
        return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        max_len = max_completion_length if max_completion_length is not None else L
        return (per_token_loss * mask).sum() / (B * max_len)
    elif loss_type == "dapo":
        return (per_token_loss * mask).sum() / _compute_dapo_normalizer(mask)
    raise ValueError(f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo")


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
        vllm_is_ratio=None,  # vLLM importance sampling ratio (B, L) or (B, 1) or None
    ):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
        assert importance_sampling_level in ("token", "sequence"), (
            f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}"
        )

        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        if completion_mask is not None:
            assert completion_mask.is_contiguous()

        mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=logits.device)

        # Handle vLLM IS ratio
        vllm_is_ratio_ptr = None
        vllm_is_ratio_stride = L  # default to per-token
        if vllm_is_ratio is not None:
            vllm_is_ratio = vllm_is_ratio.contiguous()
            vllm_is_ratio_ptr = vllm_is_ratio
            # Determine stride: L for per-token (B, L), 1 for per-sequence (B, 1)
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

            # Compute is_clipped at sequence level
            is_clipped_seq = ((coef_1 < 1 - eps_low) & (advantages < 0)) | ((coef_1 > 1 + eps_high) & (advantages > 0))
            is_clipped_seq = is_clipped_seq.float()  # (B,)

            # Step 3: Run Triton kernel with pre-computed coefficients
            kwargs = {"BLOCK_N": 2048, "num_stages": 2, "num_warps": 1}
            _grpo_loss_fwd_kernel_seq[(B, L)](
                logits,
                ref_logp,
                completion_ids,
                completion_mask,
                advantages,
                coef_1.contiguous(),
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
            # Token-level: use optimized Triton kernel
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
                L,
                N,
                **kwargs,
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
            max_completion_length,
            B,
            L,
            importance_sampling_level,
            vllm_is_ratio_stride,
            reduce,
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
            max_completion_length,
            B,
            L,
            importance_sampling_level,
            vllm_is_ratio_stride,
            reduce,
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
        elif loss_type == "grpo":
            seq_lens_bwd = mask.sum(-1, keepdim=True).clamp(min=1.0)
            dloss = dloss_input * mask / (seq_lens_bwd * B)
        elif loss_type == "bnpo":
            dloss = dloss_input * mask / mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            max_len = max_completion_length if max_completion_length is not None else L
            dloss = dloss_input * mask / (B * max_len)
        elif loss_type == "dapo":
            dloss = dloss_input * mask / _compute_dapo_normalizer(mask)
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
                *dloss.stride(),
                L,
                N,
                **kwargs,
            )
        else:
            # Token-level backward kernel
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
                *dloss.stride(),
                L,
                N,
                **kwargs,
            )

        dlogits[:, -1, :] = 0
        # Return gradients for all forward inputs: dlogits + 15 None for non-differentiable params
        return dlogits, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
