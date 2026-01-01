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
    ):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True

        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        if completion_mask is not None:
            assert completion_mask.is_contiguous()

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
        # Get mask for loss computation
        mask = completion_mask.float() if completion_mask is not None else torch.ones_like(loss)

        if not reduce:
            # Return per-token loss (original behavior for backward compatibility)
            # Apply mask to zero out padded positions
            loss = loss * mask
            if kl is not None:
                kl = kl * mask
            is_clipped = is_clipped * mask

            ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse)
            ctx.infos = (temperature, beta, eps_low, eps_high, inplace, False, None, B, L)
            return loss, kl, is_clipped

        # Apply loss reduction based on loss_type
        if loss_type == "grpo":
            # Average per-sequence loss, then batch mean
            reduced_loss = ((loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        elif loss_type == "bnpo":
            # Batch normalized per-token
            reduced_loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            # Dimension-reduced (use max_completion_length or L)
            max_len = max_completion_length if max_completion_length is not None else L
            reduced_loss = (loss * mask).sum() / (B * max_len)
        elif loss_type == "dapo":
            # Distributed-aware normalization (same as bnpo for single GPU)
            normalizer = _compute_dapo_normalizer(mask)
            reduced_loss = (loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo")

        # Compute aggregated metrics
        mask_sum = mask.sum().clamp(min=1.0)
        kl_mean = (kl * mask).sum() / mask_sum if kl is not None else None
        clip_ratio = (is_clipped.float() * mask).sum() / mask_sum

        # Save for backward - need per-token loss gradient scaling
        ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse, mask)
        ctx.infos = (temperature, beta, eps_low, eps_high, inplace, loss_type, max_completion_length, B, L)

        return reduced_loss, kl_mean, clip_ratio

    @staticmethod
    def backward(ctx, *args):
        dloss_input = args[0]  # Gradient - either scalar (reduce=True) or (B, L) tensor (reduce=False)
        saved_tensors = ctx.saved_tensors
        temperature, beta, eps_low, eps_high, inplace, loss_type, max_completion_length, B, L = ctx.infos
        _, L_ADD_1, N = saved_tensors[0].shape  # logits shape

        if loss_type is False:
            # reduce=False case: dloss_input is (B, L) per-token gradients
            logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse = saved_tensors
            dloss = dloss_input
        else:
            # reduce=True case: dloss_input is scalar, need to compute per-token gradients
            logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse, mask = saved_tensors

            # Compute per-token gradient scaling based on loss_type
            # dloss_per_token = dloss_scalar * d(reduced_loss)/d(per_token_loss)
            if loss_type == "grpo":
                # For grpo: reduced = mean over batch of (sum over seq / seq_len)
                # d(reduced)/d(per_token) = mask / (seq_len * B)
                seq_lens = mask.sum(-1, keepdim=True).clamp(min=1.0)  # (B, 1)
                dloss = dloss_input * mask / (seq_lens * B)
            elif loss_type == "bnpo":
                # For bnpo: reduced = sum(loss * mask) / sum(mask)
                # d(reduced)/d(per_token) = mask / sum(mask)
                dloss = dloss_input * mask / mask.sum().clamp(min=1.0)
            elif loss_type == "dr_grpo":
                # For dr_grpo: reduced = sum(loss * mask) / (B * max_len)
                max_len = max_completion_length if max_completion_length is not None else L
                dloss = dloss_input * mask / (B * max_len)
            elif loss_type == "dapo":
                # For dapo: reduced = sum(loss * mask) / normalizer
                # normalizer is computed with all_reduce, gradient is mask / normalizer
                normalizer = _compute_dapo_normalizer(mask)
                dloss = dloss_input * mask / normalizer
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

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
        # Return gradients for all inputs (None for non-differentiable params)
        return dlogits, None, None, None, None, None, None, None, None, None, None, None, None, None
