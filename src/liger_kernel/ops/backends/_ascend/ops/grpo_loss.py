import functools

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import get_npu_core_count


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


@triton.jit
def _grpo_loss_fwd_kernel(
    LOGITS, OLD_LOGP, REF_LOGP, INPUT_IDS, COMPLETION_MASK, ADVANTAGES,
    LOSS, LSE, KL, IS_CLIPPED, TEMPERATURE, BETA: tl.constexpr, EPS_LOW, EPS_HIGH,
    TOTAL_ROWS: tl.constexpr, VOCAB_SIZE: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_OLD_LOGP: tl.constexpr, HAS_REF_LOGP: tl.constexpr, HAS_COMPLETION_MASK: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    thread_id = tl.program_id(0)
    num_threads = tl.num_programs(0)
    INV_TEMP = 1.0 / TEMPERATURE

    for row_id in tl.range(thread_id, TOTAL_ROWS, num_threads, num_stages=NUM_STAGES):
        off_b = row_id // L
        off_l = row_id % L
        logits_row_base = off_b * ((L + 1) * VOCAB_SIZE) + off_l * VOCAB_SIZE
        input_row_idx = row_id

        do_compute = 1
        if HAS_COMPLETION_MASK:
            mask_ptr = COMPLETION_MASK + input_row_idx
            do_compute = tl.load(mask_ptr)

        if do_compute != 0:
            LOGITS_PTR = LOGITS + logits_row_base
            INPUT_IDS_PTR = INPUT_IDS + input_row_idx
            ADVANTAGES_PTR = ADVANTAGES + off_b
            LOSS_PTR = LOSS + input_row_idx
            LSE_PTR = LSE + input_row_idx
            IS_CLIPPED_PTR = IS_CLIPPED + input_row_idx

            m_i = float("-inf")
            l_i = 0.0
            for start in range(0, VOCAB_SIZE, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < VOCAB_SIZE
                x = tl.load(LOGITS_PTR + cols, mask=mask, other=float("-inf")).to(tl.float32) * INV_TEMP
                new_m = tl.maximum(m_i, tl.max(x))
                alpha = tl.exp(m_i - new_m)
                l_i = l_i * alpha + tl.sum(tl.exp(x - new_m), axis=0)
                m_i = new_m

            lse_val = m_i + tl.log(l_i)
            tl.store(LSE_PTR, lse_val)

            idx = tl.load(INPUT_IDS_PTR).to(tl.int64)
            logit_x = tl.load(LOGITS_PTR + idx).to(tl.float32) * INV_TEMP
            logp = logit_x - lse_val

            if HAS_OLD_LOGP:
                old_logp = tl.load(OLD_LOGP + input_row_idx).to(tl.float32)
            else:
                old_logp = logp

            ratio = tl.exp(logp - old_logp)
            clipped_ratio = tl.clamp(ratio, 1.0 - EPS_LOW, 1.0 + EPS_HIGH)
            advantage = tl.load(ADVANTAGES_PTR).to(tl.float32)

            loss1 = ratio * advantage
            loss2 = clipped_ratio * advantage
            per_token_loss = -tl.minimum(loss1, loss2)

            is_low_clipped = (ratio < 1.0 - EPS_LOW) & (advantage < 0)
            is_high_clipped = (ratio > 1.0 + EPS_HIGH) & (advantage > 0)
            is_clipped = is_low_clipped | is_high_clipped

            if BETA != 0.0 and HAS_REF_LOGP:
                ref_logp = tl.load(REF_LOGP + input_row_idx).to(tl.float32)
                kl_div = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
                per_token_loss += BETA * kl_div
                tl.store(KL + input_row_idx, kl_div)

            tl.store(LOSS_PTR, per_token_loss)
            tl.store(IS_CLIPPED_PTR, is_clipped)


@triton.jit
def _grpo_loss_bwd_kernel(
    DLOSS, DLOGITS, LOGITS, OLD_LOGP, REF_LOGP, INPUT_IDS, ADVANTAGES, COMPLETION_MASK,
    LSE, TEMPERATURE, BETA: tl.constexpr, EPS_LOW, EPS_HIGH,
    TOTAL_ROWS: tl.constexpr, VOCAB_SIZE: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_OLD_LOGP: tl.constexpr, HAS_REF_LOGP: tl.constexpr, HAS_COMPLETION_MASK: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    thread_id = tl.program_id(0)
    num_threads = tl.num_programs(0)
    INV_TEMP = 1.0 / TEMPERATURE

    for row_id in tl.range(thread_id, TOTAL_ROWS, num_threads, num_stages=NUM_STAGES):
        off_b = row_id // L
        off_l = row_id % L
        logits_row_base = off_b * ((L + 1) * VOCAB_SIZE) + off_l * VOCAB_SIZE
        input_row_idx = row_id

        DLOGITS_PTR = DLOGITS + logits_row_base

        do_compute = 1
        if HAS_COMPLETION_MASK:
            mask_ptr = COMPLETION_MASK + input_row_idx
            do_compute = tl.load(mask_ptr)

        if do_compute == 0:
            for start in range(0, VOCAB_SIZE, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < VOCAB_SIZE
                tl.store(DLOGITS_PTR + cols, 0.0, mask=mask)

        else:
            LOGITS_PTR = LOGITS + logits_row_base
            INPUT_IDS_PTR = INPUT_IDS + input_row_idx
            ADVANTAGES_PTR = ADVANTAGES + off_b
            LSE_PTR = LSE + input_row_idx
            DLOSS_PTR = DLOSS + input_row_idx

            dloss = tl.load(DLOSS_PTR).to(tl.float32)
            lse_val = tl.load(LSE_PTR).to(tl.float32)

            idx = tl.load(INPUT_IDS_PTR).to(tl.int64)
            x = tl.load(LOGITS_PTR + idx).to(tl.float32) * INV_TEMP
            logp = x - lse_val

            if HAS_OLD_LOGP:
                old_logp = tl.load(OLD_LOGP + input_row_idx).to(tl.float32)
            else:
                old_logp = logp

            ratio = tl.exp(logp - old_logp)
            clipped_ratio = tl.clamp(ratio, 1.0 - EPS_LOW, 1.0 + EPS_HIGH)
            advantage = tl.load(ADVANTAGES_PTR).to(tl.float32)

            loss1 = ratio * advantage
            loss2 = clipped_ratio * advantage
            use_clipped = loss2 <= loss1

            dlogp_unscaled = tl.where(use_clipped, -clipped_ratio, -ratio)
            if BETA != 0.0 and HAS_REF_LOGP:
                ref_logp = tl.load(REF_LOGP + input_row_idx).to(tl.float32)
                kl_grad = 1.0 - tl.exp(ref_logp - logp)
                dlogp_unscaled += BETA * kl_grad

            dlogp = dlogp_unscaled * dloss * INV_TEMP

            for start in range(0, VOCAB_SIZE, BLOCK_N):
                cols = start + tl.arange(0, BLOCK_N)
                mask = cols < VOCAB_SIZE

                logits_block = tl.load(LOGITS_PTR + cols, mask=mask, other=-float("inf")).to(tl.float32) * INV_TEMP
                probs = tl.exp(logits_block - lse_val)

                dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
                tl.store(DLOGITS_PTR + cols, dlogits, mask=mask)


def grpo_loss_forward_triton(
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
    inplace=False,
):
    assert logits.is_contiguous(), "logits must be contiguous"
    assert completion_ids.is_contiguous(), "completion_ids must be contiguous"
    if old_logp is not None:
        assert old_logp.is_contiguous(), "old_logp must be contiguous"
    if beta != 0.0:
        assert ref_logp is not None and ref_logp.is_contiguous(), "ref_logp required when beta != 0"

    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1  # usually one extra for shift
    TOTAL_ROWS = B * L

    if completion_mask is not None:
        assert completion_mask.is_contiguous()

    # =============================
    #   Dynamic Tiling Strategy
    # =============================
    shapes = ((N,),)
    dtype_size = 4
    memory_multiplier = 5.0
    safety_margin = 0.8

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=safety_margin,
        dtype_size=dtype_size,
        memory_multiplier=memory_multiplier,
        shapes=shapes,
        tiling_dims=(0,),
    )

    if tile_shapes is not None and len(tile_shapes[0]) > 0:
        BLOCK_N = tile_shapes[0][0]
    else:
        BLOCK_N = min(8192, triton.next_power_of_2(N))

    # =============================
    #   Parallel Workgroup Setup
    # =============================
    num_cores = get_npu_core_count()
    grid_dim0 = min(num_cores, TOTAL_ROWS)

    NUM_STAGES = 3 if TOTAL_ROWS < 2048 else 4

    # Prepare outputs
    loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    lse = torch.zeros_like(loss)
    is_clipped = torch.zeros_like(loss, dtype=torch.bool)
    kl = torch.zeros_like(loss) if beta != 0.0 else None

    # Launch kernel
    _grpo_loss_fwd_kernel[(grid_dim0,)](
        LOGITS=logits,
        OLD_LOGP=old_logp,
        REF_LOGP=ref_logp,
        INPUT_IDS=completion_ids,
        COMPLETION_MASK=completion_mask,
        ADVANTAGES=advantages,
        LOSS=loss,
        LSE=lse,
        KL=kl,
        IS_CLIPPED=is_clipped,
        TEMPERATURE=temperature,
        BETA=beta,
        EPS_LOW=eps_low,
        EPS_HIGH=eps_high,
        TOTAL_ROWS=TOTAL_ROWS,
        VOCAB_SIZE=N,
        L=L,  # [FIX]: 传入序列长度 L
        BLOCK_N=BLOCK_N,
        HAS_OLD_LOGP=old_logp is not None,
        HAS_REF_LOGP=ref_logp is not None,
        HAS_COMPLETION_MASK=completion_mask is not None,
        NUM_STAGES=NUM_STAGES,
        num_warps=8 if BLOCK_N >= 2048 else 4,
        num_stages=NUM_STAGES,
    )

    ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse)
    ctx.infos = (temperature, beta, eps_low, eps_high, inplace, B, L, N)
    ctx.mark_non_differentiable(is_clipped)

    return loss, kl, is_clipped


def grpo_loss_backward_triton(ctx, *args):
    dloss = args[0]
    saved_tensors = ctx.saved_tensors
    logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse = saved_tensors
    temperature, beta, eps_low, eps_high, inplace, B, L, N = ctx.infos

    TOTAL_ROWS = B * L
    VOCAB_SIZE = N

    shapes = ((N,),)
    dtype_size = 4
    memory_multiplier = 5.0
    safety_margin = 0.8

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=safety_margin,
        dtype_size=dtype_size,
        memory_multiplier=memory_multiplier,
        shapes=shapes,
        tiling_dims=(0,),
    )

    if tile_shapes is not None and len(tile_shapes[0]) > 0:
        BLOCK_N = tile_shapes[0][0]
    else:
        BLOCK_N = min(8192, triton.next_power_of_2(N))

    num_cores = get_npu_core_count()
    grid_dim0 = min(num_cores, TOTAL_ROWS)
    NUM_STAGES = 3 if TOTAL_ROWS < 2048 else 4

    dlogits = logits.data if inplace else torch.empty_like(logits, dtype=torch.float32)

    _grpo_loss_bwd_kernel[(grid_dim0,)](
        DLOSS=dloss,
        DLOGITS=dlogits,
        LOGITS=logits,
        OLD_LOGP=old_logp,
        REF_LOGP=ref_logp,
        INPUT_IDS=completion_ids,
        ADVANTAGES=advantages,
        COMPLETION_MASK=completion_mask,
        LSE=lse,
        TEMPERATURE=temperature,
        BETA=beta,
        EPS_LOW=eps_low,
        EPS_HIGH=eps_high,
        TOTAL_ROWS=TOTAL_ROWS,
        VOCAB_SIZE=VOCAB_SIZE,
        L=L,  # [FIX]: 传入序列长度 L
        BLOCK_N=BLOCK_N,
        HAS_OLD_LOGP=old_logp is not None,
        HAS_REF_LOGP=ref_logp is not None,
        HAS_COMPLETION_MASK=completion_mask is not None,
        NUM_STAGES=NUM_STAGES,
        num_warps=8 if BLOCK_N >= 2048 else 4,
        num_stages=NUM_STAGES,
    )

    dlogits[:, -1, :] = 0

    return (dlogits, None, None, None, None, None, None, None, None, None, None)


class GrpoLossFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta,
                eps_low, eps_high, inplace=False):
        return grpo_loss_forward_triton(ctx, logits, old_logp, ref_logp, completion_ids, advantages, completion_mask,
                                        temperature, beta, eps_low, eps_high, inplace)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *args):
        return grpo_loss_backward_triton(ctx, *args)
