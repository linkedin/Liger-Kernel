import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# Loss type mapping for Triton constexpr branching
# GRPO/DAPO/BNPO/DR_GRPO share identical per-token loss computation (standard PPO clipping)
_TYPE_GRPO: tl.constexpr = tl.constexpr(0)
_TYPE_CISPO: tl.constexpr = tl.constexpr(1)
_TYPE_SAPO: tl.constexpr = tl.constexpr(2)

_str_to_loss_type = {
    "grpo": _TYPE_GRPO.value,
    "dapo": _TYPE_GRPO.value,
    "bnpo": _TYPE_GRPO.value,
    "dr_grpo": _TYPE_GRPO.value,
    "luspo": _TYPE_GRPO.value,
    "cispo": _TYPE_CISPO.value,
    "sapo": _TYPE_SAPO.value,
}


def calculate_tile_count_2d(batch_size, seq_len, num_cores):
    """Compute optimal grid configuration for parallel processing."""
    grid_batch = batch_size
    cores_per_sample = min(seq_len, num_cores // batch_size)
    cores_per_sample = max(1, cores_per_sample)
    grid_seq = cores_per_sample
    total = grid_batch * grid_seq
    if total > num_cores:
        grid_seq = max(1, num_cores // grid_batch)
    return (grid_batch, grid_seq)


def compute_block_size_softmax(seq_vocab_size):
    """Determine optimal block size for selective log-softmax kernel."""
    multiplier = 6.0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((seq_vocab_size,),), tiling_dims=(0,)
    )
    if tile_shapes and len(tile_shapes) > 0:
        return tile_shapes[0][0]
    return 2048


def compute_block_size_forward(seq_vocab_size):
    """Determine optimal block size for forward pass kernel."""
    multiplier = 10.0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((seq_vocab_size,),), tiling_dims=(0,)
    )
    if tile_shapes and len(tile_shapes) > 0:
        return tile_shapes[0][0]
    return 2048


def compute_block_size_backward(seq_vocab_size):
    """Determine optimal block size for backward pass kernel."""
    multiplier = 12.0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((seq_vocab_size,),), tiling_dims=(0,)
    )
    if tile_shapes and len(tile_shapes) > 0:
        return tile_shapes[0][0]
    return 2048


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

            idx = tl.load(INPUT_IDS_local).to(tl.int32)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse
            if OLD_LOGP is None:
                old_logp = logp
            else:
                OLD_LOGP_local = OLD_LOGP + token_idx
                old_logp = tl.load(OLD_LOGP_local).to(tl.float32)
            coef_1 = tl.exp(logp - old_logp)
            advantage = tl.load(ADVANTAGES_local).to(tl.float32)

            if LOSS_TYPE == 0:  # GRPO/DAPO/BNPO/DR_GRPO
                coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
                is_low_clipped = (coef_1 < 1 - EPS_LOW) & (advantage < 0)
                is_high_clipped = (coef_1 > 1 + EPS_HIGH) & (advantage > 0)
                is_clipped = is_low_clipped | is_high_clipped
                if DELTA != 0.0:
                    coef_1 = tl.minimum(coef_1, DELTA)
                per_token_loss1 = coef_1 * advantage
                per_token_loss2 = coef_2 * advantage
                per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)

            elif LOSS_TYPE == 1:  # CISPO
                coef_2 = tl.minimum(coef_1, EPS_HIGH)
                per_token_loss = -coef_2 * advantage * logp
                is_clipped = (coef_1 > EPS_HIGH) & (advantage > 0)

            elif LOSS_TYPE == 2:  # SAPO
                temperature = tl.where(advantage > 0, SAPO_TEMP_POS, SAPO_TEMP_NEG)
                sigmoid_input = temperature * (coef_1 - 1.0)
                sapo_coef = tl.sigmoid(sigmoid_input) * 4.0 / temperature
                per_token_loss = -sapo_coef * advantage
                is_clipped = 0.0

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
                    kl = kl * tl.exp(logp - old_logp)
                per_token_loss += BETA * kl
                tl.store(KL_local, kl)

            tl.store(LOSS_local, per_token_loss)
            tl.store(LSE_local, lse)
            tl.store(IS_CLIPPED_local, is_clipped)


@triton.jit
def _grpo_loss_fwd_kernel_seq(
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    COMPLETION_MASK,
    ADVANTAGES,
    COEF_1,
    COEF_2,
    IS_CLIPPED_SEQ,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
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

            idx = tl.load(INPUT_IDS_local).to(tl.int32)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse

            coef_1 = tl.load(COEF_1_local).to(tl.float32)
            coef_2 = tl.load(COEF_2_local).to(tl.float32)
            is_clipped_seq = tl.load(IS_CLIPPED_SEQ_local)

            advantage = tl.load(ADVANTAGES_local).to(tl.float32)
            per_token_loss1 = coef_1 * advantage
            per_token_loss2 = coef_2 * advantage
            per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)

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
                    if OLD_LOGP is None:
                        old_logp = logp
                    else:
                        old_logp = tl.load(OLD_LOGP + token_idx).to(tl.float32)
                    kl = kl * tl.exp(logp - old_logp)
                per_token_loss += BETA * kl
                tl.store(KL_local, kl)

            tl.store(LOSS_local, per_token_loss)
            tl.store(LSE_local, lse)
            tl.store(IS_CLIPPED_local, is_clipped_seq)


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
    COEF_1,
    SEQ_LEN,
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

        should_process = 1
        if COMPLETION_MASK is not None:
            COMPLETION_MASK_local = COMPLETION_MASK + off_b * L + off_l
            not_skip = tl.load(COMPLETION_MASK_local)
            should_process = not_skip

        if should_process == 0:
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

            idx = tl.load(INPUT_IDS_local).to(tl.int32)
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

            dlogp = -coef_1 * advantage / seq_len * is_unclipped * dloss_sum
            if DELTA != 0.0:
                dlogp = dlogp * (coef_1 <= DELTA)

            if BETA != 0.0:
                REF_LOGP_local = REF_LOGP + token_idx
                ref_logp = tl.load(REF_LOGP_local).to(tl.float32)
                if USE_BIAS_CORRECTION_KL:
                    if OLD_LOGP is None:
                        old_logp = logp
                    else:
                        old_logp = tl.load(OLD_LOGP + token_idx).to(tl.float32)
                    token_coef_1 = tl.exp(logp - old_logp)
                    dlogp += BETA * token_coef_1 * (logp - ref_logp) * dloss
                else:
                    dlogp += BETA * (1 - tl.exp(ref_logp - logp)) * dloss

            dlogp = dlogp / TEMPERATURE
            for start_n in tl.range(0, N, BLOCK_N):
                cols = start_n + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
                probs = tl.exp(logits - lse)
                cols_idx = cols == idx
                dlogits = (cols_idx - probs) * dlogp
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

        should_process = 1
        if COMPLETION_MASK is not None:
            COMPLETION_MASK_local = COMPLETION_MASK + off_b * L + off_l
            not_skip = tl.load(COMPLETION_MASK_local)
            should_process = not_skip

        if should_process == 0:
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

            idx = tl.load(INPUT_IDS_local).to(tl.int32)
            x = tl.load(LOGITS_local + idx).to(tl.float32) / TEMPERATURE
            logp = x - lse
            if OLD_LOGP is None:
                old_logp = logp
            else:
                OLD_LOGP_local = OLD_LOGP + token_idx
                old_logp = tl.load(OLD_LOGP_local).to(tl.float32)
            coef_1 = tl.exp(logp - old_logp)
            advantage = tl.load(ADVANTAGES_local).to(tl.float32)

            if LOSS_TYPE == 0:  # GRPO/DAPO/BNPO/DR_GRPO
                coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
                if DELTA != 0.0:
                    coef_1_for_loss = tl.minimum(coef_1, DELTA)
                else:
                    coef_1_for_loss = coef_1
                per_token_loss1 = coef_1_for_loss * advantage
                per_token_loss2 = coef_2 * advantage
                mask = per_token_loss2 >= per_token_loss1
                dlogp = -coef_1 * advantage * mask
                if DELTA != 0.0:
                    dlogp = dlogp * (coef_1 <= DELTA)

            elif LOSS_TYPE == 1:  # CISPO
                coef_2 = tl.minimum(coef_1, EPS_HIGH)
                dlogp = -coef_2 * advantage

            elif LOSS_TYPE == 2:  # SAPO
                temperature = tl.where(advantage > 0, SAPO_TEMP_POS, SAPO_TEMP_NEG)
                sigmoid_input = temperature * (coef_1 - 1.0)
                sigmoid_val = tl.sigmoid(sigmoid_input)
                d_sapo_d_coef1 = 4.0 * sigmoid_val * (1.0 - sigmoid_val)
                dlogp = -advantage * d_sapo_d_coef1 * coef_1

            if VLLM_IS_RATIO is not None:
                vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
                    tl.float32
                )
                dlogp = dlogp * vllm_is_ratio

            if BETA != 0.0:
                REF_LOGP_local = REF_LOGP + token_idx
                ref_logp = tl.load(REF_LOGP_local).to(tl.float32)
                if USE_BIAS_CORRECTION_KL:
                    dlogp += BETA * coef_1 * (logp - ref_logp)
                else:
                    dlogp += BETA * (1 - tl.exp(ref_logp - logp))

            dlogp = dlogp * dloss / TEMPERATURE
            for start_n in tl.range(0, N, BLOCK_N):
                cols = start_n + tl.arange(0, BLOCK_N)
                logits = tl.load(LOGITS_local + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
                probs = tl.exp(logits - lse)
                cols_idx = cols == idx
                dlogits = (cols_idx - probs) * dlogp
                tl.store(DLOGITS_local + cols, dlogits, mask=cols < N)


@torch.no_grad
def fused_selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 0.9, mask=None):
    """Compute log probabilities for specific token IDs with selective masking."""
    assert logits.is_contiguous()
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    input_ids = input_ids[:, -L:]
    if mask is not None:
        mask = mask[:, -L:]
    log_p = torch.zeros(B, L, dtype=torch.float32, device=logits.device)

    block_n = compute_block_size_softmax(N)
    num_cores = get_npu_core_count()
    grid = calculate_tile_count_2d(B, L, num_cores)
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


def compute_distribution_normalizer(completion_mask):
    """Calculate global active token count for distributed loss normalization."""
    normalizer = completion_mask.to(torch.float32).sum()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        normalizer = normalizer.clone()
        torch.distributed.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    normalizer = normalizer / world_size
    return torch.clamp(normalizer, min=1.0)


def reduce_loss(per_token_loss, mask, loss_type, max_completion_length, batch_size, seq_len):
    """Apply reduction strategy based on specified loss type."""
    if loss_type == "grpo" or loss_type == "sapo":
        return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        max_len = max_completion_length if max_completion_length is not None else seq_len
        return (per_token_loss * mask).sum() / (batch_size * max_len)
    elif loss_type == "dapo" or loss_type == "cispo":
        return (per_token_loss * mask).sum() / compute_distribution_normalizer(mask)
    elif loss_type == "luspo":
        return (per_token_loss * mask.sum(-1, keepdim=True)).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}. Expected one of: grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo")


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
    """Forward pass computation for GRPO loss."""
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
    assert importance_sampling_level in ("token", "sequence"), (
        f"importance_sampling_level must be 'token' or 'sequence', got {importance_sampling_level}"
    )

    if loss_type not in _str_to_loss_type:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Supported types: {list(_str_to_loss_type.keys())}")

    if delta is not None and loss_type in ("cispo", "sapo"):
        raise ValueError(f"delta (two-sided clipping) is not supported for loss_type='{loss_type}'.")

    delta_val = 0.0 if delta is None else float(delta)

    if importance_sampling_level == "sequence" and loss_type in ("cispo", "sapo"):
        raise ValueError(
            f"Sequence-level importance sampling is not supported for loss_type='{loss_type}'. "
            f"Use importance_sampling_level='token' instead."
        )

    if loss_type == "sapo":
        if sapo_temperature_pos <= 0:
            raise ValueError(f"sapo_temperature_pos must be positive, got {sapo_temperature_pos}")
        if sapo_temperature_neg <= 0:
            raise ValueError(f"sapo_temperature_neg must be positive, got {sapo_temperature_neg}")

    loss_type_int = _str_to_loss_type[loss_type]

    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1

    if completion_mask is not None:
        assert completion_mask.is_contiguous()

    mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=logits.device)

    vllm_is_ratio_ptr = None
    vllm_is_ratio_stride = L
    if vllm_is_ratio is not None:
        assert vllm_is_ratio.dim() in (1, 2), (
            f"vllm_is_ratio must be 1D (B,) or 2D (B, L) / (B, 1), got {vllm_is_ratio.dim()}D"
        )
        if vllm_is_ratio.dim() == 2:
            assert vllm_is_ratio.shape[0] == B and vllm_is_ratio.shape[1] in (1, L), (
                f"vllm_is_ratio shape must be ({B}, 1) or ({B}, {L}), got {tuple(vllm_is_ratio.shape)}"
            )
        else:
            assert vllm_is_ratio.shape[0] == B, f"vllm_is_ratio shape must be ({B},), got {tuple(vllm_is_ratio.shape)}"
        vllm_is_ratio = vllm_is_ratio.contiguous()
        vllm_is_ratio_ptr = vllm_is_ratio
        vllm_is_ratio_stride = vllm_is_ratio.shape[1] if vllm_is_ratio.dim() > 1 else 1

    loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    lse = torch.zeros_like(loss)
    is_clipped = torch.zeros_like(loss)
    kl = torch.zeros_like(loss) if beta != 0.0 else None

    block_n = compute_block_size_forward(N)
    num_cores = get_npu_core_count()
    grid = calculate_tile_count_2d(B, L, num_cores)

    if importance_sampling_level == "sequence":
        per_token_logps = fused_selective_log_softmax(logits, completion_ids, temperature, completion_mask)

        if old_logp is None:
            log_ratio = torch.zeros_like(per_token_logps)
        else:
            log_ratio = per_token_logps - old_logp

        seq_lens = mask.sum(-1).clamp(min=1.0)
        seq_log_importance = (log_ratio * mask).sum(-1) / seq_lens
        coef_1 = torch.exp(seq_log_importance)
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)

        is_clipped_seq = ((coef_1 < 1 - eps_low) & (advantages < 0)) | ((coef_1 > 1 + eps_high) & (advantages > 0))
        is_clipped_seq = is_clipped_seq.float()

        if delta is not None:
            coef_1_for_loss = torch.clamp(coef_1, max=delta)
        else:
            coef_1_for_loss = coef_1

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

    mask_sum = mask.sum().clamp(min=1.0)
    kl_mean = (kl * mask).sum() / mask_sum if kl is not None else None
    clip_ratio = (is_clipped.float() * mask).sum() / mask_sum

    if not reduce:
        loss_out = loss * mask
        kl_out = kl * mask if kl is not None else None
        is_clipped_out = is_clipped * mask
        return loss_out, kl_out, is_clipped_out

    reduced_loss = reduce_loss(loss, mask, loss_type, max_completion_length, B, L)
    return reduced_loss, kl_mean, clip_ratio


def grpo_loss_backward_triton(ctx, *args):
    """Backward pass computation for GRPO loss."""
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
        dloss = dloss_input * mask / compute_distribution_normalizer(mask)
    elif loss_type == "luspo":
        seq_lens_bwd = mask.sum(-1, keepdim=True).clamp(min=1.0)
        dloss = dloss_input * seq_lens_bwd / (B * L)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    dlogits = logits.data if inplace else torch.empty_like(logits)

    block_n = compute_block_size_backward(N)
    num_cores = get_npu_core_count()
    grid = calculate_tile_count_2d(B, L, num_cores)

    if importance_sampling_level == "sequence":
        if vllm_is_ratio is None:
            dloss_sum = dloss.sum(-1).contiguous()
        else:
            if vllm_is_ratio.dim() == 1:
                ratio = vllm_is_ratio.unsqueeze(-1)
            else:
                ratio = vllm_is_ratio
            dloss_sum = (dloss * ratio).sum(-1).contiguous()
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
        None,
    )


class GrpoLossFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, *args):
        return grpo_loss_forward_triton(ctx, *args)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *args):
        return grpo_loss_backward_triton(ctx, *args)
