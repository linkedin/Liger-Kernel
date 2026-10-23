import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings

LOG2_E: tl.constexpr = 1.4426950408889634


@triton.jit
def _selective_logprob_fwd_kernel(
    LOGITS_ptr,
    LOGITS_stride,
    TARGETS_ptr,
    LOGPROBS_ptr,
    LOG_Z_ptr,
    BIAS_ptr,
    n_cols,
    temperature,
    HAS_BIAS: tl.constexpr,
    STORE_LOG_Z: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = LOGITS_ptr + row_idx * LOGITS_stride

    target = tl.load(TARGETS_ptr + row_idx)
    if (target == IGNORE_INDEX) | (target < 0) | (target >= n_cols):
        tl.store(LOGPROBS_ptr + row_idx, 0.0)
        if STORE_LOG_Z:
            tl.store(LOG_Z_ptr + row_idx, 0.0)
        return

    inv_temp = 1.0 / temperature

    # Pass 1: compute online softmax (max and sum-exp)
    m = float("-inf")
    d = 0.0
    for offset in range(0, n_cols, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_BIAS:
            val += tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = val * inv_temp
        new_m = tl.maximum(m, tl.max(val))
        d = d * tl.exp2((m - new_m) * LOG2_E) + tl.sum(tl.exp2((val - new_m) * LOG2_E))
        m = new_m

    lse = m + tl.log(d)
    if STORE_LOG_Z:
        tl.store(LOG_Z_ptr + row_idx, lse)

    target_logit = tl.load(logits_row_ptr + target).to(tl.float32)
    if HAS_BIAS:
        target_logit += tl.load(BIAS_ptr + target).to(tl.float32)
    target_logit = target_logit * inv_temp
    tl.store(LOGPROBS_ptr + row_idx, target_logit - lse)


@triton.jit
def _selective_logprob_bwd_kernel(
    LOGITS_ptr,
    LOGITS_stride,
    TARGETS_ptr,
    GRAD_LOGPROBS_ptr,
    BIAS_ptr,
    n_cols,
    temperature,
    HAS_BIAS: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = LOGITS_ptr + row_idx * LOGITS_stride

    target = tl.load(TARGETS_ptr + row_idx)
    g = tl.load(GRAD_LOGPROBS_ptr + row_idx).to(tl.float32)

    is_invalid = (g == 0.0) | (target == IGNORE_INDEX) | (target < 0) | (target >= n_cols)
    if is_invalid:
        for offset in range(0, n_cols, BLOCK_N):
            cols = offset + tl.arange(0, BLOCK_N)
            tl.store(logits_row_ptr + cols, 0.0, mask=cols < n_cols)
        return

    inv_temp = 1.0 / temperature

    # Pass 1: compute m and d for online softmax
    m = float("-inf")
    d = 0.0
    for offset in range(0, n_cols, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_BIAS:
            val += tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = val * inv_temp
        new_m = tl.maximum(m, tl.max(val))
        d = d * tl.exp2((m - new_m) * LOG2_E) + tl.sum(tl.exp2((val - new_m) * LOG2_E))
        m = new_m

    scaled_g = g * inv_temp

    # Pre-read target logit to compute stable true-class gradient
    val_target = tl.load(logits_row_ptr + target).to(tl.float32)
    if HAS_BIAS:
        val_target += tl.load(BIAS_ptr + target).to(tl.float32)
    val_target = val_target * inv_temp

    # Pass 2: grad_logits = -scaled_g * softmax(val)
    for offset in range(0, n_cols, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_BIAS:
            val += tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = val * inv_temp
        prob = tl.exp2((val - m) * LOG2_E) / d
        grad_val = -scaled_g * prob
        tl.store(logits_row_ptr + cols, grad_val, mask=mask)

    tl.debug_barrier()

    # Stable true-class gradient: scaled_g * (1 - prob_target)
    prob_target = tl.exp2((val_target - m) * LOG2_E) / d
    dx_target = scaled_g * (1.0 - prob_target)
    tl.store(logits_row_ptr + target, dx_target)


def selective_logprob_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor = None,
    temperature: float = 1.0,
    ignore_index: int = -100,
    return_log_z: bool = False,
):
    """Compute selective log-probabilities for each row using Triton.

    Args:
        logits: (n_rows, n_cols) tensor of logits.
        targets: (n_rows,) int64 tensor of target indices.
        bias: optional (n_cols,) bias tensor.
        temperature: temperature scaling factor.
        ignore_index: target value indicating masked/ignored tokens.
        return_log_z: whether to also return per-row log-sum-exp (LSE).

    Returns:
        logprobs: (n_rows,) float32 tensor of selective log-probabilities.
        log_z: (n_rows,) float32 tensor of LSE (if return_log_z is True).
    """
    assert logits.is_contiguous(), "logits must be contiguous"
    assert targets.is_contiguous(), "targets must be contiguous"
    n_rows, n_cols = logits.shape

    logprobs = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=logits.device, dtype=torch.float32) if return_log_z else None

    BLOCK_N, num_warps = calculate_settings(n_cols)
    BLOCK_N = min(BLOCK_N, 4096)

    _selective_logprob_fwd_kernel[(n_rows,)](
        logits,
        logits.stride(0),
        targets,
        logprobs,
        log_z if return_log_z else logits,
        bias if bias is not None else logits,
        n_cols,
        float(temperature),
        HAS_BIAS=bias is not None,
        STORE_LOG_Z=return_log_z,
        IGNORE_INDEX=ignore_index,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    if return_log_z:
        return logprobs, log_z
    return logprobs


def selective_logprob_backward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grad_logprobs: torch.Tensor,
    bias: torch.Tensor = None,
    temperature: float = 1.0,
    ignore_index: int = -100,
):
    """Compute gradients with respect to logits in-place.

    Args:
        logits: (n_rows, n_cols) tensor of logits (overwritten in-place with grad_logits).
        targets: (n_rows,) int64 tensor of target indices.
        grad_logprobs: (n_rows,) float32 tensor of upstream gradients w.r.t. logprobs.
        bias: optional (n_cols,) bias tensor.
        temperature: temperature scaling factor.
        ignore_index: target value indicating masked/ignored tokens.

    Returns:
        logits: (n_rows, n_cols) tensor containing grad_logits (modified in-place).
    """
    assert logits.is_contiguous(), "logits must be contiguous"
    assert targets.is_contiguous(), "targets must be contiguous"
    n_rows, n_cols = logits.shape

    BLOCK_N, num_warps = calculate_settings(n_cols)
    BLOCK_N = min(BLOCK_N, 4096)

    grad_logprobs = grad_logprobs.contiguous().to(torch.float32)

    _selective_logprob_bwd_kernel[(n_rows,)](
        logits,
        logits.stride(0),
        targets,
        grad_logprobs,
        bias if bias is not None else logits,
        n_cols,
        float(temperature),
        HAS_BIAS=bias is not None,
        IGNORE_INDEX=ignore_index,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return logits
