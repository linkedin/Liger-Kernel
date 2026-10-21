"""Fused lm_head + score centering loss (https://arxiv.org/abs/2609.20807).

REINFORCE with an optional token-level importance-sampling correction w(p/q), minus the expected score under the
effective sampling mass q * w. The sampler ships only its top-k head (ids and logprobs) and its tail is modeled as
rho * p, with rho matching the tail mass. Every ratio correction is constant on that tail, so its centering term is
alpha * E_p[log p], whose gradient is zero: the loss only depends, through its gradient, on the logprobs of the sampled
token and of the k head tokens. With g_j = d loss / d logp_j for those k + 1 ids, the logits gradient is
scatter(g) - (sum_j g_j) * softmax, which a single kernel writes in place over each chunk of logits. As in the fused
linear cross entropy, gradients are computed in the forward pass and the [N, V] logits are never materialized.
"""

import math

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Partial
    from torch.distributed.tensor import Replicate
except ImportError:  # torch without DTensor
    DTensor = None

MAX_FUSED_SIZE = 2048 if infer_device() == "npu" else 16384
# Rows per logits chunk. Each chunk re-reads the lm_head weight twice and updates its gradient, so the fused linear
# cross entropy's N x H memory bound (32 rows at V = 128k, H = 4k) leaves the GEMMs memory-bound; 1024 rows keep them
# compute-bound for a 256 MiB bf16 logits chunk at V = 128k.
MIN_CHUNK_SIZE = 1024
DUAL_CLIP = 3.0  # PPO's ratio bound for negative advantages (arXiv:1912.09729)
_SIGNS = {"both": 0, "neg": 1, "pos": 2, "ppo": 3}


class ISCorrection(NamedTuple):
    """Token-level importance-sampling correction w(r) on the ratio r = p / q, as in the reference implementation.

    TIS: ISCorrection(high=2.0, outside="clamp"); MIS: ISCorrection(low=0.5, high=5.0);
    PPO: ISCorrection(low=0.8, high=1.2, sign="ppo"); DAPO: ISCorrection(low=0.8, high=1.28, sign="ppo");
    naive IS: ISCorrection().
    """

    low: float = 0.0  # band [low, high] on r; 0 / inf leave a side open
    high: float = math.inf
    outside: str = "drop"  # clamp: w = nearest bound | drop: w = 0
    sign: str = "both"  # which advantages get corrected: both | neg | pos (others w = 1)
    # | ppo: block only moves away from q; A>0: (0, high], A<0: [low, DUAL_CLIP]

    def check(self):
        assert self.outside in ("clamp", "drop"), "outside must be 'clamp' or 'drop'"
        assert self.sign in _SIGNS, "sign must be one of both | neg | pos | ppo"
        return self


@triton.jit
def _log_mass(
    log_p,
    log_q,
    positive,
    LOG_LOW: tl.constexpr,
    LOG_HIGH: tl.constexpr,
    HAS_CORRECTION: tl.constexpr,
    HAS_LOW: tl.constexpr,
    HAS_HIGH: tl.constexpr,
    CLAMP: tl.constexpr,
    SIGN: tl.constexpr,
):
    """log(q * w(p/q)): a token's effective sampling mass (-inf where w = 0, log q without a correction).

    It gives the IS weight w(r) as the mass at p = r, q = 1, the centering mass q * w at (p, q), and the top-k tail
    factor rho * w(1 / rho) at p = 1, q = rho.
    """
    if HAS_CORRECTION:
        if HAS_LOW:
            lo = log_q + LOG_LOW
        else:
            lo = log_q - float("inf")
        if HAS_HIGH:
            hi = log_q + LOG_HIGH
        else:
            hi = log_q + float("inf")
        if SIGN == 3:
            in_band = tl.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + 1.0986122886681098))
        else:
            in_band = (log_p >= lo) & (log_p <= hi)
        if CLAMP:
            outside = tl.minimum(tl.maximum(log_p, lo), hi)
        else:
            outside = log_p - float("inf")
        log_m = tl.where(in_band, log_p, outside)
        if SIGN == 1:
            log_m = tl.where(positive, log_q, log_m)
        if SIGN == 2:
            log_m = tl.where(positive, log_m, log_q)
    else:
        log_m = log_q + tl.zeros_like(log_p)
    return log_m


@triton.jit
def _score_centering_kernel(
    X_ptr,
    X_stride,
    IDS_ptr,
    IDS_stride,
    SAMP_TOKEN_ptr,
    SAMP_HEAD_ptr,
    SAMP_HEAD_stride,
    ADV_ptr,
    LOSS_ptr,
    ENTROPY_ptr,
    KL_HEAD_ptr,
    MASKED_ptr,
    n_cols,
    n_head,
    inv_temperature,
    grad_scale,
    eps,
    LOG_LOW: tl.constexpr,
    LOG_HIGH: tl.constexpr,
    HAS_CORRECTION: tl.constexpr,
    HAS_LOW: tl.constexpr,
    HAS_HIGH: tl.constexpr,
    CLAMP: tl.constexpr,
    SIGN: tl.constexpr,
    SCORE_CENTER: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """One row of logits: the loss and metrics, and (in place) d loss / d logits scaled by grad_scale.

    IDS holds the sampled token in column 0 and the sampler's head in columns 1..n_head.
    """
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    IDS_ptr += row * IDS_stride
    SAMP_HEAD_ptr += row * SAMP_HEAD_stride

    # Pass 1: online logsumexp and sum exp(z - m) * z (for the entropy).
    m = float("-inf")
    d = 0.0
    s = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_cols
        z = tl.load(X_ptr + offsets, mask=valid, other=float("-inf")).to(tl.float32) * inv_temperature
        m_new = tl.maximum(m, tl.max(z))
        rescale = tl.exp(m - m_new)
        e = tl.exp(z - m_new)
        d = d * rescale + tl.sum(e)
        s = s * rescale + tl.sum(tl.where(valid, e * z, 0.0))
        m = m_new
    lse = m + tl.log(d)
    entropy = lse - s / d

    # Gather the sampled token and the head.
    token = tl.load(IDS_ptr)
    logp_token = tl.load(X_ptr + token).to(tl.float32) * inv_temperature - lse
    head_offsets = tl.arange(0, BLOCK_HEAD)
    head_valid = head_offsets < n_head
    head_ids = tl.load(IDS_ptr + 1 + head_offsets, mask=head_valid, other=0)
    logp_head = tl.load(X_ptr + head_ids, mask=head_valid, other=0.0).to(tl.float32) * inv_temperature - lse
    logq_head = tl.load(SAMP_HEAD_ptr + head_offsets, mask=head_valid, other=0.0).to(tl.float32)
    logq_token = tl.load(SAMP_TOKEN_ptr + row).to(tl.float32)
    advantage = tl.load(ADV_ptr + row).to(tl.float32)
    positive = advantage >= 0

    zero = tl.zeros_like(lse)
    log_ratio = tl.minimum(tl.maximum(logp_token - logq_token, -20.0), 20.0)
    token_weight = tl.exp(
        _log_mass(log_ratio, zero, positive, LOG_LOW, LOG_HIGH, HAS_CORRECTION, HAS_LOW, HAS_HIGH, CLAMP, SIGN)
    )

    p_head = tl.where(head_valid, tl.exp(logp_head), 0.0)
    q_head = tl.where(head_valid, tl.exp(logq_head), 0.0)
    q_tail = 1.0 - tl.sum(q_head)
    rho = tl.maximum(q_tail, 0.0) / tl.maximum(1.0 - tl.sum(p_head), eps)
    kl_head = tl.sum(tl.where(head_valid, q_head * (logq_head - logp_head), 0.0)) + q_tail * tl.log(
        tl.maximum(rho, 1e-9)
    )

    score = token_weight * logp_token
    if SCORE_CENTER:
        alpha = tl.exp(
            _log_mass(zero, tl.log(rho), positive, LOG_LOW, LOG_HIGH, HAS_CORRECTION, HAS_LOW, HAS_HIGH, CLAMP, SIGN)
        )
        mass = tl.exp(
            _log_mass(logp_head, logq_head, positive, LOG_LOW, LOG_HIGH, HAS_CORRECTION, HAS_LOW, HAS_HIGH, CLAMP, SIGN)
        )
        residual = tl.where(head_valid, mass - alpha * p_head, 0.0)
        # alpha * E_p[log p] = -alpha * entropy has zero gradient; it only makes the loss value fully centered
        score = score - tl.sum(residual * logp_head) + alpha * entropy
    else:
        residual = tl.zeros((BLOCK_HEAD,), dtype=tl.float32)

    tl.store(LOSS_ptr + row, -advantage * score)
    tl.store(ENTROPY_ptr + row, entropy)
    tl.store(KL_HEAD_ptr + row, kl_head)
    tl.store(MASKED_ptr + row, (token_weight == 0).to(tl.float32))

    if HAS_GRADIENTS:
        # d loss / d logp of the sampled token and of the head, scaled by grad_scale / temperature.
        g_token = -advantage * token_weight * grad_scale
        g_head = advantage * residual * grad_scale
        g_sum = g_token + tl.sum(g_head)
        # Head ids are distinct, but the sampled token may be one of them: fold it into that head slot.
        token_in_head = head_valid & (head_ids == token)
        g_head += tl.where(token_in_head, g_token, 0.0)
        g_token = tl.where(tl.sum(token_in_head.to(tl.int32)) > 0, 0.0, g_token)

        # Pass 2: the dense part -g_sum * softmax, in place.
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            valid = offsets < n_cols
            z = tl.load(X_ptr + offsets, mask=valid, other=float("-inf")).to(tl.float32) * inv_temperature
            grad = -g_sum * tl.exp(z - lse) * inv_temperature
            tl.store(X_ptr + offsets, grad.to(X_ptr.dtype.element_ty), mask=valid)
        tl.debug_barrier()
        # The sparse part: overwrite the k + 1 slots with g - g_sum * p.
        head_grad = (g_head - g_sum * p_head) * inv_temperature
        tl.store(X_ptr + head_ids, head_grad.to(X_ptr.dtype.element_ty), mask=head_valid)
        token_grad = (g_token - g_sum * tl.exp(logp_token)) * inv_temperature
        tl.store(X_ptr + token, token_grad.to(X_ptr.dtype.element_ty), mask=g_token != 0)


def _replicate(param):
    """Gather a sharded DTensor parameter (FSDP2 / HSDP lm_head) into a local tensor.

    The gradient is marked Partial("avg") so its backward reduce-scatters the average over ranks, which is what
    DDP / FSDP do for every other parameter. DTensor.full_tensor() would instead assume the gradient is already
    identical on every rank and keep only each rank's local slice.
    """
    if DTensor is None or not isinstance(param, DTensor):
        return param
    n_dims = param.device_mesh.ndim
    return param.redistribute(placements=[Replicate()] * n_dims).to_local(grad_placements=[Partial("avg")] * n_dims)


def _loss_normalizer(mask, num_items_in_batch):
    """Tokens / world_size, so that DDP / FSDP gradient averaging gives the global token mean (as in DAPO / GRPO).

    num_items_in_batch counts completion tokens over all processes and grad-accum micro-batches; if None, the current
    micro-batch's tokens are all-reduced over processes.
    """
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    if num_items_in_batch is not None:
        normalizer = torch.as_tensor(num_items_in_batch, dtype=torch.float32, device=mask.device)
    else:
        normalizer = mask.sum().to(torch.float32)
        if distributed:
            torch.distributed.all_reduce(normalizer)
    return (normalizer / world_size).clamp_min(1.0)


def fused_linear_score_centering_forward(
    _input,
    weight,
    ids,
    advantages,
    sampler_token_logps,
    sampler_topk_logps,
    bias,
    normalizer,
    score_center,
    correction,
    temperature,
    eps,
    compute_grad_input,
    compute_grad_weight,
    compute_grad_bias,
    chunk_size=None,
):
    """Rows are completion tokens only: _input [N, H], ids [N, 1 + k], the rest [N] or [N, k]."""
    device = _input.device
    N, H = _input.shape
    V = weight.shape[0]
    n_head = ids.shape[1] - 1
    has_gradients = compute_grad_input or compute_grad_weight or compute_grad_bias

    if chunk_size is None:
        # The fused linear cross entropy's N x H bound on the transient logits, but at least MIN_CHUNK_SIZE rows.
        chunk_size = max(triton.next_power_of_2(triton.cdiv(N, triton.cdiv(V, H))), MIN_CHUNK_SIZE)
    chunk_size = max(min(chunk_size, N), 1)

    loss_1d = torch.empty(N, dtype=torch.float32, device=device)
    entropy_1d = torch.empty(N, dtype=torch.float32, device=device)
    kl_head_1d = torch.empty(N, dtype=torch.float32, device=device)
    masked_1d = torch.empty(N, dtype=torch.float32, device=device)
    grad_input = torch.empty_like(_input) if compute_grad_input else None
    # Accumulated in the weight dtype, as the fused linear cross entropy does by default.
    grad_weight = torch.empty_like(weight) if compute_grad_weight else None
    grad_bias = torch.zeros(V, dtype=torch.float32, device=device) if compute_grad_bias else None

    c = correction
    flags = dict(
        LOG_LOW=math.log(c.low) if c is not None and c.low > 0 else 0.0,
        LOG_HIGH=math.log(c.high) if c is not None and c.high < math.inf else 0.0,
        HAS_CORRECTION=c is not None,
        HAS_LOW=c is not None and c.low > 0,
        HAS_HIGH=c is not None and c.high < math.inf,
        CLAMP=c is not None and c.outside == "clamp",
        SIGN=_SIGNS[c.sign] if c is not None else 0,
    )
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    grad_scale = 1.0 / normalizer

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        input_chunk = _input[start:end]
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        logits_chunk = logits_chunk.contiguous()
        _score_centering_kernel[(end - start,)](
            logits_chunk,
            logits_chunk.stride(0),
            ids[start:end],
            ids.stride(0),
            sampler_token_logps[start:end],
            sampler_topk_logps[start:end],
            sampler_topk_logps.stride(0),
            advantages[start:end],
            loss_1d[start:end],
            entropy_1d[start:end],
            kl_head_1d[start:end],
            masked_1d[start:end],
            V,
            n_head,
            1.0 / temperature,
            grad_scale,
            eps,
            SCORE_CENTER=score_center,
            HAS_GRADIENTS=has_gradients,
            BLOCK_SIZE=block_size,
            BLOCK_HEAD=triton.next_power_of_2(max(n_head, 1)),
            num_warps=32 if not is_hip() else 16,
            **flags,
        )
        grad_logits = logits_chunk  # overwritten in place by the kernel
        if compute_grad_input:
            torch.mm(grad_logits, weight.to(grad_logits.dtype), out=grad_input[start:end])
        if compute_grad_weight:
            if start == 0:  # the first chunk initializes the gradient instead of zero-filling it first
                torch.mm(grad_logits.t(), input_chunk.to(grad_logits.dtype), out=grad_weight)
            else:
                torch.addmm(grad_weight, grad_logits.t(), input_chunk.to(grad_logits.dtype), out=grad_weight)
        if compute_grad_bias:
            grad_bias.add_(grad_logits.sum(dim=0, dtype=torch.float32))
        del logits_chunk, grad_logits

    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)
    return loss_1d, entropy_1d, kl_head_1d, masked_1d, grad_input, grad_weight, grad_bias


class LigerFusedLinearScoreCenteringFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        sampler_token_logps,
        sampler_topk_ids,
        sampler_topk_logps,
        bias=None,
        score_center=True,
        correction=None,
        temperature=1.0,
        eps=1e-6,
        num_items_in_batch=None,
        chunk_size=None,
    ):
        """
        Args:
            _input: [B, T, H] hidden states aligned with selected_token_ids (already shifted).
            weight: [V, H] lm_head weight. bias: optional [V].
            selected_token_ids: [B, T] sampled tokens.
            attention_mask: [B, T], nonzero on completion tokens only.
            advantages: [B] or [B, T].
            sampler_token_logps: [B, T] sampler logprob of the sampled token, kept even when it is outside the head.
            sampler_topk_ids, sampler_topk_logps: [B, T, k] sampler head, not renormalized. k = V gives exact
                centering with the full sampler distribution.
            correction: optional ISCorrection, e.g. ISCorrection(high=2.0, outside="clamp") for TIS.
            num_items_in_batch: completion tokens over all processes and grad-accum micro-batches (see
                _loss_normalizer).
            chunk_size: rows of logits materialized at once (default: see MIN_CHUNK_SIZE).
        Returns:
            loss, and the masked means of the entropy, kl_head and the fraction of tokens with w = 0.
        """
        if correction is not None:
            correction.check()
        mask = attention_mask.bool()
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        # Only completion tokens go through the lm_head.
        ids = torch.cat([selected_token_ids.unsqueeze(-1), sampler_topk_ids], dim=-1)[mask].contiguous()
        normalizer = _loss_normalizer(mask, num_items_in_batch)
        n_tokens = mask.sum().clamp_min(1)

        loss_1d, entropy_1d, kl_head_1d, masked_1d, grad_input, grad_weight, grad_bias = (
            fused_linear_score_centering_forward(
                _input[mask].contiguous(),
                weight,
                ids,
                advantages.expand_as(mask)[mask].float().contiguous(),
                sampler_token_logps[mask].float().contiguous(),
                sampler_topk_logps[mask].float().contiguous(),
                bias,
                normalizer.item(),
                score_center,
                correction,
                temperature,
                eps,
                compute_grad_input=_input.requires_grad,
                compute_grad_weight=weight.requires_grad,
                compute_grad_bias=bias is not None and bias.requires_grad,
                chunk_size=chunk_size,
            )
        )
        if grad_input is not None:
            full_grad_input = torch.zeros_like(_input)
            full_grad_input[mask] = grad_input
            grad_input = full_grad_input
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)

        loss = loss_1d.sum() / normalizer
        entropy = entropy_1d.sum() / n_tokens
        kl_head = kl_head_1d.sum() / n_tokens
        frac_masked = masked_1d.sum() / n_tokens
        ctx.mark_non_differentiable(entropy, kl_head, frac_masked)
        return loss, entropy, kl_head, frac_masked

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, *_grad_metrics):
        grads = ctx.saved_tensors
        # The loss is usually the last op: skip rescaling the weight-sized gradient by 1.
        if not torch.equal(grad_output, torch.ones_like(grad_output)):
            grads = [g * grad_output if g is not None else None for g in grads]
        grad_input, grad_weight, grad_bias = grads
        return grad_input, grad_weight, *([None] * 6), grad_bias, *([None] * 6)


def fused_linear_score_centering(
    _input,
    weight,
    selected_token_ids,
    attention_mask,
    advantages,
    sampler_token_logps,
    sampler_topk_ids,
    sampler_topk_logps,
    bias=None,
    score_center=True,
    correction=None,
    temperature=1.0,
    eps=1e-6,
    num_items_in_batch=None,
    chunk_size=None,
):
    """Returns (loss, metrics); weight and bias may be sharded DTensors (FSDP2)."""
    loss, entropy, kl_head, frac_masked = LigerFusedLinearScoreCenteringFunction.apply(
        _input,
        _replicate(weight),
        selected_token_ids,
        attention_mask,
        advantages,
        sampler_token_logps,
        sampler_topk_ids,
        sampler_topk_logps,
        _replicate(bias),
        score_center,
        correction,
        temperature,
        eps,
        num_items_in_batch,
        chunk_size,
    )
    return loss, {"entropy": entropy, "kl_head": kl_head, "frac_masked": frac_masked}
