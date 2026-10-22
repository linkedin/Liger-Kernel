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
except ImportError:
    DTensor = None

# Two passes over each row keep more live registers than cross entropy; 16384 measured faster than 32768.
MAX_FUSED_SIZE = 2048 if infer_device() == "npu" else 16384
# Each chunk re-reads the weight twice and updates its gradient. FLCE's BT x H budget gives 32-row chunks at
# V = 128k, H = 4k, which leaves those GEMMs memory-bound; 1024 rows keep them compute-bound.
MIN_CHUNK_SIZE = 1024
# PPO's ratio bound for negative advantages (https://arxiv.org/abs/1912.09729)
DUAL_CLIP = 3.0
_LOG_DUAL_CLIP: tl.constexpr = tl.constexpr(math.log(DUAL_CLIP))


class ISCorrection(NamedTuple):
    """
    Token-level importance-sampling correction w(r) on the ratio r = p / q: w = r inside the band [low, high] and,
    outside it, the nearest bound (outside="clamp") or 0 (outside="drop").

    TIS: ISCorrection(high=2.0, outside="clamp"), MIS: ISCorrection(low=0.5, high=5.0),
    PPO: ISCorrection(low=0.8, high=1.2, sign="ppo"), DAPO: ISCorrection(low=0.8, high=1.28, sign="ppo").
    """

    low: float = 0.0
    high: float = math.inf
    outside: str = "drop"
    # "ppo" only blocks moves away from q: (0, high] for A > 0 and [low, DUAL_CLIP] for A < 0
    sign: str = "both"

    def check(self):
        assert self.outside in ("clamp", "drop"), f"outside must be 'clamp' or 'drop'. Got: {self.outside}"
        assert self.sign in ("both", "ppo"), f"sign must be 'both' or 'ppo'. Got: {self.sign}"
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
    PPO: tl.constexpr,
):
    """
    log(q * w(p / q)), the effective sampling mass of a token. The same function gives the IS weight w(r) at
    (p, q) = (r, 1) and the top-k tail factor rho * w(1 / rho) at (p, q) = (1, rho).
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
        if PPO:
            in_band = tl.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + _LOG_DUAL_CLIP))
        else:
            in_band = (log_p >= lo) & (log_p <= hi)
        if CLAMP:
            outside = tl.minimum(tl.maximum(log_p, lo), hi)
        else:
            outside = log_p - float("inf")
        log_m = tl.where(in_band, log_p, outside)
    else:
        log_m = log_q + tl.zeros_like(log_p)
    return log_m


@triton.jit
def _score_centering_kernel(
    X_ptr,
    X_stride,
    ids_ptr,
    ids_stride,
    sampler_token_logps_ptr,
    sampler_topk_logps_ptr,
    sampler_topk_logps_stride,
    advantages_ptr,
    loss_ptr,
    entropy_ptr,
    kl_head_ptr,
    masked_ptr,
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
    PPO: tl.constexpr,
    SCORE_CENTER: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """
    Computes the loss and metrics of one row of logits and, if HAS_GRADIENTS, overwrites the row with
    grad_scale * d loss / d logits. ids holds the sampled token in column 0 and the sampler's top-k head after it.
    """
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    ids_ptr += row * ids_stride
    sampler_topk_logps_ptr += row * sampler_topk_logps_stride

    # online logsumexp, plus sum(exp(z - m) * z) for the entropy
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

    token = tl.load(ids_ptr)
    logp_token = tl.load(X_ptr + token).to(tl.float32) * inv_temperature - lse
    head_offsets = tl.arange(0, BLOCK_HEAD)
    head_valid = head_offsets < n_head
    head_ids = tl.load(ids_ptr + 1 + head_offsets, mask=head_valid, other=0)
    logp_head = tl.load(X_ptr + head_ids, mask=head_valid, other=0.0).to(tl.float32) * inv_temperature - lse
    logq_head = tl.load(sampler_topk_logps_ptr + head_offsets, mask=head_valid, other=0.0).to(tl.float32)
    logq_token = tl.load(sampler_token_logps_ptr + row).to(tl.float32)
    advantage = tl.load(advantages_ptr + row).to(tl.float32)
    positive = advantage >= 0

    zero = tl.zeros_like(lse)
    log_ratio = tl.minimum(tl.maximum(logp_token - logq_token, -20.0), 20.0)
    token_weight = tl.exp(
        _log_mass(log_ratio, zero, positive, LOG_LOW, LOG_HIGH, HAS_CORRECTION, HAS_LOW, HAS_HIGH, CLAMP, PPO)
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
            _log_mass(zero, tl.log(rho), positive, LOG_LOW, LOG_HIGH, HAS_CORRECTION, HAS_LOW, HAS_HIGH, CLAMP, PPO)
        )
        mass = tl.exp(
            _log_mass(logp_head, logq_head, positive, LOG_LOW, LOG_HIGH, HAS_CORRECTION, HAS_LOW, HAS_HIGH, CLAMP, PPO)
        )
        residual = tl.where(head_valid, mass - alpha * p_head, 0.0)
        # The sampler's tail is modeled as rho * p, so its centering term alpha * E_p[log p] = -alpha * entropy has
        # zero gradient: it only makes the loss value fully centered.
        score = score - tl.sum(residual * logp_head) + alpha * entropy
    else:
        residual = tl.zeros((BLOCK_HEAD,), dtype=tl.float32)

    tl.store(loss_ptr + row, -advantage * score)
    tl.store(entropy_ptr + row, entropy)
    tl.store(kl_head_ptr + row, kl_head)
    tl.store(masked_ptr + row, (token_weight == 0).to(tl.float32))

    if HAS_GRADIENTS:
        # With g_j = d loss / d logp_j for the k + 1 gathered ids, d loss / d logits = scatter(g) - sum(g) * softmax.
        g_token = -advantage * token_weight * grad_scale
        g_head = advantage * residual * grad_scale
        g_sum = g_token + tl.sum(g_head)
        # Head ids are distinct but the sampled token can be one of them, so fold it into that head slot to avoid
        # two stores to the same address.
        token_in_head = head_valid & (head_ids == token)
        g_head += tl.where(token_in_head, g_token, 0.0)
        g_token = tl.where(tl.sum(token_in_head.to(tl.int32)) > 0, 0.0, g_token)

        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            valid = offsets < n_cols
            z = tl.load(X_ptr + offsets, mask=valid, other=float("-inf")).to(tl.float32) * inv_temperature
            grad = -g_sum * tl.exp(z - lse) * inv_temperature
            tl.store(X_ptr + offsets, grad.to(X_ptr.dtype.element_ty), mask=valid)
        # The sparse stores below overwrite slots written by other threads in the dense loop above.
        tl.debug_barrier()
        head_grad = (g_head - g_sum * p_head) * inv_temperature
        tl.store(X_ptr + head_ids, head_grad.to(X_ptr.dtype.element_ty), mask=head_valid)
        token_grad = (g_token - g_sum * tl.exp(logp_token)) * inv_temperature
        tl.store(X_ptr + token, token_grad.to(X_ptr.dtype.element_ty), mask=g_token != 0)


def _replicate(param):
    """
    Gathers a sharded DTensor parameter (e.g. an FSDP2 lm_head) into a local tensor whose gradient is reduce-scattered
    as an average over ranks. DTensor.full_tensor() would instead assume the gradient is identical on every rank and
    keep only each rank's local slice of it.
    """
    if DTensor is None or not isinstance(param, DTensor):
        return param
    n_dims = param.device_mesh.ndim
    return param.redistribute(placements=[Replicate()] * n_dims).to_local(grad_placements=[Partial("avg")] * n_dims)


def _loss_normalizer(mask, num_items_in_batch):
    """
    Tokens / world_size, so that DDP / FSDP gradient averaging gives the global token mean (as in DAPO / GRPO).
    num_items_in_batch counts completion tokens over all processes and grad-accum micro-batches; without it, the
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
    return normalizer.clamp_min(1.0) / world_size


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
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0]
    n_head = ids.shape[1] - 1

    if chunk_size is None:
        chunk_size = max(triton.next_power_of_2(triton.cdiv(BT, triton.cdiv(V, H))), MIN_CHUNK_SIZE)
    chunk_size = max(min(chunk_size, BT), 1)

    loss_1d = torch.empty(BT, dtype=torch.float32, device=device)
    entropy_1d = torch.empty(BT, dtype=torch.float32, device=device)
    kl_head_1d = torch.empty(BT, dtype=torch.float32, device=device)
    masked_1d = torch.empty(BT, dtype=torch.float32, device=device)
    grad_input = torch.empty_like(_input) if compute_grad_input else None
    # the first chunk overwrites grad_weight, so it only needs zeroing when there are no active tokens
    grad_weight = (torch.empty_like(weight) if BT > 0 else torch.zeros_like(weight)) if compute_grad_weight else None
    grad_bias = torch.zeros(V, dtype=torch.float32, device=device) if compute_grad_bias else None

    c = correction
    correction_flags = dict(
        LOG_LOW=math.log(c.low) if c is not None and c.low > 0 else 0.0,
        LOG_HIGH=math.log(c.high) if c is not None and c.high < math.inf else 0.0,
        HAS_CORRECTION=c is not None,
        HAS_LOW=c is not None and c.low > 0,
        HAS_HIGH=c is not None and c.high < math.inf,
        CLAMP=c is not None and c.outside == "clamp",
        PPO=c is not None and c.sign == "ppo",
    )

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        input_chunk = _input[start:end]
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias

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
            1.0 / normalizer,
            eps,
            SCORE_CENTER=score_center,
            HAS_GRADIENTS=compute_grad_input or compute_grad_weight or compute_grad_bias,
            BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(V)),
            BLOCK_HEAD=triton.next_power_of_2(max(n_head, 1)),
            num_warps=32 if not is_hip() else 16,
            **correction_flags,
        )
        grad_logits_chunk = logits_chunk

        if compute_grad_input:
            torch.mm(grad_logits_chunk, weight.to(grad_logits_chunk.dtype), out=grad_input[start:end])
        if compute_grad_weight:
            # the first chunk writes grad_weight directly instead of zero-filling a weight-sized buffer first
            input_chunk = input_chunk.to(grad_logits_chunk.dtype)
            if start == 0:
                torch.mm(grad_logits_chunk.t(), input_chunk, out=grad_weight)
            else:
                torch.addmm(grad_weight, grad_logits_chunk.t(), input_chunk, out=grad_weight)
        if compute_grad_bias:
            grad_bias.add_(grad_logits_chunk.sum(dim=0, dtype=torch.float32))
        del logits_chunk, grad_logits_chunk, input_chunk

    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None
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
        Fusing the last linear layer with the score centering loss.
            Reference: https://arxiv.org/abs/2609.20807, https://github.com/martin-marek/score-centering

        REINFORCE with an optional token-level IS correction, minus the expected score under the effective sampling
        mass q * w. The sampler only provides its top-k logprobs, so its tail is modeled as rho * p. The gradient then
        only depends on the logprobs of the sampled token and the k head tokens, which lets us compute it in the
        forward pass chunk by chunk, as in the fused linear cross entropy, without materializing the full logits.

        _input: (B, T, H) hidden states, already aligned with selected_token_ids.
        weight: (V, H). bias: (V,)
        selected_token_ids: (B, T) sampled tokens.
        attention_mask: (B, T), nonzero on completion tokens; only these go through the lm_head.
        advantages: (B,) or (B, T)
        sampler_token_logps: (B, T) sampler logprob of the sampled token, also when it is outside the head.
        sampler_topk_ids, sampler_topk_logps: (B, T, k) sampler head, not renormalized. k = V gives exact centering.
        score_center (bool): subtract the expected score; False gives plain (IS-corrected) REINFORCE.
        correction (ISCorrection): token-level IS correction, None for no correction.
        temperature (float): sampling temperature the logits are divided by.
        eps (float): floor on the trainer's tail mass.
        num_items_in_batch: completion tokens over all processes and grad-accum micro-batches.
        chunk_size (int): rows of logits materialized at once. Default: at least MIN_CHUNK_SIZE.

        Returns the loss and the masked means of the entropy, the head KL(q || p) and the fraction of tokens with
        w = 0.
        """
        if correction is not None:
            correction.check()
        mask = attention_mask.bool()
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        ids = torch.cat([selected_token_ids.unsqueeze(-1), sampler_topk_ids], dim=-1)[mask]
        normalizer = _loss_normalizer(mask, num_items_in_batch)

        loss_1d, entropy_1d, kl_head_1d, masked_1d, grad_input, grad_weight, grad_bias = (
            fused_linear_score_centering_forward(
                _input[mask],
                weight,
                ids,
                advantages.expand_as(mask)[mask].float(),
                sampler_token_logps[mask].float(),
                sampler_topk_logps[mask].float(),
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
            grad_input = torch.zeros_like(_input).index_put_((mask,), grad_input)
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)

        n_tokens = mask.sum().clamp_min(1)
        loss = loss_1d.sum() / normalizer
        entropy = entropy_1d.sum() / n_tokens
        kl_head = kl_head_1d.sum() / n_tokens
        frac_masked = masked_1d.sum() / n_tokens
        ctx.mark_non_differentiable(entropy, kl_head, frac_masked)
        return loss, entropy, kl_head, frac_masked

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, *grad_metrics):
        grads = ctx.saved_tensors
        # If the loss is the last layer, grad_output is 1.0. Skip the mul to save time
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
