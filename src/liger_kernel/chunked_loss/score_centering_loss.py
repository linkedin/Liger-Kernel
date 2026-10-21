"""Score centering loss (https://arxiv.org/abs/2609.20807) fused with the lm_head.

REINFORCE with an optional truncated (TIS) or masked (MIS) importance-sampling correction, minus the expected
score under the effective sampling mass q * w(p/q). The sampler only ships its top-k head (ids and logprobs); its
tail is modeled as rho * p, with rho matching the tail mass. Every ratio correction is constant on that tail, so its
centering term is alpha * E_p[log p], whose gradient is zero. The gradient therefore only touches the logprobs of the
sampled token and the k head tokens, which lets us stream over (sequence x vocab) chunks and never materialize the
full [N, V] logits.
"""

import math

from typing import NamedTuple
from typing import Optional

import torch

from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Partial
    from torch.distributed.tensor import Replicate
except ImportError:  # torch without DTensor
    DTensor = None

_VOCAB_CHUNK_SIZE = 4096
_SEQ_CHUNK_SIZE = 2048


def _gather_logprob_forward(hidden, weight, ids, bias, temperature, vocab_chunk_size, seq_chunk_size):
    """Log-probabilities of ids [N, K] and the entropy of each row, with an online logsumexp over vocab chunks."""
    device = hidden.device
    n_rows, n_ids = ids.shape
    vocab_size = weight.shape[0]
    inv_t = 1.0 / temperature

    logprobs = torch.empty((n_rows, n_ids), device=device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=device, dtype=torch.float32)
    entropy = torch.empty((n_rows,), device=device, dtype=torch.float32)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        n_chunk = seq_end - seq_start
        hidden_chunk = hidden[seq_start:seq_end]
        ids_chunk = ids[seq_start:seq_end]

        running_max = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        sum_exp_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)  # sum exp(z - m) * z
        id_logits = torch.zeros((n_chunk, n_ids), device=device, dtype=torch.float32)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            logits_chunk = (hidden_chunk @ weight[vocab_start:vocab_end].to(hidden.dtype).t()).float()
            if bias is not None:
                logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
            logits_chunk.mul_(inv_t)

            new_max = torch.maximum(running_max, logits_chunk.amax(dim=-1))
            rescale = torch.exp(running_max - new_max)
            chunk_exp = torch.exp(logits_chunk - new_max.unsqueeze(-1))
            sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
            sum_exp_logit = sum_exp_logit * rescale + (chunk_exp * logits_chunk).sum(dim=-1)
            running_max = new_max

            in_chunk = (ids_chunk >= vocab_start) & (ids_chunk < vocab_end)
            local_ids = (ids_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1)
            id_logits += logits_chunk.gather(1, local_ids) * in_chunk

        log_z_chunk = running_max + torch.log(sum_exp)
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = id_logits - log_z_chunk.unsqueeze(-1)
        entropy[seq_start:seq_end] = log_z_chunk - sum_exp_logit / sum_exp

    return logprobs, log_z, entropy


def _gather_logprob_backward(
    hidden,
    weight,
    ids,
    bias,
    log_z,
    grad_logprobs,
    temperature,
    vocab_chunk_size,
    seq_chunk_size,
    compute_grad_weight,
    compute_grad_bias,
):
    """d logp_k / d z = onehot_k - p, so grad_logits = scatter(g) - (sum_k g_k) * p, recomputed per chunk.

    Vocab chunks are the outer loop so each [vocab_chunk, H] slice of the weight gradient is finished in float32 and
    written straight into a weight-dtype buffer, instead of keeping a float32 [V, H] accumulator alive.
    """
    inv_t = 1.0 / temperature
    n_rows = hidden.shape[0]
    vocab_size = weight.shape[0]

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    grad_weight = torch.empty_like(weight) if compute_grad_weight else None
    grad_bias = torch.empty_like(bias) if compute_grad_bias else None
    grad_logprobs = grad_logprobs.to(torch.float32)
    grad_sum = grad_logprobs.sum(dim=-1, keepdim=True)

    for vocab_start in range(0, vocab_size, vocab_chunk_size):
        vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
        weight_chunk = weight[vocab_start:vocab_end]
        grad_weight_chunk = torch.zeros(weight_chunk.shape, device=weight.device, dtype=torch.float32)
        grad_bias_chunk = torch.zeros((vocab_end - vocab_start,), device=weight.device, dtype=torch.float32)

        for seq_start in range(0, n_rows, seq_chunk_size):
            seq_end = min(seq_start + seq_chunk_size, n_rows)
            hidden_chunk = hidden[seq_start:seq_end]
            ids_chunk = ids[seq_start:seq_end]
            grad_chunk = grad_logprobs[seq_start:seq_end]

            logits_chunk = (hidden_chunk @ weight_chunk.to(hidden.dtype).t()).float()
            if bias is not None:
                logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
            logits_chunk.mul_(inv_t)

            grad_logits = torch.exp(logits_chunk - log_z[seq_start:seq_end].unsqueeze(-1))
            grad_logits.mul_(-grad_sum[seq_start:seq_end])
            in_chunk = (ids_chunk >= vocab_start) & (ids_chunk < vocab_end)
            local_ids = (ids_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1)
            grad_logits.scatter_add_(1, local_ids, grad_chunk * in_chunk)
            grad_logits.mul_(inv_t)

            grad_hidden[seq_start:seq_end].add_(grad_logits @ weight_chunk.float())
            if compute_grad_weight:
                grad_weight_chunk.add_(grad_logits.t() @ hidden_chunk.float())
            if compute_grad_bias:
                grad_bias_chunk.add_(grad_logits.sum(dim=0))

        if compute_grad_weight:
            grad_weight[vocab_start:vocab_end].copy_(grad_weight_chunk)
        if compute_grad_bias:
            grad_bias[vocab_start:vocab_end].copy_(grad_bias_chunk)

    return grad_hidden, grad_weight, grad_bias


class _ChunkedGatherLogProbFunction(torch.autograd.Function):
    """log_softmax(hidden @ weight.T + bias)[ids] for ids [N, K] without materializing [N, V].

    Also returns the per-row entropy (non-differentiable), which is free to accumulate in the forward pass.
    """

    @staticmethod
    def forward(ctx, hidden, weight, ids, bias, temperature, vocab_chunk_size, seq_chunk_size):
        logprobs, log_z, entropy = _gather_logprob_forward(
            hidden, weight, ids, bias, temperature, vocab_chunk_size, seq_chunk_size
        )
        ctx.has_bias = bias is not None
        ctx.save_for_backward(hidden, weight, ids, bias if ctx.has_bias else hidden.new_empty((0,)), log_z)
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.seq_chunk_size = seq_chunk_size
        ctx.mark_non_differentiable(entropy)
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, _grad_entropy):
        hidden, weight, ids, bias, log_z = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _gather_logprob_backward(
            hidden,
            weight,
            ids,
            bias if ctx.has_bias else None,
            log_z,
            grad_logprobs,
            ctx.temperature,
            ctx.vocab_chunk_size,
            ctx.seq_chunk_size,
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.has_bias and ctx.needs_input_grad[3],
        )
        return (
            grad_hidden.to(hidden.dtype),
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
        )


DUAL_CLIP = 3.0  # PPO's ratio bound for negative advantages (arXiv:1912.09729)


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
        assert self.sign in ("both", "neg", "pos", "ppo"), "sign must be one of both | neg | pos | ppo"
        return self


def _log_mass(log_p, log_q, positive, c):
    """log(q * w(p/q)): a token's effective sampling mass under correction c (-inf where w = 0, log q for none).

    The same function gives the IS weight w(r) as the mass at p = r, q = 1, the centering mass q * w as the mass
    at (p, q), and the top-k tail factor rho * w(1 / rho) as the mass at p = 1, q = rho.
    """
    ref = log_p if torch.is_tensor(log_p) else log_q
    log_p, log_q = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=ref.dtype, device=ref.device) for x in (log_p, log_q))
    )
    if c is None:
        return log_q
    lo = log_q + math.log(c.low) if c.low > 0 else torch.full_like(log_q, -math.inf)
    hi = log_q + math.log(c.high) if c.high < math.inf else torch.full_like(log_q, math.inf)
    if c.sign == "ppo":
        in_band = torch.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + math.log(DUAL_CLIP)))
    else:
        in_band = (log_p >= lo) & (log_p <= hi)
    outside = torch.clamp(log_p, lo, hi) if c.outside == "clamp" else torch.full_like(log_p, -math.inf)
    log_m = torch.where(in_band, log_p, outside)
    if c.sign == "neg":
        log_m = torch.where(positive, log_q, log_m)
    if c.sign == "pos":
        log_m = torch.where(positive, log_m, log_q)
    return log_m


def score_centering_loss_from_logps(
    token_logps,
    head_logps,
    entropy,
    sampler_token_logps,
    sampler_topk_logps,
    advantages,
    score_center=True,
    correction=None,
    eps=1e-6,
):
    """Per-token loss from trainer logprobs of the sampled token [N] and of the sampler's top-k head [N, k]."""
    positive = advantages >= 0
    token_logps_d, head_logps_d = token_logps.detach(), head_logps.detach()
    log_ratio = (token_logps_d - sampler_token_logps).clamp(-20.0, 20.0)
    token_weight = _log_mass(log_ratio, 0.0, positive, correction).exp()
    score = token_weight * token_logps

    p_head, q_head = head_logps_d.exp(), sampler_topk_logps.exp()
    q_tail = 1.0 - q_head.sum(-1)
    rho = q_tail.clamp_min(0.0) / (1.0 - p_head.sum(-1)).clamp_min(eps)
    if score_center:
        alpha = _log_mass(0.0, rho.log(), positive, correction).exp()
        mass = _log_mass(head_logps_d, sampler_topk_logps, positive.unsqueeze(-1), correction).exp()
        residual = mass - alpha.unsqueeze(-1) * p_head
        # alpha * E_p[log p] = -alpha * entropy has zero gradient; it only makes the loss value fully centered
        score = score - (residual * head_logps).sum(-1) + alpha * entropy

    loss = -advantages * score
    kl_head = (q_head * (sampler_topk_logps - head_logps_d)).sum(-1) + q_tail * rho.clamp_min(1e-9).log()
    return loss, {"kl_head": kl_head, "masked": token_weight == 0}


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


class LigerFusedLinearScoreCenteringFunction:
    @staticmethod
    def apply(
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
        vocab_chunk_size=_VOCAB_CHUNK_SIZE,
        seq_chunk_size=_SEQ_CHUNK_SIZE,
    ):
        """
        Args:
            _input: [B, T, H] hidden states aligned with selected_token_ids (already shifted).
            weight: [V, H] lm_head weight. bias: optional [V]. Either may be a sharded DTensor (FSDP2), see _replicate.
            selected_token_ids: [B, T] sampled tokens.
            attention_mask: [B, T], nonzero on completion tokens only.
            advantages: [B] or [B, T].
            sampler_token_logps: [B, T] sampler logprob of the sampled token, kept even when it is outside the head.
            sampler_topk_ids, sampler_topk_logps: [B, T, k] sampler head, not renormalized. k = V gives exact
                centering with the full sampler distribution.
            correction: optional ISCorrection, e.g. ISCorrection(high=2.0, outside="clamp") for TIS.
            num_items_in_batch: completion tokens over all processes and grad-accum micro-batches. If None, the
                current micro-batch's tokens are all-reduced over processes. Either way the loss is divided by
                (tokens / world_size), so that DDP / FSDP gradient averaging gives the global token mean.
        Returns:
            (loss, metrics) where metrics holds the masked means of entropy, kl_head and frac_masked.
        """
        if correction is not None:
            correction.check()
        weight, bias = _replicate(weight), _replicate(bias)
        mask = attention_mask.bool()
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        advantages = advantages.expand_as(mask)[mask].float()

        # Only completion tokens go through the lm_head.
        ids = torch.cat([selected_token_ids.unsqueeze(-1), sampler_topk_ids], dim=-1)[mask].long()
        logps, entropy = _ChunkedGatherLogProbFunction.apply(
            _input[mask], weight, ids, bias, temperature, vocab_chunk_size, seq_chunk_size
        )
        token_loss, metrics = score_centering_loss_from_logps(
            logps[:, 0],
            logps[:, 1:],
            entropy,
            sampler_token_logps[mask].float(),
            sampler_topk_logps[mask].float(),
            advantages,
            score_center=score_center,
            correction=correction,
            eps=eps,
        )

        # Global token mean under data parallelism, as in DAPO / GRPO (see _compute_dapo_normalizer)
        loss = token_loss.sum() / LigerFusedLinearPPOBase._compute_dapo_normalizer(mask, num_items_in_batch)
        n_tokens = mask.sum().clamp_min(1)
        metrics = {
            "entropy": entropy.sum() / n_tokens,
            "kl_head": metrics["kl_head"].sum() / n_tokens,
            "frac_masked": metrics["masked"].float().sum() / n_tokens,
        }
        return loss, metrics


class LigerFusedLinearScoreCenteringLoss(torch.nn.Module):
    """Fused lm_head + score centering loss, optionally on top of a token-level IS correction (TIS, MIS, PPO, ...)."""

    def __init__(
        self,
        score_center: bool = True,
        correction: Optional[ISCorrection] = None,
        temperature: float = 1.0,
        eps: float = 1e-6,
        vocab_chunk_size: int = _VOCAB_CHUNK_SIZE,
        seq_chunk_size: int = _SEQ_CHUNK_SIZE,
    ):
        """
        Args:
            score_center (bool): Subtract the expected score under q * w. False gives plain (IS-corrected) REINFORCE.
            correction (ISCorrection, optional): Token-level IS correction; None is plain REINFORCE.
            temperature (float): Sampling temperature the logits are divided by.
            eps (float): Floor on the trainer and sampler tail masses.
            vocab_chunk_size, seq_chunk_size (int): Peak temporary memory is seq_chunk_size x vocab_chunk_size.
        """
        super().__init__()
        assert temperature != 0, "Temperature cannot be 0."
        self.score_center = score_center
        self.correction = correction.check() if correction is not None else None
        self.temperature = temperature
        self.eps = eps
        self.vocab_chunk_size = vocab_chunk_size
        self.seq_chunk_size = seq_chunk_size

    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        sampler_token_logps,
        sampler_topk_ids,
        sampler_topk_logps,
        bias=None,
        num_items_in_batch=None,
    ):
        return LigerFusedLinearScoreCenteringFunction.apply(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            sampler_token_logps,
            sampler_topk_ids,
            sampler_topk_logps,
            bias,
            self.score_center,
            self.correction,
            self.temperature,
            self.eps,
            num_items_in_batch,
            self.vocab_chunk_size,
            self.seq_chunk_size,
        )
