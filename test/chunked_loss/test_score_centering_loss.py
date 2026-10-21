import math
import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from liger_kernel.chunked_loss import ISCorrection
from liger_kernel.chunked_loss import LigerFusedLinearScoreCenteringLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_score_centering
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

set_seed()

# Token-level points of the reference implementation's rl.is.* grid (README table of the score centering repo).
CORRECTIONS = {
    "pg": None,
    "naive_is": ISCorrection(),
    "tis": ISCorrection(high=2.0, outside="clamp"),
    "mis": ISCorrection(low=0.5, high=5.0),
    "ppo": ISCorrection(low=0.8, high=1.2, sign="ppo"),
    "dapo": ISCorrection(low=0.8, high=1.28, sign="ppo"),
    "tis_neg": ISCorrection(high=1.0, outside="clamp", sign="neg"),
    "mis_pos": ISCorrection(low=0.9, high=1.1, sign="pos"),
}


def ref_log_mass(log_p, log_q, positive, c):
    """Port of losses.log_mass from https://github.com/martin-marek/score-centering (stat=ratio)."""
    log_p, log_q = torch.broadcast_tensors(torch.as_tensor(log_p, device=device), torch.as_tensor(log_q, device=device))
    if c is None:
        return log_q
    lo = log_q + math.log(c.low) if c.low > 0 else torch.full_like(log_q, -math.inf)
    hi = log_q + math.log(c.high) if c.high < math.inf else torch.full_like(log_q, math.inf)
    if c.sign == "ppo":
        in_band = torch.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + math.log(3.0)))
    else:
        in_band = (log_p >= lo) & (log_p <= hi)
    log_m = torch.where(in_band, log_p, torch.clamp(log_p, lo, hi) if c.outside == "clamp" else -math.inf)
    if c.sign == "neg":
        log_m = torch.where(positive, log_q, log_m)
    if c.sign == "pos":
        log_m = torch.where(positive, log_m, log_q)
    return log_m


def ref_loss_fn_rl(logits, batch, correction, score_center):
    """Port of the token-level path of losses.loss_fn_rl, on full [B, T, V] logits.

    Inputs are already aligned with the logits (no roll). With batch["sampling_vocab_logprobs"] it is the exact,
    full-distribution score centering; with batch["center_ids"] the top-k head with a rho * p tail.
    """
    mask = batch["mask"].float()
    positive = (batch["advantages"] >= 0)[:, None]
    logp_vocab = logits.float().log_softmax(-1)
    logp = logp_vocab.gather(-1, batch["targets"][..., None])[..., 0]
    logq = batch["sampling_token_logprobs"]
    w = ref_log_mass((logp - logq).clamp(-20.0, 20.0), 0.0, positive, correction).exp().detach()
    plogp = (logp_vocab.exp() * logp_vocab).sum(-1).detach()
    score = w * logp
    if score_center and "sampling_vocab_logprobs" in batch:
        mass = ref_log_mass(logp_vocab, batch["sampling_vocab_logprobs"], positive[..., None], correction).exp()
        score = score - (mass.detach() * logp_vocab).sum(-1)
    kl_head = None
    if "center_ids" in batch:
        logp_head, logq_head = logp_vocab.gather(-1, batch["center_ids"]), batch["center_logprobs"]
        p_head, q_head = logp_head.exp().detach(), logq_head.exp()
        q_tail = 1.0 - q_head.sum(-1)
        rho = q_tail.clamp_min(0.0) / (1.0 - p_head.sum(-1)).clamp_min(1e-6)
        kl_head = ((q_head * (logq_head - logp_head)).sum(-1) + q_tail * rho.clamp_min(1e-9).log()).detach()
        if score_center:
            alpha = ref_log_mass(0.0, rho.log(), positive, correction).exp()
            mass = ref_log_mass(logp_head, logq_head, positive[..., None], correction).exp()
            residual = mass - alpha[..., None] * p_head
            score = score - ((residual.detach() * logp_head).sum(-1) + (alpha * plogp).detach())
    n_tokens = mask.sum().clamp_min(1)
    loss = -(batch["advantages"] * (score * mask).sum(-1)).sum() / n_tokens
    metrics = {"entropy": -(plogp * mask).sum() / n_tokens, "frac_masked": ((w == 0) * mask).sum() / n_tokens}
    if kl_head is not None:
        metrics["kl_head"] = (kl_head * mask).sum() / n_tokens
    return loss, metrics


def make_batch(lin, B, T, k, temperature=1.0, noise=0.5):
    """A sampler that is a perturbed copy of the trainer, as under a train / inference mismatch."""
    H, V = lin.in_features, lin.out_features
    x = torch.randn(B, T, H, device=device, dtype=lin.weight.dtype)
    with torch.no_grad():
        trainer_logits = lin(x).float() / temperature
    sampler_logp = (trainer_logits + noise * torch.randn(B, T, V, device=device)).log_softmax(-1)
    targets = torch.distributions.Categorical(logits=sampler_logp).sample()
    center_logprobs, center_ids = sampler_logp.topk(k, dim=-1)
    mask = torch.ones(B, T, device=device, dtype=torch.long)
    mask[:, : T // 4] = 0  # prompt tokens
    mask[0, -3:] = 0  # padding
    batch = {
        "targets": targets,
        "mask": mask,
        "advantages": torch.randn(B, device=device),
        "sampling_token_logprobs": sampler_logp.gather(-1, targets[..., None])[..., 0],
        "sampling_vocab_logprobs": sampler_logp,
        "center_ids": center_ids,
        "center_logprobs": center_logprobs,
    }
    return x, batch


def liger_args(batch, k=None):
    ids, logps = batch["center_ids"], batch["center_logprobs"]
    if k is not None:
        ids, logps = ids[..., :k], logps[..., :k]
    return (batch["targets"], batch["mask"], batch["advantages"], batch["sampling_token_logprobs"], ids, logps)


@pytest.mark.parametrize("B, T, H, V, k", [(4, 32, 64, 300, 8), (2, 47, 31, 129, 16)])
@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float32, 1e-5, 1e-4), (torch.bfloat16, 5e-2, 5e-1)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("name", CORRECTIONS)
@pytest.mark.parametrize("score_center", [True, False])
def test_correctness(B, T, H, V, k, dtype, atol, rtol, bias, name, score_center):
    correction = CORRECTIONS[name]
    lin = torch.nn.Linear(H, V, bias=bias, dtype=dtype, device=device)
    liger_lin = torch.nn.Linear(H, V, bias=bias, dtype=dtype, device=device)
    liger_lin.load_state_dict(lin.state_dict())
    x, batch = make_batch(lin, B, T, k)
    del batch["sampling_vocab_logprobs"]  # the top-k path
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)

    loss1, metrics1 = ref_loss_fn_rl(lin(x1), batch, correction, score_center)
    liger_loss = LigerFusedLinearScoreCenteringLoss(
        score_center=score_center,
        correction=correction,
        vocab_chunk_size=64,  # several vocab and sequence chunks even at test sizes
        seq_chunk_size=16,
    )
    loss2, metrics2 = liger_loss(x2, liger_lin.weight, *liger_args(batch), liger_lin.bias)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
    for key in metrics1:
        assert_verbose_allclose(metrics1[key], metrics2[key], atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(lin.weight.grad, liger_lin.weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(lin.bias.grad, liger_lin.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("name", ["pg", "tis", "mis", "ppo"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_full_vocab_head_is_exact_centering(name, temperature):
    """With k = V the tail is empty, so top-k centering equals exact centering with the full sampler logprobs."""
    B, T, H, V = 2, 24, 32, 100
    correction = CORRECTIONS[name]
    lin = torch.nn.Linear(H, V, device=device)
    x, batch = make_batch(lin, B, T, V, temperature)
    del batch["center_ids"], batch["center_logprobs"]
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)

    loss1, _ = ref_loss_fn_rl(lin(x1) / temperature, batch, correction, score_center=True)
    sampler_logp = batch["sampling_vocab_logprobs"]
    ids = sampler_logp.argsort(-1, descending=True)
    args = (*liger_args(batch | {"center_ids": ids, "center_logprobs": sampler_logp.gather(-1, ids)}),)
    loss2, _ = liger_fused_linear_score_centering(
        x2, lin.weight, *args, lin.bias, True, correction, temperature, 1e-6, None, 32, 16
    )
    assert_verbose_allclose(loss1, loss2, atol=1e-5, rtol=1e-4)
    loss1.backward()
    loss2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-4)


def test_frozen_weight_and_num_items_in_batch():
    B, T, H, V, k = 2, 16, 32, 100, 4
    lin = torch.nn.Linear(H, V, bias=False, device=device).requires_grad_(False)
    x, batch = make_batch(lin, B, T, k)
    x.requires_grad_(True)
    loss, metrics = liger_fused_linear_score_centering(x, lin.weight, *liger_args(batch))
    loss.backward()
    assert x.grad is not None and lin.weight.grad is None
    assert metrics["entropy"] > 0

    n_tokens = batch["mask"].sum()
    loss_scaled, _ = liger_fused_linear_score_centering(
        x, lin.weight, *liger_args(batch), None, True, None, 1.0, 1e-6, 2 * n_tokens
    )
    assert_verbose_allclose(loss_scaled, loss / 2, atol=1e-6, rtol=1e-5)


# Score centering cancels drift: with a constant advantage, the expected (corrected) REINFORCE gradient under a
# mismatched sampler q is zero. Port of tests/test_loss.py from the score centering repo.
TRAINER_LOGITS = torch.tensor([0.5, -0.2, 1.1])
SAMPLER_LOGP = (TRAINER_LOGITS + torch.tensor([0.1, 0.3, -0.2])).log_softmax(-1)
DRIFT_CORRECTIONS = {
    "pg": None,
    "tis": ISCorrection(high=2.0, outside="clamp"),
    "mis": ISCorrection(low=0.8, high=1.1),  # this band drops some of the tokens
}


def expected_gradient(correction, k):
    """k = V - 1: the tail is one token, so modeling it as rho * p is exact; k = V: the full sampler distribution."""
    weight = torch.eye(3, device=device)  # hidden states are the logits
    head_logp, head_ids = SAMPLER_LOGP.topk(k)
    grad = torch.zeros(3, device=device)
    for token in range(3):
        logits = TRAINER_LOGITS.to(device).view(1, 1, 3).clone().requires_grad_(True)
        loss, _ = liger_fused_linear_score_centering(
            logits,
            weight,
            torch.tensor([[token]], device=device),
            torch.ones(1, 1, device=device),
            torch.ones(1, device=device),
            SAMPLER_LOGP[token].to(device).view(1, 1),
            head_ids.to(device).view(1, 1, k),
            head_logp.to(device).view(1, 1, k),
            None,
            k > 0,
            correction,
        )
        loss.backward()
        grad += SAMPLER_LOGP[token].exp().item() * logits.grad.view(3)
    return grad


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("name", DRIFT_CORRECTIONS)
def test_score_centering_cancels_drift(name, k):
    assert expected_gradient(DRIFT_CORRECTIONS[name], k).abs().max() < 1e-6


def test_drift_without_centering():  # plain REINFORCE does drift
    assert expected_gradient(None, k=0).abs().max() > 1e-3


@pytest.mark.skipif(device == "cpu", reason="peak memory is measured on an accelerator")
def test_peak_memory_does_not_scale_with_tokens_times_vocab():
    B, T, H, V, k = 2, 512, 256, 32768, 16
    lin = torch.nn.Linear(H, V, bias=False, device=device, dtype=torch.bfloat16)
    x, batch = make_batch(lin, B, T, k)
    del batch["sampling_vocab_logprobs"]
    x.requires_grad_(True)
    loss_fn = LigerFusedLinearScoreCenteringLoss(seq_chunk_size=256)
    device_module = getattr(torch, device)
    device_module.synchronize()
    device_module.empty_cache()
    device_module.reset_peak_memory_stats()
    base = device_module.memory_allocated()
    loss_fn(x, lin.weight, *liger_args(batch))[0].backward()
    device_module.synchronize()
    peak = device_module.max_memory_allocated() - base
    # The weight gradient is unavoidable; one float32 [N, V] logits tensor alone would be 2x the allowance.
    allowance = lin.weight.numel() * lin.weight.element_size() + B * T * V * 4 // 2
    assert peak < allowance, f"peak {peak / 2**20:.1f} MiB >= {allowance / 2**20:.1f} MiB"


class _Policy(torch.nn.Module):
    def __init__(self, H, V):
        super().__init__()
        self.body = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias=False)
        self.loss = LigerFusedLinearScoreCenteringLoss(
            correction=CORRECTIONS["tis"], vocab_chunk_size=16, seq_chunk_size=8
        )

    def forward(self, x, *args):
        return self.loss(self.body(x), self.lm_head.weight, *args)[0]


def _policy_batch(B, T, H, V, k):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, H, generator=g)
    sampler_logp = torch.randn(B, T, V, generator=g).log_softmax(-1)
    targets = torch.multinomial(sampler_logp.exp().view(-1, V), 1, generator=g).view(B, T)
    head_logp, head_ids = sampler_logp.topk(k, -1)
    mask = torch.ones(B, T)
    mask[:, :3] = 0
    mask[0, -4:] = 0  # ranks hold different numbers of completion tokens
    args = (targets, mask, torch.randn(B, generator=g), sampler_logp.gather(-1, targets[..., None])[..., 0])
    return x, (*args, head_ids, head_logp)


def _fsdp2_worker(rank, world_size, file_name, loss_in_forward, shape, ref_grads):
    from torch.distributed.fsdp import fully_shard

    torch.distributed.init_process_group("gloo", init_method=f"file://{file_name}", rank=rank, world_size=world_size)
    mesh = torch.distributed.device_mesh.init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    model = _Policy(*shape[2:4])
    if not loss_in_forward:
        fully_shard(model.lm_head, mesh=mesh)
        fully_shard(model.body, mesh=mesh)
    fully_shard(model, mesh=mesh)
    x, args = _policy_batch(*shape)
    rows = slice(rank * shape[0] // world_size, (rank + 1) * shape[0] // world_size)
    x, args = x[rows], tuple(a[rows] for a in args)
    if loss_in_forward:
        loss = model(x, *args)
    else:  # the sharded lm_head.weight DTensor goes straight into the loss
        loss = model.loss(model.body(x), model.lm_head.weight, *args)[0]
    loss.backward()
    torch.testing.assert_close(model.lm_head.weight.grad.full_tensor(), ref_grads[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.body.weight.grad.full_tensor(), ref_grads[1], atol=1e-6, rtol=1e-5)
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("loss_in_forward", [True, False])
def test_fsdp2_matches_single_process(loss_in_forward):
    """Data-parallel FSDP2 on CPU: the gradients equal those of the global token-mean loss on one process."""
    pytest.importorskip("torch.distributed.fsdp", reason="needs FSDP2")
    shape, world_size = (4, 12, 16, 64, 8), 2
    torch.manual_seed(0)
    model = _Policy(*shape[2:4])
    x, args = _policy_batch(*shape)
    model(x, *args).backward()
    ref_grads = (model.lm_head.weight.grad, model.body.weight.grad)
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _fsdp2_worker,
            args=(world_size, f.name, loss_in_forward, shape, ref_grads),
            nprocs=world_size,
            join=True,
        )
