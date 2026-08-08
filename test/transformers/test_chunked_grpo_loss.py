"""Tests for the chunked Triton GRPO loss (fused lm_head, logits never materialized).

Checks, at multiple context lengths and across loss configurations:
  1. Intermediates: per-token logp/lse vs an fp32 ground truth and vs the
     non-chunked Triton path's fused_selective_log_softmax.
  2. Per-token results (reduce=False): per_token_loss / per_token_kl /
     is_clipped vs the non-chunked triton_grpo_loss.
  3. End results (reduce=True): loss, metrics, grad_hidden and grad_weight vs
     three references: a plain torch implementation from logits (TRL-style,
     "non-chunked torch"), the non-chunked Triton kernel, and the chunked
     torch LigerFusedLinearGRPOLoss.
  4. Bitwise determinism across reruns (no atomics, fixed launch configs).
"""

import pytest
import torch

from test.utils import infer_device
from test.utils import set_seed

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from liger_kernel.ops.chunked_grpo_loss import chunked_selective_log_softmax_with_lse
from liger_kernel.ops.grpo_loss import fused_selective_log_softmax
from liger_kernel.transformers.chunked_grpo_loss import chunked_triton_grpo_loss
from liger_kernel.transformers.grpo_loss import _reduce_grpo_loss
from liger_kernel.transformers.grpo_loss import triton_grpo_loss

device = infer_device()

HIDDEN_SIZE = 2048
VOCAB_SIZE = 248320  # Qwen3.5-MoE vocab; divisible by the kernel's BN
ODD_VOCAB_SIZE = 50257  # gpt2 vocab; exercises the vocab-tail masking path


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


@torch.no_grad()
def fp32_logp_lse(hidden, weight, targets, temperature):
    """fp32 ground-truth selective log-softmax, row-chunked to bound memory."""
    n = hidden.shape[0]
    logp = torch.empty(n, dtype=torch.float32, device=hidden.device)
    lse = torch.empty(n, dtype=torch.float32, device=hidden.device)
    chunk = 8192
    w = weight.float()
    for i in range(0, n, chunk):
        logits = (hidden[i : i + chunk].float() @ w.t()) / temperature
        row_lse = torch.logsumexp(logits, dim=-1)
        tgt = logits.gather(-1, targets[i : i + chunk].unsqueeze(-1)).squeeze(-1)
        logp[i : i + chunk] = tgt - row_lse
        lse[i : i + chunk] = row_lse
    return logp, lse


def torch_grpo_loss_from_logits(
    logits,  # (B, L, V), already sliced to completion positions
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    mask,
    *,
    temperature,
    beta,
    eps_low,
    eps_high,
    loss_type,
    max_completion_length,
    importance_sampling_level,
    sapo_temperature_pos=1.0,
    sapo_temperature_neg=1.05,
    vllm_is_ratio=None,
    delta=None,
    use_bias_correction_kl=False,
    num_items_in_batch=None,
):
    """Plain torch reference (TRL _compute_loss math), written independently of
    the implementation under test."""
    logp = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    logp = logp.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

    old = old_logp.float() if old_logp is not None else logp.detach()
    log_ratio = logp - old
    if importance_sampling_level == "sequence":
        log_ratio = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
    coef_1 = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1).float()

    if loss_type == "cispo":
        per_token_loss = -torch.clamp(coef_1, max=eps_high).detach() * adv * logp
    elif loss_type == "sapo":
        temps = torch.where(adv > 0, sapo_temperature_pos, sapo_temperature_neg)
        per_token_loss = -torch.sigmoid(temps * (coef_1 - 1)) * 4 / temps * adv
    else:
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
        c1 = torch.clamp(coef_1, max=delta) if delta is not None else coef_1
        per_token_loss = -torch.min(c1 * adv, coef_2 * adv)

    if vllm_is_ratio is not None:
        per_token_loss = per_token_loss * vllm_is_ratio
    if beta != 0.0:
        kl = torch.exp(ref_logp.float() - logp) - (ref_logp.float() - logp) - 1
        if use_bias_correction_kl:
            kl = kl * coef_1
        per_token_loss = per_token_loss + beta * kl

    if per_token_loss.shape[1] == 1:
        per_token_loss = per_token_loss.expand_as(logp)
    return _reduce_grpo_loss(per_token_loss, mask, loss_type, max_completion_length, num_items_in_batch)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_inputs(batch, seq_len, vocab, *, seed=0, dtype=torch.bfloat16):
    set_seed(seed)
    # (B, L+1, H): logits-based paths use all L+1 positions and slice
    # internally; hidden-based paths take [:, :-1, :].
    hidden = (torch.randn(batch, seq_len + 1, HIDDEN_SIZE, device=device) * 0.02).to(dtype)
    weight = (torch.randn(vocab, HIDDEN_SIZE, device=device) * 0.02).to(dtype)
    completion_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    lengths = torch.randint(seq_len // 2, seq_len + 1, (batch,), device=device)
    mask = (torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    advantages = torch.randn(batch, device=device)

    logp_true, _ = fp32_logp_lse(
        hidden[:, :-1, :].reshape(-1, HIDDEN_SIZE).contiguous(), weight, completion_ids.reshape(-1), 1.0
    )
    logp_true = logp_true.view(batch, seq_len)
    # old: sizeable perturbation so clipping actually triggers; ref: small one
    old_logp = logp_true + torch.randn_like(logp_true) * 0.3
    ref_logp = logp_true + torch.randn_like(logp_true) * 0.1
    return {
        "hidden": hidden,
        "weight": weight,
        "completion_ids": completion_ids,
        "mask": mask,
        "advantages": advantages,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
    }


# their production config + variations exercising every supported branch
CONFIGS = {
    "dapo_seq_onpolicy": dict(loss_type="dapo", importance_sampling_level="sequence", beta=0.0, use_old=False),
    "dapo_seq_offpolicy_niib": dict(
        loss_type="dapo", importance_sampling_level="sequence", beta=0.0, use_old=True, use_niib=True
    ),
    "dapo_token_kl_biascorr": dict(
        loss_type="dapo",
        importance_sampling_level="token",
        beta=0.04,
        use_old=True,
        use_bias_correction_kl=True,
    ),
    "grpo_token_kl": dict(loss_type="grpo", importance_sampling_level="token", beta=0.04, use_old=True),
    "bnpo_token": dict(loss_type="bnpo", importance_sampling_level="token", beta=0.0, use_old=True),
    "dr_grpo_token": dict(
        loss_type="dr_grpo", importance_sampling_level="token", beta=0.0, use_old=True, use_max_len=True
    ),
    "cispo_token": dict(loss_type="cispo", importance_sampling_level="token", beta=0.0, use_old=True, eps_high=4.0),
    "sapo_token": dict(loss_type="sapo", importance_sampling_level="token", beta=0.0, use_old=True),
    "dapo_token_delta": dict(loss_type="dapo", importance_sampling_level="token", beta=0.0, use_old=True, delta=4.0),
    "dapo_seq_vllm_ratio": dict(
        loss_type="dapo", importance_sampling_level="sequence", beta=0.0, use_old=True, use_vllm_ratio=True
    ),
}


def config_kwargs(cfg, inputs, seq_len):
    kwargs = dict(
        temperature=1.0,
        beta=cfg["beta"],
        eps_low=cfg.get("eps_low", 0.2),
        eps_high=cfg.get("eps_high", 0.2),
        loss_type=cfg["loss_type"],
        max_completion_length=seq_len if cfg.get("use_max_len") else None,
        importance_sampling_level=cfg["importance_sampling_level"],
        delta=cfg.get("delta"),
        use_bias_correction_kl=cfg.get("use_bias_correction_kl", False),
        num_items_in_batch=inputs["mask"].sum() if cfg.get("use_niib") else None,
        vllm_is_ratio=(
            (1 + 0.05 * torch.randn(inputs["mask"].shape[0], 1, device=device)).clamp(0.8, 1.2)
            if cfg.get("use_vllm_ratio")
            else None
        ),
    )
    old = inputs["old_logp"] if cfg["use_old"] else None
    ref = inputs["ref_logp"] if cfg["beta"] != 0.0 else None
    return kwargs, old, ref


# ---------------------------------------------------------------------------
# 1. intermediates: logp / lse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, seq_len, vocab",
    [
        (4, 128, VOCAB_SIZE),
        (4, 1024, VOCAB_SIZE),
        (4, 4096, VOCAB_SIZE),
        (2, 16384, VOCAB_SIZE),
        (4, 512, ODD_VOCAB_SIZE),
        (3, 300, ODD_VOCAB_SIZE),  # ragged batch/seq/vocab all at once
    ],
)
@pytest.mark.parametrize("temperature", [1.0, 0.9])
def test_logp_lse_intermediates(batch, seq_len, vocab, temperature):
    inputs = make_inputs(batch, seq_len, vocab)
    hidden2d = inputs["hidden"][:, :-1, :].reshape(-1, HIDDEN_SIZE).contiguous()
    ids = inputs["completion_ids"].reshape(-1)

    logp, lse = chunked_selective_log_softmax_with_lse(hidden2d, inputs["weight"], ids, temperature)
    logp_ref, lse_ref = fp32_logp_lse(hidden2d, inputs["weight"], ids, temperature)

    # fp32 ground truth: only bf16-GEMM accumulation-order noise separates them
    assert torch.allclose(logp, logp_ref, atol=1e-2, rtol=1e-3), (
        f"logp vs fp32 ref: max diff {(logp - logp_ref).abs().max().item():.2e}"
    )
    assert torch.allclose(lse, lse_ref, atol=1e-2, rtol=1e-3), (
        f"lse vs fp32 ref: max diff {(lse - lse_ref).abs().max().item():.2e}"
    )

    # non-chunked Triton intermediate (from materialized bf16 logits)
    logits = inputs["hidden"] @ inputs["weight"].t()  # (B, L+1, V)
    logp_triton = fused_selective_log_softmax(logits, inputs["completion_ids"], temperature)
    assert torch.allclose(logp.view(batch, seq_len), logp_triton, atol=3e-2, rtol=1e-3), (
        f"logp vs non-chunked triton: max diff {(logp.view(batch, seq_len) - logp_triton).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 2. per-token results (reduce=False) vs non-chunked Triton
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_per_token_results_vs_triton(config_name):
    cfg = CONFIGS[config_name]
    batch, seq_len = 4, 1024
    inputs = make_inputs(batch, seq_len, VOCAB_SIZE)
    kwargs, old, ref = config_kwargs(cfg, inputs, seq_len)

    ptl_c, kl_c, clipped_c = chunked_triton_grpo_loss(
        inputs["hidden"][:, :-1, :].contiguous(),
        inputs["weight"],
        old,
        ref,
        inputs["completion_ids"],
        inputs["advantages"],
        inputs["mask"],
        reduce=False,
        **kwargs,
    )
    logits = inputs["hidden"] @ inputs["weight"].t()
    ptl_t, kl_t, clipped_t = triton_grpo_loss(
        logits,
        old,
        ref,
        inputs["completion_ids"],
        inputs["advantages"],
        inputs["mask"],
        inplace=False,
        reduce=False,
        **kwargs,
    )

    mask = inputs["mask"].bool()
    assert torch.allclose(ptl_c[mask], ptl_t[mask], atol=2e-2, rtol=2e-2), (
        f"per_token_loss: max diff {(ptl_c[mask] - ptl_t[mask]).abs().max().item():.2e}"
    )
    if cfg["beta"] != 0.0:
        assert torch.allclose(kl_c[mask], kl_t[mask], atol=2e-2, rtol=2e-2), (
            f"per_token_kl: max diff {(kl_c[mask] - kl_t[mask]).abs().max().item():.2e}"
        )
    # clipping indicators may flip at ratio~threshold boundaries due to bf16
    # rounding differences; require near-total agreement instead of equality
    mismatch = (clipped_c[mask].bool() != clipped_t[mask].bool()).float().mean().item()
    assert mismatch < 0.01, f"is_clipped mismatch fraction {mismatch:.4f}"


# ---------------------------------------------------------------------------
# 3. end results: loss, metrics, grads vs all three references
# ---------------------------------------------------------------------------


def run_chunked_triton(inputs, old, ref, kwargs):
    hidden, weight = inputs["hidden"], inputs["weight"]
    loss, metrics = chunked_triton_grpo_loss(
        hidden[:, :-1, :].contiguous(),
        weight,
        old,
        ref,
        inputs["completion_ids"],
        inputs["advantages"],
        inputs["mask"],
        reduce=True,
        **kwargs,
    )
    return loss, metrics


def run_triton(inputs, old, ref, kwargs):
    logits = inputs["hidden"] @ inputs["weight"].t()
    loss, metrics = triton_grpo_loss(
        logits,
        old,
        ref,
        inputs["completion_ids"],
        inputs["advantages"],
        inputs["mask"],
        inplace=False,
        reduce=True,
        **kwargs,
    )
    return loss, metrics


def run_chunked_torch(inputs, old, ref, kwargs):
    module = LigerFusedLinearGRPOLoss(
        beta=kwargs["beta"],
        compiled=False,
        use_ref_model=kwargs["beta"] != 0.0,
        epsilon_low=kwargs["eps_low"],
        epsilon_high=kwargs["eps_high"],
        loss_type=kwargs["loss_type"],
        max_completion_length=kwargs["max_completion_length"],
        importance_sampling_level=kwargs["importance_sampling_level"],
        temperature=kwargs["temperature"],
        delta=kwargs["delta"],
        use_bias_correction_kl=kwargs["use_bias_correction_kl"],
    )
    loss, metrics = module(
        inputs["hidden"][:, :-1, :],
        inputs["weight"],
        inputs["completion_ids"],
        inputs["mask"],
        inputs["advantages"],
        ref_per_token_logps=ref,
        old_per_token_logps=old,
        vllm_is_ratio=kwargs["vllm_is_ratio"],
        num_items_in_batch=kwargs["num_items_in_batch"],
    )
    return loss, metrics


def run_torch_reference(inputs, old, ref, kwargs):
    logits = (inputs["hidden"] @ inputs["weight"].t())[:, :-1, :]
    loss = torch_grpo_loss_from_logits(
        logits,
        old,
        ref,
        inputs["completion_ids"],
        inputs["advantages"],
        inputs["mask"],
        **kwargs,
    )
    return loss, None


IMPLEMENTATIONS = {
    "torch_ref": run_torch_reference,
    "triton": run_triton,
    "chunked_torch": run_chunked_torch,
    "chunked_triton": run_chunked_triton,
}


def compute_all(inputs, cfg, seq_len):
    kwargs, old, ref = config_kwargs(cfg, inputs, seq_len)
    results = {}
    for name, fn in IMPLEMENTATIONS.items():
        inputs["hidden"].grad = None
        inputs["weight"].grad = None
        inputs["hidden"].requires_grad_(True)
        inputs["weight"].requires_grad_(True)
        loss, metrics = fn(inputs, old, ref, kwargs)
        loss.backward()
        results[name] = {
            "loss": loss.item(),
            "metrics": [m.item() for m in metrics] if metrics is not None else None,
            "grad_hidden": inputs["hidden"].grad.float().flatten().clone(),
            "grad_weight": inputs["weight"].grad.float().flatten().clone(),
        }
        inputs["hidden"].requires_grad_(False)
        inputs["weight"].requires_grad_(False)
    return results


def assert_end_results_match(results):
    ref = results["torch_ref"]
    for name, res in results.items():
        if name == "torch_ref":
            continue
        rel = abs(res["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-6)
        assert rel < 2e-2, f"{name} loss {res['loss']:.6f} vs torch_ref {ref['loss']:.6f} (rel {rel:.2e})"
        for key in ("grad_hidden", "grad_weight"):
            cos = torch.nn.functional.cosine_similarity(res[key], ref[key], dim=0).item()
            norm_ratio = (res[key].norm() / ref[key].norm().clamp(min=1e-12)).item()
            assert cos > 0.999, f"{name} {key} cosine {cos:.6f}"
            assert abs(norm_ratio - 1) < 2e-2, f"{name} {key} norm ratio {norm_ratio:.4f}"
    # metrics parity between the two Triton-based implementations
    m_ct, m_t = results["chunked_triton"]["metrics"], results["triton"]["metrics"]
    assert len(m_ct) == len(m_t)
    for a, b in zip(m_ct, m_t):
        assert abs(a - b) < 2e-2, f"metrics mismatch: chunked_triton {m_ct} vs triton {m_t}"


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_end_results_all_configs(config_name):
    seq_len = 1024
    inputs = make_inputs(4, seq_len, VOCAB_SIZE, seed=1)
    assert_end_results_match(compute_all(inputs, CONFIGS[config_name], seq_len))


@pytest.mark.parametrize(
    "batch, seq_len",
    [(4, 128), (4, 1024), (4, 4096), (2, 16384)],
)
def test_end_results_context_lengths(batch, seq_len):
    """Production config (dapo, sequence-level IS, beta=0) across context lengths."""
    inputs = make_inputs(batch, seq_len, VOCAB_SIZE, seed=2)
    cfg = CONFIGS["dapo_seq_offpolicy_niib"]
    assert_end_results_match(compute_all(inputs, cfg, seq_len))


def test_end_results_odd_vocab():
    seq_len = 512
    inputs = make_inputs(4, seq_len, ODD_VOCAB_SIZE, seed=3)
    cfg = CONFIGS["dapo_seq_offpolicy_niib"]
    assert_end_results_match(compute_all(inputs, cfg, seq_len))


# ---------------------------------------------------------------------------
# 4. edge shapes, dtypes, and extreme values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, seq_len, vocab, dtype",
    [
        (1, 1, 256, torch.bfloat16),  # single token
        (1, 100, 8192, torch.bfloat16),  # N below one row tile (BM=128)
        (1, 129, 8192, torch.bfloat16),  # one over a row tile
        (1, 4097, 8192, torch.bfloat16),  # one over the backward chunk (4096)
        (3, 2731, 8192, torch.bfloat16),  # ragged multi-chunk (N=8193)
        (3, 77, 250, torch.bfloat16),  # vocab smaller than the vocab tile (BN=256)
        (3, 77, 251, torch.bfloat16),  # prime vocab
        (2, 33, 2, torch.bfloat16),  # degenerate two-token vocab
        (64, 3, 8192, torch.bfloat16),  # wide batch, tiny sequences
        (4, 512, 50257, torch.float16),
        (4, 512, 50257, torch.float32),  # needs the 2-stage fp32 SMEM config
        (3, 77, 250, torch.float32),  # fp32 + sub-tile vocab
    ],
)
def test_edge_shapes_and_dtypes(batch, seq_len, vocab, dtype):
    set_seed(7)
    hidden = (torch.randn(batch, seq_len + 1, HIDDEN_SIZE, device=device) * 0.02).to(dtype)
    weight = (torch.randn(vocab, HIDDEN_SIZE, device=device) * 0.02).to(dtype)
    completion_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    # pin some targets to the vocab boundaries (exercises tail masking)
    completion_ids[:, 0] = 0
    completion_ids[:, -1] = vocab - 1
    lengths = torch.randint(max(seq_len // 2, 1), seq_len + 1, (batch,), device=device)
    mask = (torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    advantages = torch.randn(batch, device=device)

    hidden2d = hidden[:, :-1, :].reshape(-1, HIDDEN_SIZE).contiguous()
    ids_flat = completion_ids.reshape(-1)
    logp, lse = chunked_selective_log_softmax_with_lse(hidden2d, weight, ids_flat, 1.0)
    logp_ref, lse_ref = fp32_logp_lse(hidden2d, weight, ids_flat, 1.0)
    atol = 1e-2 if dtype != torch.float32 else 1e-3
    assert torch.allclose(logp, logp_ref, atol=atol, rtol=1e-3)
    assert torch.allclose(lse, lse_ref, atol=atol, rtol=1e-3)

    old = logp_ref.view(batch, seq_len) + torch.randn(batch, seq_len, device=device) * 0.3
    kwargs = dict(
        temperature=1.0,
        beta=0.0,
        eps_low=0.2,
        eps_high=0.2,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="sequence",
        delta=None,
        use_bias_correction_kl=False,
        num_items_in_batch=mask.sum(),
        vllm_is_ratio=None,
    )
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    loss, _ = chunked_triton_grpo_loss(
        hidden[:, :-1, :].contiguous(),
        weight,
        old,
        None,
        completion_ids,
        advantages,
        mask,
        reduce=True,
        **kwargs,
    )
    loss.backward()
    gh, gw = hidden.grad.float().flatten().clone(), weight.grad.float().flatten().clone()
    hidden.grad = None
    weight.grad = None
    logits = (hidden @ weight.t())[:, :-1, :]
    loss_ref = torch_grpo_loss_from_logits(logits, old, None, completion_ids, advantages, mask, **kwargs)
    loss_ref.backward()
    rel = abs(loss.item() - loss_ref.item()) / max(abs(loss_ref.item()), 1e-6)
    assert rel < 2e-2, f"loss {loss.item():.6f} vs ref {loss_ref.item():.6f}"
    for got, ref in ((gh, hidden.grad), (gw, weight.grad)):
        ref = ref.float().flatten()
        if got.norm() == 0 and ref.norm() == 0:
            continue
        cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
        assert cos > 0.999, f"grad cosine {cos:.6f}"
    hidden.requires_grad_(False)
    weight.requires_grad_(False)


def test_zero_mask_and_extreme_scale():
    set_seed(8)
    batch, seq_len, vocab = 4, 256, 8192
    for mask_mode, scale in (("zeros", 0.02), ("ones", 5.0)):
        hidden = (torch.randn(batch, seq_len, HIDDEN_SIZE, device=device) * scale).to(torch.bfloat16)
        weight = (torch.randn(vocab, HIDDEN_SIZE, device=device) * scale).to(torch.bfloat16)
        ids = torch.randint(0, vocab, (batch, seq_len), device=device)
        mask = (
            torch.zeros(batch, seq_len, device=device)
            if mask_mode == "zeros"
            else torch.ones(batch, seq_len, device=device)
        )
        adv = torch.randn(batch, device=device)
        loss, _ = chunked_triton_grpo_loss(
            hidden,
            weight,
            None,
            None,
            ids,
            adv,
            mask,
            temperature=1.0,
            beta=0.0,
            eps_low=0.2,
            eps_high=0.2,
            loss_type="dapo",
            importance_sampling_level="sequence",
            reduce=True,
            num_items_in_batch=mask.sum(),
        )
        assert torch.isfinite(loss), f"non-finite loss with mask={mask_mode}, scale={scale}"
        if mask_mode == "zeros":
            assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 5. bitwise determinism
# ---------------------------------------------------------------------------


def test_bitwise_determinism():
    seq_len = 2048
    inputs = make_inputs(4, seq_len, VOCAB_SIZE, seed=4)
    cfg = CONFIGS["dapo_seq_offpolicy_niib"]
    kwargs, old, ref = config_kwargs(cfg, inputs, seq_len)

    runs = []
    for _ in range(2):
        inputs["hidden"].grad = None
        inputs["weight"].grad = None
        inputs["hidden"].requires_grad_(True)
        inputs["weight"].requires_grad_(True)
        loss, _ = run_chunked_triton(inputs, old, ref, kwargs)
        loss.backward()
        runs.append((loss.item(), inputs["hidden"].grad.clone(), inputs["weight"].grad.clone()))
        inputs["hidden"].requires_grad_(False)
        inputs["weight"].requires_grad_(False)

    assert runs[0][0] == runs[1][0], "loss not bitwise deterministic"
    assert torch.equal(runs[0][1], runs[1][1]), "grad_hidden not bitwise deterministic"
    assert torch.equal(runs[0][2], runs[1][2]), "grad_weight not bitwise deterministic"
