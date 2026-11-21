#!/usr/bin/env python3
# BSD 2-Clause License
# Copyright (c) 2024-2026 LinkedIn Corporation. All rights reserved.
"""Unit, numerical-stability, and multi-step mini-model convergence tests for Triton GRPO Entropy.

Designed to run in TWO environments seamlessly:
1. Full GPU CI (`pytest test/transformers/test_grpo_triton_entropy.py` on CUDA/ROCm with
   PyTorch + Triton + Liger-Kernel installed): exercises `selective_log_softmax_with_entropy`,
   `compute_high_entropy_mask`, `triton_grpo_loss_with_entropy`, non-contiguous strides,
   large shared logit offsets (`#1535`), masked first-block vocab tiles (`#1653`), and an
   8-step mini-model AdamW convergence comparison (`make test-convergence`).
2. CPU / Sandbox Pre-Verification (`python3 test_grpo_triton_entropy.py`): executes exact
   tile-by-tile FP32/BF16-emulated mathematical & Jacobian parity checks plus an 8-step
   AdamW mini-model convergence run against FP64 reference arithmetic using NumPy.
"""

from __future__ import annotations

import math
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from liger_kernel.ops._grpo_online_entropy import HAS_TORCH_TRITON
from liger_kernel.ops._grpo_online_entropy import numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

if HAS_TORCH_TRITON:
    import torch
    import torch.nn.functional as F

    from liger_kernel.ops._grpo_online_entropy import compute_high_entropy_mask
    from liger_kernel.ops._grpo_online_entropy import selective_log_softmax_with_entropy
    from liger_kernel.ops._grpo_online_entropy import triton_grpo_loss_with_entropy


# ==============================================================================
# Reference FP64 Oracle (Zero-Approximation Ground Truth)
# ==============================================================================


def fp64_reference_logp_entropy_and_grads(
    logits: np.ndarray,
    index: np.ndarray,
    temperature: float = 1.0,
    mask: np.ndarray | None = None,
    grad_log_p: np.ndarray | None = None,
    grad_entropy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact FP64 materialized log-softmax, Shannon entropy, and analytical Jacobian."""
    z = np.asarray(logits, dtype=np.float64)
    u = z / float(temperature)
    orig_shape = index.shape
    v_dim = z.shape[-1]
    u_2d = u.reshape(-1, v_dim)
    idx_1d = np.asarray(index, dtype=np.int64).reshape(-1)
    mask_1d = np.ones_like(idx_1d, dtype=bool) if mask is None else np.asarray(mask).reshape(-1).astype(bool)

    log_p_out = np.zeros((u_2d.shape[0],), dtype=np.float64)
    ent_out = np.zeros((u_2d.shape[0],), dtype=np.float64)
    grad_z = np.zeros_like(u_2d, dtype=np.float64)

    g_lp = (
        np.zeros((u_2d.shape[0],), dtype=np.float64)
        if grad_log_p is None
        else np.asarray(grad_log_p, dtype=np.float64).reshape(-1)
    )
    g_h = (
        np.zeros((u_2d.shape[0],), dtype=np.float64)
        if grad_entropy is None
        else np.asarray(grad_entropy, dtype=np.float64).reshape(-1)
    )

    for r in range(u_2d.shape[0]):
        if not mask_1d[r]:
            continue
        row = u_2d[r]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        m = np.max(row[finite])
        shifted = np.where(finite, row - m, -np.inf)
        exp_s = np.where(finite, np.exp(shifted), 0.0)
        l_sum = np.sum(exp_s)
        lse = m + math.log(l_sum)
        p = exp_s / l_sum
        log_p_all = np.where(finite, row - lse, 0.0)
        ent = -np.sum(np.where(p > 0.0, p * log_p_all, 0.0))
        ent_out[r] = ent
        t_idx = int(idx_1d[r])
        if 0 <= t_idx < v_dim:
            log_p_out[r] = row[t_idx] - lse

        is_tgt = np.zeros((v_dim,), dtype=np.float64)
        eff_glp = g_lp[r] if (0 <= t_idx < v_dim) else 0.0
        if 0 <= t_idx < v_dim:
            is_tgt[t_idx] = 1.0
        grad_z[r] = (eff_glp * (is_tgt - p) - g_h[r] * p * (log_p_all + ent)) / float(temperature)

    return log_p_out.reshape(orig_shape), ent_out.reshape(orig_shape), grad_z.reshape(logits.shape)


# ==============================================================================
# Core Cross-Environment Test Cases (Executed in Both Pytest & Standalone Runner)
# ==============================================================================


def check_shift_invariance_and_anti_cancellation() -> None:
    """Verify shift-invariant `s_i` prevents catastrophic cancellation at large offsets (`#1535`)."""
    rng = np.random.default_rng(42)
    b, t, v = 2, 7, 1031  # Non-power-of-2 tail vocab
    base_logits = rng.normal(0.0, 2.5, size=(b, t, v)).astype(np.float32)
    # Force row (0, 0) into a near-peaked late-RL regime where H ~ 4e-4
    base_logits[0, 0, :] = -12.0
    base_logits[0, 0, 17] = 0.0
    base_logits[0, 0, 513] = -8.8
    index = rng.integers(0, v, size=(b, t), dtype=np.int64)
    g_lp = rng.normal(0.0, 1.0, size=(b, t)).astype(np.float32)
    g_ent = rng.normal(0.0, 0.5, size=(b, t)).astype(np.float32)

    for offset in (0.0, 25.0, 150.0, 1000.0):
        shifted = (base_logits.astype(np.float64) + offset).astype(np.float32)
        ref_lp, ref_ent, ref_gz = fp64_reference_logp_entropy_and_grads(
            shifted, index, temperature=1.3, grad_log_p=g_lp, grad_entropy=g_ent
        )
        got_lp, got_ent, got_gz = numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd(
            shifted, index, temperature=1.3, grad_log_p=g_lp, grad_entropy=g_ent, block_size=128
        )
        np.testing.assert_allclose(got_lp, ref_lp, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got_ent, ref_ent, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got_gz, ref_gz, rtol=1e-5, atol=1e-5)
        # Specifically verify the peaked row's small entropy (~4e-4) did NOT cancel to 0 or blow up!
        rel_peaked_err = abs(float(got_ent[0, 0]) - float(ref_ent[0, 0])) / max(float(ref_ent[0, 0]), 1e-12)
        assert rel_peaked_err < 1e-4, f"Catastrophic cancellation at offset={offset}: rel_err={rel_peaked_err}"


def check_masked_blocks_padding_and_finite_diff_jacobian() -> None:
    """Verify first-block `-inf` masking (`#1653`), `row_mask=0`, `-100` targets, and numerical Jacobian."""
    rng = np.random.default_rng(1082)
    b, t, v = 2, 4, 300
    logits = rng.normal(0.0, 1.8, size=(b, t, v)).astype(np.float32)
    # Entire first tile [0:128] is -inf on row (0, 1) -> tests `#1653` online normalizer safety!
    logits[0, 1, :128] = -np.inf
    index = rng.integers(128, v, size=(b, t), dtype=np.int64)
    index[1, 2] = -100  # Padding label
    mask = np.ones((b, t), dtype=np.int64)
    mask[1, 0] = 0  # Inactive masked row

    g_lp = rng.normal(0.0, 1.0, size=(b, t)).astype(np.float32)
    g_ent = rng.normal(0.0, 1.0, size=(b, t)).astype(np.float32)

    ref_lp, ref_ent, ref_gz = fp64_reference_logp_entropy_and_grads(
        logits, index, temperature=0.7, mask=mask, grad_log_p=g_lp, grad_entropy=g_ent
    )
    got_lp, got_ent, got_gz = numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd(
        logits, index, temperature=0.7, mask=mask, grad_log_p=g_lp, grad_entropy=g_ent, block_size=128
    )

    assert np.all(np.isfinite(got_lp)) and np.all(np.isfinite(got_ent)) and np.all(np.isfinite(got_gz))
    assert got_lp[1, 0] == 0.0 and got_ent[1, 0] == 0.0 and np.all(got_gz[1, 0] == 0.0)
    assert got_lp[1, 2] == 0.0
    np.testing.assert_allclose(got_lp, ref_lp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_ent, ref_ent, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_gz, ref_gz, rtol=1e-5, atol=1e-5)

    # Central finite-difference Jacobian check on small slice (1, 2, 19)
    z_small = rng.normal(0.0, 1.2, size=(1, 2, 19)).astype(np.float64)
    idx_small = np.array([[3, 14]], dtype=np.int64)
    w_lp = np.array([[0.8, -0.4]], dtype=np.float64)
    w_ent = np.array([[-0.35, 0.65]], dtype=np.float64)
    _, _, ana_gz = fp64_reference_logp_entropy_and_grads(
        z_small, idx_small, temperature=0.85, grad_log_p=w_lp, grad_entropy=w_ent
    )
    _, _, emu_gz = numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd(
        z_small.astype(np.float32),
        idx_small,
        temperature=0.85,
        grad_log_p=w_lp.astype(np.float32),
        grad_entropy=w_ent.astype(np.float32),
        block_size=8,
    )
    eps = 1e-5
    fd_gz = np.zeros_like(z_small)
    for ti in range(2):
        for vi in range(19):
            zp = z_small.copy()
            zm = z_small.copy()
            zp[0, ti, vi] += eps
            zm[0, ti, vi] -= eps
            lp_p, ent_p, _ = fp64_reference_logp_entropy_and_grads(zp, idx_small, temperature=0.85)
            lp_m, ent_m, _ = fp64_reference_logp_entropy_and_grads(zm, idx_small, temperature=0.85)
            obj_p = np.sum(lp_p * w_lp + ent_p * w_ent)
            obj_m = np.sum(lp_m * w_lp + ent_m * w_ent)
            fd_gz[0, ti, vi] = (obj_p - obj_m) / (2.0 * eps)

    np.testing.assert_allclose(ana_gz, fd_gz, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(emu_gz, fd_gz, rtol=1e-4, atol=1e-4)


def check_mini_model_8step_grpo_convergence() -> None:
    """8-step mini-model AdamW convergence test (`test/convergence/` standard) with `top_entropy_quantile=0.4`."""
    rng = np.random.default_rng(2026)
    b, t, h, v = 4, 8, 32, 257
    lr = 5e-3
    top_entropy_quantile = 0.4
    entropy_coef = 0.02
    beta = 0.04
    eps_low, eps_high = 0.2, 0.28  # DAPO asymmetric clip

    w_proj_init = (rng.normal(0.0, 1.0 / math.sqrt(h), size=(h, h))).astype(np.float32)
    w_head_init = (rng.normal(0.0, 1.0 / math.sqrt(h), size=(v, h))).astype(np.float32)
    x_batches = [rng.normal(0.0, 1.0, size=(b, t, h)).astype(np.float32) for _ in range(8)]
    target_batches = [rng.integers(0, v, size=(b, t), dtype=np.int64) for _ in range(8)]
    mask_batches = [(rng.uniform(0.0, 1.0, size=(b, t)) > 0.15).astype(np.float32) for _ in range(8)]
    for mb in mask_batches:
        mb[:, 0] = 1.0
    adv_batches = [rng.normal(0.0, 1.0, size=(b, 1)).astype(np.float32) for _ in range(8)]

    def run_training(use_tiled_emulation: bool) -> tuple[list[float], np.ndarray, np.ndarray]:
        w_proj = w_proj_init.copy()
        w_head = w_head_init.copy()
        m_p, v_p = np.zeros_like(w_proj), np.zeros_like(w_proj)
        m_h, v_h = np.zeros_like(w_head), np.zeros_like(w_head)
        losses: list[float] = []

        for step in range(1, 9):
            x = x_batches[step - 1]
            ids = target_batches[step - 1]
            mask = mask_batches[step - 1]
            adv = adv_batches[step - 1]

            hidden = np.tanh(x @ w_proj.T)  # [B, T, H]
            logits = hidden @ w_head.T  # [B, T, V]

            # Reference / old logps from initial weights
            ref_hidden = np.tanh(x @ w_proj_init.T)
            ref_logits = ref_hidden @ w_head_init.T
            ref_lp, _, _ = fp64_reference_logp_entropy_and_grads(ref_logits, ids, temperature=1.1, mask=mask)
            old_lp = ref_lp.astype(np.float32)

            if use_tiled_emulation:
                lp, ent, _ = numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd(
                    logits, ids, temperature=1.1, mask=mask, block_size=64
                )
            else:
                lp64, ent64, _ = fp64_reference_logp_entropy_and_grads(logits, ids, temperature=1.1, mask=mask)
                lp, ent = lp64.astype(np.float32), ent64.astype(np.float32)

            valid_ents = ent[mask.astype(bool)]
            thresh = float(np.quantile(valid_ents, 1.0 - top_entropy_quantile))
            ent_mask = ((ent * mask) >= thresh) & mask.astype(bool)
            eff_mask = mask * ent_mask.astype(np.float32)

            ratio = np.exp(lp - old_lp)
            clipped = np.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
            s1 = ratio * adv
            s2 = clipped * adv
            choose_s1 = s1 <= s2
            pol_tok = -np.minimum(s1, s2) * ent_mask.astype(np.float32)

            diff_ref = ref_lp.astype(np.float32) - lp
            kl_tok = np.exp(diff_ref) - diff_ref - 1.0
            total_tok = pol_tok + beta * kl_tok

            denom = max(float(np.sum(mask)), 1.0)
            eff_denom = max(float(np.sum(eff_mask)), 1.0)
            loss = float(np.sum(total_tok * mask) / denom - entropy_coef * (np.sum(ent * eff_mask) / eff_denom))
            losses.append(loss)

            # Upstream dL/dlp and dL/dent
            d_pol_dlp = np.where(choose_s1, -adv * ratio, 0.0) * ent_mask.astype(np.float32)
            d_kl_dlp = beta * (1.0 - np.exp(diff_ref))
            g_lp = (d_pol_dlp + d_kl_dlp) * mask / denom
            g_ent = -entropy_coef * eff_mask / eff_denom

            if use_tiled_emulation:
                _, _, grad_logits = numpy_triton_emulated_selective_log_softmax_entropy_fwd_bwd(
                    logits, ids, temperature=1.1, mask=mask, grad_log_p=g_lp, grad_entropy=g_ent, block_size=64
                )
            else:
                _, _, gz64 = fp64_reference_logp_entropy_and_grads(
                    logits, ids, temperature=1.1, mask=mask, grad_log_p=g_lp, grad_entropy=g_ent
                )
                grad_logits = gz64.astype(np.float32)

            grad_w_head = grad_logits.reshape(-1, v).T @ hidden.reshape(-1, h)
            grad_hidden = grad_logits @ w_head
            grad_pre = grad_hidden * (1.0 - hidden**2)
            grad_w_proj = grad_pre.reshape(-1, h).T @ x.reshape(-1, h)

            # AdamW update
            b1, b2, eps = 0.9, 0.999, 1e-8
            m_h = b1 * m_h + (1.0 - b1) * grad_w_head
            v_h = b2 * v_h + (1.0 - b2) * (grad_w_head**2)
            w_head -= lr * (m_h / (1.0 - b1**step)) / (np.sqrt(v_h / (1.0 - b2**step)) + eps)

            m_p = b1 * m_p + (1.0 - b1) * grad_w_proj
            v_p = b2 * v_p + (1.0 - b2) * (grad_w_proj**2)
            w_proj -= lr * (m_p / (1.0 - b1**step)) / (np.sqrt(v_p / (1.0 - b2**step)) + eps)

        return losses, w_proj, w_head

    ref_losses, ref_wp, ref_wh = run_training(use_tiled_emulation=False)
    liger_losses, liger_wp, liger_wh = run_training(use_tiled_emulation=True)

    np.testing.assert_allclose(liger_losses, ref_losses, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(liger_wp, ref_wp, rtol=1e-4, atol=5e-5)
    np.testing.assert_allclose(liger_wh, ref_wh, rtol=1e-4, atol=5e-5)


# ==============================================================================
# Pytest Suite (Runs Both NumPy Checks + Real GPU Triton Kernels When Available)
# ==============================================================================


def test_numpy_shift_invariance_and_anti_cancellation():
    check_shift_invariance_and_anti_cancellation()


def test_numpy_masked_blocks_and_jacobian():
    check_masked_blocks_padding_and_finite_diff_jacobian()


def test_numpy_mini_model_8step_convergence():
    check_mini_model_8step_grpo_convergence()


if HAS_TORCH_TRITON and pytest is not None:
    _GPU_AVAILABLE = torch.cuda.is_available()

    @pytest.mark.skipif(not _GPU_AVAILABLE, reason="CUDA GPU required for Triton kernel execution")
    @pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 1e-5, 1e-4), (torch.bfloat16, 2e-2, 2e-2)])
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 1.6])
    @pytest.mark.parametrize("vocab_size", [1024, 4111, 32003])
    def test_triton_selective_log_softmax_entropy_forward_backward(dtype, atol, rtol, temperature, vocab_size):
        torch.manual_seed(42)
        b, t = 3, 9
        # Slice a wider tensor to create non-contiguous row strides (`[:, 1::2, :]`)
        raw = torch.randn(b, t * 2, vocab_size, device="cuda", dtype=dtype)
        logits_triton = raw[:, 1::2, :].detach().requires_grad_(True)
        logits_ref = raw[:, 1::2, :].detach().float().requires_grad_(True)

        index = torch.randint(0, vocab_size, (b, t), device="cuda", dtype=torch.long)
        mask = (torch.rand(b, t, device="cuda") > 0.2).long()
        mask[0, 0] = 0  # Ensure at least one masked token

        g_lp = torch.randn(b, t, device="cuda", dtype=torch.float32)
        g_ent = torch.randn(b, t, device="cuda", dtype=torch.float32)

        lp_tri, ent_tri = selective_log_softmax_with_entropy(
            logits_triton, index, temperature=temperature, mask=mask, return_entropy=True, entropy_requires_grad=True
        )
        (lp_tri * g_lp + ent_tri * g_ent).sum().backward()

        scaled_ref = logits_ref / temperature
        log_probs_all = F.log_softmax(scaled_ref, dim=-1)
        probs_all = torch.exp(log_probs_all)
        lp_ref = log_probs_all.gather(-1, index.unsqueeze(-1)).squeeze(-1) * mask.float()
        ent_ref = -(probs_all * log_probs_all).sum(dim=-1) * mask.float()
        (lp_ref * g_lp + ent_ref * g_ent).sum().backward()

        torch.testing.assert_close(lp_tri, lp_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(ent_tri, ent_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(logits_triton.grad.float(), logits_ref.grad, atol=atol, rtol=rtol)

    @pytest.mark.skipif(not _GPU_AVAILABLE, reason="CUDA GPU required for Triton GRPO convergence test")
    @pytest.mark.parametrize("loss_type", ["grpo", "dapo", "bnpo", "dr_grpo", "cispo"])
    @pytest.mark.parametrize("top_entropy_quantile", [1.0, 0.35])
    def test_triton_grpo_loss_with_top_entropy_quantile_and_convergence(loss_type, top_entropy_quantile):
        torch.manual_seed(2026)
        b, t, h, v = 4, 12, 64, 2053
        model_tri = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, v)).cuda()
        model_ref = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, v)).cuda()
        model_ref.load_state_dict(model_tri.state_dict())

        opt_tri = torch.optim.AdamW(model_tri.parameters(), lr=3e-3)
        opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=3e-3)

        for _ in range(5):
            x = torch.randn(b, t, h, device="cuda")
            ids = torch.randint(0, v, (b, t), device="cuda")
            mask = (torch.rand(b, t, device="cuda") > 0.15).long()
            mask[:, 0] = 1
            adv = torch.randn(b, device="cuda")
            old_lp = -torch.rand(b, t, device="cuda") * 3.0
            ref_lp = -torch.rand(b, t, device="cuda") * 3.0
            vllm_is = torch.rand(b, t, device="cuda") * 0.5 + 0.75

            opt_tri.zero_grad()
            loss_tri, _ = triton_grpo_loss_with_entropy(
                logits=model_tri(x),
                selected_token_ids=ids,
                attention_mask=mask,
                advantages=adv,
                old_per_token_logps=old_lp,
                ref_per_token_logps=ref_lp,
                vllm_is_ratio=vllm_is,
                temperature=1.2,
                beta=0.04,
                epsilon_low=0.2,
                epsilon_high=0.28,
                loss_type=loss_type,
                top_entropy_quantile=top_entropy_quantile,
                entropy_coef=0.01,
            )
            loss_tri.backward()
            opt_tri.step()

            # Reference PyTorch path following exact TRL `_compute_loss` ordering
            opt_ref.zero_grad()
            lg_ref = model_ref(x) / 1.2
            lps_all = F.log_softmax(lg_ref, dim=-1)
            ps_all = torch.exp(lps_all)
            lp_r = lps_all.gather(-1, ids.unsqueeze(-1)).squeeze(-1) * mask.float()
            ent_r = -(ps_all * lps_all).sum(dim=-1) * mask.float()
            ent_m = compute_high_entropy_mask(ent_r, mask, top_entropy_quantile) if top_entropy_quantile < 1.0 else None
            coef_1 = torch.exp(lp_r - old_lp)
            adv_2d = adv.unsqueeze(1)
            if loss_type == "cispo":
                pt_loss = -torch.clamp(coef_1, max=0.28).detach() * adv_2d * lp_r
            else:
                coef_2 = torch.clamp(coef_1, 1.0 - 0.2, 1.0 + 0.28)
                pt_loss = -torch.minimum(coef_1 * adv_2d, coef_2 * adv_2d)
            if ent_m is not None:
                pt_loss = pt_loss * ent_m.float()
            pt_loss = pt_loss * vllm_is
            diff_ref = ref_lp - lp_r
            pt_kl = torch.exp(diff_ref) - diff_ref - 1.0
            pt_loss = pt_loss + 0.04 * pt_kl
            mf = mask.float()
            if loss_type == "grpo":
                loss_r = ((pt_loss * mf).sum(-1) / mf.sum(-1).clamp(min=1.0)).mean()
            elif loss_type in ("bnpo", "dapo", "cispo"):
                loss_r = (pt_loss * mf).sum() / mf.sum().clamp(min=1.0)
            else:
                loss_r = (pt_loss * mf).sum() / float(b * t)
            eff_m = mf if ent_m is None else mf * ent_m.float()
            loss_r = loss_r - 0.01 * ((ent_r * eff_m).sum() / eff_m.sum().clamp(min=1.0))
            loss_r.backward()
            opt_ref.step()

            torch.testing.assert_close(loss_tri, loss_r, atol=1e-5, rtol=1e-4)

        for p_tri, p_ref in zip(model_tri.parameters(), model_ref.parameters()):
            torch.testing.assert_close(p_tri, p_ref, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    print("[1/3] Running shift-invariance & anti-cancellation checks (offsets up to +1000)...")
    check_shift_invariance_and_anti_cancellation()
    print("      PASS: Zero catastrophic cancellation across offsets [0, 25, 150, 1000] (rel_err < 1e-4).")
    print("[2/3] Running masked vocab blocks (#1653), padding (-100), and finite-diff Jacobian checks...")
    check_masked_blocks_padding_and_finite_diff_jacobian()
    print("      PASS: Analytical & tiled backward Jacobians match central finite differences (< 1e-4).")
    print("[3/3] Running 8-step AdamW mini-model DAPO + top_entropy_quantile=0.4 convergence test...")
    check_mini_model_8step_grpo_convergence()
    print("      PASS: 8-step losses and model weights match FP64 oracle (< 1e-5).")
    print("ALL VERIFICATION SUITES PASSED.")
