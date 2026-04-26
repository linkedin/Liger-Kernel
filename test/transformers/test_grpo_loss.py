import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed

from liger_kernel.ops.grpo_loss import fused_selective_log_softmax
from liger_kernel.transformers.grpo_loss import triton_grpo_loss


@torch.no_grad
def selective_log_softmax(logits, input_ids, temperature=0.9):
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    logits_to_keep = logits.size(1)
    index = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    logits = logits / temperature

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def _get_log_probs(logits, input_ids):
    """Helper function to compute per-token log probabilities."""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids[:, -logits.size(1) :]):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def torch_grpo_loss(
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
    delta=None,
    use_bias_correction_kl=False,
):
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
    logits = logits[:, :-1]

    per_token_logps = _get_log_probs(logits / temperature, completion_ids)
    ref_per_token_logps = ref_logp

    if old_logp is None:
        old_logp = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_logp)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    if delta is not None:
        coef_1 = torch.clamp(coef_1, max=delta)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    per_token_loss = per_token_loss * completion_mask if completion_mask is not None else per_token_loss

    per_token_kl = None
    if beta != 0.0:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if use_bias_correction_kl:
            per_token_kl = per_token_kl * torch.exp(per_token_logps - old_logp)
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl
    is_clipped = (per_token_loss1 < per_token_loss2).float()
    return per_token_loss, per_token_kl, is_clipped


def torch_cispo_loss(
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask,
    temperature,
    beta,
    eps_high,
    use_bias_correction_kl=False,
):
    """Reference implementation for CISPO loss.

    CISPO (Clipped Importance Sampling Policy Optimization) uses:
    - Upper-bound only clipping (no lower bound)
    - Detached clipped coefficient (no gradient through clipping)
    - Loss includes per_token_logps multiplication

    Reference: MiniMax-M1 technical report
    """
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
    logits = logits[:, :-1]

    per_token_logps = _get_log_probs(logits / temperature, completion_ids)
    ref_per_token_logps = ref_logp

    if old_logp is None:
        old_logp = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_logp)
    # CISPO: upper-bound only clipping with detach
    coef_2 = torch.clamp(coef_1, max=eps_high).detach()
    # CISPO loss includes per_token_logps
    per_token_loss = -coef_2 * advantages.unsqueeze(1) * per_token_logps
    per_token_loss = per_token_loss * completion_mask if completion_mask is not None else per_token_loss

    per_token_kl = None
    if beta != 0.0:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if use_bias_correction_kl:
            per_token_kl = per_token_kl * torch.exp(per_token_logps - old_logp)
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl
    is_clipped = ((coef_1 > eps_high) & (advantages.unsqueeze(1) > 0)).float()
    return per_token_loss, per_token_kl, is_clipped


def torch_sapo_loss(
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    completion_mask,
    temperature,
    beta,
    sapo_temperature_pos,
    sapo_temperature_neg,
    use_bias_correction_kl=False,
):
    """Reference implementation for SAPO loss.

    SAPO (Soft Adaptive Policy Optimization) uses:
    - Sigmoid-based soft gating instead of hard clipping
    - Different temperatures for positive/negative advantages

    Reference: https://huggingface.co/papers/2511.20347
    """
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
    logits = logits[:, :-1]

    per_token_logps = _get_log_probs(logits / temperature, completion_ids)
    ref_per_token_logps = ref_logp

    if old_logp is None:
        old_logp = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_logp)

    # SAPO: sigmoid-based soft gating
    # Select temperature based on advantage sign
    temp = torch.where(advantages.unsqueeze(1) > 0, sapo_temperature_pos, sapo_temperature_neg)
    sigmoid_input = temp * (coef_1 - 1.0)
    sapo_coef = torch.sigmoid(sigmoid_input) * 4.0 / temp
    per_token_loss = -sapo_coef * advantages.unsqueeze(1)
    per_token_loss = per_token_loss * completion_mask if completion_mask is not None else per_token_loss

    per_token_kl = None
    if beta != 0.0:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if use_bias_correction_kl:
            per_token_kl = per_token_kl * torch.exp(per_token_logps - old_logp)
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl
    # SAPO has no clipping concept
    is_clipped = torch.zeros_like(per_token_loss)
    return per_token_loss, per_token_kl, is_clipped


set_seed(42)
device = infer_device()


@pytest.mark.parametrize(
    "temperature, B, T, V",
    [
        (0.9, 1, 1024, 64000),
        (0.7, 1, 1024, 151936),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
def test_selective_log_softmax(B, T, V, temperature, dtype, atol, rtol):
    # logits_to_keep + 1
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logit1 = _input.clone()
    logit2 = _input.clone()
    logit3 = _input.clone().float()

    # we set the length of prompt_ids is 100 and the length of completion_ids is T
    input_ids = torch.randint(0, V - 1, (B, 100 + T), dtype=torch.int64, device=device)

    torch_bf16_logp = selective_log_softmax(logit1, input_ids, temperature)
    triton_bf16_logp = fused_selective_log_softmax(logit2, input_ids, temperature)
    torch_fp32_logp = selective_log_softmax(logit3, input_ids, temperature)

    assert_verbose_allclose(torch_bf16_logp, torch_fp32_logp.to(dtype), rtol=rtol, atol=atol)
    assert_verbose_allclose(triton_bf16_logp, torch_fp32_logp.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "temperature, num_iteration, beta, eps_low, eps_high",
    [(0.7, num_iteration, beta, 0.2, 0.4) for num_iteration in [1, 5] for beta in [0.0, 0.04]],
)
@pytest.mark.parametrize(
    "B, T, V",
    [
        (1, 1024, 151936),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
def test_grpo_loss(B, T, V, temperature, num_iteration, beta, eps_low, eps_high, dtype, atol, rtol):
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logits1 = _input.clone().requires_grad_(True)
    logits2 = _input.clone().requires_grad_(True)
    logits3 = _input.clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, T), dtype=torch.int64, device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    # we set num_padding is 100
    completion_mask[:, -100:] = 0

    # we set these in fp32, because fused_selective_log_softmax retutn fp32 logp, although logits in bf16
    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32) if beta != 0.0 else None
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32) if num_iteration > 1 else None
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    loss1, kl1, is_clipped1 = torch_grpo_loss(
        logits1, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_low, eps_high
    )

    loss2, kl2, is_clipped2 = triton_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace=True,
    )

    loss3, kl3, is_clipped3 = torch_grpo_loss(
        logits3, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_low, eps_high
    )

    dy = torch.randn_like(loss3)
    loss1.backward(dy)
    loss2.backward(dy)
    loss3.backward(dy)

    assert_verbose_allclose(loss1, loss3, atol=atol, rtol=rtol)
    if kl1 is not None and kl3 is not None:
        assert_verbose_allclose(kl1, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits1.grad, logits3.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(loss2, loss3, atol=atol, rtol=rtol)
    if kl2 is not None and kl3 is not None:
        assert_verbose_allclose(kl2, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits2.grad, logits3.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("delta", [1.5, 2.0])
@pytest.mark.parametrize(
    "temperature, num_iteration, beta, eps_low, eps_high",
    [(0.7, 5, beta, 0.2, 0.4) for beta in [0.0, 0.04]],
)
@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 128, 1000),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
def test_grpo_loss_with_delta(B, T, V, temperature, num_iteration, beta, eps_low, eps_high, dtype, atol, rtol, delta):
    """Test delta (two-sided clipping) support for standard PPO loss types."""
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logits1 = _input.clone().requires_grad_(True)
    logits2 = _input.clone().requires_grad_(True)
    logits3 = _input.clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, T), dtype=torch.int64, device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    completion_mask[:, -20:] = 0

    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32) if beta != 0.0 else None
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32) if num_iteration > 1 else None
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    loss1, kl1, is_clipped1 = torch_grpo_loss(
        logits1,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        delta=delta,
    )

    loss2, kl2, is_clipped2 = triton_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace=True,
        delta=delta,
    )

    loss3, kl3, is_clipped3 = torch_grpo_loss(
        logits3,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        delta=delta,
    )

    dy = torch.randn_like(loss3)
    loss1.backward(dy)
    loss2.backward(dy)
    loss3.backward(dy)

    assert_verbose_allclose(loss1, loss3, atol=atol, rtol=rtol)
    if kl1 is not None and kl3 is not None:
        assert_verbose_allclose(kl1, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits1.grad, logits3.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(loss2, loss3, atol=atol, rtol=rtol)
    if kl2 is not None and kl3 is not None:
        assert_verbose_allclose(kl2, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits2.grad, logits3.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "temperature, num_iteration, eps_low, eps_high",
    [(0.7, 5, 0.2, 0.4)],
)
@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 128, 1000),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
def test_grpo_loss_with_bias_correction_kl(B, T, V, temperature, num_iteration, eps_low, eps_high, dtype, atol, rtol):
    """Test use_bias_correction_kl (importance-sampling-corrected KL from DeepSeek-V3.2)."""
    beta = 0.04  # Must be non-zero for KL to matter
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logits1 = _input.clone().requires_grad_(True)
    logits2 = _input.clone().requires_grad_(True)
    logits3 = _input.clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, T), dtype=torch.int64, device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    completion_mask[:, -20:] = 0

    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    loss1, kl1, is_clipped1 = torch_grpo_loss(
        logits1,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        use_bias_correction_kl=True,
    )

    loss2, kl2, is_clipped2 = triton_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace=True,
        use_bias_correction_kl=True,
    )

    loss3, kl3, is_clipped3 = torch_grpo_loss(
        logits3,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        use_bias_correction_kl=True,
    )

    dy = torch.randn_like(loss3)
    loss1.backward(dy)
    loss2.backward(dy)
    loss3.backward(dy)

    assert_verbose_allclose(loss1, loss3, atol=atol, rtol=rtol)
    if kl1 is not None and kl3 is not None:
        assert_verbose_allclose(kl1, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits1.grad, logits3.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(loss2, loss3, atol=atol, rtol=rtol)
    if kl2 is not None and kl3 is not None:
        assert_verbose_allclose(kl2, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits2.grad, logits3.grad, atol=atol, rtol=rtol)


def trl_reference_grpo_loss(
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
    loss_type,
    importance_sampling_level,
    delta=None,
    use_bias_correction_kl=False,
    vespo_k_pos=2.0,
    vespo_lambda_pos=3.0,
    vespo_k_neg=3.0,
    vespo_lambda_neg=2.0,
):
    """TRL reference implementation from grpo_trainer.py"""
    from liger_kernel.chunked_loss.grpo_loss import get_gamma_weights

    B, L_ADD_1, V = logits.shape
    L = L_ADD_1 - 1

    logits_scaled = logits[:, :-1, :] / temperature
    log_probs = torch.log_softmax(logits_scaled.float(), dim=-1)
    per_token_logps = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

    if old_logp is None:
        old_logp = per_token_logps.detach()

    log_ratio = per_token_logps - old_logp

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    else:  # sequence
        log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)

    coef_1 = torch.exp(log_importance_weights)

    if loss_type == "vespo":
        # VESPO: detached gamma weighting on per-token logp, no clipping.
        # phi_seq replaces the (coef_1, coef_2) clipping pair.
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=completion_mask,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )  # (B, 1)
        per_token_loss = -phi_seq * advantages.unsqueeze(-1) * per_token_logps
    else:
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
        if delta is not None:
            coef_1 = torch.clamp(coef_1, max=delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(-1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(-1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if importance_sampling_level == "sequence":
            per_token_loss = per_token_loss.expand(B, L)

    if beta != 0.0:
        kl = torch.exp(ref_logp - per_token_logps) - (ref_logp - per_token_logps) - 1.0
        if use_bias_correction_kl:
            # TRL: kl *= coef_1 with shape matching importance_sampling_level
            # (token: (B, T); sequence: (B, 1)).
            kl = kl * torch.exp(log_importance_weights)
        per_token_loss = per_token_loss + beta * kl

    # Loss reduction
    if loss_type == "grpo":
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss = (per_token_loss * completion_mask).sum() / (B * L)
    elif loss_type == "dapo" or loss_type == "vespo":
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    elif loss_type == "luspo":
        loss = (per_token_loss * completion_mask.sum(-1, keepdim=True)).mean()

    return loss


@pytest.mark.parametrize("delta", [None, 1.5])
@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "luspo", "vespo"])
@pytest.mark.parametrize("beta,use_bias_correction_kl", [(0.0, False), (0.04, False), (0.04, True)])
@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 128, 1000),
    ],
)
def test_grpo_loss_vs_trl(B, T, V, beta, use_bias_correction_kl, loss_type, importance_sampling_level, delta):
    """Test that triton_grpo_loss matches TRL's exact implementation."""
    if importance_sampling_level == "token" and loss_type == "luspo":
        pytest.skip("Token-level importance sampling is not supported for loss_type='luspo'")
    if importance_sampling_level == "sequence" and loss_type == "vespo":
        pytest.skip("Sequence-level importance sampling is not supported for loss_type='vespo'")
    if delta is not None and loss_type == "vespo":
        pytest.skip("delta (two-sided clipping) is not supported for loss_type='vespo'")
    torch.manual_seed(42)

    logits = torch.randn(B, T + 1, V, device=device, dtype=torch.float32)
    completion_ids = torch.randint(0, V, (B, T), device=device)
    completion_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    # Compute realistic old_logp and ref_logp
    with torch.no_grad():
        log_probs = torch.log_softmax(logits[:, :-1, :] / 0.9, dim=-1)
        current_logp = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        old_logp = current_logp + torch.randn_like(current_logp) * 0.3
        ref_logp = current_logp + torch.randn_like(current_logp) * 0.2 if beta != 0.0 else None

    temperature = 0.9
    eps_low, eps_high = 0.2, 0.4

    # TRL reference
    trl_loss = trl_reference_grpo_loss(
        logits.clone(),
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        loss_type,
        importance_sampling_level,
        delta=delta,
        use_bias_correction_kl=use_bias_correction_kl,
    )

    # Triton implementation
    logits_triton = logits.clone().requires_grad_(True)
    triton_loss, _ = triton_grpo_loss(
        logits_triton,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level=importance_sampling_level,
        loss_type=loss_type,
        max_completion_length=T,
        reduce=True,
        delta=delta,
        use_bias_correction_kl=use_bias_correction_kl,
    )

    # Verify forward match
    torch.testing.assert_close(triton_loss, trl_loss, rtol=1e-4, atol=1e-4)

    # Verify backward works
    triton_loss.backward()
    assert logits_triton.grad is not None
    assert not torch.isnan(logits_triton.grad).any()


@pytest.mark.parametrize("loss_type", ["dapo", "cispo"])
def test_triton_num_items_in_batch_normalizer(loss_type):
    """``num_items_in_batch`` overrides the dapo/cispo normalizer in the triton path.

    Mirrors the chunked-loss test: in single-process world, passing
    ``num_items_in_batch=mask.sum()`` matches the default normalizer; doubling
    the value halves both the loss and the input gradient.
    """
    torch.manual_seed(0)
    B, T, V = 2, 64, 256

    completion_ids = torch.randint(0, V, (B, T), device=device)
    completion_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    completion_mask[:, -8:] = 0
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32)

    eps_low, eps_high = (0.2, 0.4) if loss_type == "dapo" else (0.0, 5.0)
    mask_sum = completion_mask.sum().item()
    base_logits = torch.randn(B, T + 1, V, device=device, dtype=torch.float32)

    def _run(num_items_in_batch):
        logits = base_logits.clone().requires_grad_(True)
        loss, _ = triton_grpo_loss(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature=0.9,
            beta=0.04,
            eps_low=eps_low,
            eps_high=eps_high,
            inplace=False,
            loss_type=loss_type,
            max_completion_length=T,
            reduce=True,
            num_items_in_batch=num_items_in_batch,
        )
        loss.backward()
        return loss.detach(), logits.grad.detach().clone()

    loss_default, grad_default = _run(num_items_in_batch=None)
    loss_match, grad_match = _run(num_items_in_batch=mask_sum)
    loss_double, grad_double = _run(num_items_in_batch=mask_sum * 2)

    assert_verbose_allclose(loss_default, loss_match, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(grad_default, grad_match, atol=1e-5, rtol=1e-5)

    assert_verbose_allclose(loss_double * 2, loss_default, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(grad_double * 2, grad_default, atol=1e-5, rtol=1e-5)


def trl_reference_grpo_loss_with_vllm_is(
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
    loss_type,
    importance_sampling_level,
    vllm_is_ratio,
    delta=None,
    use_bias_correction_kl=False,
):
    """TRL reference implementation with vLLM IS ratio correction."""
    B, L_ADD_1, V = logits.shape
    L = L_ADD_1 - 1

    logits_scaled = logits[:, :-1, :] / temperature
    log_probs = torch.log_softmax(logits_scaled.float(), dim=-1)
    per_token_logps = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

    if old_logp is None:
        old_logp = per_token_logps.detach()

    log_ratio = per_token_logps - old_logp

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    else:  # sequence
        log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)

    coef_1 = torch.exp(log_importance_weights)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    if delta is not None:
        coef_1 = torch.clamp(coef_1, max=delta)

    per_token_loss1 = coef_1 * advantages.unsqueeze(-1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(-1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    if importance_sampling_level == "sequence":
        per_token_loss = per_token_loss.expand(B, L)

    # Apply vLLM IS ratio BEFORE KL penalty (matches TRL)
    if vllm_is_ratio is not None:
        per_token_loss = per_token_loss * vllm_is_ratio

    if beta != 0.0:
        kl = torch.exp(ref_logp - per_token_logps) - (ref_logp - per_token_logps) - 1.0
        if use_bias_correction_kl:
            # TRL: kl *= coef_1 with shape matching importance_sampling_level
            # (token: (B, T); sequence: (B, 1)).
            kl = kl * torch.exp(log_importance_weights)
        per_token_loss = per_token_loss + beta * kl

    # Loss reduction
    if loss_type == "grpo":
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss = (per_token_loss * completion_mask).sum() / (B * L)
    elif loss_type == "dapo":
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    elif loss_type == "luspo":
        loss = (per_token_loss * completion_mask.sum(-1, keepdim=True)).mean()

    return loss


def torch_grpo_loss_with_vllm_is(
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
    vllm_is_ratio,
    loss_type="grpo",
    sapo_temperature_pos=1.0,
    sapo_temperature_neg=1.05,
    delta=None,
    use_bias_correction_kl=False,
):
    """Reference implementation with vLLM IS ratio correction for all loss types."""
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    logits = logits[:, :-1]
    per_token_logps = _get_log_probs(logits / temperature, completion_ids)
    ref_per_token_logps = ref_logp
    if old_logp is None:
        old_logp = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_logp)

    if loss_type == "cispo":
        coef_2 = torch.clamp(coef_1, max=eps_high).detach()
        per_token_loss = -coef_2 * advantages.unsqueeze(1) * per_token_logps
        is_clipped = ((coef_1 > eps_high) & (advantages.unsqueeze(1) > 0)).float()
    elif loss_type == "sapo":
        temp = torch.where(advantages.unsqueeze(1) > 0, sapo_temperature_pos, sapo_temperature_neg)
        sigmoid_input = temp * (coef_1 - 1.0)
        sapo_coef = torch.sigmoid(sigmoid_input) * 4.0 / temp
        per_token_loss = -sapo_coef * advantages.unsqueeze(1)
        is_clipped = torch.zeros_like(per_token_loss)
    else:
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
        if delta is not None:
            coef_1 = torch.clamp(coef_1, max=delta)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        is_clipped = (per_token_loss1 < per_token_loss2).float()

    # Apply vLLM IS correction BEFORE KL penalty
    if vllm_is_ratio is not None:
        per_token_loss = per_token_loss * vllm_is_ratio
    per_token_loss = per_token_loss * completion_mask if completion_mask is not None else per_token_loss
    per_token_kl = None
    if beta != 0.0:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if use_bias_correction_kl:
            per_token_kl = per_token_kl * torch.exp(per_token_logps - old_logp)
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl
    return per_token_loss, per_token_kl, is_clipped


@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "luspo"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 128, 1000),
    ],
)
def test_grpo_loss_with_vllm_is_ratio_reduced(B, T, V, beta, loss_type, importance_sampling_level):
    """Test that triton_grpo_loss with vllm_is_ratio matches TRL's behavior with reduce=True."""
    if importance_sampling_level == "token" and loss_type == "luspo":
        pytest.skip("Token-level importance sampling is not supported for loss_type='luspo'")
    torch.manual_seed(42)

    logits = torch.randn(B, T + 1, V, device=device, dtype=torch.float32)
    completion_ids = torch.randint(0, V, (B, T), device=device)
    completion_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    # Compute realistic old_logp and ref_logp
    with torch.no_grad():
        log_probs = torch.log_softmax(logits[:, :-1, :] / 0.9, dim=-1)
        current_logp = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        old_logp = current_logp + torch.randn_like(current_logp) * 0.3
        ref_logp = current_logp + torch.randn_like(current_logp) * 0.2 if beta != 0.0 else None

    # Create vLLM IS ratio (random values between 0.5 and 1.5)
    vllm_is_ratio = torch.rand(B, T, device=device, dtype=torch.float32) + 0.5

    temperature = 0.9
    eps_low, eps_high = 0.2, 0.4

    # TRL reference with vLLM IS ratio
    trl_loss = trl_reference_grpo_loss_with_vllm_is(
        logits.clone(),
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        loss_type,
        importance_sampling_level,
        vllm_is_ratio,
    )

    # Triton implementation with vLLM IS ratio
    logits_triton = logits.clone().requires_grad_(True)
    triton_loss, _ = triton_grpo_loss(
        logits_triton,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level=importance_sampling_level,
        loss_type=loss_type,
        max_completion_length=T,
        reduce=True,
        vllm_is_ratio=vllm_is_ratio,
    )

    # Verify forward match
    torch.testing.assert_close(triton_loss, trl_loss, rtol=1e-4, atol=1e-4)

    # Verify backward works
    triton_loss.backward()
    assert logits_triton.grad is not None
    assert not torch.isnan(logits_triton.grad).any()

    # Also verify that vllm_is_ratio=None gives same result as vllm_is_ratio=1
    logits_no_ratio = logits.clone().requires_grad_(True)
    loss_no_ratio, _ = triton_grpo_loss(
        logits_no_ratio,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level=importance_sampling_level,
        loss_type=loss_type,
        max_completion_length=T,
        reduce=True,
        vllm_is_ratio=None,
    )

    logits_ones_ratio = logits.clone().requires_grad_(True)
    loss_ones_ratio, _ = triton_grpo_loss(
        logits_ones_ratio,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level=importance_sampling_level,
        loss_type=loss_type,
        max_completion_length=T,
        reduce=True,
        vllm_is_ratio=torch.ones(B, T, device=device),
    )

    torch.testing.assert_close(loss_no_ratio, loss_ones_ratio, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "temperature, num_iteration, beta, eps_low, eps_high",
    [(0.7, num_iteration, beta, 0.2, 0.4) for num_iteration in [1, 5] for beta in [0.0, 0.04]],
)
@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 128, 1000),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
@pytest.mark.parametrize("loss_type", ["grpo", "cispo", "sapo"])
def test_grpo_loss_with_vllm_is_ratio(
    B, T, V, temperature, num_iteration, beta, eps_low, eps_high, dtype, atol, rtol, loss_type
):
    """Test that triton_grpo_loss with vllm_is_ratio matches PyTorch reference for all loss types."""
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logits1 = _input.clone().requires_grad_(True)
    logits2 = _input.clone().requires_grad_(True)
    logits3 = _input.clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, T), dtype=torch.int64, device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    completion_mask[:, -20:] = 0

    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32) if beta != 0.0 else None
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32) if num_iteration > 1 else None
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    # Create vLLM IS ratio (random values between 0.001 and 1.0 to simulate typical IS correction)
    vllm_is_ratio = torch.rand(B, T, device=device, dtype=torch.float32) * 0.999 + 0.001

    loss1, kl1, _ = torch_grpo_loss_with_vllm_is(
        logits1,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        vllm_is_ratio,
        loss_type=loss_type,
    )
    loss2, kl2, _ = triton_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace=True,
        vllm_is_ratio=vllm_is_ratio,
        loss_type=loss_type,
    )
    loss3, kl3, _ = torch_grpo_loss_with_vllm_is(
        logits3,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        vllm_is_ratio,
        loss_type=loss_type,
    )

    dy = torch.randn_like(loss3)
    loss1.backward(dy)
    loss2.backward(dy)
    loss3.backward(dy)

    # Compare triton bf16 vs torch fp32
    assert_verbose_allclose(loss2, loss3, atol=atol, rtol=rtol)
    if kl2 is not None and kl3 is not None:
        assert_verbose_allclose(kl2, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits2.grad, logits3.grad, atol=atol, rtol=rtol)

    # Verify vllm_is_ratio=None gives same result as vllm_is_ratio=ones
    logits_none = _input.clone().float().requires_grad_(True)
    logits_ones = _input.clone().float().requires_grad_(True)
    loss_none, _, _ = triton_grpo_loss(
        logits_none,
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
        vllm_is_ratio=None,
        loss_type=loss_type,
    )
    loss_ones, _, _ = triton_grpo_loss(
        logits_ones,
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
        vllm_is_ratio=torch.ones(B, T, device=device, dtype=torch.float32),
        loss_type=loss_type,
    )
    assert_verbose_allclose(loss_none, loss_ones, atol=1e-5, rtol=1e-5)

    # Verify (B, 1) shape gives same result as (B, T) with uniform value
    uniform_val = 0.42
    logits_b1 = _input.clone().float().requires_grad_(True)
    logits_bt = _input.clone().float().requires_grad_(True)
    loss_b1, _, _ = triton_grpo_loss(
        logits_b1,
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
        vllm_is_ratio=torch.full((B, 1), uniform_val, device=device, dtype=torch.float32),
        loss_type=loss_type,
    )
    loss_bt, _, _ = triton_grpo_loss(
        logits_bt,
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
        vllm_is_ratio=torch.full((B, T), uniform_val, device=device, dtype=torch.float32),
        loss_type=loss_type,
    )
    loss_b1.backward(dy)
    loss_bt.backward(dy)
    assert_verbose_allclose(loss_b1, loss_bt, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(logits_b1.grad, logits_bt.grad, atol=1e-5, rtol=1e-5)

    # Verify 1D (B,) shape gives same result as (B, 1)
    logits_1d = _input.clone().float().requires_grad_(True)
    logits_2d = _input.clone().float().requires_grad_(True)
    loss_1d, _, _ = triton_grpo_loss(
        logits_1d,
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
        vllm_is_ratio=torch.full((B,), uniform_val, device=device, dtype=torch.float32),
        loss_type=loss_type,
    )
    loss_2d, _, _ = triton_grpo_loss(
        logits_2d,
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
        vllm_is_ratio=torch.full((B, 1), uniform_val, device=device, dtype=torch.float32),
        loss_type=loss_type,
    )
    loss_1d.backward(dy)
    loss_2d.backward(dy)
    assert_verbose_allclose(loss_1d, loss_2d, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(logits_1d.grad, logits_2d.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_grpo_loss_sequence_backward_matches_reference(beta):
    """Sequence-level importance sampling should match reference gradients."""
    pytest.importorskip("triton")
    torch.manual_seed(0)

    B, T, V = 2, 8, 32
    logits = torch.randn(B, T + 1, V, device=device, dtype=torch.float32)
    completion_ids = torch.randint(0, V, (B, T), device=device)
    completion_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    with torch.no_grad():
        log_probs = torch.log_softmax(logits[:, :-1, :] / 1.1, dim=-1)
        current_logp = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        old_logp = current_logp + torch.randn_like(current_logp) * 0.2
        ref_logp = current_logp + torch.randn_like(current_logp) * 0.1 if beta != 0.0 else None

    temperature = 1.1
    eps_low, eps_high = 0.2, 0.4

    logits_triton = logits.clone().requires_grad_(True)
    triton_loss, _ = triton_grpo_loss(
        logits_triton,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        importance_sampling_level="sequence",
        loss_type="grpo",
        max_completion_length=T,
        reduce=True,
    )
    triton_loss.backward()

    logits_ref = logits.clone().requires_grad_(True)
    reference_loss = trl_reference_grpo_loss(
        logits_ref,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        loss_type="grpo",
        importance_sampling_level="sequence",
    )
    reference_loss.backward()

    torch.testing.assert_close(triton_loss, reference_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logits_triton.grad, logits_ref.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "temperature, num_iteration, beta, eps_high",
    [(0.7, num_iteration, beta, 5.0) for num_iteration in [1, 5] for beta in [0.0, 0.04]],
)
@pytest.mark.parametrize(
    "B, T, V",
    [
        (1, 1024, 151936),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
def test_cispo_loss(B, T, V, temperature, num_iteration, beta, eps_high, dtype, atol, rtol):
    """Test CISPO loss type support in Triton kernel."""
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logits1 = _input.clone().requires_grad_(True)
    logits2 = _input.clone().requires_grad_(True)
    logits3 = _input.clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, T), dtype=torch.int64, device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    completion_mask[:, -100:] = 0

    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32) if beta != 0.0 else None
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32) if num_iteration > 1 else None
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    loss1, kl1, is_clipped1 = torch_cispo_loss(
        logits1, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_high
    )

    loss2, kl2, is_clipped2 = triton_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low=0.2,  # not used for CISPO
        eps_high=eps_high,
        inplace=True,
        loss_type="cispo",
    )

    loss3, kl3, is_clipped3 = torch_cispo_loss(
        logits3, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_high
    )

    dy = torch.randn_like(loss3)
    loss1.backward(dy)
    loss2.backward(dy)
    loss3.backward(dy)

    assert_verbose_allclose(loss1, loss3, atol=atol, rtol=rtol)
    if kl1 is not None and kl3 is not None:
        assert_verbose_allclose(kl1, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits1.grad, logits3.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(loss2, loss3, atol=atol, rtol=rtol)
    if kl2 is not None and kl3 is not None:
        assert_verbose_allclose(kl2, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits2.grad, logits3.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "temperature, num_iteration, beta, sapo_temp_pos, sapo_temp_neg",
    [(0.7, num_iteration, beta, 1.0, 1.05) for num_iteration in [1, 5] for beta in [0.0, 0.04]],
)
@pytest.mark.parametrize(
    "B, T, V",
    [
        (1, 1024, 151936),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 5e-2, 5e-1),
    ],
)
def test_sapo_loss(B, T, V, temperature, num_iteration, beta, sapo_temp_pos, sapo_temp_neg, dtype, atol, rtol):
    """Test SAPO loss type support in Triton kernel."""
    _input = torch.randn(B, T + 1, V, device=device, dtype=dtype)

    logits1 = _input.clone().requires_grad_(True)
    logits2 = _input.clone().requires_grad_(True)
    logits3 = _input.clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, T), dtype=torch.int64, device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    completion_mask[:, -100:] = 0

    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32) if beta != 0.0 else None
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32) if num_iteration > 1 else None
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    loss1, kl1, is_clipped1 = torch_sapo_loss(
        logits1,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        sapo_temp_pos,
        sapo_temp_neg,
    )

    loss2, kl2, is_clipped2 = triton_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low=0.2,  # not used for SAPO
        eps_high=0.4,  # not used for SAPO
        inplace=True,
        loss_type="sapo",
        sapo_temperature_pos=sapo_temp_pos,
        sapo_temperature_neg=sapo_temp_neg,
    )

    loss3, kl3, is_clipped3 = torch_sapo_loss(
        logits3,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        sapo_temp_pos,
        sapo_temp_neg,
    )

    dy = torch.randn_like(loss3)
    loss1.backward(dy)
    loss2.backward(dy)
    loss3.backward(dy)

    assert_verbose_allclose(loss1, loss3, atol=atol, rtol=rtol)
    if kl1 is not None and kl3 is not None:
        assert_verbose_allclose(kl1, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits1.grad, logits3.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(loss2, loss3, atol=atol, rtol=rtol)
    if kl2 is not None and kl3 is not None:
        assert_verbose_allclose(kl2, kl3, atol=atol, rtol=rtol)
    assert_verbose_allclose(logits2.grad, logits3.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("loss_type", ["cispo", "sapo"])
def test_triton_sequence_level_rejects_unsupported_loss_types(loss_type):
    """Sequence-level importance sampling should raise ValueError for cispo and sapo."""
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T + 1, V, device=device, dtype=torch.float32).contiguous()
    completion_ids = torch.randint(0, V, (B, T), device=device)
    completion_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="Sequence-level importance sampling is not supported"):
        triton_grpo_loss(
            logits,
            old_logp,
            None,
            completion_ids,
            advantages,
            completion_mask,
            temperature=0.9,
            beta=0.0,
            eps_low=0.2,
            eps_high=0.4,
            importance_sampling_level="sequence",
            loss_type=loss_type,
            reduce=True,
        )
