import pytest
import torch
import torch.nn.functional as F

from test.utils import infer_device
from test.utils import set_seed

from liger_kernel.ops.grpo_loss import fused_selective_log_softmax
from liger_kernel.transformers.grpo_loss import triton_grpo_loss


def compare(x, y, extra_str=""):
    if x is None or y is None:
        return
    if any([x.dtype == torch.float32, y.dtype == torch.float32]):
        x, y = x.float(), y.float()
    diff = (x - y).abs()
    diff = diff / (torch.max(x.abs(), y.abs()) + 1e-5)
    print(f"{extra_str}Max difference: {diff.max().item()}, Mean difference: {diff.mean().item()}")


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


def torch_grpo_loss(
    logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_low, eps_high
):
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True
    logits = logits[:, :-1]

    def get_log_probs(logits, input_ids):
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids[:, -logits.size(1) :]):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    per_token_logps = get_log_probs(logits / temperature, completion_ids)
    # return per_token_logps, None, None
    ref_per_token_logps = ref_logp

    if old_logp is None:
        old_logp = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_logp)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    per_token_loss = per_token_loss * completion_mask if completion_mask is not None else per_token_loss

    per_token_kl = None
    if beta != 0.0:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl
    is_clipped = (per_token_loss1 < per_token_loss2).float()
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
        (torch.bfloat16, 1e-5, 1e-5),
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

    # assert_verbose_allclose(torch_bf16_logp, torch_fp32_logp, rtol=rtol, atol=atol)
    # assert_verbose_allclose(triton_bf16_logp, torch_fp32_logp, rtol=rtol, atol=atol)
    print("\n" + "=" * 20 + " selective_log_softmax " + "=" * 20)
    compare(torch_bf16_logp, torch_fp32_logp, "torch-bf16 vs torch-fp32, ")
    compare(triton_bf16_logp, torch_fp32_logp, "triton-bf16 vs torch-fp32, ")


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
        (torch.bfloat16, 1e-5, 1e-5),
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

    print("\n" + "=" * 20 + " grpo_loss " + "=" * 20)
    compare(loss1, loss3, "per_token_loss: torch-bf16 vs torch-fp32, ")
    compare(kl1, kl3, "per_token_kl: torch-bf16 vs torch-fp32, ")
    compare(logits1.grad, logits3.grad, "logits.grad: torch-bf16 vs torch-fp32, ")
    compare(loss2, loss3, "per_token_loss: triton-bf16 vs torch-fp32, ")
    compare(kl2, kl3, "per_token_kl: triton-bf16 vs torch-fp32, ")
    compare(logits2.grad, logits3.grad, "logits.grad: triton-bf16 vs torch-fp32, ")
