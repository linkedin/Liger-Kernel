import pytest
import torch
import torch.nn.functional as F

from liger_kernel.ops.selective_logprob import selective_logprob_backward
from liger_kernel.ops.selective_logprob import selective_logprob_forward
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()


@pytest.mark.parametrize("B, V", [(4, 128), (16, 1024), (8, 4096), (2, 32000), (3, 5000)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("temperature", [0.8, 1.0, 1.5])
def test_selective_logprob_forward_backward(B, V, dtype, has_bias, temperature):
    set_seed()
    if device == "cpu" and dtype == torch.bfloat16:
        pytest.skip("bfloat16 not well supported on CPU")

    logits_ref = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)
    logits_act = logits_ref.detach().clone().requires_grad_(False)
    bias = torch.randn(V, device=device, dtype=dtype) if has_bias else None

    # Targets with some ignore_index (-100)
    targets = torch.randint(0, V, (B,), device=device, dtype=torch.long)
    targets[0] = -100

    # Reference computation in float32 for high accuracy
    ref_x = logits_ref.float()
    if has_bias:
        ref_x = ref_x + bias.float()
    ref_x = ref_x / temperature
    ref_lse = torch.logsumexp(ref_x, dim=-1)
    ref_log_softmax = F.log_softmax(ref_x, dim=-1)

    valid_mask = targets != -100
    safe_targets = torch.where(valid_mask, targets, 0)
    ref_logprob = ref_log_softmax.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    ref_logprob = torch.where(valid_mask, ref_logprob, torch.zeros_like(ref_logprob))

    # Kernel forward
    act_logprob, act_log_z = selective_logprob_forward(
        logits_act,
        targets,
        bias=bias,
        temperature=temperature,
        ignore_index=-100,
        return_log_z=True,
    )

    atol = 1e-4 if dtype == torch.float32 else 2e-2
    rtol = 1e-4 if dtype == torch.float32 else 2e-2

    assert_verbose_allclose(act_logprob, ref_logprob, atol=atol, rtol=rtol)
    assert_verbose_allclose(act_log_z[valid_mask], ref_lse[valid_mask], atol=atol, rtol=rtol)

    # Test backward
    grad_logprobs = torch.randn(B, device=device, dtype=torch.float32)
    # Target loss: sum(grad_logprobs * logprob)
    loss_ref = (ref_logprob * grad_logprobs).sum()
    loss_ref.backward()
    ref_grad_logits = logits_ref.grad

    # Act backward (modifies logits_act in-place)
    act_grad_logits = selective_logprob_backward(
        logits_act,
        targets,
        grad_logprobs,
        bias=bias,
        temperature=temperature,
        ignore_index=-100,
    )

    assert_verbose_allclose(act_grad_logits, ref_grad_logits, atol=atol, rtol=rtol)
    # Check ignored token row gradient is all zeros
    assert torch.all(act_grad_logits[0] == 0.0)
