import pytest
import torch

from liger_kernel.ops.cutedsl.ops.grpo_loss import fused_linear_selective_logprob

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CuTe DSL GRPO requires CUDA"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="CuTe DSL GRPO requires an SM100 GPU",
    ),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("hidden_size", [96, 128])
def test_fused_linear_selective_logprob(dtype, with_bias, hidden_size):
    torch.manual_seed(42)
    token_count, vocab_size = 33, 513
    x_master = 0.1 * torch.randn(token_count, hidden_size, device="cuda", dtype=torch.float32)
    w_master = 0.1 * torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32)
    b_master = 0.1 * torch.randn(vocab_size, device="cuda", dtype=torch.float32) if with_bias else None
    target = torch.randint(0, vocab_size, (token_count,), device="cuda")
    grad_output = torch.randn(token_count, device="cuda", dtype=torch.float32)

    x = x_master.to(dtype).requires_grad_()
    w = w_master.to(dtype).requires_grad_()
    b = b_master.to(dtype).requires_grad_() if b_master is not None else None
    actual = fused_linear_selective_logprob(x, w, target, b)
    actual.backward(grad_output)

    x_ref = x_master.requires_grad_()
    w_ref = w_master.requires_grad_()
    b_ref = b_master.requires_grad_() if b_master is not None else None
    logits = x_ref @ w_ref.t()
    if b_ref is not None:
        logits = logits + b_ref
    expected = torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    expected.backward(grad_output)

    atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-2)
    torch.testing.assert_close(x.grad.float(), x_ref.grad, atol=atol, rtol=5e-2)
    torch.testing.assert_close(w.grad.float(), w_ref.grad, atol=atol, rtol=5e-2)
    if with_bias:
        torch.testing.assert_close(b.grad.float(), b_ref.grad, atol=atol, rtol=5e-2)
