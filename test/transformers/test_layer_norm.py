import pytest
import torch

from liger_kernel.transformers.layer_norm import LigerLayerNorm


@pytest.mark.parametrize(
    "hidden_size",
    [
        64,
        128,
        256,
        512,
    ],
)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (2, 8),
        (4, 16),
        (8, 32),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_layer_norm(batch_size, seq_len, hidden_size, dtype, atol, rtol):
    torch.manual_seed(0)

    x = torch.randn(
        batch_size, seq_len, hidden_size, dtype=dtype, device="cuda", requires_grad=True
    )
    liger_ln = LigerLayerNorm(hidden_size, eps=1e-6).to(dtype).cuda()
    torch_ln = torch.nn.LayerNorm(hidden_size, eps=1e-6).to(dtype).cuda()

    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(x)
    torch_output = torch_ln(x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)

    assert torch.allclose(x.grad, x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(
        liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol
    )
    assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol)
