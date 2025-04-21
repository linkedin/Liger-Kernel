import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.element_wise_dyt import LigerDyT


# @torch.compile    
def torch_dyt_with_beta(x, alpha, gamma, beta):
    return gamma * torch.tanh(x * alpha) + beta

# @torch.compile    
def torch_dyt_without_beta(x, alpha, gamma):
    return gamma * torch.tanh(x * alpha)

class TorchDyT(torch.nn.Module):
    def __init__(self, hidden_size, beta=True, init_alpha=0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = None
        if beta:
            self.beta = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        if self.beta is None:
            return torch_dyt_without_beta(x, self.alpha, self.gamma)
        return torch_dyt_with_beta(x, self.alpha, self.gamma, self.beta)


set_seed(42)
device = infer_device()


@pytest.mark.parametrize("have_beta", [False, True])
@pytest.mark.parametrize("init_alpha", [0.5])
@pytest.mark.parametrize(
    "B, T, hidden_size",
    [
        (2, 8, 4096),
        (4, 16, 2048),
        (1, 1, 1023),  # Minimal batch/seq with near power-of-2 hidden
        (3, 7, 256),  # Prime numbers for batch/seq
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_dyt_correctness(B, T, hidden_size, have_beta, init_alpha, dtype, atol, rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    alpha = torch.randn(1, device=device, dtype=dtype)
    gamma = torch.randn(hidden_size, device=device, dtype=dtype)
    beta = torch.randn(hidden_size, device=device, dtype=dtype)

    torch_dyt = TorchDyT(hidden_size=hidden_size, beta=have_beta, init_alpha=init_alpha).to(device).to(dtype)
    torch_dyt.alpha.data = alpha.clone()
    torch_dyt.gamma.data = gamma.clone()
    if have_beta:
        torch_dyt.beta.data = beta.clone()

    liger_dyt = LigerDyT(hidden_size=hidden_size, beta=have_beta, init_alpha=init_alpha).to(device).to(dtype)
    liger_dyt.alpha.data = alpha.clone()
    liger_dyt.gamma.data = gamma.clone()
    if have_beta:
        liger_dyt.beta.data = beta.clone()

    torch_output = torch_dyt(x1)
    liger_output = liger_dyt(x2)

    assert_verbose_allclose(torch_output, liger_output, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(_input)
    torch_output.backward(grad_output)
    liger_output.backward(grad_output)

    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(torch_dyt.alpha.grad, liger_dyt.alpha.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(torch_dyt.gamma.grad, liger_dyt.gamma.grad, rtol=rtol, atol=atol)
    if have_beta:
        assert_verbose_allclose(torch_dyt.beta.grad, liger_dyt.beta.grad, rtol=rtol, atol=atol)
