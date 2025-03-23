import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops.dyt import LigerDyTFunction
from liger_kernel.transformers.dyt import LigerDyT
from liger_kernel.transformers.functional import liger_dyt


class TorchDyT(nn.Module):
    def __init__(self, hidden_size, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


set_seed(42)
device = infer_device()


@pytest.mark.parametrize("init_alpha", [0.5, 0.2, 1.0])
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
def test_liger_dyt_correctness(B, T, hidden_size, init_alpha, dtype, atol, rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    alpha = torch.randn(1, device=device, dtype=dtype)
    gamma = torch.randn(hidden_size, device=device, dtype=dtype)
    beta = torch.randn(hidden_size, device=device, dtype=dtype)

    torch_dyt = TorchDyT(hidden_size=hidden_size, init_alpha=init_alpha).to(device).to(dtype)
    torch_dyt.alpha.data = alpha.clone()
    torch_dyt.gamma.data = gamma.clone()
    torch_dyt.beta.data = beta.clone()

    liger_dyt = LigerDyT(hidden_size=hidden_size, init_alpha=init_alpha).to(device).to(dtype)
    liger_dyt.alpha.data = alpha.clone()
    liger_dyt.gamma.data = gamma.clone()
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
    assert_verbose_allclose(torch_dyt.beta.grad, liger_dyt.beta.grad, rtol=rtol, atol=atol)


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
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_liger_dyt_functional(B, T, hidden_size, dtype, atol, rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    alpha = torch.randn(1, device=device, dtype=dtype)
    gamma = torch.randn(hidden_size, device=device, dtype=dtype)
    beta = torch.randn(hidden_size, device=device, dtype=dtype)

    alpha1 = alpha.clone().requires_grad_(True)
    gamma1 = gamma.clone().requires_grad_(True)
    beta1 = beta.clone().requires_grad_(True)

    alpha2 = alpha.clone().requires_grad_(True)
    gamma2 = gamma.clone().requires_grad_(True)
    beta2 = beta.clone().requires_grad_(True)

    output1 = liger_dyt(x1, alpha=alpha1, gamma=gamma1, beta=beta1)
    output2 = LigerDyTFunction.apply(x2, alpha2, gamma2, beta2)

    assert_verbose_allclose(output1, output2, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(_input)
    output1.backward(grad_output)
    output2.backward(grad_output)

    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(alpha1.grad, alpha2.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(gamma1.grad, gamma2.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(beta1.grad, beta2.grad, rtol=rtol, atol=atol)
