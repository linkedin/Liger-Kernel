import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerDyTFunction
from liger_kernel.transformers.dyt import LigerDyT
from liger_kernel.transformers.functional import liger_dyt


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


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("init_alpha", [0.5])
@pytest.mark.parametrize(
    "B, T, hidden_size",
    [
        (2, 8, 4096),
        (4, 16, 2048),
        (1, 1, 1023),  # Minimal batch/seq with near power-of-2 hidden
        (3, 7, 256),  # Prime numbers for batch/seq
        (2, 4099, 16),  # Small hidden, several rows per program
        (1, 1001, 61),  # Small odd hidden, ragged row blocks
        (1, 65537, 16),  # More rows than CUDA grid dimension 1 allows
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol, alpha_atol, alpha_rtol",
    [
        (torch.float32, 1e-5, 1e-5, 1e-5, 1e-3),
    ],
)
def test_liger_dyt_correctness(B, T, hidden_size, beta, init_alpha, dtype, atol, rtol, alpha_atol, alpha_rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    alpha = torch.randn(1, device=device, dtype=dtype)
    gamma = torch.randn(hidden_size, device=device, dtype=dtype)
    beta_data = torch.randn(hidden_size, device=device, dtype=dtype)

    torch_dyt = TorchDyT(hidden_size=hidden_size, beta=beta, init_alpha=init_alpha).to(device).to(dtype)
    torch_dyt.alpha.data = alpha.clone()
    torch_dyt.gamma.data = gamma.clone()
    if beta:
        torch_dyt.beta.data = beta_data.clone()

    liger_dyt = LigerDyT(hidden_size=hidden_size, beta=beta, init_alpha=init_alpha).to(device).to(dtype)
    liger_dyt.alpha.data = alpha.clone()
    liger_dyt.gamma.data = gamma.clone()
    if beta:
        liger_dyt.beta.data = beta_data.clone()

    torch_output = torch_dyt(x1)
    liger_output = liger_dyt(x2)

    assert_verbose_allclose(torch_output, liger_output, rtol=rtol, atol=atol, extra_info="[output]")

    grad_output = torch.randn_like(_input)
    torch_output.backward(grad_output)
    liger_output.backward(grad_output)

    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol, extra_info="[input.grad]")
    assert_verbose_allclose(
        torch_dyt.alpha.grad, liger_dyt.alpha.grad, rtol=alpha_rtol, atol=alpha_atol, extra_info="[alpha.grad]"
    )
    assert_verbose_allclose(torch_dyt.gamma.grad, liger_dyt.gamma.grad, rtol=rtol, atol=atol, extra_info="[gamma.grad]")
    if beta:
        assert_verbose_allclose(
            torch_dyt.beta.grad, liger_dyt.beta.grad, rtol=rtol, atol=atol, extra_info="[beta.grad]"
        )


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize(
    "B, T, hidden_size",
    [
        (2, 8, 4096),
        (4, 16, 2048),
        (1, 1, 1023),  # Minimal batch/seq with near power-of-2 hidden
        (3, 7, 256),  # Prime numbers for batch/seq
        (2, 4099, 16),  # Small hidden, several rows per program
        (1, 1001, 61),  # Small odd hidden, ragged row blocks
        (1, 65537, 16),  # More rows than CUDA grid dimension 1 allows
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
def test_liger_dyt_functional(B, T, hidden_size, beta, dtype, atol, rtol):
    _input = torch.randn(B, T, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    alpha = torch.randn(1, device=device, dtype=dtype)
    gamma = torch.randn(hidden_size, device=device, dtype=dtype)
    beta_data = torch.randn(hidden_size, device=device, dtype=dtype)

    alpha1 = alpha.clone().requires_grad_(True)
    gamma1 = gamma.clone().requires_grad_(True)
    beta1 = beta_data.clone().requires_grad_(True) if beta else None

    alpha2 = alpha.clone().requires_grad_(True)
    gamma2 = gamma.clone().requires_grad_(True)

    beta2 = beta_data.clone().requires_grad_(True) if beta else None

    output1 = liger_dyt(x1, alpha=alpha1, gamma=gamma1, beta=beta1)
    output2 = LigerDyTFunction.apply(x2, alpha2, gamma2, beta2)

    assert_verbose_allclose(output1, output2, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(_input)
    output1.backward(grad_output)
    output2.backward(grad_output)

    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(alpha1.grad, alpha2.grad, rtol=rtol, atol=atol)
    assert_verbose_allclose(gamma1.grad, gamma2.grad, rtol=rtol, atol=atol)
    if beta:
        assert_verbose_allclose(beta1.grad, beta2.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rows", [1, 257])
def test_liger_dyt_ragged_rows_with_infinite_alpha(rows):
    # 257 rows tile raggedly when several rows share a program. Padded rows
    # must not poison the parameter gradients: tanh(inf * 0) is NaN.
    x = torch.ones(rows, 16, device=device)
    alpha = torch.full((1,), float("inf"), device=device, requires_grad=True)
    gamma = torch.ones(16, device=device, requires_grad=True)
    beta = torch.zeros(16, device=device, requires_grad=True)
    output = LigerDyTFunction.apply(x, alpha, gamma, beta)
    dalpha, dgamma, dbeta = torch.autograd.grad(output, (alpha, gamma, beta), torch.ones_like(output))
    assert_verbose_allclose(dalpha, torch.zeros_like(alpha))
    assert_verbose_allclose(dgamma, torch.full_like(gamma, rows))
    assert_verbose_allclose(dbeta, torch.full_like(beta, rows))


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_liger_dyt_bfloat16_input_float32_parameters():
    # 257 is not representable in bfloat16 (rounds to 256), so rounding
    # dalpha or dbeta through x.dtype loses integer precision on float32
    # parameters.
    rows, channels = 257, 16
    x = torch.ones(rows, channels, device=device, dtype=torch.bfloat16, requires_grad=True)
    alpha = torch.zeros(1, device=device, dtype=torch.float32, requires_grad=True)
    gamma = torch.ones(channels, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.zeros(channels, device=device, dtype=torch.float32, requires_grad=True)
    output = LigerDyTFunction.apply(x, alpha, gamma, beta)
    assert output.dtype == torch.bfloat16
    dalpha, dgamma, dbeta = torch.autograd.grad(output, (alpha, gamma, beta), torch.ones_like(output))
    assert dalpha.dtype == torch.float32
    assert dgamma.dtype == torch.float32
    assert dbeta.dtype == torch.float32
    assert_verbose_allclose(dalpha, torch.full_like(alpha, rows * channels))
    assert_verbose_allclose(dgamma, torch.zeros_like(gamma))
    assert_verbose_allclose(dbeta, torch.full_like(beta, rows))
