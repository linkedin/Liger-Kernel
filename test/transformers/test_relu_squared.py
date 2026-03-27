import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops.relu_squared import LigerReLUSquaredFunction
from liger_kernel.transformers.functional import liger_relu_squared
from liger_kernel.transformers.relu_squared import LigerReLUSquared
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


# ---- PyTorch Reference ----
class TorchReLUSquared(torch.nn.Module):
    def forward(self, x):
        relu_applied = torch.nn.functional.relu(x)
        return torch.square(relu_applied)


# ---- Correctness Test (Module) ----


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8, 4096),
        (4, 16, 2048),
        (1, 1, 1023),
        (3, 7, 256),
        (2, 8),
        (1, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_relu_squared_correctness(shape, dtype, atol, rtol):
    _input = torch.randn(*shape, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    torch_layer = TorchReLUSquared()
    liger_layer = LigerReLUSquared().to(device)

    # Forward
    torch_output = torch_layer(x1)
    liger_output = liger_layer(x2)
    assert_verbose_allclose(torch_output, liger_output, rtol=rtol, atol=atol, extra_info="[output]")

    # Backward
    grad_output = torch.randn_like(_input)
    torch_output.backward(grad_output)
    liger_output.backward(grad_output)
    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol, extra_info="[input.grad]")


# ---- Functional API Test ----


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8, 4096),
        (1, 1, 1023),
        (2, 8),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_liger_relu_squared_functional(shape, dtype, atol, rtol):
    _input = torch.randn(*shape, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    output1 = liger_relu_squared(x1)
    output2 = LigerReLUSquaredFunction.apply(x2)

    assert_verbose_allclose(output1, output2, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(_input)
    output1.backward(grad_output)
    output2.backward(grad_output)
    assert_verbose_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
