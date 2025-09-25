import pytest
import torch
import torch.nn as nn

from test.utils import supports_bfloat16
from liger_kernel.ops.silu import LigerSiLUFunction
from liger_kernel.transformers.functional import liger_silu
from liger_kernel.utils import infer_device

device = infer_device()


class StandardSiLUMLP(nn.Module):
    """Standard MLP with SiLU activation"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.silu(self.gate_proj(x)))


class LigerSiLUMLP(nn.Module):
    """Liger-optimized MLP with SiLU activation"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate_output = self.gate_proj(x)
        silu_output = LigerSiLUFunction.apply(gate_output)
        return self.down_proj(silu_output)


SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # SiLU is simpler than SwiGLU, so we can use tighter tolerances
        (torch.float32, 1e-2, 1e-4),
        pytest.param(
            torch.bfloat16,
            1e-0,
            1e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
def test_correctness_silu_mlp(
    bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol
):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    W1 = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    W2 = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    standard_mlp = StandardSiLUMLP(hidden_size, intermediate_size).to(device).to(dtype)
    standard_mlp.gate_proj.weight.data = W1.T
    standard_mlp.down_proj.weight.data = W2.T

    liger_mlp = LigerSiLUMLP(hidden_size, intermediate_size).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = W1.T
    liger_mlp.down_proj.weight.data = W2.T

    y1 = standard_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        standard_mlp.gate_proj.weight.grad,
        liger_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        standard_mlp.down_proj.weight.grad,
        liger_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 8, 8),
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-3,
            1e-3,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
def test_correctness_functional_silu(bsz, seq_len, size, dtype, atol, rtol):
    """Test functional SiLU implementation"""
    input_tensor = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)

    x1 = input_tensor.clone().requires_grad_(True)
    x2 = input_tensor.clone().requires_grad_(True)

    # Compare against PyTorch's built-in SiLU
    y1 = torch.nn.functional.silu(x1)
    y2 = LigerSiLUFunction.apply(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    # Test backward pass
    grad_output = torch.randn_like(y1)
    y1.backward(grad_output.clone())
    y2.backward(grad_output.clone())

    # Check if gradients are close
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 4),
        (5, 7),
        (100,),
        (2, 3, 4, 5),
    ],
)
def test_silu_various_shapes(shape):
    """Test SiLU with various tensor shapes"""
    x = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)

    # Standard SiLU
    y_std = torch.nn.functional.silu(x)

    # Liger SiLU
    x_liger = x.clone().detach().requires_grad_(True)
    y_liger = LigerSiLUFunction.apply(x_liger)

    assert torch.allclose(y_std, y_liger, atol=1e-6, rtol=1e-6)

    # Test gradients
    grad_out = torch.randn_like(y_std)
    y_std.backward(grad_out.clone())
    y_liger.backward(grad_out.clone())

    assert torch.allclose(x.grad, x_liger.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 8, 8),
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-6, 1e-5),
        (torch.bfloat16, 1e-3, 1e-3),
    ],
)
def test_correctness_functional(bsz, seq_len, size, dtype, atol, rtol):
    input_tensor = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)

    x1 = input_tensor.clone().requires_grad_(True)
    x2 = input_tensor.clone().requires_grad_(True)

    y1 = liger_silu(x1)
    y2 = LigerSiLUFunction.apply(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    # Test backward pass
    grad_output = torch.randn_like(y1)
    y1.backward(grad_output)
    y2.backward(grad_output)

    # Check if gradients are close
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
