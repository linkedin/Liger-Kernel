import math

import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.transformers.functional import liger_geglu
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="gelu_pytorch_tanh",
)
SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2048, 2048, 4096),
        # weird shapes
        (9, 41, 341, 4231),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 2e-6),
        pytest.param(
            torch.bfloat16,
            # For NPU: use quack's distance-based comparison method (tolerance params not used)
            # Reference for quack method: https://github.com/Dao-AILab/quack/blob/9a333c70288a07e135e415f9c2ae96520178ecf5/tests/test_linear.py#L65
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    # For NPU + bfloat16: use quack's distance-based comparison method
    # For GPU + bfloat16: use direct comparison
    # For float32: use direct comparison
    if dtype == torch.bfloat16 and device == "npu":
        _test_correctness_quack_method(bsz, seq_len, hidden_size, intermediate_size)
    else:
        # For GPU + bfloat16 or float32, use direct comparison
        _test_correctness_direct(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol)


def _test_correctness_quack_method(bsz, seq_len, hidden_size, intermediate_size):
    """Test using quack's distance-based comparison method."""
    torch.manual_seed(0)

    # Create inputs in fp32, then convert to bf16
    _input_fp32 = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=torch.float32)
    _input_bf16 = _input_fp32.to(torch.bfloat16)

    x_fp32 = _input_fp32.clone().requires_grad_(True)
    x_bf16_ref = _input_bf16.clone().requires_grad_(True)
    x_bf16_custom = _input_bf16.clone().requires_grad_(True)

    # Initialize weights with scaled initialization (following quack: 1/sqrt(in_features))
    scale_g = 1.0 / math.sqrt(hidden_size)
    scale_u = 1.0 / math.sqrt(hidden_size)
    scale_d = 1.0 / math.sqrt(intermediate_size)

    G_fp32 = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float32) * scale_g
    U_fp32 = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float32) * scale_u
    D_fp32 = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float32) * scale_d

    G_bf16 = G_fp32.to(torch.bfloat16)
    U_bf16 = U_fp32.to(torch.bfloat16)
    D_bf16 = D_fp32.to(torch.bfloat16)

    # Reference implementations
    llama_mlp_fp32 = LlamaMLP(config=LLAMA_CONFIG).to(device).to(torch.float32)
    llama_mlp_fp32.gate_proj.weight.data = G_fp32.T
    llama_mlp_fp32.up_proj.weight.data = U_fp32.T
    llama_mlp_fp32.down_proj.weight.data = D_fp32.T

    llama_mlp_bf16 = LlamaMLP(config=LLAMA_CONFIG).to(device).to(torch.bfloat16)
    llama_mlp_bf16.gate_proj.weight.data = G_bf16.T
    llama_mlp_bf16.up_proj.weight.data = U_bf16.T
    llama_mlp_bf16.down_proj.weight.data = D_bf16.T

    liger_mlp_bf16 = LigerGEGLUMLP(config=LLAMA_CONFIG).to(device).to(torch.bfloat16)
    liger_mlp_bf16.gate_proj.weight.data = G_bf16.T
    liger_mlp_bf16.up_proj.weight.data = U_bf16.T
    liger_mlp_bf16.down_proj.weight.data = D_bf16.T

    # Forward pass
    y_fp32_ref = llama_mlp_fp32(x_fp32)
    y_bf16_ref = llama_mlp_bf16(x_bf16_ref)
    y_bf16_custom = liger_mlp_bf16(x_bf16_custom)

    # Quack's method: compare distances to fp32 reference
    # Custom bf16 distance to fp32 should be < 2 * ref bf16 distance to fp32 + 1e-6
    dist_custom = (y_bf16_custom.float() - y_fp32_ref).abs()
    dist_ref = (y_bf16_ref.float() - y_fp32_ref).abs()
    max_dist_custom = dist_custom.max().item()
    max_dist_ref = dist_ref.max().item()

    assert max_dist_custom < 2 * max_dist_ref + 1e-6, (
        f"Output distance to fp32 reference too large: "
        f"custom={max_dist_custom:.6e}, ref={max_dist_ref:.6e}, "
        f"threshold={2 * max_dist_ref + 1e-6:.6e}"
    )

    # Backward pass
    dy_fp32 = torch.randn_like(y_fp32_ref)
    dy_bf16 = dy_fp32.to(torch.bfloat16)

    y_fp32_ref.backward(dy_fp32.clone(), retain_graph=True)
    y_bf16_ref.backward(dy_bf16.clone(), retain_graph=True)
    y_bf16_custom.backward(dy_bf16.clone(), retain_graph=True)

    # Check gradients using quack's method
    def _check_grad_quack(grad_custom, grad_ref_bf16, grad_ref_fp32, name):
        dist_custom = (grad_custom.float() - grad_ref_fp32).abs()
        dist_ref = (grad_ref_bf16.float() - grad_ref_fp32).abs()
        max_dist_custom = dist_custom.max().item()
        max_dist_ref = dist_ref.max().item()
        assert max_dist_custom < 2 * max_dist_ref + 1e-6, (
            f"{name} gradient distance to fp32 reference too large: "
            f"custom={max_dist_custom:.6e}, ref={max_dist_ref:.6e}, "
            f"threshold={2 * max_dist_ref + 1e-6:.6e}"
        )

    _check_grad_quack(
        liger_mlp_bf16.gate_proj.weight.grad,
        llama_mlp_bf16.gate_proj.weight.grad,
        llama_mlp_fp32.gate_proj.weight.grad,
        "gate_proj.weight",
    )

    _check_grad_quack(
        liger_mlp_bf16.up_proj.weight.grad,
        llama_mlp_bf16.up_proj.weight.grad,
        llama_mlp_fp32.up_proj.weight.grad,
        "up_proj.weight",
    )

    _check_grad_quack(
        liger_mlp_bf16.down_proj.weight.grad,
        llama_mlp_bf16.down_proj.weight.grad,
        llama_mlp_fp32.down_proj.weight.grad,
        "down_proj.weight",
    )

    _check_grad_quack(x_bf16_custom.grad, x_bf16_ref.grad, x_fp32.grad, "input")


def _test_correctness_direct(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    """Test using direct comparison (for GPU + bfloat16 or float32)."""
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # Initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    llama_mlp = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    llama_mlp.gate_proj.weight.data = G.T
    llama_mlp.up_proj.weight.data = U.T
    llama_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerGEGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = G.T
    liger_mlp.up_proj.weight.data = U.T
    liger_mlp.down_proj.weight.data = D.T

    y1 = llama_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol) is True

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert (
        torch.allclose(
            llama_mlp.gate_proj.weight.grad,
            liger_mlp.gate_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )
    assert (
        torch.allclose(
            llama_mlp.up_proj.weight.grad,
            liger_mlp.up_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )
    assert (
        torch.allclose(
            llama_mlp.down_proj.weight.grad,
            liger_mlp.down_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol) is True


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_correctness_functional(bsz, seq_len, size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)
    _b = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    b1 = _b.clone().requires_grad_(True)
    b2 = _b.clone().requires_grad_(True)

    y1 = liger_geglu(a=x1, b=b1)
    y2 = LigerGELUMulFunction.apply(x2, b2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)
