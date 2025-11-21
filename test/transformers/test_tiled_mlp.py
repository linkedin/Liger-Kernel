import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig

from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (1, 1024, 128, 256),  # num_shards=8 if auto
        (2, 1024, 64, 256),  # num_shards=16 if auto
        # weird shapes
        (4, 127, 128, 256),
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
            1e-0,
            1e-0,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
@pytest.mark.parametrize("check_2d", [True, False])
def test_tiled_geglu_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, check_2d):
    """Test that TiledGEGLUMLP produces similar results as regular GEGLUMLP."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh",
    )

    # scale input so that the numerical errors are accumulated less
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    # Convert to 2D input for MoE experts testing
    if check_2d:
        x1 = x1.view(-1, hidden_size)
        x2 = x2.view(-1, hidden_size)

    # Initialize weights
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    # Regular GEGLU MLP
    regular_mlp = LigerGEGLUMLP(config=config).to(device).to(dtype)
    regular_mlp.gate_proj.weight.data = G
    regular_mlp.up_proj.weight.data = U
    regular_mlp.down_proj.weight.data = D

    # Tiled GEGLU MLP
    tiled_mlp = LigerTiledGEGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
    tiled_mlp.gate_proj.weight.data = G
    tiled_mlp.up_proj.weight.data = U
    tiled_mlp.down_proj.weight.data = D

    # Forward pass
    y1 = regular_mlp(x1)
    y2 = tiled_mlp(x2)

    torch.testing.assert_close(y1, y2, atol=atol, rtol=rtol, msg="Forward outputs don't match")

    # Backward pass
    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # Check gradients
    torch.testing.assert_close(
        regular_mlp.gate_proj.weight.grad,
        tiled_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        msg="gate_proj weight gradients don't match",
    )

    torch.testing.assert_close(
        regular_mlp.up_proj.weight.grad,
        tiled_mlp.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        msg="up_proj weight gradients don't match",
    )

    torch.testing.assert_close(
        regular_mlp.down_proj.weight.grad,
        tiled_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        msg="down_proj weight gradients don't match",
    )

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol, msg="Input gradients don't match")


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 512, 512, 1024),
        (1, 1024, 256, 512),
        # weird shapes
        (4, 127, 128, 256),
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
            1e-0,
            1e-0,
            marks=pytest.mark.skip(reason="bfloat16 tests disabled due to numerical instability"),
        ),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
@pytest.mark.parametrize("check_2d", [True, False])
def test_tiled_swiglu_correctness(
    bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards, check_2d
):
    """Test that TiledSwiGLUMLP produces similar results as regular SwiGLUMLP."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    # scale input so that the numerical errors are accumulated less
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1

    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    # Convert to 2D input for MoE experts testing
    if check_2d:
        x1 = x1.view(-1, hidden_size)
        x2 = x2.view(-1, hidden_size)

    # Initialize weights
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    # Regular SwiGLU MLP
    regular_mlp = LigerSwiGLUMLP(config=config).to(device).to(dtype)
    regular_mlp.gate_proj.weight.data = G
    regular_mlp.up_proj.weight.data = U
    regular_mlp.down_proj.weight.data = D

    # Tiled SwiGLU MLP
    tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
    tiled_mlp.gate_proj.weight.data = G
    tiled_mlp.up_proj.weight.data = U
    tiled_mlp.down_proj.weight.data = D

    # Forward pass
    y1 = regular_mlp(x1)
    y2 = tiled_mlp(x2)

    torch.testing.assert_close(y1, y2, atol=atol, rtol=rtol, msg="Forward outputs don't match")

    # Backward pass
    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # Check gradients
    torch.testing.assert_close(
        regular_mlp.gate_proj.weight.grad,
        tiled_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        msg="gate_proj weight gradients don't match",
    )

    torch.testing.assert_close(
        regular_mlp.up_proj.weight.grad,
        tiled_mlp.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        msg="up_proj weight gradients don't match",
    )

    torch.testing.assert_close(
        regular_mlp.down_proj.weight.grad,
        tiled_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
        msg="down_proj weight gradients don't match",
    )

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol, msg="Input gradients don't match")
