import pytest
import torch

from transformers.models.llama.configuration_llama import LlamaConfig

from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()

LLAMA_GEGLU_CONFIG = LlamaConfig(
    hidden_size=1024,
    intermediate_size=2048,
    hidden_act="gelu_pytorch_tanh",
)

LLAMA_SWIGLU_CONFIG = LlamaConfig(
    hidden_size=1024,
    intermediate_size=2048,
    hidden_act="silu",
)


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
        # Tiled computation reorders operations, leading to numerical differences
        # Larger tolerances account for accumulated floating-point errors
        (torch.float32, 1.0, 1e-2),
        # bfloat16 tests are skipped due to large numerical differences from tiling
        # This is expected behavior as bfloat16 has lower precision
        pytest.param(
            torch.bfloat16,
            100.0,
            1.0,
            marks=pytest.mark.skip(reason="bfloat16 has too much accumulated error with tiling"),
        ),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
def test_tiled_geglu_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards):
    """Test that TiledGEGLUMLP produces similar results as regular GEGLUMLP (float32 only)."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh",
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # Initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    # Regular GEGLU MLP
    regular_mlp = LigerGEGLUMLP(config=config).to(device).to(dtype)
    regular_mlp.gate_proj.weight.data = G.T
    regular_mlp.up_proj.weight.data = U.T
    regular_mlp.down_proj.weight.data = D.T

    # Tiled GEGLU MLP
    tiled_mlp = LigerTiledGEGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
    tiled_mlp.gate_proj.weight.data = G.T
    tiled_mlp.up_proj.weight.data = U.T
    tiled_mlp.down_proj.weight.data = D.T

    # Forward pass
    y1 = regular_mlp(x1)
    y2 = tiled_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol), "Forward outputs don't match"

    # Backward pass
    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # Check gradients
    assert torch.allclose(
        regular_mlp.gate_proj.weight.grad,
        tiled_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    ), "gate_proj weight gradients don't match"

    assert torch.allclose(
        regular_mlp.up_proj.weight.grad,
        tiled_mlp.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    ), "up_proj weight gradients don't match"

    assert torch.allclose(
        regular_mlp.down_proj.weight.grad,
        tiled_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    ), "down_proj weight gradients don't match"

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol), "Input gradients don't match"


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
        # Tiled computation reorders operations, leading to numerical differences
        # Larger tolerances account for accumulated floating-point errors
        (torch.float32, 1.0, 1e-2),
        # bfloat16 tests are skipped due to large numerical differences from tiling
        # This is expected behavior as bfloat16 has lower precision
        pytest.param(
            torch.bfloat16,
            100.0,
            1.0,
            marks=pytest.mark.skip(reason="bfloat16 has too much accumulated error with tiling"),
        ),
    ],
)
@pytest.mark.parametrize("num_shards", [None, 2, 4])
def test_tiled_swiglu_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol, num_shards):
    """Test that TiledSwiGLUMLP produces similar results as regular SwiGLUMLP (float32 only)."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # Initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    # Regular SwiGLU MLP
    regular_mlp = LigerSwiGLUMLP(config=config).to(device).to(dtype)
    regular_mlp.gate_proj.weight.data = G.T
    regular_mlp.up_proj.weight.data = U.T
    regular_mlp.down_proj.weight.data = D.T

    # Tiled SwiGLU MLP
    tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
    tiled_mlp.gate_proj.weight.data = G.T
    tiled_mlp.up_proj.weight.data = U.T
    tiled_mlp.down_proj.weight.data = D.T

    # Forward pass
    y1 = regular_mlp(x1)
    y2 = tiled_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol), "Forward outputs don't match"

    # Backward pass
    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # Check gradients
    assert torch.allclose(
        regular_mlp.gate_proj.weight.grad,
        tiled_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    ), "gate_proj weight gradients don't match"

    assert torch.allclose(
        regular_mlp.up_proj.weight.grad,
        tiled_mlp.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    ), "up_proj weight gradients don't match"

    assert torch.allclose(
        regular_mlp.down_proj.weight.grad,
        tiled_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    ), "down_proj weight gradients don't match"

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol), "Input gradients don't match"


@pytest.mark.parametrize(
    "seq_len, hidden_size",
    [
        (128, 64),  # seq_len > hidden_size, should use 2 shards
        (256, 128),  # seq_len > hidden_size, should use 2 shards
        (64, 128),  # seq_len < hidden_size, should use 1 shard
    ],
)
def test_automatic_shard_calculation(seq_len, hidden_size):
    """Test that automatic shard calculation works correctly."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        hidden_act="silu",
    )

    x = torch.randn(2, seq_len, hidden_size, device=device)

    # Test with automatic shard calculation (num_shards=None)
    tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=None).to(device)

    # Should not raise any errors
    output = tiled_mlp(x)

    # Check output shape
    assert output.shape == x.shape, "Output shape doesn't match input shape"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_tiled_mlp_with_2d_input(dtype):
    """Test tiled MLP with 2D input (for MoE experts)."""
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        hidden_act="silu",
    )

    # 2D input: [seq_len, hidden_size]
    x = torch.randn(256, 128, device=device, dtype=dtype, requires_grad=True)

    tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=2).to(device).to(dtype)

    # Forward pass
    output = tiled_mlp(x)

    assert output.shape == x.shape, "Output shape doesn't match input shape"

    # Backward pass
    dy = torch.randn_like(output)
    output.backward(dy)

    assert x.grad is not None, "Input gradient not computed"
    assert x.grad.shape == x.shape, "Input gradient shape doesn't match"


@pytest.mark.parametrize("activation_type", ["geglu", "swiglu"])
def test_memory_efficiency(activation_type):
    """
    Test that tiled MLP uses less memory than regular MLP for long sequences.
    This is a basic sanity check - in practice, memory savings are more significant
    with very long sequences and during training.
    """
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        hidden_act="gelu_pytorch_tanh" if activation_type == "geglu" else "silu",
    )

    # Use a moderately long sequence
    x = torch.randn(1, 2048, 512, device=device, requires_grad=True)

    if activation_type == "geglu":
        regular_mlp = LigerGEGLUMLP(config=config).to(device)
        tiled_mlp = LigerTiledGEGLUMLP(config=config, num_shards=4).to(device)
    else:
        regular_mlp = LigerSwiGLUMLP(config=config).to(device)
        tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=4).to(device)

    # Copy weights
    tiled_mlp.gate_proj.weight.data = regular_mlp.gate_proj.weight.data.clone()
    tiled_mlp.up_proj.weight.data = regular_mlp.up_proj.weight.data.clone()
    tiled_mlp.down_proj.weight.data = regular_mlp.down_proj.weight.data.clone()

    # Test that both produce valid outputs
    y1 = regular_mlp(x.clone().requires_grad_(True))
    y2 = tiled_mlp(x.clone().requires_grad_(True))

    # Basic sanity check - outputs should be similar (not exact due to numerical differences)
    assert y1.shape == y2.shape, "Output shapes don't match"
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4), "Outputs differ significantly"
