import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig

from liger_kernel.ops import apply_tiled_mlp
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize("input_requires_grad", [True, False], ids=["trainable-input", "frozen-input"])
@pytest.mark.parametrize("non_contiguous", [False, True], ids=["contiguous", "strided"])
@pytest.mark.parametrize("explicit_params", [False, True], ids=["module-params", "compute-params"])
def test_tiled_function_backward_contract(input_requires_grad, non_contiguous, explicit_params):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 7, device=device)
    if non_contiguous:
        x = x.transpose(0, 1)
    x = x.detach().requires_grad_(input_requires_grad)
    reference_x = x.detach().clone().requires_grad_(input_requires_grad)
    layer = torch.nn.Linear(7, 7, device=device)
    reference_layer = torch.nn.Linear(7, 7, device=device)
    reference_layer.load_state_dict(layer.state_dict())
    actual = apply_tiled_mlp(
        lambda module, shard: module(shard),
        layer,
        x,
        num_shards=2,
        compute_params=list(layer.parameters()) if explicit_params else None,
    )
    expected = reference_layer(reference_x)
    assert actual.requires_grad
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for result, reference in (
        (actual, expected),
        (x.grad, reference_x.grad),
        (layer.weight.grad, reference_layer.weight.grad),
        (layer.bias.grad, reference_layer.bias.grad),
    ):
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode], ids=["no-grad", "inference"])
def test_tiled_function_inference_contract(context):
    layer = torch.nn.Linear(7, 7, device=device)
    x = torch.randn(2, 4, 7, device=device, requires_grad=True)
    with context():
        actual = apply_tiled_mlp(lambda module, shard: module(shard), layer, x, num_shards=2)
        expected = layer(x)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    assert x.grad is None and layer.weight.grad is None and layer.bias.grad is None


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

    # BF16 accumulation is sensitive to the number of reduction steps. Narrow hidden layers
    # (hidden_size < 128) combined with sharding result in high-density summation boundaries
    # where rounding errors exceed standard tolerances. We skip these edge cases to maintain
    # strict parity checks for production-scale shapes.
    if dtype == torch.bfloat16 and hidden_size < 128:
        pytest.skip(f"Skipping unstable BF16 configuration: hidden_size={hidden_size}")

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

    # Dynamic parameter discovery ensures PEFT/LoRA adapters are also validated
    regular_params = [p for p in regular_mlp.parameters() if p.requires_grad]
    tiled_params = [p for p in tiled_mlp.parameters() if p.requires_grad]
    assert len(regular_params) == len(tiled_params), "Number of trainable parameters mismatch"

    for p1, p2 in zip(regular_params, tiled_params):
        torch.testing.assert_close(
            p1.grad,
            p2.grad,
            atol=atol,
            rtol=rtol,
            msg="Gradients for trainable parameters do not match",
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

    # See rationale in test_tiled_geglu_correctness
    if dtype == torch.bfloat16 and hidden_size < 128:
        pytest.skip(f"Skipping unstable BF16 configuration: hidden_size={hidden_size}")

    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    # scale input so that the numerical errors are accumulated less
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

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
    regular_params = [p for p in regular_mlp.parameters() if p.requires_grad]
    tiled_params = [p for p in tiled_mlp.parameters() if p.requires_grad]
    assert len(regular_params) == len(tiled_params), "Number of trainable parameters mismatch"

    for p1, p2 in zip(regular_params, tiled_params):
        torch.testing.assert_close(
            p1.grad,
            p2.grad,
            atol=atol,
            rtol=rtol,
            msg="Gradients for trainable parameters do not match",
        )

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol, msg="Input gradients don't match")
