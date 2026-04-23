import math
from types import SimpleNamespace

import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from liger_kernel.ops import LigerGELUMulFunction
from liger_kernel.transformers.functional import liger_geglu
from liger_kernel.transformers.geglu import LigerGEGLUMLP, LigerGEGLUMLPForGemma4
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


# ---------------------------------------------------------------------------
# Gemma 4 double-wide MLP edge cases
# ---------------------------------------------------------------------------


def _make_gemma4_config(
    hidden_size=2048,
    intermediate_size=4096,
    num_hidden_layers=32,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    hidden_activation="gelu_pytorch_tanh",
):
    """Minimal fake config matching Gemma4TextConfig's MLP-relevant fields."""
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_kv_shared_layers=num_kv_shared_layers,
        use_double_wide_mlp=use_double_wide_mlp,
        hidden_activation=hidden_activation,
    )


def test_gemma4_mlp_no_doubling_without_layer_idx():
    """layer_idx=None (default) → standard intermediate_size, no doubling."""
    cfg = _make_gemma4_config(use_double_wide_mlp=True, num_kv_shared_layers=8)
    mlp = LigerGEGLUMLPForGemma4(cfg)
    assert mlp.intermediate_size == cfg.intermediate_size


def test_gemma4_mlp_no_doubling_when_flag_false():
    """use_double_wide_mlp=False (31B production) → never doubles."""
    cfg = _make_gemma4_config(use_double_wide_mlp=False)
    for layer_idx in [0, 15, 31]:
        mlp = LigerGEGLUMLPForGemma4(cfg, layer_idx=layer_idx)
        assert mlp.intermediate_size == cfg.intermediate_size


def test_gemma4_mlp_no_doubling_for_non_kv_shared_layer():
    """Early layers (before KV-sharing starts) → standard size."""
    cfg = _make_gemma4_config(
        num_hidden_layers=32,
        num_kv_shared_layers=8,
        use_double_wide_mlp=True,
    )
    # first_kv_shared = 32 - 8 = 24.  Layer 0 and 23 are NOT shared.
    for layer_idx in [0, 10, 23]:
        mlp = LigerGEGLUMLPForGemma4(cfg, layer_idx=layer_idx)
        assert mlp.intermediate_size == cfg.intermediate_size, (
            f"Layer {layer_idx} should NOT be doubled"
        )


def test_gemma4_mlp_doubles_for_kv_shared_layer():
    """KV-shared layers with use_double_wide_mlp=True → doubled intermediate_size."""
    cfg = _make_gemma4_config(
        hidden_size=2048,
        intermediate_size=4096,
        num_hidden_layers=32,
        num_kv_shared_layers=8,
        use_double_wide_mlp=True,
    )
    # first_kv_shared = 32 - 8 = 24.  Layers 24-31 are KV-shared → doubled.
    for layer_idx in [24, 28, 31]:
        mlp = LigerGEGLUMLPForGemma4(cfg, layer_idx=layer_idx)
        assert mlp.intermediate_size == cfg.intermediate_size * 2, (
            f"Layer {layer_idx} should be doubled"
        )
        assert mlp.gate_proj.in_features == cfg.hidden_size
        assert mlp.gate_proj.out_features == cfg.intermediate_size * 2
        assert mlp.up_proj.out_features == cfg.intermediate_size * 2
        assert mlp.down_proj.in_features == cfg.intermediate_size * 2
        assert mlp.down_proj.out_features == cfg.hidden_size


def test_gemma4_mlp_doubled_forward_backward():
    """Doubled MLP produces correct-shaped output and gradients flow."""
    cfg = _make_gemma4_config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
    )
    # layer 2 is KV-shared (first_kv_shared = 4-2 = 2)
    mlp = LigerGEGLUMLPForGemma4(cfg, layer_idx=2).to(device)
    x = torch.randn(2, 8, 64, device=device, requires_grad=True)
    y = mlp(x)
    assert y.shape == (2, 8, 64), f"Expected (2, 8, 64), got {y.shape}"
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gemma4_mlp_no_doubling_when_zero_kv_shared():
    """num_kv_shared_layers=0 → never doubles, even with use_double_wide_mlp=True."""
    cfg = _make_gemma4_config(
        num_hidden_layers=32,
        num_kv_shared_layers=0,
        use_double_wide_mlp=True,
    )
    for layer_idx in [0, 15, 31]:
        mlp = LigerGEGLUMLPForGemma4(cfg, layer_idx=layer_idx)
        assert mlp.intermediate_size == cfg.intermediate_size
