import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from liger_kernel.transformers.mlp import LigerMLP
from liger_kernel.utils import infer_device

device = infer_device()

FP32_DIFF_THRESHOLD = 1e-5


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:  # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _assert_close(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype, atol, rtol):
    if dtype == torch.float32:
        assert calc_diff(x, y) < FP32_DIFF_THRESHOLD
    else:
        assert torch.allclose(x, y, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes (TMA requires every row stride to be 16-byte aligned,
        # so hidden/intermediate sizes are kept multiples of 8)
        (6, 42, 128, 432),
        (3, 37, 96, 264),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, None, None),
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e2, 1e-2),
    ],
)
def test_correctness_llamamlp(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_attention_heads=1,
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights directly in their final (contiguous) orientation
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    llama_mlp = LlamaMLP(config=config).to(device).to(dtype)
    llama_mlp.gate_proj.weight.data = G
    llama_mlp.up_proj.weight.data = U
    llama_mlp.down_proj.weight.data = D

    liger_mlp = LigerMLP(config=config).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = G
    liger_mlp.up_proj.weight.data = U
    liger_mlp.down_proj.weight.data = D

    y1 = llama_mlp(x1)
    y2 = liger_mlp(x2)

    _assert_close(y1, y2, dtype, atol, rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    _assert_close(llama_mlp.gate_proj.weight.grad, liger_mlp.gate_proj.weight.grad, dtype, atol, rtol)
    _assert_close(llama_mlp.up_proj.weight.grad, liger_mlp.up_proj.weight.grad, dtype, atol, rtol)
    _assert_close(llama_mlp.down_proj.weight.grad, liger_mlp.down_proj.weight.grad, dtype, atol, rtol)

    _assert_close(x1.grad, x2.grad, dtype, atol, rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes (TMA requires every row stride to be 16-byte aligned,
        # so hidden/intermediate sizes are kept multiples of 8)
        (6, 42, 128, 432),
        (3, 37, 96, 264),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, None, None),
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e2, 1e-2),
    ],
)
def test_correctness_inference_mode(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    """Exercise the inference forward path (no GU saved for backward)."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_attention_heads=1,
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    # initialize weights directly in their final (contiguous) orientation
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    llama_mlp = LlamaMLP(config=config).to(device).to(dtype)
    llama_mlp.gate_proj.weight.data = G
    llama_mlp.up_proj.weight.data = U
    llama_mlp.down_proj.weight.data = D

    liger_mlp = LigerMLP(config=config).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = G
    liger_mlp.up_proj.weight.data = U
    liger_mlp.down_proj.weight.data = D

    with torch.no_grad():
        y1 = llama_mlp(_input)
        y2 = liger_mlp(_input)

    _assert_close(y1, y2, dtype, atol, rtol)


@pytest.mark.parametrize(
    "hidden_act",
    [
        "gelu",
        "relu",
        "tanh",
    ],
)
def test_invalid_hidden_act_raises(hidden_act):
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        hidden_act=hidden_act,
    )
    with pytest.raises(ValueError):
        LigerMLP(config=config)


@pytest.mark.parametrize("hidden_act", ["silu", "swish"])
def test_supported_hidden_act(hidden_act):
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        hidden_act=hidden_act,
    )
    LigerMLP(config=config)


@pytest.mark.xfail(
    reason="TMA descriptors require 3 * intermediate_size * elem_bytes to be 16-byte aligned; "
    "intermediate_size=431 with bf16 (stride 2586 bytes) violates this",
    raises=ValueError,
    strict=True,
)
def test_misaligned_intermediate_size_not_supported():
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=431,
        hidden_act="silu",
        num_attention_heads=1,
    )
    liger_mlp = LigerMLP(config=config).to(device).to(torch.bfloat16)
    x = torch.randn(2, 8, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    liger_mlp(x)
