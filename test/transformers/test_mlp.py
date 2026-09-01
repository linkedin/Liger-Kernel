import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from liger_kernel.transformers.mlp import LigerFalconH1MLP
from liger_kernel.transformers.mlp import LigerMLP
from liger_kernel.utils import infer_device

try:
    from transformers.models.falcon_h1.configuration_falcon_h1 import FalconH1Config
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1MLP

    FALCON_H1_AVAILABLE = True
except ImportError:
    FALCON_H1_AVAILABLE = False

device = infer_device()

pytest.importorskip("triton.tools.tensor_descriptor", reason="LigerMLP requires triton.tools.tensor_descriptor")

DIFF_THRESHOLD = 1e-5
# Why the FalconH1 tests use a looser tolerance only for bf16.
#
# The reference implementation and the fused kernel round at different places:
# the reference rounds every intermediate result to the compute dtype (the
# gate/up GEMM outputs, the scaled gate g * gate_mult, the silu output, the
# silu(g) * up product...), while the kernel keeps everything in fp32 until the
# very end.
#
# Without multipliers there are fewer such differences (one less intermediate
# rounding), so bf16 stays at ~8e-6 and fits within 1e-5. With multipliers, the
# extra g * gate_mult product adds one more rounding step, pushing bf16 to
# ~1.02e-5, just over 1e-5 - hence the 2e-5 relaxation for bf16 only. fp32
# (~1.1e-6) and fp16 (~1.6e-7) are far below 1e-5 either way and keep the
# strict threshold, which also guards against formula errors.
FALCON_H1_BF16_DIFF_THRESHOLD = 2e-5


def falcon_h1_threshold(dtype):
    return FALCON_H1_BF16_DIFF_THRESHOLD if dtype == torch.bfloat16 else DIFF_THRESHOLD


falcon_h1_unavailable = pytest.mark.skipif(not FALCON_H1_AVAILABLE, reason="falcon_h1 module not available")


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:  # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _assert_close(x: torch.Tensor, y: torch.Tensor, threshold: float = DIFF_THRESHOLD):
    assert calc_diff(x, y) < threshold


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
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float16,
    ],
)
def test_correctness_llamamlp(bsz, seq_len, hidden_size, intermediate_size, dtype):
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

    _assert_close(y1, y2)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    _assert_close(llama_mlp.gate_proj.weight.grad, liger_mlp.gate_proj.weight.grad)
    _assert_close(llama_mlp.up_proj.weight.grad, liger_mlp.up_proj.weight.grad)
    _assert_close(llama_mlp.down_proj.weight.grad, liger_mlp.down_proj.weight.grad)

    _assert_close(x1.grad, x2.grad)


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
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float16,
    ],
)
def test_correctness_inference_mode(bsz, seq_len, hidden_size, intermediate_size, dtype):
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

    _assert_close(y1, y2)


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


@falcon_h1_unavailable
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
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float16,
    ],
)
@pytest.mark.parametrize("mlp_multipliers", [[1.0, 1.0], [1.7, 0.85]])
def test_correctness_falconh1mlp(bsz, seq_len, hidden_size, intermediate_size, dtype, mlp_multipliers):
    config = FalconH1Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        mlp_bias=False,
        mlp_multipliers=mlp_multipliers,
        num_attention_heads=1,
        num_key_value_heads=1,
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    falcon_mlp = FalconH1MLP(config=config).to(device).to(dtype)
    falcon_mlp.gate_proj.weight.data = G
    falcon_mlp.up_proj.weight.data = U
    falcon_mlp.down_proj.weight.data = D

    liger_mlp = LigerFalconH1MLP(config=config).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = G
    liger_mlp.up_proj.weight.data = U
    liger_mlp.down_proj.weight.data = D

    y1 = falcon_mlp(x1)
    y2 = liger_mlp(x2)

    threshold = falcon_h1_threshold(dtype)
    _assert_close(y1, y2, threshold)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    _assert_close(falcon_mlp.gate_proj.weight.grad, liger_mlp.gate_proj.weight.grad, threshold)
    _assert_close(falcon_mlp.up_proj.weight.grad, liger_mlp.up_proj.weight.grad, threshold)
    _assert_close(falcon_mlp.down_proj.weight.grad, liger_mlp.down_proj.weight.grad, threshold)

    _assert_close(x1.grad, x2.grad, threshold)


@falcon_h1_unavailable
@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        (6, 42, 128, 432),
        (3, 37, 96, 264),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        torch.float16,
    ],
)
@pytest.mark.parametrize("mlp_multipliers", [[1.0, 1.0], [1.7, 0.85]])
def test_correctness_falconh1mlp_inference_mode(bsz, seq_len, hidden_size, intermediate_size, dtype, mlp_multipliers):
    """Exercise the inference forward path (no GU saved for backward)."""
    config = FalconH1Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        mlp_bias=False,
        mlp_multipliers=mlp_multipliers,
        num_attention_heads=1,
        num_key_value_heads=1,
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    falcon_mlp = FalconH1MLP(config=config).to(device).to(dtype)
    falcon_mlp.gate_proj.weight.data = G
    falcon_mlp.up_proj.weight.data = U
    falcon_mlp.down_proj.weight.data = D

    liger_mlp = LigerFalconH1MLP(config=config).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = G
    liger_mlp.up_proj.weight.data = U
    liger_mlp.down_proj.weight.data = D

    with torch.no_grad():
        y1 = falcon_mlp(_input)
        y2 = liger_mlp(_input)

    _assert_close(y1, y2, falcon_h1_threshold(dtype))
