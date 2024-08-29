import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3MLP

from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP, LigerSwiGLUMLP

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="silu",
)
PHI3_CONFIG = Phi3Config(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="silu",
)
SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2048, 4096, 11008),
        (2, 2048, 2048, 4096),
        # weird shapes
        (9, 41, 341, 4231),
        (6, 42, 256, 2048),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        (torch.bfloat16, 1e4, 1e-2),
    ],
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 8,
    reason=f"Test requires GPU Ampere or newer (Found: {torch.cuda.get_device_name()})",
)
def test_correctness_llamamlp(
    bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol
):

    _input = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    G = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)

    llama_mlp = LlamaMLP(config=LLAMA_CONFIG).to("cuda").to(dtype)
    llama_mlp.gate_proj.weight.data = G.T
    llama_mlp.up_proj.weight.data = U.T
    llama_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerSwiGLUMLP(config=LLAMA_CONFIG).to("cuda").to(dtype)
    liger_mlp.gate_proj.weight.data = G.T
    liger_mlp.up_proj.weight.data = U.T
    liger_mlp.down_proj.weight.data = D.T

    y1 = llama_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        llama_mlp.gate_proj.weight.grad,
        liger_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        llama_mlp.up_proj.weight.grad,
        liger_mlp.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        llama_mlp.down_proj.weight.grad,
        liger_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2048, 4096, 11008),
        (2, 2048, 2048, 4096),
        # weird shapes
        (9, 41, 341, 4231),
        (6, 42, 256, 2048),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        (torch.bfloat16, 1e4, 1e-2),
    ],
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 8,
    reason=f"Test requires GPU Ampere or newer (Found: {torch.cuda.get_device_name()})",
)
def test_correctness_phi3mlp(
    bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol
):

    _input = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    GU = torch.randn(hidden_size, intermediate_size * 2, device="cuda", dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)

    phi3_mlp = Phi3MLP(config=PHI3_CONFIG).to("cuda").to(dtype)
    phi3_mlp.gate_up_proj.weight.data = GU.T
    phi3_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerPhi3SwiGLUMLP(config=PHI3_CONFIG).to("cuda").to(dtype)
    liger_mlp.gate_up_proj.weight.data = GU.T
    liger_mlp.down_proj.weight.data = D.T

    y1 = phi3_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        phi3_mlp.gate_up_proj.weight.grad,
        liger_mlp.gate_up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        phi3_mlp.down_proj.weight.grad,
        liger_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
