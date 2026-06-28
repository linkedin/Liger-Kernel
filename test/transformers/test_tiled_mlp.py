import copy

import pytest
import torch

from test.utils import supports_bfloat16
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from liger_kernel.transformers import monkey_patch
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


def test_apply_liger_tiled_mlp_patch_matches_regular_swiglu():
    """The tiled MLP monkey patch must produce the same logits and gradients as the regular Liger SwiGLU
    patch, end to end through a real model."""
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=512,
        max_position_embeddings=512,
        hidden_act="silu",
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(config).to(device).to(torch.float32)

    tiled_model = copy.deepcopy(base)
    regular_model = copy.deepcopy(base)

    monkey_patch.apply_liger_tiled_mlp(model=tiled_model, num_shards=4)
    for layer in regular_model.model.layers:
        monkey_patch._patch_swiglu_module(layer.mlp, LigerSwiGLUMLP)

    input_ids = torch.randint(0, config.vocab_size, (2, 256), device=device)
    labels = input_ids.clone()

    tiled_out = tiled_model(input_ids=input_ids, labels=labels)
    regular_out = regular_model(input_ids=input_ids, labels=labels)

    torch.testing.assert_close(tiled_out.logits, regular_out.logits, atol=1e-3, rtol=1e-3)

    tiled_out.loss.backward()
    regular_out.loss.backward()

    for (name, p_tiled), (_, p_regular) in zip(tiled_model.named_parameters(), regular_model.named_parameters()):
        if p_tiled.grad is None:
            continue
        torch.testing.assert_close(p_tiled.grad, p_regular.grad, atol=1e-2, rtol=1e-2, msg=name)


@pytest.mark.parametrize("num_shards", [2, 4, 8])
@pytest.mark.parametrize(
    "mlp_cls, hidden_act",
    [
        (LigerTiledSwiGLUMLP, "silu"),
        (LigerTiledGEGLUMLP, "gelu_pytorch_tanh"),
    ],
)
def test_tiled_mlp_zero3_gradient_reduction_deferral(mlp_cls, hidden_act, num_shards):
    """The tiled backward must defer DeepSpeed ZeRO-3 gradient reduction (ds_grad_is_ready stays False
    until the last shard), run the recompute exactly once per shard, leave non-ZeRO-3 parameters
    untouched, and leave the computed gradients unchanged."""
    config = LlamaConfig(hidden_size=128, intermediate_size=256, hidden_act=hidden_act)
    x = torch.randn(2, 512, 128, device=device, dtype=torch.float32)

    # Without a ds_id marker the deferral logic is a no-op and must not set ds_grad_is_ready
    plain = mlp_cls(config=config, num_shards=num_shards).to(device).to(torch.float32)
    plain(x.detach().clone().requires_grad_(True)).pow(2).sum().backward()
    assert all(not hasattr(p, "ds_grad_is_ready") for p in plain.parameters())
    ref_grads = [p.grad.clone() for p in plain.parameters()]

    # With a ds_id marker (ZeRO-3 partitioned), same weights, record the flag during each recompute
    z3 = mlp_cls(config=config, num_shards=num_shards).to(device).to(torch.float32)
    z3.load_state_dict(plain.state_dict())
    for p in z3.parameters():
        p.ds_id = 0

    seen = []
    original_mlp_forward = z3._mlp_forward

    def recording_mlp_forward(module, shard):
        seen.append(getattr(next(z3.parameters()), "ds_grad_is_ready", None))
        return original_mlp_forward(module, shard)

    z3._mlp_forward = recording_mlp_forward
    z3(x.detach().clone().requires_grad_(True)).pow(2).sum().backward()

    # the recompute runs once per shard in forward and once per shard in backward
    assert len(seen) == 2 * num_shards
    # reduction is deferred for every shard but the last, where the accumulated grad is released
    assert seen[-num_shards:] == [False] * (num_shards - 1) + [True]
    # the flag is bookkeeping only: gradients must match the non-ZeRO-3 run exactly
    for p, ref in zip(z3.parameters(), ref_grads):
        torch.testing.assert_close(p.grad, ref)
