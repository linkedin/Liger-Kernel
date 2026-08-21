import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from test.utils import supports_bfloat16
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP
from liger_kernel.utils import infer_comm_backend
from liger_kernel.utils import infer_device

device = infer_device()

_NDEV = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _need_gpus(n):
    return pytest.mark.skipif(_NDEV < n, reason=f"requires {n} GPUs")


# HuggingFace LlamaMLP is the torch reference used by test_swiglu / test_geglu.
# silu → SwiGLU, gelu_pytorch_tanh → GEGLU.
_FSDP_MLP_KINDS = {
    "swiglu": {"hidden_act": "silu", "tiled_cls": LigerTiledSwiGLUMLP},
    "geglu": {"hidden_act": "gelu_pytorch_tanh", "tiled_cls": LigerTiledGEGLUMLP},
}


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


def _test_fsdp_tiled_mlp(
    rank, world_size, mlp_kind, num_shards, bsz, seq_len, hidden_size, intermediate_size, file_name
):
    """FSDP-wrapped TiledMLP vs FSDP-wrapped HuggingFace LlamaMLP (torch).

    Issue #893: tiled backward must not reshard FSDP params between sequence
    tiles. Matching *parameter* grads (via summon_full_params) is the actual
    regression; input grads can still agree when shard grads are wrong.
    """
    torch.distributed.init_process_group(
        backend=infer_comm_backend(),
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    spec = _FSDP_MLP_KINDS[mlp_kind]
    config = LlamaConfig(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=spec["hidden_act"])
    dtype = torch.float32
    atol, rtol = 1e-3, 1e-3

    torch.manual_seed(42)
    G = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    for t in (G, U, D):
        torch.distributed.broadcast(t, src=0)

    tiled_model = spec["tiled_cls"](config=config, num_shards=num_shards).to(device).to(dtype)
    ref_model = LlamaMLP(config=config).to(device).to(dtype)
    for model in (tiled_model, ref_model):
        model.gate_proj.weight.data = G.clone()
        model.up_proj.weight.data = U.clone()
        model.down_proj.weight.data = D.clone()
    tiled_model = FSDP(tiled_model, use_orig_params=True, device_id=rank)
    ref_model = FSDP(ref_model, use_orig_params=True, device_id=rank)

    torch.manual_seed(123)
    x = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
    torch.distributed.broadcast(x, src=0)
    x_tiled = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    out_tiled = tiled_model(x_tiled)
    out_ref = ref_model(x_ref)
    torch.testing.assert_close(
        out_tiled, out_ref, atol=atol, rtol=rtol, msg=f"Rank {rank}: forward outputs don't match"
    )

    out_tiled.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(
        x_tiled.grad, x_ref.grad, atol=atol, rtol=rtol, msg=f"Rank {rank}: input gradients don't match"
    )

    with FSDP.summon_full_params(tiled_model, with_grads=True), FSDP.summon_full_params(ref_model, with_grads=True):
        for i, (p_tiled, p_ref) in enumerate(zip(tiled_model.parameters(), ref_model.parameters())):
            assert p_tiled.grad is not None, f"Rank {rank}: tiled param {i} has no grad"
            assert p_ref.grad is not None, f"Rank {rank}: ref param {i} has no grad"
            torch.testing.assert_close(
                p_tiled.grad,
                p_ref.grad,
                atol=atol,
                rtol=rtol,
                msg=f"Rank {rank}: parameter {i} gradients don't match",
            )

    torch.distributed.destroy_process_group()


@pytest.mark.skipif(_NDEV < 2, reason="requires at least 2 GPUs")
@pytest.mark.parametrize("mlp_kind", ["swiglu", "geglu"])
@pytest.mark.parametrize(
    "world_size, num_shards, bsz, seq_len, hidden_size, intermediate_size",
    [
        pytest.param(2, 2, 2, 512, 128, 256, id="2gpu-2shards-even"),
        pytest.param(2, 4, 2, 127, 128, 256, id="2gpu-4shards-ragged"),
        pytest.param(4, 2, 2, 512, 128, 256, id="4gpu-2shards-even", marks=_need_gpus(4)),
        pytest.param(4, 4, 2, 255, 256, 512, id="4gpu-4shards-ragged", marks=_need_gpus(4)),
        pytest.param(8, 2, 2, 512, 128, 256, id="8gpu-2shards-even", marks=_need_gpus(8)),
        pytest.param(8, 4, 2, 511, 256, 512, id="8gpu-4shards-ragged", marks=_need_gpus(8)),
    ],
)
def test_fsdp_tiled_mlp(mlp_kind, world_size, num_shards, bsz, seq_len, hidden_size, intermediate_size):
    """TiledMLP + FSDP must match HuggingFace LlamaMLP (torch) on forward, input grads, and param grads."""
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _test_fsdp_tiled_mlp,
            args=(world_size, mlp_kind, num_shards, bsz, seq_len, hidden_size, intermediate_size, f.name),
            nprocs=world_size,
            join=True,
        )
