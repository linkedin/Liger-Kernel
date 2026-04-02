import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import transformers

from packaging import version
from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3MLP

from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.transformers.functional import liger_swiglu
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP
from liger_kernel.transformers.swiglu import LigerExperts
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.utils import infer_comm_backend
from liger_kernel.utils import infer_device

IS_TRANSFORMERS_V5_OR_LATER = version.parse(transformers.__version__) >= version.parse("5.0.0")
if IS_TRANSFORMERS_V5_OR_LATER:
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts
else:
    from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP

device = infer_device()

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
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_llamamlp(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    llama_mlp = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    llama_mlp.gate_proj.weight.data = G.T
    llama_mlp.up_proj.weight.data = U.T
    llama_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerSwiGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
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


@pytest.mark.skipif(IS_TRANSFORMERS_V5_OR_LATER, reason="Skip for transformers >= v5.0.0")
@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_mixtralblocksparsetop2mlp(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    MIXTRAL_CONFIG = MixtralConfig(
        num_local_experts=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_experts_per_tok=2,
    )

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)
    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    D = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    mixtral_blocksparsetop2mlp = MixtralBlockSparseTop2MLP(config=MIXTRAL_CONFIG).to(device).to(dtype)
    mixtral_blocksparsetop2mlp.w1.weight.data = G.T
    mixtral_blocksparsetop2mlp.w2.weight.data = U.T
    mixtral_blocksparsetop2mlp.w3.weight.data = D.T

    liger_blocksparsetop2mlp = LigerBlockSparseTop2MLP(config=MIXTRAL_CONFIG).to(device).to(dtype)
    liger_blocksparsetop2mlp.w1.weight.data = G.T
    liger_blocksparsetop2mlp.w2.weight.data = U.T
    liger_blocksparsetop2mlp.w3.weight.data = D.T

    y1 = mixtral_blocksparsetop2mlp(x1)
    y2 = liger_blocksparsetop2mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        mixtral_blocksparsetop2mlp.w1.weight.grad,
        liger_blocksparsetop2mlp.w1.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        mixtral_blocksparsetop2mlp.w2.weight.grad,
        liger_blocksparsetop2mlp.w2.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        mixtral_blocksparsetop2mlp.w3.weight.grad,
        liger_blocksparsetop2mlp.w3.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_TRANSFORMERS_V5_OR_LATER, reason="Skip for transformers < v5.0.0")
@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # TF32 accumulation order differences between Triton and cuBLAS, propagated
        # through 2 GEMMs (error ~ sqrt(I) * gate_proj_error), cause ~3-5% relative
        # error on small-valued outputs and ~5 absolute error near zero.
        (torch.float32, 30, 5e-2),
        # bf16: same Triton-vs-cuBLAS divergence plus bf16 quantization of intermediate
        # activations. Near-zero outputs can differ by ~8-16 absolute due to cancellation
        # at 7-bit mantissa precision.
        pytest.param(
            torch.bfloat16,
            100.0,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_mixtralexperts(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    MIXTRAL_CONFIG = MixtralConfig(
        num_local_experts=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        experts_implementation="eager",
        hidden_act="silu",
        num_experts_per_tok=2,
    )

    _input = torch.randn(bsz * seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # match shape: (num_experts, 2 * intermediate_dim, hidden_dim)
    GU = torch.randn(
        MIXTRAL_CONFIG.num_local_experts,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    # match shape: (num_experts, hidden_dim, intermediate_dim)
    D = torch.randn(
        MIXTRAL_CONFIG.num_local_experts, hidden_size, intermediate_size, device=device, dtype=dtype, requires_grad=True
    )

    # Generate random router logits and do topk
    router_logits = torch.randn(bsz * seq_len, MIXTRAL_CONFIG.num_local_experts, device=device, dtype=dtype)
    router_logits = router_logits.softmax(dim=-1)
    top_k_weights, top_k_index = router_logits.topk(k=MIXTRAL_CONFIG.num_experts_per_tok, dim=-1)
    top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

    mixtral_experts = MixtralExperts(config=MIXTRAL_CONFIG).to(device).to(dtype)
    mixtral_experts.gate_up_proj.data = GU.clone().detach()
    mixtral_experts.down_proj.data = D.clone().detach()

    liger_experts = LigerExperts(config=MIXTRAL_CONFIG).to(device).to(dtype)
    liger_experts.gate_up_proj.data = GU.clone().detach()
    liger_experts.down_proj.data = D.clone().detach()

    mixtral_experts.gate_up_proj.requires_grad_()
    mixtral_experts.down_proj.requires_grad_()
    liger_experts.gate_up_proj.requires_grad_()
    liger_experts.down_proj.requires_grad_()

    y1 = mixtral_experts(x1, top_k_index, top_k_weights)
    y2 = liger_experts(x2, top_k_index, top_k_weights)

    def _assert_close(a, b, label):
        af, bf = a.float(), b.float()
        diff = (af - bf).abs()
        rel = diff / (bf.abs() + 1e-9)
        abs_idx = diff.argmax()
        rel_idx = rel.argmax()
        print(
            f"\n  [{label}]"
            f"\n    max_abs={diff.max():.4g}  ref={af.flatten()[abs_idx]:.4g}  liger={bf.flatten()[abs_idx]:.4g}"
            f"\n    max_rel={rel.max():.4g}   ref={af.flatten()[rel_idx]:.4g}  liger={bf.flatten()[rel_idx]:.4g}"
        )
        assert torch.allclose(a, b, atol=atol, rtol=rtol), (
            f"{label}: max_abs={diff.max():.4g}  max_rel={rel.max():.4g}  (atol={atol}, rtol={rtol})"
        )

    _assert_close(y1, y2, "forward output")

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    _assert_close(mixtral_experts.gate_up_proj.grad, liger_experts.gate_up_proj.grad, "gate_up_proj.grad")
    _assert_close(mixtral_experts.down_proj.grad, liger_experts.down_proj.grad, "down_proj.grad")
    _assert_close(x1.grad, x2.grad, "x.grad")


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_phi3mlp(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    GU = torch.randn(hidden_size, intermediate_size * 2, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    phi3_mlp = Phi3MLP(config=PHI3_CONFIG).to(device).to(dtype)
    phi3_mlp.gate_up_proj.weight.data = GU.T
    phi3_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerPhi3SwiGLUMLP(config=PHI3_CONFIG).to(device).to(dtype)
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


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 8, 8),
        (9, 7, 41),
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
def test_correctness_functional(bsz, seq_len, size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)
    _b = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    b1 = _b.clone().requires_grad_(True)
    b2 = _b.clone().requires_grad_(True)

    y1 = liger_swiglu(a=x1, b=b1)
    y2 = LigerSiLUMulFunction.apply(x2, b2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    # Test backward pass
    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    # Check if gradients are close for x
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)


def _test_dtensor_liger_silumul(rank, world_size, bsz, seq_len, hidden_size, dtype, atol, rtol, file_name):
    torch.distributed.init_process_group(
        backend=infer_comm_backend(),
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=world_size,
    )
    device = f"{infer_device()}:{rank}" if infer_device() != "cpu" else "cpu"
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        infer_device(), mesh_shape=(world_size,), mesh_dim_names=("tp",)
    )

    _a = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)
    _b = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    # Broadcast from rank 0 so all ranks operate on identical tensors
    torch.distributed.broadcast(_a, src=0)
    torch.distributed.broadcast(_b, src=0)

    assert hidden_size % world_size == 0, f"hidden_size ({hidden_size}) must be divisible by world_size ({world_size})"

    # DTensor path: shard inputs along the hidden dim
    a1 = _a.clone().detach().requires_grad_(True)
    b1 = _b.clone().detach().requires_grad_(True)
    da = torch.distributed.tensor.distribute_tensor(
        a1, device_mesh=device_mesh, placements=[torch.distributed.tensor.Shard(2)]
    )
    db = torch.distributed.tensor.distribute_tensor(
        b1, device_mesh=device_mesh, placements=[torch.distributed.tensor.Shard(2)]
    )

    # Regular tensor path
    a2 = _a.clone().detach().requires_grad_(True)
    b2 = _b.clone().detach().requires_grad_(True)

    c1 = LigerSiLUMulFunction.apply(da, db)
    c2 = LigerSiLUMulFunction.apply(a2, b2)

    torch.testing.assert_close(c1.full_tensor(), c2, atol=atol, rtol=rtol)

    grad = torch.randn_like(c2)
    torch.distributed.broadcast(grad, src=0)
    dgrad = torch.distributed.tensor.distribute_tensor(
        grad, device_mesh=device_mesh, placements=[torch.distributed.tensor.Shard(2)]
    )

    c1.backward(dgrad)
    c2.backward(grad)

    torch.testing.assert_close(da.grad.full_tensor(), a2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(db.grad.full_tensor(), b2.grad, atol=atol, rtol=rtol)


@pytest.mark.xfail(
    torch.cuda.device_count() < 8,
    reason="Pending multi-GPU host support. This test is expected to pass when run with multi-GPU host.",
)
@pytest.mark.parametrize(
    "world_size, bsz, seq_len, hidden_size",
    [
        (4, 2, 2, 8),
        (8, 9, 7, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
def test_dtensor_liger_silumul(world_size, bsz, seq_len, hidden_size, dtype, atol, rtol):
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _test_dtensor_liger_silumul,
            args=(world_size, bsz, seq_len, hidden_size, dtype, atol, rtol, f.name),
            nprocs=world_size,
            join=True,
        )
