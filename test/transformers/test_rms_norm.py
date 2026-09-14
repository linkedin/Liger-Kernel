import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

# torch.distributed.tensor is a lazy submodule on torch 2.12+; bind it once so
# downstream ``torch.distributed.tensor.distribute_tensor`` / ``.Shard`` access
# doesn't AttributeError before any explicit import has happened.
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerRMSNormFunction
from liger_kernel.ops.rms_norm import rms_norm_backward
from liger_kernel.ops.rms_norm import rms_norm_forward
from liger_kernel.transformers.functional import liger_rms_norm
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import infer_comm_backend
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

#  Only setting torch.use_deterministic_algorithms(True) might throw the following error:
#  RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
#  but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an
#  environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information,
#  go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SLEEP_SECONDS = 0.1


def test_ordinary_forward_preserves_historical_backend_function_abi(monkeypatch):
    import liger_kernel.transformers.rms_norm as rms_norm_transformers

    calls = []

    class HistoricalBackendRMSNormFunction:
        @staticmethod
        def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
            raise AssertionError("the test backend forward should only be reached through apply")

        @staticmethod
        def apply(X, W, eps, offset, casting_mode, in_place, row_mode):
            calls.append((X, W, eps, offset, casting_mode, in_place, row_mode))
            return X

    monkeypatch.setattr(rms_norm_transformers, "LigerRMSNormFunction", HistoricalBackendRMSNormFunction)

    module = LigerRMSNorm(8)
    hidden_states = torch.randn(2, 8)

    assert module(hidden_states) is hidden_states
    assert len(calls) == 1
    assert len(calls[0]) == 7


class BaseRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.elementwise_affine:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L112
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.elementwise_affine:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L122
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.elementwise_affine:
            output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        input_dtype = x.dtype
        grouped_x = x.float().reshape(*x.shape[:-1], -1, self.group_size)
        output = grouped_x * torch.rsqrt(grouped_x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output.flatten(-2) * (1.0 + self.weight.float())
        return output.to(input_dtype)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        # weird shapes
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaRMSNorm, 0.0, "llama"),
        (GemmaRMSNorm, 1.0, "gemma"),
        pytest.param(
            BaseRMSNorm,
            0.0,
            "none",
            marks=pytest.mark.skipif(device == "npu", reason="Ascend NPU does not support this test"),
        ),
    ],
)
@pytest.mark.parametrize(
    "in_place",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "elementwise_affine",
    [
        True,
        False,
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, in_place, elementwise_affine):
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    # reference (llama or gemma)
    ref_rms = reference(hidden_size=hd, elementwise_affine=elementwise_affine).to(device).to(dtype)
    ref_o = ref_rms(h1)
    ref_o.backward(do)

    # triton
    triton_rms = (
        LigerRMSNorm(
            hidden_size=hd,
            offset=offset,
            casting_mode=casting_mode,
            in_place=in_place,
            elementwise_affine=elementwise_affine,
        )
        .to(device)
        .to(dtype)
    )
    triton_o = triton_rms(h2)
    # clone since in_place=True lets backward overwrite the grad-output buffer
    triton_o.backward(do.clone())

    assert_verbose_allclose(ref_o, triton_o, atol=atol, rtol=rtol)
    if elementwise_affine:
        assert_verbose_allclose(ref_rms.weight.grad, triton_rms.weight.grad, atol=atol, rtol=rtol)
    print(f"{h1.grad=}")
    print(f"{h2.grad=}")
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol, max_print=20)


@pytest.mark.parametrize(
    "bs, sl, hd, group_size",
    [
        (2, 3, 128, 32),
        (5, 7, 123, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 2e-5, 2e-5),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_grouped_correctness(bs, sl, hd, group_size, dtype, atol, rtol):
    tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    h1 = tensor.clone().requires_grad_(True)
    h2 = tensor.clone().requires_grad_(True)
    grad = torch.randn_like(tensor)

    reference = GroupedGemmaRMSNorm(hd, group_size).to(device=device, dtype=dtype)
    triton = LigerRMSNorm(
        hidden_size=hd,
        offset=1.0,
        casting_mode="gemma",
        init_fn="zeros",
        in_place=False,
        group_size=group_size,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        reference.weight.normal_()
        triton.weight.copy_(reference.weight)

    reference_output = reference(h1)
    triton_output = triton(h2)
    reference_output.backward(grad)
    triton_output.backward(grad.clone())

    assert_verbose_allclose(reference_output, triton_output, atol=atol, rtol=rtol)
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(reference.weight.grad, triton.weight.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_grouped_backward_fully_overwrites_empty_dx(monkeypatch, dtype):
    shape = (17, 128)
    n_groups = 4
    tensor = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(shape[-1], device=device, dtype=dtype)
    grad = torch.randn_like(tensor)
    forward = rms_norm_forward(tensor, weight, 1e-6, 1.0, "gemma", row_mode=None, n_groups=n_groups)
    _, saved_x, rstd, block_size, num_warps, casting_mode = forward
    backward_args = (
        grad,
        saved_x,
        weight,
        rstd,
        1.0,
        casting_mode,
        block_size,
        num_warps,
        False,
        None,
        n_groups,
    )
    expected_dx, expected_dw = rms_norm_backward(*backward_args)

    original_empty_like = torch.empty_like
    original_empty = torch.empty

    def poisoned_empty_like(*args, **kwargs):
        return original_empty_like(*args, **kwargs).fill_(torch.nan)

    def poisoned_empty(*args, **kwargs):
        return original_empty(*args, **kwargs).fill_(torch.nan)

    monkeypatch.setattr(torch, "empty_like", poisoned_empty_like)
    monkeypatch.setattr(torch, "empty", poisoned_empty)
    actual_dx, actual_dw = rms_norm_backward(*backward_args)

    assert not torch.isnan(actual_dx).any()
    assert not torch.isnan(actual_dw).any()
    assert_verbose_allclose(actual_dx, expected_dx, atol=0.0, rtol=0.0)
    assert_verbose_allclose(actual_dw, expected_dw, atol=0.0, rtol=0.0)


def test_grouped_requires_elementwise_affine_weight():
    with pytest.raises(ValueError, match="group_size requires elementwise_affine=True"):
        LigerRMSNorm(hidden_size=128, group_size=32, elementwise_affine=False)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"group_size": 0}, "group_size must be a positive integer"),
        ({"group_size": 24}, "hidden_size .* must be divisible"),
    ],
)
def test_rms_norm_rejects_invalid_public_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LigerRMSNorm(hidden_size=128, **kwargs)


@pytest.mark.parametrize("n_groups", [0, -1, 1.5])
def test_grouped_rejects_invalid_group_count(n_groups):
    x = torch.randn(2, 128, device=device)
    weight = torch.ones(128, device=device)
    with pytest.raises(ValueError, match="n_groups must be a positive integer"):
        LigerRMSNormFunction.apply(x, weight, 1e-6, 0.0, "gemma", False, None, n_groups)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaRMSNorm, 0.0, "llama"),
        (GemmaRMSNorm, 1.0, "gemma"),
    ],
)
@pytest.mark.parametrize(
    "elementwise_affine",
    [
        True,
        False,
    ],
)
def test_correctness_functional(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, elementwise_affine):
    # h
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    if elementwise_affine:
        w = torch.randn(hd, device=device, dtype=dtype)
    else:
        w = None

    y1 = liger_rms_norm(X=h1, W=w, eps=1e-6, offset=offset, casting_mode=casting_mode)
    y2 = LigerRMSNormFunction.apply(h2, w, 1e-6, offset, casting_mode)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y2)

    # Clone grad since in_place=True lets backward overwrite the grad-output buffer
    y1.backward(grad.clone())
    y2.backward(grad.clone())

    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol)


def _test_dtensor_rms_norm(rank, world_size, bs, sl, hd, dtype, atol, rtol, offset, casting_mode, file_name):
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
    t = torch.randn(bs, sl, hd, device=device, dtype=dtype, requires_grad=True)
    dt = torch.distributed.tensor.distribute_tensor(
        t,
        device_mesh=device_mesh,
        placements=[torch.distributed.tensor.Shard(2)],
    )
    w = torch.randn(hd, device=device, dtype=dtype, requires_grad=True)
    w1 = w.detach().clone()
    w2 = w.detach().clone()

    y1 = liger_rms_norm(X=dt, W=w1, eps=1e-6, offset=offset, casting_mode=casting_mode)
    y2 = liger_rms_norm(X=t, W=w2, eps=1e-6, offset=offset, casting_mode=casting_mode)
    torch.testing.assert_close(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y2)
    dgrad = torch.distributed.tensor.distribute_tensor(
        grad,
        device_mesh=device_mesh,
        placements=[torch.distributed.tensor.Shard(2)],
    )

    y1.backward(dgrad)
    y2.backward(grad)
    torch.testing.assert_close(w1.grad, w2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dt.grad, t.grad, atol=atol, rtol=rtol)


@pytest.mark.xfail(
    torch.cuda.device_count() < 8,
    reason="Pending multi-GPU host support. This test is expected to pass when run with multi-GPU host.",
)
@pytest.mark.parametrize(
    "world_size, bs, sl, hd",
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
@pytest.mark.parametrize(
    "offset, casting_mode",
    [
        (0.0, "llama"),
        (1.0, "gemma"),
    ],
)
def test_dtensor_rms_norm(world_size, bs, sl, hd, dtype, atol, rtol, offset, casting_mode):
    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _test_dtensor_rms_norm,
            args=(world_size, bs, sl, hd, dtype, atol, rtol, offset, casting_mode, f.name),
            nprocs=world_size,
            join=True,
        )
