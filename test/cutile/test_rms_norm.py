import importlib.util

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import supports_bfloat16


class BaseRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
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
        return hidden_states.to(input_dtype)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
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
        return hidden_states.to(input_dtype)


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states):
        output = hidden_states.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)


def _has_cuda_tile():
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:
        return False


_CUDA_TILE_AVAILABLE = _has_cuda_tile()
if _CUDA_TILE_AVAILABLE:
    from liger_kernel.ops.cutile.ops.rms_norm import LigerRMSNormFunction as CuTileRMSNormFunction
else:
    CuTileRMSNormFunction = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile RMSNorm requires CUDA"),
    pytest.mark.skipif(not _CUDA_TILE_AVAILABLE, reason="cuda-tile is not installed"),
]


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
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
        (BaseRMSNorm, 0.0, "none"),
    ],
)
@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_cutile_rms_norm_correctness(
    bs,
    sl,
    hd,
    dtype,
    atol,
    rtol,
    reference,
    offset,
    casting_mode,
    in_place,
    elementwise_affine,
):
    tensor = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)
    reference_input = tensor.clone().requires_grad_(True)
    cutile_input = tensor.clone().requires_grad_(True)
    grad_output = torch.randn_like(tensor)

    reference_rms = reference(hidden_size=hd, elementwise_affine=elementwise_affine).cuda().to(dtype)
    reference_output = reference_rms(reference_input)
    reference_output.backward(grad_output)

    cutile_weight = reference_rms.weight.detach().clone().requires_grad_(True) if elementwise_affine else None
    cutile_output = CuTileRMSNormFunction.apply(
        cutile_input,
        cutile_weight,
        1e-6,
        offset,
        casting_mode,
        in_place,
        None,
    )
    cutile_output.backward(grad_output.clone())

    assert_verbose_allclose(reference_output, cutile_output, atol=atol, rtol=rtol)
    assert_verbose_allclose(reference_input.grad, cutile_input.grad, atol=atol, rtol=rtol, max_print=20)
    if elementwise_affine:
        assert_verbose_allclose(reference_rms.weight.grad, cutile_weight.grad, atol=atol, rtol=rtol)
