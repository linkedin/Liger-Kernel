import os

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.transformers.functional import liger_rms_norm
from liger_kernel.transformers.rms_norm import LigerRMSNorm
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


class BaseRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L112
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L122
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


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
        pytest.param(BaseRMSNorm, 0.0, "none", marks=pytest.mark.skipif(device="xpu", reason="skip for XPU")),
    ],
)
@pytest.mark.parametrize(
    "in_place",
    [
        True,
        False,
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, in_place):
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    # reference (llama or gemma)
    ref_rms = reference(hidden_size=hd).to(device).to(dtype)
    ref_o = ref_rms(h1)
    ref_o.backward(do, retain_graph=True)

    # triton
    triton_rms = (
        LigerRMSNorm(hidden_size=hd, offset=offset, casting_mode=casting_mode, in_place=in_place).to(device).to(dtype)
    )
    triton_o = triton_rms(h2)
    triton_o.backward(do, retain_graph=True)

    assert_verbose_allclose(ref_o, triton_o, atol=atol, rtol=rtol)
    assert_verbose_allclose(ref_rms.weight.grad, triton_rms.weight.grad, atol=atol, rtol=rtol)
    print(f"{h1.grad=}")
    print(f"{h2.grad=}")
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol, max_print=20)


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
def test_correctness_functional(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode):
    # h
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    w = torch.randn(hd, device=device, dtype=dtype)

    y1 = liger_rms_norm(X=h1, W=w, eps=1e-6, offset=offset, casting_mode=casting_mode)
    y2 = LigerRMSNormFunction.apply(h2, w, 1e-6, offset, casting_mode)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y2)

    y1.backward(grad)
    y2.backward(grad)

    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol)
