from test.utils import assert_verbose_allclose

import pytest
import torch
import torch.nn as nn

from liger_kernel.transformers.rms_norm import LigerRMSNorm

torch.use_deterministic_algorithms(True)

SLEEP_SECONDS = 0.1


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


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        (4, 256, 1024),
        (8, 512, 2048),
        (16, 1024, 4096),
        # # weird shapes
        (3, 423, 213),
        (5, 123, 123),
        (7, 341, 234),
        (9, 236, 345),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 1e-5),
        (torch.float16, 2e-1, 1e-5),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaRMSNorm, 0.0, "llama"),
        (GemmaRMSNorm, 1.0, "gemma"),
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode):
    # h
    _tensor = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

    # reference (llama or gemma)
    ref_rms = reference(hidden_size=hd).to("cuda").to(dtype)
    ref_o = ref_rms(h1)
    ref_o.backward(do.clone(), retain_graph=True)

    # triton
    triton_rms = (
        LigerRMSNorm(hidden_size=hd, offset=offset, casting_mode=casting_mode)
        .to("cuda")
        .to(dtype)
    )
    triton_o = triton_rms(h2)
    triton_o.backward(do.clone(), retain_graph=True)

    assert_verbose_allclose(ref_o, triton_o, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        ref_rms.weight.grad, triton_rms.weight.grad, atol=atol, rtol=rtol
    )
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol)
