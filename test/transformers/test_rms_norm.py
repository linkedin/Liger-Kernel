import pytest
import torch
import torch.nn as nn

from liger_kernel.transformers.rms_norm import LigerRMSNorm

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
        (torch.float32, 1e-4, 1e-7),
        (torch.bfloat16, 5.0, 1e-5),
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol):
    # h
    _tensor = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

    # llama
    llama_rms = LlamaRMSNorm(hidden_size=hd).to("cuda").to(dtype)
    llama_o = llama_rms(h1)
    llama_o.backward(do.clone(), retain_graph=True)

    # triton
    triton_rms = LigerRMSNorm(hidden_size=hd).to("cuda").to(dtype)
    triton_o = triton_rms(h2)
    triton_o.backward(do.clone(), retain_graph=True)

    assert torch.allclose(llama_o, triton_o, atol=atol, rtol=rtol) is True
    assert (
        torch.allclose(
            llama_rms.weight.grad, triton_rms.weight.grad, atol=atol, rtol=rtol
        )
        is True
    )
    # import pdb; pdb.set_trace()
    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol) is True
