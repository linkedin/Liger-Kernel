from test.utils import assert_verbose_allclose

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from liger_kernel.ops.attention import LigerLlamaFlexAttention

LLAMA_CONFIG = LlamaConfig(
    hidden_size=1024,
    intermediate_size=1024,
    hidden_act="gelu_pytorch_tanh",
)
SLEEP_SECONDS = 0.1


def _match_params(src, tgt):
    for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
        tgt_param.data.copy_(src_param.data)


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 512, 1024, 1024),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 2e-6),
    ],
)
def test_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):

    _input = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)

    llama_attn = LlamaAttention(config=LLAMA_CONFIG, layer_idx=0).to("cuda").to(dtype)
    liger_llama_attn = (
        LigerLlamaFlexAttention(config=LLAMA_CONFIG, layer_idx=0).to("cuda").to(dtype)
    )

    _match_params(llama_attn, liger_llama_attn)

    y1 = llama_attn(x1, position_ids=position_ids)[0]
    y2 = liger_llama_attn(x2, position_ids=position_ids)[0]

    assert_verbose_allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    # TODO -- test grad of weights
