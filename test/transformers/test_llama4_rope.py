import pytest
import torch

from test.utils import supports_bfloat16

from liger_kernel.ops import LigerLlama4RopeFunction
from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb
from liger_kernel.utils import infer_device

try:
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding
    from transformers.models.llama4.modeling_llama4 import apply_rotary_emb

    IS_LLAMA4_AVAILABLE = True
except Exception:
    IS_LLAMA4_AVAILABLE = False

device = infer_device()


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird shapes
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config, device=device)

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).to(dtype)
    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)
    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    freqs_cis = rotary_emb(q1, pos_ids)

    hf_q, hf_k = apply_rotary_emb(q1, k1, freqs_cis)
    tt_q, tt_k = liger_llama4_text_rotary_pos_emb(q2, k2, freqs_cis)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # backward
    dq, dk = torch.randn_like(hf_q, device=device), torch.randn_like(hf_k, device=device).to(dtype)

    q1_grad, k1_grad = torch.autograd.grad((hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 2, 2, 2, 8),
        (1, 2, 1, 2, 8),
        (9, 7, 41, 41, 40),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
def test_functional_correctness(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config, device=device)

    _q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)
    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    freqs_cis = rotary_emb(q1, pos_ids)

    functional_q, functional_k = liger_llama4_text_rotary_pos_emb(q1, k1, freqs_cis)
    class_q, class_k = LigerLlama4RopeFunction.apply(q2, k2, freqs_cis)

    assert torch.allclose(functional_q, class_q, atol=atol, rtol=rtol)
    assert torch.allclose(functional_k, class_k, atol=atol, rtol=rtol)

    dq, dk = torch.randn_like(functional_q), torch.randn_like(functional_k)
    dq1, dk1 = dq.clone(), dk.clone()
    dq2, dk2 = dq.clone(), dk.clone()

    q1_grad, k1_grad = torch.autograd.grad(
        (functional_q, functional_k),
        (q1, k1),
        (dq1, dk1),
        allow_unused=True,
    )

    q2_grad, k2_grad = torch.autograd.grad(
        (class_q, class_k),
        (q2, k2),
        (dq2, dk2),
        allow_unused=True,
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)
