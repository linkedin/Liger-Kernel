import pytest
import torch
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q/k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird shapes
        # HuggingFace llama/mistral source code doesn't support odd head dimension
        # so we don't test it here
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
def test_correctness(
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol
):

    rotary_emb = LlamaRotaryEmbedding(head_dim, device="cuda")

    _tensor_q = (
        torch.randn((bsz, seq_len, num_q_heads, head_dim), device="cuda")
        .transpose(1, 2)
        .to(dtype)
    )

    _tensor_k = (
        torch.randn((bsz, seq_len, num_kv_heads, head_dim), device="cuda")
        .transpose(1, 2)
        .to(dtype)
    )

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k1, pos_ids)

    # validate forward pass
    hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin, pos_ids)
    tt_q, tt_k = liger_rotary_pos_emb(q2, k2, cos, sin)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # validate backward pass
    dq, dk = torch.randn_like(hf_q, device="cuda"), torch.randn_like(
        hf_k, device="cuda"
    ).to(dtype)

    q1_grad, k1_grad = torch.autograd.grad(
        (hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True
    )
    q2_grad, k2_grad = torch.autograd.grad(
        (tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)
