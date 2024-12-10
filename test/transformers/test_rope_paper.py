from test.utils import supports_bfloat16

import pytest
import torch
from transformers.models.roformer.modeling_roformer import (
    RoFormerSelfAttention,
    RoFormerSinusoidalPositionalEmbedding,
)

from liger_kernel.ops.rope_paper import LigerRopePaperFunction
from liger_kernel.transformers.functional import liger_rope_paper
from liger_kernel.transformers.rope_paper import liger_rotary_paper_pos_emb
from liger_kernel.utils import infer_device

device = infer_device()

SLEEP_SECONDS = 0.1

apply_rotary_pos_emb = RoFormerSelfAttention.apply_rotary_position_embeddings


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
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
def test_correctness(
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol
):

    _tensor_q = (
        torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device)
        .transpose(1, 2)
        .to(dtype)
    )

    _tensor_k = (
        torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device)
        .transpose(1, 2)
        .to(dtype)
    )

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    rotary_emb = RoFormerSinusoidalPositionalEmbedding(
        num_positions=seq_len, embedding_dim=head_dim
    ).to(device)
    sinusoidal_pos = rotary_emb((bsz, seq_len))[None, :, :].to(dtype)

    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # validate forward pass
    hf_q, hf_k = apply_rotary_pos_emb(sinusoidal_pos[None, :, :, :], q1, k1)
    tt_q, tt_k = liger_rotary_paper_pos_emb(q2, k2, cos, sin)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # validate backward pass
    dq, dk = (
        torch.randn_like(hf_q, device=device),
        torch.randn_like(hf_k, device=device).to(dtype),
    )

    q1_grad, k1_grad = torch.autograd.grad(
        (hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True
    )
    q2_grad, k2_grad = torch.autograd.grad(
        (tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 2, 2, 2, 8),
        (1, 2, 1, 2, 8),
        # weird shapes
        (9, 7, 41, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
def test_functional_correctness(
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol
):
    _q = torch.randn((bsz, num_q_heads, seq_len, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)

    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    rotary_emb = RoFormerSinusoidalPositionalEmbedding(
        num_positions=seq_len, embedding_dim=head_dim
    ).to(device)

    sinusoidal_pos = rotary_emb((bsz, seq_len))[None, None, :, :].to(dtype)

    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    functional_q, functional_k = liger_rope_paper(q1, k1, cos, sin)
    class_q, class_k = LigerRopePaperFunction.apply(q2, k2, cos, sin)

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
