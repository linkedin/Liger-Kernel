"""Correctness tests for the CuteDSL RoPE kernel.

These tests require the optional ``nvidia-cutlass-dsl`` package and an NVIDIA
GPU; they are skipped otherwise. Numerics are checked against the existing
Triton kernel (``LigerRopeFunction``) and HuggingFace's ``apply_rotary_pos_emb``
using *real* rotary embeddings (whose two head-dim halves are duplicated).
"""

import importlib.util

import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()

cutedsl_available = importlib.util.find_spec("cutlass") is not None and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not cutedsl_available,
    reason="nvidia-cutlass-dsl + CUDA GPU required for CuteDSL RoPE",
)

if cutedsl_available:
    from liger_kernel.ops.cutedsl.ops.rope import LigerRopeCuteDSLFunction


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q / k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird, non-tile-aligned shapes (exercise predicated kernel)
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
        # long sequence (the perf-relevant shape)
        (1, 8192, 32, 8, 128),
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
@pytest.mark.parametrize("expand_position_ids", [True, False])
def test_cutedsl_matches_triton_and_hf(
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol, expand_position_ids
):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    _q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)
    _k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q_hf = _q.clone().requires_grad_(True)
    k_hf = _k.clone().requires_grad_(True)
    q_tt = _q.clone().requires_grad_(True)
    k_tt = _k.clone().requires_grad_(True)
    q_cu = _q.clone().requires_grad_(True)
    k_cu = _k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k_hf, pos_ids)

    # forward
    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    tt_q, tt_k = LigerRopeFunction.apply(q_tt, k_tt, cos, sin)
    cu_q, cu_k = LigerRopeCuteDSLFunction.apply(q_cu, k_cu, cos, sin)

    assert torch.allclose(cu_q, hf_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, hf_k, atol=atol, rtol=rtol)
    assert torch.allclose(cu_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, tt_k, atol=atol, rtol=rtol)

    # backward
    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_cu_grad, k_cu_grad = torch.autograd.grad((cu_q, cu_k), (q_cu, k_cu), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_cu_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_cu_grad, k_hf_grad, atol=atol, rtol=rtol)
