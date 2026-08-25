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
        (torch.float16, 1e-2, 1e-5),
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


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 8, 64),
        (1, 2048, 32, 8, 128),
        # token-fallback head_dim (no TMA): fast path must still be correct
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
        (torch.float16, 1e-2, 1e-5),
    ],
)
def test_cutedsl_backward_contiguous_grad(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    """Exercise the no-copy backward fast path.

    When the incoming gradient is already ``(bsz, n_head, seq, hd)``-contiguous
    (e.g. eager attention, or any model that calls ``.contiguous()`` after RoPE),
    ``rope_backward`` rotates it in place with the TMA kernel in ``SEQ_INNER`` mode
    instead of materialising a token-major copy. The autograd-driven test above
    only feeds transpose-view grads (the fallback path), so this covers the fast
    path directly and checks numerics against HuggingFace.
    """
    from liger_kernel.ops.cutedsl.ops.rope import rope_backward

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

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k_hf, pos_ids)

    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)

    # A genuinely (bsz, n_head, seq, hd)-CONTIGUOUS gradient triggers the fast path.
    dq = torch.randn(bsz, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    dk = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    assert dq.is_contiguous() and dk.is_contiguous()

    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_cu_grad, k_cu_grad = rope_backward(dq.clone(), dk.clone(), cos, sin)

    assert torch.allclose(q_cu_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_cu_grad, k_hf_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim",
    [
        # TMA fast path (even head_dim)
        (128, 32, 64),
        (256, 16, 128),
        # token fallback (head_dim=92 is not TMA-vectorizable)
        (423, 73, 92),
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
        (torch.float16, 1e-2, 1e-5),
    ],
)
def test_cutedsl_vision_2d_cos_sin(seq_len, num_heads, head_dim, dtype, atol, rtol):
    """Vision RoPE feeds a 2-D ``(seq_len, head_dim)`` cos/sin table (no batch dim).

    ``liger_rotary_pos_emb_vision`` transposes q/k to 4-D but forwards cos/sin
    *unchanged* as rank-2 tensors, so the kernel must accept a rank-2 table (the
    CuteDSL ``rope_forward``/``rope_backward`` handle this via a ``cos.dim()==2``
    unsqueeze). Numerics are checked against HuggingFace (fed the equivalent 3-D
    table) and the Triton kernel (fed the same 2-D table), for both the TMA and
    the token-fallback code paths.
    """
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_heads, head_dim=head_dim), "device": device},
    )

    _q = torch.randn((seq_len, num_heads, head_dim), device=device).to(dtype)
    _k = torch.randn((seq_len, num_heads, head_dim), device=device).to(dtype)

    def to_4d(x):
        return x.unsqueeze(0).transpose(1, 2).contiguous()

    q_hf = to_4d(_q).clone().requires_grad_(True)
    k_hf = to_4d(_k).clone().requires_grad_(True)
    q_tt = to_4d(_q).clone().requires_grad_(True)
    k_tt = to_4d(_k).clone().requires_grad_(True)
    q_cu = to_4d(_q).clone().requires_grad_(True)
    k_cu = to_4d(_k).clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos_3d, sin_3d = rotary_emb(k_hf, pos_ids)  # (1, seq_len, head_dim)
    cos_2d, sin_2d = cos_3d[0], sin_3d[0]  # (seq_len, head_dim) -- the vision table

    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos_3d, sin_3d)
    tt_q, tt_k = LigerRopeFunction.apply(q_tt, k_tt, cos_2d, sin_2d)
    cu_q, cu_k = LigerRopeCuteDSLFunction.apply(q_cu, k_cu, cos_2d, sin_2d)

    assert torch.allclose(cu_q, hf_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, hf_k, atol=atol, rtol=rtol)
    assert torch.allclose(cu_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, tt_k, atol=atol, rtol=rtol)

    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_cu_grad, k_cu_grad = torch.autograd.grad((cu_q, cu_k), (q_cu, k_cu), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_cu_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_cu_grad, k_hf_grad, atol=atol, rtol=rtol)
