import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops import LigerRopeFunction
from liger_kernel.transformers.functional import liger_rope
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()

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
        # long sequence (perf-relevant; also exercises the int64 program-id path)
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
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    expand_position_ids,
    atol,
    rtol,
):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)

    # validate forward pass
    hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin)
    tt_q, tt_k = liger_rotary_pos_emb(q2, k2, cos, sin)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # validate backward pass
    dq, dk = (
        torch.randn_like(hf_q, device=device),
        torch.randn_like(hf_k, device=device).to(dtype),
    )

    q1_grad, k1_grad = torch.autograd.grad((hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

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
        (torch.float16, 1e-2, 1e-5),
    ],
)
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_functional_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    expand_position_ids,
    dtype,
    atol,
    rtol,
):
    _q = torch.randn((bsz, num_q_heads, seq_len, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)

    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)

    functional_q, functional_k = liger_rope(q=q1, k=k1, cos=cos, sin=sin)
    class_q, class_k = LigerRopeFunction.apply(q2, k2, cos, sin)

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


@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim",
    [
        (128, 32, 64),
        (256, 16, 128),
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
def test_vision_2d_cos_sin(seq_len, num_heads, head_dim, dtype, atol, rtol):
    """Vision RoPE feeds a 2-D ``(seq_len, head_dim)`` cos/sin table (no batch dim).

    ``liger_rotary_pos_emb_vision`` transposes q/k to 4-D but forwards cos/sin
    *unchanged* as rank-2 tensors, so the kernel must accept a rank-2 table. This
    guards the Triton kernel's 2-D handling (it survives because the per-batch cos
    offset multiplies by ``batch_idx == 0``). Numerics are checked against
    HuggingFace, which is fed the equivalent 3-D table.
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

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos_3d, sin_3d = rotary_emb(k_hf, pos_ids)  # (1, seq_len, head_dim)
    cos_2d, sin_2d = cos_3d[0], sin_3d[0]  # (seq_len, head_dim) -- the vision table

    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos_3d, sin_3d)
    tt_q, tt_k = liger_rotary_pos_emb(q_tt, k_tt, cos_2d, sin_2d)

    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_tt_grad, k_tt_grad = torch.autograd.grad((tt_q, tt_k), (q_tt, k_tt), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_hf_grad, q_tt_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_hf_grad, k_tt_grad, atol=atol, rtol=rtol)


def _hf_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_partial_rope(q, k, cos, sin):
    cos_u = cos.unsqueeze(1) if cos.ndim == 3 else cos
    sin_u = sin.unsqueeze(1) if sin.ndim == 3 else sin
    rotary_dim = cos_u.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos_u) + (_hf_rotate_half(q_rot) * sin_u)
    k_embed = (k_rot * cos_u) + (_hf_rotate_half(k_rot) * sin_u)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim, rotary_dim",
    [
        (2, 128, 32, 32, 128, 32),  # GPT-NeoX style
        (2, 128, 32, 32, 80, 20),  # Phi-2 style
        (2, 128, 32, 8, 64, 16),  # StableLM style
        (2, 128, 16, 4, 128, 64),  # 50% partial RoPE
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
def test_partial_rope_transformers_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    dtype,
    atol,
    rtol,
):
    torch.manual_seed(0)
    _tensor_q = torch.randn((bsz, num_q_heads, seq_len, head_dim), device=device, dtype=dtype)
    _tensor_k = torch.randn((bsz, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)
    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    freqs = torch.randn(bsz, seq_len, rotary_dim // 2, device=device, dtype=torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    ref_q, ref_k = _hf_partial_rope(q1, k1, cos, sin)
    tt_q, tt_k = liger_rotary_pos_emb(q2, k2, cos, sin)

    assert torch.allclose(ref_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(ref_k, tt_k, atol=atol, rtol=rtol)

    dq = torch.randn_like(ref_q)
    dk = torch.randn_like(ref_k)

    q1_grad, k1_grad = torch.autograd.grad((ref_q, ref_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)
