import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops import LigerRopeFunction
from liger_kernel.ops import LigerRopeTHDFunction
from liger_kernel.transformers.functional import liger_rope
from liger_kernel.transformers.functional import liger_rope_thd
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.rope import liger_rotary_pos_emb_thd
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
    "total_tokens, num_q_heads, num_kv_heads, head_dim",
    [
        (128, 32, 32, 64),
        (128, 32, 8, 64),
        (423, 73, 155, 92),
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
def test_thd_correctness(
    total_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    atol,
    rtol,
):
    _q = torch.randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    _k = torch.randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    pos_ids = torch.arange(total_tokens, device=device, dtype=torch.long).unsqueeze(0)
    _cos, _sin = rotary_emb(_k.transpose(0, 1).unsqueeze(0), pos_ids)
    _cos = _cos.squeeze(0)
    _sin = _sin.squeeze(0)

    q1 = _q.clone().requires_grad_(True)
    k1 = _k.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    ref_q, ref_k = apply_rotary_pos_emb(
        q1.transpose(0, 1).unsqueeze(0),
        k1.transpose(0, 1).unsqueeze(0),
        _cos.unsqueeze(0),
        _sin.unsqueeze(0),
    )
    ref_q = ref_q.squeeze(0).transpose(0, 1)
    ref_k = ref_k.squeeze(0).transpose(0, 1)
    tt_q, tt_k = liger_rotary_pos_emb_thd(q2, k2, _cos, _sin)

    assert torch.allclose(ref_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(ref_k, tt_k, atol=atol, rtol=rtol)

    dq, dk = torch.randn_like(ref_q), torch.randn_like(ref_k)

    q1_grad, k1_grad = torch.autograd.grad((ref_q, ref_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "total_tokens, num_q_heads, num_kv_heads, head_dim",
    [
        (7, 2, 2, 8),
        (7, 1, 2, 8),
        (11, 41, 41, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
def test_thd_functional_correctness(
    total_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    atol,
    rtol,
):
    _q = torch.randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    _k = torch.randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    pos_ids = torch.arange(total_tokens, device=device, dtype=torch.long).unsqueeze(0)
    _cos, _sin = rotary_emb(_k.transpose(0, 1).unsqueeze(0), pos_ids)
    _cos = _cos.squeeze(0)
    _sin = _sin.squeeze(0)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)
    q3 = _q.clone().requires_grad_(True)

    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)
    k3 = _k.clone().requires_grad_(True)

    functional_q, functional_k = liger_rope_thd(q=q1, k=k1, cos=_cos, sin=_sin)
    class_q, class_k = LigerRopeTHDFunction.apply(q2, k2, _cos, _sin)
    wrapper_q, wrapper_k = liger_rotary_pos_emb_thd(q3, k3, _cos, _sin)

    assert torch.allclose(functional_q, class_q, atol=atol, rtol=rtol)
    assert torch.allclose(functional_k, class_k, atol=atol, rtol=rtol)
    assert torch.allclose(functional_q, wrapper_q, atol=atol, rtol=rtol)
    assert torch.allclose(functional_k, wrapper_k, atol=atol, rtol=rtol)

    dq, dk = torch.randn_like(functional_q), torch.randn_like(functional_k)
    dq1, dk1 = dq.clone(), dk.clone()
    dq2, dk2 = dq.clone(), dk.clone()
    dq3, dk3 = dq.clone(), dk.clone()

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
    q3_grad, k3_grad = torch.autograd.grad(
        (wrapper_q, wrapper_k),
        (q3, k3),
        (dq3, dk3),
        allow_unused=True,
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(q1_grad, q3_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k3_grad, atol=atol, rtol=rtol)


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
