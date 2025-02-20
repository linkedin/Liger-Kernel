import pytest
import torch

from test.utils import supports_bfloat16

try:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

    IS_QWEN_AVAILABLE = True
except Exception:
    IS_QWEN_AVAILABLE = False

from liger_kernel.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
from liger_kernel.transformers.functional import liger_qwen2vl_mrope
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.skipif(not IS_QWEN_AVAILABLE, reason="Qwen is not available in transformers.")
@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 131])
@pytest.mark.parametrize("num_q_heads, num_kv_heads", [(64, 8), (28, 4), (12, 2)])
@pytest.mark.parametrize(
    "head_dim, mrope_section",
    [
        (128, [16, 24, 24]),
        (96, [16, 16, 16]),
        (64, [8, 12, 12]),
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
def test_correctness(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, mrope_section, dtype, atol, rtol):
    rotary_emb = Qwen2VLRotaryEmbedding(config=Qwen2VLConfig(head_dim=head_dim), device=device)

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    # NOTE: this position ids distribution is different from the real one, just to test op correctness
    pos_ids = torch.arange(seq_len * 3 * bsz, device=device, dtype=torch.long).view(3, bsz, seq_len)
    cos, sin = rotary_emb(k1, pos_ids)

    # validate forward pass
    hf_q, hf_k = apply_multimodal_rotary_pos_emb(q1, k1, cos, sin, mrope_section)
    tt_q, tt_k = liger_multimodal_rotary_pos_emb(q2, k2, cos, sin, mrope_section)
    torch.testing.assert_close(hf_q, tt_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(hf_k, tt_k, atol=atol, rtol=rtol)

    # validate backward pass
    dq, dk = (
        torch.randn_like(hf_q, device=device),
        torch.randn_like(hf_k, device=device).to(dtype),
    )

    q1_grad, k1_grad = torch.autograd.grad((hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    torch.testing.assert_close(q1_grad, q2_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k1_grad, k2_grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_QWEN_AVAILABLE, reason="Qwen is not available in transformers.")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim, mrope_section",
    [
        (1, 2, 2, 2, 8, [2, 1, 1]),
        (1, 2, 1, 2, 8, [2, 1, 1]),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
def test_functional_correctness(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, mrope_section, dtype, atol, rtol):
    _q = torch.randn((bsz, num_q_heads, seq_len, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)

    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    rotary_emb = Qwen2VLRotaryEmbedding(config=Qwen2VLConfig(head_dim=head_dim), device=device)

    pos_ids = torch.arange(seq_len * 3 * bsz, device=device, dtype=torch.long).view(3, bsz, seq_len)
    cos, sin = rotary_emb(k1, pos_ids)

    functional_q, functional_k = liger_qwen2vl_mrope(q1, k1, cos, sin, mrope_section)
    class_q, class_k = LigerQwen2VLMRopeFunction.apply(q2, k2, cos, sin, mrope_section)

    torch.testing.assert_close(functional_q, class_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(functional_k, class_k, atol=atol, rtol=rtol)

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

    torch.testing.assert_close(q1_grad, q2_grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k1_grad, k2_grad, atol=atol, rtol=rtol)
