import pytest
import torch

from test.utils import supports_bfloat16

from liger_kernel.ops import LigerLlama4RopeFunction
from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb
from liger_kernel.utils import get_total_gpu_memory
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
    rotary_emb = Llama4TextRotaryEmbedding(config=config)

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
    rotary_emb = Llama4TextRotaryEmbedding(config=config)

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


# The kernel indexes the flattened batch*seq dimension as
# `base_offset * q_row_stride`, whose largest value is
# `(bsz * seq_len - 1) * n_q_heads * head_dim`. That product is one row short of
# `q.numel()`, so it only exceeds int32 once q itself holds more than 2**31
# elements. There is no cheaper shape: the bound is the element count.
_HEAD_DIM = 128
_N_Q_HEADS = 64
_ROW = _N_Q_HEADS * _HEAD_DIM
_SEQ_LEN = (2**31 - 1) // _ROW + 2  # smallest seq_len whose last offset wraps int32


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        pytest.param(
            1,
            _SEQ_LEN,
            _N_Q_HEADS,
            1,
            _HEAD_DIM,
            marks=pytest.mark.skipif(
                infer_device() == "cpu" or get_total_gpu_memory() < 20,
                reason="This test requires a GPU with at least 20GB of memory",
            ),
        ),
    ],
)
def test_row_offset_does_not_wrap_int32(bsz, seq_len, num_q_heads, num_kv_heads, head_dim):
    """Regression for #1335: the last token must still be rotated, not read from a wrapped address."""
    dtype = torch.bfloat16
    n_kv_heads = num_kv_heads

    q = torch.zeros((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((bsz, seq_len, n_kv_heads, head_dim), device=device, dtype=dtype)

    # Only the final token carries a signal. It sits at the offset that wraps.
    q[0, -1].fill_(1.0)
    k[0, -1].fill_(1.0)

    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    freqs_cis = rotary_emb(q, pos_ids)

    # The kernel rotates q/k in place, so keep an unrotated copy of the last token for the reference.
    q_last = q[:, -1:].float().clone()
    k_last = k[:, -1:].float().clone()

    q_out, k_out = LigerLlama4RopeFunction.apply(q, k, freqs_cis)

    # A wrapped index leaves the last row untouched or fills it from elsewhere.
    assert torch.isfinite(q_out[0, -1]).all()
    assert torch.isfinite(k_out[0, -1]).all()
    assert q_out[0, -1].abs().sum() > 0
    assert k_out[0, -1].abs().sum() > 0

    ref_q, ref_k = apply_rotary_emb(q_last, k_last, freqs_cis[:, -1:])
    assert torch.allclose(q_out[:, -1:].float(), ref_q, atol=1e-1, rtol=1e-5)
    assert torch.allclose(k_out[:, -1:].float(), ref_k, atol=1e-1, rtol=1e-5)


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (4, 1, 16, 4, 64),  # batched single-token decode (left-padded cache_position)
        (2, 16, 8, 8, 64),
        (4, 37, 16, 4, 92),
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
def test_per_batch_row_position_ids(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    """Regression test for #1645: distinct position_ids per batch row must not reuse row 0 frequencies."""
    torch.manual_seed(0)
    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max(seq_len * 4, 64),
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config).to(device)

    # For seq_len == 1, construct via (B, H, 1, D).transpose(1, 2) to exercise the stride(1) == D decode layout.
    if seq_len == 1:
        _q = torch.randn((bsz, num_q_heads, 1, head_dim), device=device, dtype=dtype).transpose(1, 2)
        _k = torch.randn((bsz, num_kv_heads, 1, head_dim), device=device, dtype=dtype).transpose(1, 2)
    else:
        _q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=dtype)
        _k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    # Distinct per-row offsets (e.g., left-padded batched generation / ragged cache positions)
    offsets = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1) * 7
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0) + offsets

    q_hf = _q.clone().requires_grad_(True)
    k_hf = _k.clone().requires_grad_(True)
    freqs_hf = rotary_emb(q_hf, position_ids)
    hf_q, hf_k = apply_rotary_emb(q_hf, k_hf, freqs_hf)

    q_lig = _q.clone().requires_grad_(True)
    k_lig = _k.clone().requires_grad_(True)
    freqs_lig = rotary_emb(q_lig, position_ids)
    lig_q, lig_k = liger_llama4_text_rotary_pos_emb(q_lig, k_lig, freqs_lig)

    assert torch.allclose(hf_q, lig_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, lig_k, atol=atol, rtol=rtol)

    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    torch.autograd.backward([hf_q, hf_k], [dq, dk])
    torch.autograd.backward([lig_q, lig_k], [dq.clone(), dk.clone()])

    assert torch.allclose(q_hf.grad, q_lig.grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_hf.grad, k_lig.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
def test_llama4_rope_freqs_layout_variants_and_validation():
    """Verify unbatched real cache (including S_cache == B * S), 4D/5D singleton head dim, and shape validation."""
    torch.manual_seed(0)
    bsz, seq_len, n_heads, head_dim = 2, 8, 4, 32
    config = Llama4TextConfig(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=head_dim,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config).to(device)
    q = torch.randn((bsz, seq_len, n_heads, head_dim), device=device, dtype=torch.float32)
    k = torch.randn((bsz, seq_len, n_heads, head_dim), device=device, dtype=torch.float32)

    # 1. Unbatched 3D real cache with max_seq_len == bsz * seq_len (must slice [:seq_len], not treat as batched)
    cache_pos = torch.arange(bsz * seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cache_complex = rotary_emb(q[:, :1], cache_pos)[0]  # (bsz * seq_len, head_dim // 2)
    cache_real = torch.view_as_real(cache_complex)  # (bsz * seq_len, head_dim // 2, 2)
    ref_q, ref_k = apply_rotary_emb(q, k, cache_complex[:seq_len].unsqueeze(0))
    got_q, got_k = LigerLlama4RopeFunction.apply(q.clone(), k.clone(), cache_real)
    torch.testing.assert_close(got_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_k, ref_k, atol=1e-5, rtol=1e-5)

    # 2. Per-row 4D complex (B, S, 1, D/2) and 5D real (B, S, 1, D/2, 2)
    pos_ids = torch.stack([torch.arange(seq_len), torch.arange(seq_len) + 5]).to(device)
    freqs_3d = rotary_emb(q, pos_ids)
    ref_q2, ref_k2 = apply_rotary_emb(q, k, freqs_3d)
    for freqs_variant in (freqs_3d.unsqueeze(2), torch.view_as_real(freqs_3d.unsqueeze(2))):
        vq, vk = LigerLlama4RopeFunction.apply(q.clone(), k.clone(), freqs_variant)
        torch.testing.assert_close(vq, ref_q2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(vk, ref_k2, atol=1e-5, rtol=1e-5)

    # 3. Sequence broadcast (B, 1, D/2) across seq_len > 1
    freqs_bcast_seq = freqs_3d[:, :1, :]
    ref_q3, ref_k3 = apply_rotary_emb(q, k, freqs_bcast_seq)
    bq, bk = LigerLlama4RopeFunction.apply(q.clone(), k.clone(), freqs_bcast_seq)
    torch.testing.assert_close(bq, ref_q3, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(bk, ref_k3, atol=1e-5, rtol=1e-5)

    # 4. Invalid short sequence length (1 < S_f < seq_len) and mismatched head_dim_half must raise ValueError
    with pytest.raises(ValueError, match="sequence length"):
        LigerLlama4RopeFunction.apply(q.clone(), k.clone(), freqs_3d[:, :3, :])
    with pytest.raises(ValueError, match="last dim"):
        LigerLlama4RopeFunction.apply(q.clone(), k.clone(), freqs_3d[..., :4])
