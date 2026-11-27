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


def _explicit_freqs(bsz, seq_len, head_dim, device):
    """(batch, seq, head_dim/2) complex angles that differ per batch.

    HF's rotary embedding derives them from position_ids, and the existing
    tests expand one position-id range across every batch — which makes each
    batch's angles identical and hides a batch-indexing bug.
    """
    angles = torch.zeros((bsz, seq_len, head_dim // 2), dtype=torch.complex64, device=device)
    for b in range(bsz):
        angles[b] = torch.exp(1j * torch.tensor(0.1 * (b + 1), device=device))
    return angles


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [(2, 4, 2, 2, 32), (3, 5, 4, 2, 16), (4, 6, 2, 2, 8)],
)
def test_per_batch_freqs_are_not_collapsed_to_batch_zero(bsz, seq_len, num_q_heads, num_kv_heads, head_dim):
    """Each batch rotates by its own frequencies.

    freqs_cis arrives as (batch, seq, dim) with distinct per-batch angles, as
    HF produces from per-row position_ids. Flattening that to (-1, dim) and
    slicing to seq_len drops every batch but the first, so batches b >= 1 were
    silently rotated by batch 0's angles.
    """
    q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=torch.float32)
    k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=torch.float32)
    freqs_cis = _explicit_freqs(bsz, seq_len, head_dim, device)

    got_q, got_k = LigerLlama4RopeFunction.apply(q, k, freqs_cis)

    def reference(x):
        xc = torch.view_as_complex(x.reshape(bsz, seq_len, x.shape[2], -1, 2))
        return torch.view_as_real(xc * freqs_cis.unsqueeze(2)).flatten(-2)

    assert torch.allclose(got_q, reference(q), atol=1e-5, rtol=1e-5)
    assert torch.allclose(got_k, reference(k), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [(2, 4, 2, 2, 32), (1, 3, 2, 2, 16)],
)
def test_forward_does_not_mutate_inputs(bsz, seq_len, num_q_heads, num_kv_heads, head_dim):
    """Applying RoPE must leave the caller's q/k untouched.

    The kernel writes its rotated values into the buffers it is given. For an
    already-fp32 contiguous input, both the dtype cast and `.contiguous()` in
    `_cast_and_contiguous` are no-ops, so the caller's own tensors were
    rotated in place.
    """
    q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=torch.float32)
    k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=torch.float32)
    q_before, k_before = q.clone(), k.clone()
    freqs_cis = _explicit_freqs(bsz, seq_len, head_dim, device)

    LigerLlama4RopeFunction.apply(q, k, freqs_cis)

    assert torch.equal(q, q_before), "forward mutated its q input in place"
    assert torch.equal(k, k_before), "forward mutated its k input in place"


def test_backward_rotates_gradients_per_batch_and_leaves_them_intact():
    """The backward pass rotates dq/dk by each batch's own conjugate angles.

    Backward runs the same forward helper with imag_sign=-1; the batch axis
    and the no-in-place guarantee must hold there too.
    """
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 4, 2, 2, 32
    q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=torch.float32)
    k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=torch.float32)
    freqs_cis = _explicit_freqs(bsz, seq_len, head_dim, device)

    q_out, k_out = LigerLlama4RopeFunction.apply(q, k, freqs_cis)
    dq = torch.randn_like(q_out)
    dk = torch.randn_like(k_out)
    dq_before, dk_before = dq.clone(), dk.clone()

    grad_q, grad_k = torch.autograd.grad((q_out, k_out), (q, k), (dq, dk))

    def conjugated(x):
        xc = torch.view_as_complex(x.reshape(bsz, seq_len, x.shape[2], -1, 2))
        return torch.view_as_real(xc * freqs_cis.conj().unsqueeze(2)).flatten(-2)

    assert torch.allclose(grad_q, conjugated(dq), atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_k, conjugated(dk), atol=1e-5, rtol=1e-5)
    assert torch.equal(dq, dq_before), "backward mutated its dq input in place"
    assert torch.equal(dk, dk_before), "backward mutated its dk input in place"


def test_shared_seq_freqs_still_apply_to_every_batch():
    """A (seq, dim) table without a batch axis keeps working for batch > 1.

    Only tensors whose leading dimension *is* the batch get the batch axis;
    a shared table must keep rotating every batch by the same angles.
    """
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim = 3, 5, 2, 2, 16
    q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=torch.float32)
    k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=torch.float32)
    freqs_cis = _explicit_freqs(1, seq_len, head_dim, device)[0]

    got_q, _ = LigerLlama4RopeFunction.apply(q, k, freqs_cis)

    xc = torch.view_as_complex(q.reshape(bsz, seq_len, num_q_heads, -1, 2))
    expected = torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2)
    assert torch.allclose(got_q, expected, atol=1e-5, rtol=1e-5)


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
