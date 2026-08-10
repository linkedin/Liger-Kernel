"""Unit tests for Liger's Megatron-Core RoPE adapter.

These tests deliberately do not import megatron-core; the adapter's contract is
verified against a local re-implementation of Megatron's
``_apply_rotary_pos_emb_bshd`` (non-interleaved, non-multi-latent path). The
parametrization style mirrors ``test/megatron/test_rms_norm.py``.
"""

import os

import pytest
import torch

from liger_kernel.megatron.rope import liger_apply_rotary_pos_emb_bshd
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()

set_seed(42)

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------------
# Reference: copy of Megatron-Core's _apply_rotary_pos_emb_bshd (the exact
# symbol liger_apply_rotary_pos_emb_bshd is a drop-in for), stripped to the
# non-interleaved / non-multi-latent / mscale=1 path Liger routes.
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rotary_pos_emb_bshd(t, freqs):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _make_freqs(seq_len, rot_dim, dtype, base=10000.0):
    """Megatron-style doubled rotary frequencies of shape [seq, 1, 1, rot_dim].

    ``rot_dim`` must be even; the table is ``cat(theta, theta)`` so the left and
    right halves match (Liger's kernel reads only the left half and mirrors it,
    matching Megatron's ``emb = cat(freqs, freqs)`` construction).
    """
    half = rot_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    theta = torch.outer(positions, inv_freq)  # [seq, half]
    emb = torch.cat((theta, theta), dim=-1)  # [seq, rot_dim]
    return emb.reshape(seq_len, 1, 1, rot_dim).to(dtype)


# ---------------------------------------------------------------------------
# Forward + backward correctness.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(device == "cpu", reason="Liger Triton kernels require an accelerator.")
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "seq_len, batch, heads, head_dim, rot_dim",
    [
        (64, 2, 4, 32, 32),  # full rotary
        (128, 1, 8, 64, 64),  # full rotary, more heads
        (48, 3, 4, 64, 32),  # partial rotary (t_pass non-empty)
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness(seq_len, batch, heads, head_dim, rot_dim, dtype, atol, rtol):
    # Megatron layout: [seq, batch, heads, head_dim].
    _t = torch.randn(seq_len, batch, heads, head_dim, device=device, dtype=dtype)
    do = torch.randn(seq_len, batch, heads, head_dim, device=device, dtype=dtype)
    freqs = _make_freqs(seq_len, rot_dim, dtype)

    t1 = _t.clone().requires_grad_(True)
    t2 = _t.clone().requires_grad_(True)

    ref_o = _ref_apply_rotary_pos_emb_bshd(t1, freqs)
    ref_o.backward(do, retain_graph=True)

    liger_o = liger_apply_rotary_pos_emb_bshd(t2, freqs)
    liger_o.backward(do, retain_graph=True)

    assert_verbose_allclose(ref_o, liger_o, atol=atol, rtol=rtol)
    assert_verbose_allclose(t1.grad, t2.grad, atol=atol, rtol=rtol, max_print=20)


@pytest.mark.skipif(device == "cpu", reason="Liger Triton kernels require an accelerator.")
def test_forward_does_not_modify_input():
    """The adapter must not mutate its input (Megatron reuses q/k tensors)."""
    seq_len, batch, heads, head_dim = 16, 2, 4, 32
    t = torch.randn(seq_len, batch, heads, head_dim, device=device, dtype=torch.float32)
    snapshot = t.detach().clone()
    freqs = _make_freqs(seq_len, head_dim, torch.float32)
    _ = liger_apply_rotary_pos_emb_bshd(t, freqs)
    assert torch.equal(t, snapshot)


@pytest.mark.skipif(device == "cpu", reason="Liger Triton kernels require an accelerator.")
def test_output_shape_and_dtype_preserved():
    seq_len, batch, heads, head_dim, rot_dim = 32, 2, 4, 64, 32
    t = torch.randn(seq_len, batch, heads, head_dim, device=device, dtype=torch.float32)
    freqs = _make_freqs(seq_len, rot_dim, torch.float32)
    out = liger_apply_rotary_pos_emb_bshd(t, freqs)
    assert out.shape == t.shape
    assert out.dtype == t.dtype
