"""Correctness tests for the cuTile RoPE kernel.

These tests require the optional ``cuda-tile`` package and an NVIDIA GPU; they
are skipped otherwise. Numerics are checked against the existing Triton kernel
(``LigerRopeFunction`` from ``liger_kernel.ops.rope``) and HuggingFace's
``apply_rotary_pos_emb`` using *real* rotary embeddings (whose two head-dim
halves are duplicated).

The cuTile RoPE op has two internal code paths and the shape sweep below
exercises both:

* the **ALIGNED** 4-D ``ct.load``/``ct.store`` fast path, taken when
  ``head_dim // 2``, ``n_q_heads`` and ``n_kv_heads`` are all powers of two; and
* the **general** ``ct.gather``/``ct.scatter`` path, taken for any non-power-of-2
  shape (e.g. ``head_dim=92`` -> ``head_dim_half=46``, ``n_q_heads=73``).
"""

import importlib.util

import pytest
import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch
from test.utils import supports_bfloat16

device = infer_device()

cutile_available = importlib.util.find_spec("cuda.tile") is not None and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not cutile_available,
    reason="cuda-tile + CUDA GPU required for cuTile RoPE",
)

if cutile_available:
    from liger_kernel.ops.cutile.ops.rope import LigerRopeFunction as LigerRopeCuTileFunction


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        # ---- ALIGNED path (all of head_dim//2, n_q_heads, n_kv_heads are pow2) ----
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q / k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # ---- general gather/scatter path (non-pow2 heads / head_dim//2) ----
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
        # long sequence (the perf-relevant shape), ALIGNED
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
def test_cutile_matches_triton_and_hf(
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
    q_ct = _q.clone().requires_grad_(True)
    k_ct = _k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k_hf, pos_ids)

    # forward
    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    tt_q, tt_k = LigerRopeFunction.apply(q_tt, k_tt, cos, sin)
    ct_q, ct_k = LigerRopeCuTileFunction.apply(q_ct, k_ct, cos, sin)

    assert torch.allclose(ct_q, hf_q, atol=atol, rtol=rtol)
    assert torch.allclose(ct_k, hf_k, atol=atol, rtol=rtol)
    assert torch.allclose(ct_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(ct_k, tt_k, atol=atol, rtol=rtol)

    # backward
    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_ct_grad, k_ct_grad = torch.autograd.grad((ct_q, ct_k), (q_ct, k_ct), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_ct_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_ct_grad, k_hf_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        # ALIGNED path
        (1, 128, 32, 32, 64),
        (2, 128, 32, 8, 64),
        (1, 2048, 32, 8, 128),
        # general gather/scatter path
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
def test_cutile_backward_contiguous_grad(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    """Exercise the functional ``rope_forward`` + ``rope_backward`` pair directly
    with an already-contiguous incoming gradient.

    The autograd-driven test above only feeds transpose-view grads. Here we drive
    the functional API by hand -- ``rope_forward`` returns the exact tile/config
    parameters (``ALIGNED``, ``TILE_QH``, ``TILE_KH``, ``TILE_HD`` ...) that
    ``rope_backward`` needs, so this mirrors real autograd usage but lets us pass a
    genuinely ``(bsz, n_head, seq, hd)``-contiguous gradient (e.g. eager attention,
    or any model that calls ``.contiguous()`` after RoPE). Checks numerics against
    HuggingFace for both the ALIGNED and the general code paths.
    """
    from liger_kernel.ops.cutile.ops.rope import rope_backward
    from liger_kernel.ops.cutile.ops.rope import rope_forward

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

    # Run the functional forward to obtain the exact tile/config params backward needs.
    _, _, saved_cos, saved_sin, cos_bs, aligned, tile_qh, tile_kh, tile_hd, original_dtype = rope_forward(
        _q.clone(), _k.clone(), cos, sin
    )

    # A genuinely (bsz, n_head, seq, hd)-CONTIGUOUS gradient.
    dq = torch.randn(bsz, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    dk = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    assert dq.is_contiguous() and dk.is_contiguous()

    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)

    q_ct_grad, k_ct_grad = rope_backward(
        dq.clone(),
        dk.clone(),
        saved_cos,
        saved_sin,
        cos_bs,
        aligned,
        tile_qh,
        tile_kh,
        tile_hd,
        original_dtype,
        bsz,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    assert torch.allclose(q_ct_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_ct_grad, k_hf_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim",
    [
        # ALIGNED path (head_dim//2, num_heads all pow2)
        (128, 32, 64),
        (256, 16, 128),
        # general gather/scatter path (non-pow2 head_dim//2 and num_heads)
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
def test_cutile_vision_2d_cos_sin(seq_len, num_heads, head_dim, dtype, atol, rtol):
    """Vision RoPE feeds a 2-D ``(seq_len, head_dim)`` cos/sin table (no batch dim).

    ``liger_rotary_pos_emb_vision`` transposes q/k to 4-D but forwards cos/sin
    *unchanged* as rank-2 tensors. This is a regression test: the cuTile kernel
    previously raised ``TileTypeError: Index size 3 does not match the array rank
    2`` on such a table because it indexes cos/sin with a 3-tuple. Numerics are
    checked against HuggingFace (fed the equivalent 3-D table) and the Triton
    kernel (fed the same 2-D table), for both the ALIGNED and general code paths.
    """
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_heads, head_dim=head_dim), "device": device},
    )

    # Vision layout: q, k arrive as (seq_len, num_heads, head_dim); the wrapper
    # unsqueezes a batch dim and transposes to (1, num_heads, seq_len, head_dim).
    _q = torch.randn((seq_len, num_heads, head_dim), device=device).to(dtype)
    _k = torch.randn((seq_len, num_heads, head_dim), device=device).to(dtype)

    def to_4d(x):
        return x.unsqueeze(0).transpose(1, 2).contiguous()

    q_hf = to_4d(_q).clone().requires_grad_(True)
    k_hf = to_4d(_k).clone().requires_grad_(True)
    q_tt = to_4d(_q).clone().requires_grad_(True)
    k_tt = to_4d(_k).clone().requires_grad_(True)
    q_ct = to_4d(_q).clone().requires_grad_(True)
    k_ct = to_4d(_k).clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos_3d, sin_3d = rotary_emb(k_hf, pos_ids)  # (1, seq_len, head_dim)
    cos_2d, sin_2d = cos_3d[0], sin_3d[0]  # (seq_len, head_dim) -- the vision table

    # forward: HF gets the 3-D table; the kernels get the bare 2-D table.
    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos_3d, sin_3d)
    tt_q, tt_k = LigerRopeFunction.apply(q_tt, k_tt, cos_2d, sin_2d)
    ct_q, ct_k = LigerRopeCuTileFunction.apply(q_ct, k_ct, cos_2d, sin_2d)

    assert torch.allclose(ct_q, hf_q, atol=atol, rtol=rtol)
    assert torch.allclose(ct_k, hf_k, atol=atol, rtol=rtol)
    assert torch.allclose(ct_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(ct_k, tt_k, atol=atol, rtol=rtol)

    # backward
    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_ct_grad, k_ct_grad = torch.autograd.grad((ct_q, ct_k), (q_ct, k_ct), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_ct_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_ct_grad, k_hf_grad, atol=atol, rtol=rtol)
