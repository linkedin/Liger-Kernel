"""Tests for the pure-PyTorch Native Sparse Attention reference (arXiv 2502.11089).

The vectorized implementation in ``liger_kernel.transformers.native_sparse_attention``
is validated against an independent, deliberately-naive per-token oracle defined
here (``_naive_nsa``), plus degenerate-case milestones where individual branches
must reduce to ordinary causal attention.
"""

import math

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.native_sparse_attention import LigerNativeSparseAttention
from liger_kernel.transformers.native_sparse_attention import native_sparse_attention
from liger_kernel.transformers.native_sparse_attention import select_blocks
from liger_kernel.transformers.native_sparse_attention import selected_attention
from liger_kernel.transformers.native_sparse_attention import sliding_window_attention
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


# ---------------------------------------------------------------------------
# Independent naive oracle (explicit per-token loops)
# ---------------------------------------------------------------------------
def _naive_nsa(
    q,
    k,
    v,
    gate,
    k_cmp,
    v_cmp,
    *,
    num_kv_heads,
    compress_block_size,
    compress_stride,
    selection_block_size,
    num_selected_blocks,
    init_blocks,
    local_blocks,
    window_size,
    scale,
):
    batch, num_q_heads, seq_len, dim = q.shape
    group = num_q_heads // num_kv_heads
    num_cmp = k_cmp.shape[2]
    num_sel = math.ceil(seq_len / selection_block_size)
    lp = selection_block_size // compress_stride
    lc = compress_block_size // compress_stride

    cdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
    kcf, vcf = k_cmp.to(cdt), v_cmp.to(cdt)

    # Stage 1: compression scores p_cmp and compression-branch output.
    p_cmp = qf.new_zeros((batch, num_q_heads, seq_len, num_cmp))
    out_cmp = qf.new_zeros((batch, num_q_heads, seq_len, dim))
    for b in range(batch):
        for hq in range(num_q_heads):
            hkv = hq // group
            for t in range(seq_len):
                visible = [j for j in range(num_cmp) if j * compress_stride + compress_block_size - 1 <= t]
                if not visible:
                    continue
                logits = torch.stack([qf[b, hq, t] @ kcf[b, hkv, j] for j in visible]) * scale
                w = torch.softmax(logits, dim=0)
                for wi, j in enumerate(visible):
                    p_cmp[b, hq, t, j] = w[wi]
                out_cmp[b, hq, t] = sum(w[wi] * vcf[b, hkv, j] for wi, j in enumerate(visible))

    # Stage 2: selection block scores (Eq. 9 + GQA group sum Eq. 10).
    p_slc = qf.new_zeros((batch, num_kv_heads, seq_len, num_sel))
    for b in range(batch):
        for hkv in range(num_kv_heads):
            for t in range(seq_len):
                for jj in range(num_sel):
                    acc = 0.0
                    for m in range(lp):
                        for n in range(lc):
                            idx = lp * jj + m + n
                            if idx < num_cmp:
                                acc = acc + sum(p_cmp[b, hkv * group + g, t, idx] for g in range(group))
                    p_slc[b, hkv, t, jj] = acc

    # Stage 3: block selection (forced init + local, then top-n by importance).
    selected = torch.zeros((batch, num_kv_heads, seq_len, num_sel), dtype=torch.bool)
    for b in range(batch):
        for hkv in range(num_kv_heads):
            for t in range(seq_len):
                qblock = t // selection_block_size
                causal = [jj for jj in range(num_sel) if jj <= qblock]
                row = p_slc[b, hkv, t].clone()
                for jj in range(num_sel):
                    if jj > qblock:
                        row[jj] = float("-inf")
                    elif jj < init_blocks or (qblock - (local_blocks - 1) <= jj <= qblock):
                        row[jj] = float("inf")
                kk = min(num_selected_blocks, num_sel)
                top = torch.topk(row, kk).indices.tolist()
                for jj in top:
                    if jj in causal:
                        selected[b, hkv, t, jj] = True

    # Stage 4 + 5: selected and sliding attention.
    out_slc = qf.new_zeros((batch, num_q_heads, seq_len, dim))
    out_win = qf.new_zeros((batch, num_q_heads, seq_len, dim))
    for b in range(batch):
        for hq in range(num_q_heads):
            hkv = hq // group
            for t in range(seq_len):
                sel_keys = [p for p in range(t + 1) if selected[b, hkv, t, p // selection_block_size]]
                out_slc[b, hq, t] = _attend(qf[b, hq, t], kf[b, hkv], vf[b, hkv], sel_keys, scale)
                win_keys = [p for p in range(t + 1) if p > t - window_size]
                out_win[b, hq, t] = _attend(qf[b, hq, t], kf[b, hkv], vf[b, hkv], win_keys, scale)

    g = gate.to(cdt)
    out = g[..., 0:1] * out_cmp + g[..., 1:2] * out_slc + g[..., 2:3] * out_win
    return out.to(q.dtype)


def _attend(qt, k, v, key_positions, scale):
    if not key_positions:
        return torch.zeros_like(qt)
    idx = torch.tensor(key_positions, device=qt.device)
    logits = (k[idx] @ qt) * scale
    w = torch.softmax(logits, dim=0)
    return (w.unsqueeze(-1) * v[idx]).sum(dim=0)


def _full_causal_attention(q, k, v, scale, num_kv_heads):
    batch, num_q_heads, seq_len, _ = q.shape
    group = num_q_heads // num_kv_heads
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale
    causal = torch.arange(seq_len, device=q.device).view(seq_len, 1) >= torch.arange(seq_len, device=q.device).view(
        1, seq_len
    )
    scores = scores.masked_fill(~causal, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)


# ---------------------------------------------------------------------------
# Fixtures / config
# ---------------------------------------------------------------------------
# (batch, seq_len, num_q_heads, num_kv_heads, head_dim, l, d, l', n, init, local, window)
_CONFIG = dict(
    compress_block_size=4,
    compress_stride=2,
    selection_block_size=8,
    num_selected_blocks=4,
    init_blocks=1,
    local_blocks=2,
    window_size=6,
)

# The module exposes the window as `sliding_window_size`; the core function uses
# `window_size`. Derive the module kwargs from the single source of truth above.
_MODULE_CONFIG = {k: v for k, v in _CONFIG.items() if k != "window_size"}
_MODULE_CONFIG["sliding_window_size"] = _CONFIG["window_size"]


def _make_inputs(batch, num_q_heads, num_kv_heads, seq_len, dim, dtype, requires_grad=True):
    cfg = _CONFIG
    num_cmp = max(0, (seq_len - cfg["compress_block_size"]) // cfg["compress_stride"] + 1)
    q = torch.randn(batch, num_q_heads, seq_len, dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_kv_heads, seq_len, dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_kv_heads, seq_len, dim, device=device, dtype=dtype)
    k_cmp = torch.randn(batch, num_kv_heads, num_cmp, dim, device=device, dtype=dtype)
    v_cmp = torch.randn(batch, num_kv_heads, num_cmp, dim, device=device, dtype=dtype)
    gate = torch.rand(batch, num_q_heads, seq_len, 3, device=device, dtype=dtype)
    tensors = [q, k, v, gate, k_cmp, v_cmp]
    if requires_grad:
        for t in tensors:
            t.requires_grad_(True)
    return tensors


def _core(q, k, v, gate, k_cmp, v_cmp, num_kv_heads):
    return native_sparse_attention(
        q, k, v, gate, k_cmp, v_cmp, num_kv_heads=num_kv_heads, scale=1.0 / math.sqrt(q.shape[-1]), **_CONFIG
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 24, 30])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float64, 1e-10, 1e-10),
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 5e-2, 5e-2),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
        ),
    ],
)
def test_core_matches_naive_oracle(num_kv_heads, seq_len, dtype, atol, rtol):
    """Vectorized `native_sparse_attention` == independent per-token oracle (fwd + bwd)."""
    set_seed(0)
    batch, num_q_heads, dim = 2, 4, 16
    q, k, v, gate, k_cmp, v_cmp = _make_inputs(batch, num_q_heads, num_kv_heads, seq_len, dim, dtype)
    ref_inputs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, gate, k_cmp, v_cmp)]

    out = _core(q, k, v, gate, k_cmp, v_cmp, num_kv_heads)
    ref = _naive_nsa(*ref_inputs, num_kv_heads=num_kv_heads, scale=1.0 / math.sqrt(dim), **_CONFIG)
    assert_verbose_allclose(out, ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    for a, b in zip((q, k, v, gate, k_cmp, v_cmp), ref_inputs):
        assert_verbose_allclose(a.grad, b.grad, atol=atol * 10, rtol=rtol * 10)


def test_sliding_branch_reduces_to_full_causal():
    """window_size >= seq_len => sliding branch is exactly full causal attention."""
    set_seed(1)
    batch, num_q_heads, num_kv_heads, seq_len, dim = 2, 4, 2, 12, 16
    q, k, v, _, _, _ = _make_inputs(batch, num_q_heads, num_kv_heads, seq_len, dim, torch.float32, requires_grad=False)
    scale = 1.0 / math.sqrt(dim)
    out = sliding_window_attention(q, k, v, window_size=seq_len, scale=scale)
    ref = _full_causal_attention(q, k, v, scale, num_kv_heads)
    assert_verbose_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_selected_branch_reduces_to_full_causal_when_all_selected():
    """All blocks selected => selected branch is exactly full causal attention."""
    set_seed(2)
    batch, num_q_heads, num_kv_heads, seq_len, dim = 2, 4, 2, 16, 16
    q, k, v, _, _, _ = _make_inputs(batch, num_q_heads, num_kv_heads, seq_len, dim, torch.float32, requires_grad=False)
    num_sel = math.ceil(seq_len / _CONFIG["selection_block_size"])
    selected = torch.ones(batch, num_kv_heads, seq_len, num_sel, dtype=torch.bool, device=device)
    scale = 1.0 / math.sqrt(dim)
    out = selected_attention(q, k, v, selected, _CONFIG["selection_block_size"], scale)
    ref = _full_causal_attention(q, k, v, scale, num_kv_heads)
    assert_verbose_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_module_forward_backward():
    set_seed(3)
    batch, seq_len, hidden = 2, 24, 64
    model = LigerNativeSparseAttention(
        hidden_size=hidden, num_heads=4, num_kv_heads=2, head_dim=16, **_MODULE_CONFIG
    ).to(device)
    x = torch.randn(batch, seq_len, hidden, device=device, requires_grad=True)
    out = model(x)
    assert out.shape == (batch, seq_len, hidden)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_module_determinism():
    set_seed(4)
    model = LigerNativeSparseAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, **_MODULE_CONFIG).to(
        device
    )
    x = torch.randn(2, 20, 64, device=device)
    a = model(x)
    b = model(x)
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_heads=4, num_kv_heads=3), "divisible by num_kv_heads"),
        (dict(num_heads=4, num_kv_heads=2, compress_block_size=5, compress_stride=2), "compress_block_size"),
        (dict(num_heads=4, num_kv_heads=2, selection_block_size=7, compress_stride=2), "selection_block_size"),
        (
            dict(num_heads=4, num_kv_heads=2, num_selected_blocks=2, init_blocks=1, local_blocks=2),
            "num_selected_blocks",
        ),
    ],
)
def test_validation_errors(kwargs, match):
    base = dict(hidden_size=64, head_dim=16)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LigerNativeSparseAttention(**base)


def test_short_sequence_no_compression():
    """seq_len < compress_block_size: compression/selection empty, sliding still works."""
    set_seed(5)
    model = LigerNativeSparseAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, **_MODULE_CONFIG).to(
        device
    )
    x = torch.randn(1, 3, 64, device=device, requires_grad=True)  # 3 < compress_block_size (4)
    out = model(x)
    assert out.shape == (1, 3, 64)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_select_blocks_exact():
    """Hand-computed selection mask, independent of the naive oracle's topk convention."""
    # 1 batch, 1 kv head, 4 query positions, 4 selection blocks (block size 2 => positions 0..7).
    # Importance favors block 3 for the last query; forced = init block 0 + local (current) block.
    p_slc = torch.tensor([[[[9.0, 1.0, 2.0, 3.0]]]]).repeat(1, 1, 8, 1)  # [1,1,8,4]
    selected = select_blocks(p_slc, selection_block_size=2, topk=2, init_blocks=1, local_blocks=1)
    # query t=0 -> query block 0: only block 0 causal -> {0}
    assert selected[0, 0, 0].tolist() == [True, False, False, False]
    # query t=3 -> query block 1: causal blocks {0,1}; forced init {0} + local {1}; topk=2 -> {0,1}
    assert selected[0, 0, 3].tolist() == [True, True, False, False]
    # query t=7 -> query block 3: causal {0,1,2,3}; forced init {0}+local {3}; topk=2 fills highest
    # non-forced by importance among remaining {1,2}: block 2 (2.0) > block 1 (1.0) is NOT added
    # because topk=2 is already full with forced {0,3}. So exactly {0,3}.
    assert selected[0, 0, 7].tolist() == [True, False, False, True]
    # No future blocks ever selected.
    for t in range(8):
        qb = t // 2
        for j in range(4):
            if j > qb:
                assert not selected[0, 0, t, j]


def test_no_future_leakage():
    """Perturbing a future token must not change outputs at earlier positions (causality)."""
    set_seed(6)
    model = LigerNativeSparseAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, **_MODULE_CONFIG).to(
        device
    )
    x = torch.randn(1, 20, 64, device=device)
    out_a = model(x)
    x2 = x.clone()
    x2[:, 15:, :] += torch.randn_like(x2[:, 15:, :])  # change only positions >= 15
    out_b = model(x2)
    assert_verbose_allclose(out_a[:, :15], out_b[:, :15], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch, num_q_heads, num_kv_heads", [(1, 1, 1), (1, 4, 1), (3, 4, 4)])
def test_core_edge_shapes(batch, num_q_heads, num_kv_heads):
    """Single-head, batch=1, and full-MHA (group=1) all match the oracle."""
    set_seed(7)
    seq_len, dim = 20, 16
    q, k, v, gate, k_cmp, v_cmp = _make_inputs(batch, num_q_heads, num_kv_heads, seq_len, dim, torch.float32)
    ref_inputs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, gate, k_cmp, v_cmp)]
    out = _core(q, k, v, gate, k_cmp, v_cmp, num_kv_heads)
    ref = _naive_nsa(*ref_inputs, num_kv_heads=num_kv_heads, scale=1.0 / math.sqrt(dim), **_CONFIG)
    assert_verbose_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_core_no_forced_blocks():
    """init_blocks=0, local_blocks=0 exercises the pure top-k selection regime."""
    set_seed(8)
    dim = 16
    alt = dict(_CONFIG, init_blocks=0, local_blocks=0)
    q, k, v, gate, k_cmp, v_cmp = _make_inputs(2, 4, 2, 24, dim, torch.float32)
    ref_inputs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, gate, k_cmp, v_cmp)]
    out = native_sparse_attention(q, k, v, gate, k_cmp, v_cmp, num_kv_heads=2, scale=1.0 / math.sqrt(dim), **alt)
    ref = _naive_nsa(*ref_inputs, num_kv_heads=2, scale=1.0 / math.sqrt(dim), **alt)
    assert_verbose_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_fp16_no_overflow():
    """Large head_dim + large activations in fp16 must not saturate to inf pre-softmax."""
    q = torch.randn(1, 2, 8, 128, device=device, dtype=torch.float16) * 30.0
    k = torch.randn(1, 2, 8, 128, device=device, dtype=torch.float16) * 30.0
    v = torch.randn(1, 2, 8, 128, device=device, dtype=torch.float16)
    out = sliding_window_attention(q, k, v, window_size=4, scale=1.0 / math.sqrt(128))
    assert torch.isfinite(out.float()).all()


def test_compression_mlp_is_trained():
    """The compression scorer (phi MLP + intra-block PE) must receive gradient even
    though block selection is non-differentiable (grad flows via the compression branch)."""
    set_seed(9)
    model = LigerNativeSparseAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, **_MODULE_CONFIG).to(
        device
    )
    x = torch.randn(2, 24, 64, device=device)
    model(x).sum().backward()
    for name in ["k_compress.0.weight", "k_compress.2.weight", "v_compress.0.weight", "k_intra_block_pe"]:
        p = dict(model.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} received no gradient"


# ---------------------------------------------------------------------------
# Triton kernel path (validated against the pure-torch reference above)
# ---------------------------------------------------------------------------
_needs_accel = pytest.mark.skipif(device == "cpu", reason="Triton NSA kernels require an accelerator")

# Kernel-valid config: selection blocks tile in 16/32/64, head_dim padded to a pow2.
_KERNEL_CONFIG = dict(
    compress_block_size=32,
    compress_stride=16,
    selection_block_size=64,
    num_selected_blocks=6,
    init_blocks=1,
    local_blocks=2,
    window_size=48,
)


@_needs_accel
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-3),
        pytest.param(
            torch.bfloat16,
            8e-2,
            8e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
        ),
    ],
)
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [64, 130])
def test_kernel_branches_match_reference(dtype, atol, rtol, num_kv_heads, seq_len):
    """Each Triton branch kernel == its pure-torch oracle branch (fwd + bwd)."""
    from liger_kernel.ops.nsa_compressed_attention import nsa_compressed_attention
    from liger_kernel.ops.nsa_selected_attention import nsa_selected_attention
    from liger_kernel.ops.nsa_sliding_attention import nsa_sliding_attention
    from liger_kernel.transformers.native_sparse_attention import compressed_attention
    from liger_kernel.transformers.native_sparse_attention import selection_block_scores

    set_seed(20)
    batch, num_q_heads, dim = 2, 4, 32
    cfg = _KERNEL_CONFIG
    scale = 1.0 / math.sqrt(dim)
    num_cmp = max(0, (seq_len - cfg["compress_block_size"]) // cfg["compress_stride"] + 1)
    num_sel = math.ceil(seq_len / cfg["selection_block_size"])

    def mk(shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    q = mk((batch, num_q_heads, seq_len, dim))
    k = mk((batch, num_kv_heads, seq_len, dim))
    v = mk((batch, num_kv_heads, seq_len, dim))
    k_cmp = mk((batch, num_kv_heads, num_cmp, dim))
    v_cmp = mk((batch, num_kv_heads, num_cmp, dim))

    # selection mask from the (torch) scorer so kernel & oracle see the same routing
    p_cmp = compressed_attention(q, k_cmp, v_cmp, scale, cfg["compress_block_size"], cfg["compress_stride"])[1]
    p_slc = selection_block_scores(
        p_cmp, num_kv_heads, cfg["compress_block_size"], cfg["compress_stride"], cfg["selection_block_size"], num_sel
    )
    selected = select_blocks(
        p_slc, cfg["selection_block_size"], cfg["num_selected_blocks"], cfg["init_blocks"], cfg["local_blocks"]
    )

    from liger_kernel.transformers.native_sparse_attention import selected_attention as sel_ref
    from liger_kernel.transformers.native_sparse_attention import sliding_window_attention as win_ref

    cases = [
        (
            lambda a, b, c: nsa_compressed_attention(
                a, b, c, cfg["compress_block_size"], cfg["compress_stride"], scale
            ),
            lambda a, b, c: compressed_attention(a, b, c, scale, cfg["compress_block_size"], cfg["compress_stride"])[0],
            (q, k_cmp, v_cmp),
        ),
        (
            lambda a, b, c: nsa_selected_attention(a, b, c, selected, cfg["selection_block_size"], scale),
            lambda a, b, c: sel_ref(a, b, c, selected, cfg["selection_block_size"], scale),
            (q, k, v),
        ),
        (
            lambda a, b, c: nsa_sliding_attention(a, b, c, cfg["window_size"], scale),
            lambda a, b, c: win_ref(a, b, c, cfg["window_size"], scale),
            (q, k, v),
        ),
    ]
    for kern_fn, ref_fn, inputs in cases:
        ka = [t.detach().clone().requires_grad_(True) for t in inputs]
        ra = [t.detach().clone().requires_grad_(True) for t in inputs]
        ko = kern_fn(*ka)
        ro = ref_fn(*ra)
        assert_verbose_allclose(ko, ro, atol=atol, rtol=rtol)
        g = torch.randn_like(ko)
        ko.backward(g)
        ro.backward(g)
        for a, b in zip(ka, ra):
            assert_verbose_allclose(a.grad, b.grad, atol=atol * 5, rtol=rtol * 5)


@_needs_accel
@pytest.mark.parametrize(
    "dtype, otol, gtol",
    [
        (torch.float32, 2e-3, 5e-3),
        pytest.param(
            torch.bfloat16,
            8e-2,
            2e-1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this device"),
        ),
    ],
)
def test_kernel_module_matches_reference(dtype, otol, gtol):
    """Full module: kernel path == torch reference path (fwd output + every param grad)."""
    set_seed(21)
    batch, seq_len, hidden = 2, 130, 128
    model = (
        LigerNativeSparseAttention(hidden_size=hidden, num_heads=4, num_kv_heads=2, head_dim=32, **_kernel_module_cfg())
        .to(device)
        .to(dtype)
    )
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    gout = torch.randn_like(x)

    def run(use_kernel):
        model.use_kernel = use_kernel
        for p in model.parameters():
            p.grad = None
        xi = x.clone().requires_grad_(True)
        out = model(xi)
        out.backward(gout)
        return out.detach(), xi.grad.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    ok, gxk, gk = run(True)
    ot, gxt, gt = run(False)
    assert_verbose_allclose(ok, ot, atol=otol, rtol=otol)
    assert_verbose_allclose(gxk, gxt, atol=gtol, rtol=gtol)
    for name in gk:
        assert_verbose_allclose(gk[name], gt[name], atol=gtol, rtol=gtol)


def _kernel_module_cfg():
    cfg = {k: v for k, v in _KERNEL_CONFIG.items() if k != "window_size"}
    cfg["sliding_window_size"] = _KERNEL_CONFIG["window_size"]
    return cfg
