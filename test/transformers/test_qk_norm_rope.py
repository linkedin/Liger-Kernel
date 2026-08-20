"""Correctness tests for the fused QK-Norm + RoPE Triton kernel.

The reference stacks a per-head RMSNorm (over ``head_dim``) with the HuggingFace
Llama/Qwen rotary embedding, exactly as ``Qwen3Attention.forward`` does::

    q = q_norm(q_proj(x).view(B, T, n_qh, hd)).transpose(1, 2)
    k = k_norm(k_proj(x).view(B, T, n_kh, hd)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

Correctness methodology
-----------------------
A fused kernel that keeps the RMSNorm->RoPE chain in fp32 is *not* bit-identical
to running RMSNorm and RoPE as two separate ops -- in low precision it is
actually more accurate (it rounds once instead of round-tripping the normalized
Q/K through bf16).  A fixed ``atol`` on ``fused vs naive`` therefore either has
to be loose enough to be meaningless or it flags precision, not bugs.

Instead we compute a fully-fp32 "gold" result and assert the fused kernel is
**no worse than the naive same-dtype reference** relative to that gold (plus a
small absolute floor for the fp32 case where the naive path *is* the gold).
This encodes the real property we care about -- "fusing must not lose accuracy"
-- rather than a magic threshold.
"""

import pytest
import torch

from liger_kernel.ops.qk_norm_rope import LigerQkNormRopeFunction
from liger_kernel.transformers.qk_norm_rope import liger_qk_norm_rope


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rms_norm_ref(x, weight, eps):
    # llama casting: reduce/rstd in fp32, weight multiply back in input dtype
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def ref_forward(q, k, wq, wk, cos, sin, eps):
    # q, k: (B, T, n_head, hd)
    q = rms_norm_ref(q, wq, eps).transpose(1, 2)
    k = rms_norm_ref(k, wk, eps).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k


def make_cos_sin(bsz, seq_len, head_dim, device, dtype, batched=False, base=10000):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    if batched:
        cos = cos.unsqueeze(0).expand(bsz, -1, -1)
        sin = sin.unsqueeze(0).expand(bsz, -1, -1)
    else:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return cos.to(dtype), sin.to(dtype)


def _max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def _assert_not_worse(fused, ref, gold, name, slack=2.0, floor=0.0):
    """Assert the fused result is no worse than the naive reference vs fp32 gold.

    ``slack`` absorbs reduction-order differences; ``floor`` is an absolute
    allowance for the fp32 case, where the naive reference *is* the gold so its
    error is exactly zero.
    """
    fused_err = _max_abs_err(fused, gold)
    ref_err = _max_abs_err(ref, gold)
    assert fused_err <= ref_err * slack + floor, (
        f"{name}: fused err {fused_err:.3e} > ref err {ref_err:.3e} * {slack} + {floor:.3e}"
    )


def _run_gold_ref_fused(q, k, wq, wk, cos, sin, eps):
    """Run fp32-gold, naive-same-dtype, and fused paths with a shared upstream grad.

    Returns three dicts keyed by q/k/dq/dk/dwq/dwk.
    """
    dtype = q.dtype
    cos32, sin32 = cos.float(), sin.float()

    def leaves(dt):
        return (
            q.clone().to(dt).requires_grad_(True),
            k.clone().to(dt).requires_grad_(True),
            wq.clone().to(dt).requires_grad_(True),
            wk.clone().to(dt).requires_grad_(True),
        )

    # ---- fp32 gold ----
    gq, gk, gwq, gwk = leaves(torch.float32)
    gold_q, gold_k = ref_forward(gq, gk, gwq, gwk, cos32, sin32, eps)
    grad_q = torch.randn_like(gold_q)
    grad_k = torch.randn_like(gold_k)
    ((gold_q * grad_q).sum() + (gold_k * grad_k).sum()).backward()
    gold = dict(q=gold_q, k=gold_k, dq=gq.grad, dk=gk.grad, dwq=gwq.grad, dwk=gwk.grad)

    # ---- naive same-dtype reference ----
    rq, rk, rwq, rwk = leaves(dtype)
    ref_q, ref_k = ref_forward(rq, rk, rwq, rwk, cos, sin, eps)
    ((ref_q * grad_q.to(dtype)).sum() + (ref_k * grad_k.to(dtype)).sum()).backward()
    ref = dict(q=ref_q, k=ref_k, dq=rq.grad, dk=rk.grad, dwq=rwq.grad, dwk=rwk.grad)

    # ---- fused kernel ----
    fq, fk, fwq, fwk = leaves(dtype)
    fus_q, fus_k = liger_qk_norm_rope(fq, fk, fwq, fwk, cos, sin, eps)
    ((fus_q * grad_q.to(dtype)).sum() + (fus_k * grad_k.to(dtype)).sum()).backward()
    fus = dict(q=fus_q, k=fus_k, dq=fq.grad, dk=fk.grad, dwq=fwq.grad, dwk=fwk.grad)

    return gold, ref, fus


def _check_all_not_worse(gold, ref, fus, act_floor, wgrad_floor):
    # activations and input grads
    for key in ("q", "k", "dq", "dk"):
        _assert_not_worse(fus[key], ref[key], gold[key], key, slack=2.0, floor=act_floor)
    # weight grads accumulate over all tokens/heads -> large magnitude, so a
    # bigger absolute floor for the fp32 reduction-order term.
    for key in ("dwq", "dwk"):
        _assert_not_worse(fus[key], ref[key], gold[key], key, slack=2.0, floor=wgrad_floor)


# Real Qwen3 dense attention shapes (n_q_head, n_kv_head, head_dim), taken from
# the published HuggingFace ``config.json`` files.  Qwen3 decouples ``head_dim``
# (fixed at 128) from ``hidden_size / num_attention_heads`` and uses GQA with 8
# KV heads across the whole dense family; ``rms_norm_eps`` is 1e-6 everywhere.
QWEN3_REAL_CONFIGS = [
    pytest.param(16, 8, 128, id="qwen3_0.6b"),
    pytest.param(32, 8, 128, id="qwen3_4b_8b"),
    pytest.param(40, 8, 128, id="qwen3_14b"),
    pytest.param(64, 8, 128, id="qwen3_32b"),
]

# fp32 leaves the naive path == gold, so allow a small absolute reduction-order
# floor; bf16 relies purely on "fused must not be worse than naive".
DTYPE_FLOORS = [
    pytest.param(torch.float32, 1e-4, 1e-2, id="fp32"),  # (dtype, act_floor, wgrad_floor)
    pytest.param(torch.bfloat16, 0.0, 0.0, id="bf16"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n_qh, n_kh, hd", QWEN3_REAL_CONFIGS)
@pytest.mark.parametrize(
    "bsz, seq_len",
    [
        (1, 4096),  # single long-context sequence
        (4, 2048),  # typical training micro-batch
    ],
)
@pytest.mark.parametrize("dtype, act_floor, wgrad_floor", DTYPE_FLOORS)
def test_qk_norm_rope_qwen3_real_configs(n_qh, n_kh, hd, bsz, seq_len, dtype, act_floor, wgrad_floor):
    """Correctness on real Qwen3 dense attention shapes and training seq lengths.

    Uses the production ``rms_norm_eps`` (1e-6), ``rope_theta`` (1e6) and the GQA
    head layout that the fused ``qwen3_attention_forward`` monkeypatch actually
    feeds the kernel.  Asserts the fused kernel is no worse than the naive
    same-dtype path measured against a fully-fp32 gold reference.
    """
    device = "cuda"
    eps = 1e-6  # Qwen3 rms_norm_eps
    torch.manual_seed(0)

    q = torch.randn(bsz, seq_len, n_qh, hd, device=device, dtype=dtype)
    k = torch.randn(bsz, seq_len, n_kh, hd, device=device, dtype=dtype)
    # RMSNorm weights initialize to ones in Qwen3; jitter around 1.0 to exercise
    # a realistic-but-non-trivial per-channel scale.
    wq = 1.0 + 0.1 * torch.randn(hd, device=device, dtype=dtype)
    wk = 1.0 + 0.1 * torch.randn(hd, device=device, dtype=dtype)
    cos, sin = make_cos_sin(bsz, seq_len, hd, device, dtype, batched=False, base=1000000)

    gold, ref, fus = _run_gold_ref_fused(q, k, wq, wk, cos, sin, eps)

    # shape / layout
    assert fus["q"].shape == (bsz, n_qh, seq_len, hd)
    assert fus["k"].shape == (bsz, n_kh, seq_len, hd)

    _check_all_not_worse(gold, ref, fus, act_floor, wgrad_floor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bsz, seq_len", [(1, 128), (2, 200), (4, 51)])
@pytest.mark.parametrize("n_qh, n_kh, hd", [(32, 8, 128), (16, 16, 64), (28, 4, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batched_cos", [False, True])
def test_qk_norm_rope_correctness(bsz, seq_len, n_qh, n_kh, hd, dtype, batched_cos):
    """Small-shape sweep (odd seq lens, MHA/GQA mixes, batched vs shared cos/sin).

    Same "no worse than naive vs fp32 gold" methodology as the real-config test.
    """
    device = "cuda"
    eps = 1e-6
    torch.manual_seed(0)
    act_floor = 1e-4 if dtype == torch.float32 else 0.0
    wgrad_floor = 1e-2 if dtype == torch.float32 else 0.0

    q = torch.randn(bsz, seq_len, n_qh, hd, device=device, dtype=dtype)
    k = torch.randn(bsz, seq_len, n_kh, hd, device=device, dtype=dtype)
    wq = torch.randn(hd, device=device, dtype=dtype)
    wk = torch.randn(hd, device=device, dtype=dtype)
    cos, sin = make_cos_sin(bsz, seq_len, hd, device, dtype, batched=batched_cos)

    gold, ref, fus = _run_gold_ref_fused(q, k, wq, wk, cos, sin, eps)

    assert fus["q"].shape == (bsz, n_qh, seq_len, hd)
    assert fus["k"].shape == (bsz, n_kh, seq_len, hd)

    _check_all_not_worse(gold, ref, fus, act_floor, wgrad_floor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qk_norm_rope_patch_matches_reference():
    """Sanity-check the fp32 layout/values of the standalone function."""
    device = "cuda"
    eps = 1e-6
    bsz, seq_len, n_qh, n_kh, hd = 2, 64, 8, 2, 128
    dtype = torch.float32
    torch.manual_seed(0)

    q = torch.randn(bsz, seq_len, n_qh, hd, device=device, dtype=dtype)
    k = torch.randn(bsz, seq_len, n_kh, hd, device=device, dtype=dtype)
    wq = torch.randn(hd, device=device, dtype=dtype)
    wk = torch.randn(hd, device=device, dtype=dtype)
    cos, sin = make_cos_sin(bsz, seq_len, hd, device, dtype, batched=False)

    rq, rk = ref_forward(q, k, wq, wk, cos, sin, eps)
    fq, fk = LigerQkNormRopeFunction.apply(q, k, wq, wk, cos, sin, eps)

    assert torch.allclose(rq, fq, atol=1e-4, rtol=1e-4)
    assert torch.allclose(rk, fk, atol=1e-4, rtol=1e-4)
    # returned tensors must be laid out (bsz, n_head, seq_len, head_dim)
    assert fq.shape == (bsz, n_qh, seq_len, hd)
    assert fk.shape == (bsz, n_kh, seq_len, hd)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen3_attention_forward_patch_end_to_end(dtype):
    """End-to-end: the monkeypatched ``Qwen3Attention.forward`` must be no worse
    than the stock HuggingFace attention output (both measured against an fp32
    gold) on a real (scaled-down) Qwen3 config.

    We build an actual ``Qwen3Attention`` layer so the fused kernel is exercised
    through ``self.q_norm``/``self.k_norm``/``apply_rotary_pos_emb`` exactly as it
    is in a live model.
    """
    import types

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from liger_kernel.transformers.model.qwen3_attention import qwen3_attention_forward

    device = "cuda"
    torch.manual_seed(0)

    bsz, seq_len = 2, 512
    # Real Qwen3 head layout (GQA 32/8, head_dim=128), hidden trimmed so the
    # projection weights stay small; head_dim / eps / theta match production.
    config = Qwen3Config(
        hidden_size=32 * 128,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=40960,
        rope_theta=1000000,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )
    attn = Qwen3Attention(config, layer_idx=0).to(device=device).eval()
    rotary = Qwen3RotaryEmbedding(config, device=device)

    hidden = torch.randn(bsz, seq_len, config.hidden_size, device=device)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

    def run(module, h, cos, sin):
        with torch.no_grad():
            out, _ = module(h, position_embeddings=(cos, sin), attention_mask=None)
        return out

    # fp32 gold: stock forward in fp32
    cos32, sin32 = rotary(hidden, pos_ids)
    gold = run(attn, hidden, cos32, sin32)

    # cast the whole layer + inputs to the target dtype
    attn = attn.to(dtype=dtype)
    h = hidden.to(dtype)
    cos, sin = cos32.to(dtype), sin32.to(dtype)

    ref = run(attn, h, cos, sin)  # stock forward, target dtype

    attn.forward = types.MethodType(qwen3_attention_forward, attn)
    fused = run(attn, h, cos, sin)  # fused forward, target dtype

    assert fused.shape == gold.shape
    floor = 1e-4 if dtype == torch.float32 else 0.0
    _assert_not_worse(fused, ref, gold, "attn_out", slack=2.0, floor=floor)
