"""
Tests for LigerFusedMoEFunction.

Tests cover:
1. Routing metadata correctness (permutation invariants)
2. Forward correctness vs. reference Python loop
3. Backward / gradient correctness
4. Edge cases (empty experts, single expert, all tokens to one expert)
5. Multiple dtypes
"""

import pytest
import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedMoEFunction
from liger_kernel.utils import infer_device

device = infer_device()

if device == "npu":
    from liger_kernel.ops.backends._ascend.ops.fused_moe import compute_routing_metadata
else:
    from liger_kernel.ops.fused_moe import compute_routing_metadata


# ---------------------------------------------------------------------------
# Reference implementation (original Python loop)
# ---------------------------------------------------------------------------


def _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Reference: Python loop over active experts (original LigerExperts logic)."""
    T, H = x.shape
    E = gate_up_proj.shape[0]
    final = torch.zeros_like(x)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index.long(), num_classes=E)
        # top_k_index (T, K) → one_hot (T, K, E) → permute (E, K, T)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for eh in expert_hit:
        eidx = eh[0]
        top_k_pos, token_idx = torch.where(expert_mask[eidx])
        curr = x[token_idx]
        gate, up = nn.functional.linear(curr, gate_up_proj[eidx]).chunk(2, dim=-1)
        curr = nn.functional.silu(gate) * up
        curr = nn.functional.linear(curr, down_proj[eidx])
        curr = curr * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, curr.to(final.dtype))

    return final


def _make_inputs(T, E, H, intermediate_dim, K, dtype, device, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(T, H, dtype=dtype, device=device)
    gate_up_proj = torch.randn(E, 2 * intermediate_dim, H, dtype=dtype, device=device) * 0.02
    down_proj = torch.randn(E, H, intermediate_dim, dtype=dtype, device=device) * 0.02
    # Random top-k routing (uniform distribution)
    logits = torch.randn(T, E, device=device)
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)
    top_k_weights = torch.softmax(torch.gather(logits, 1, top_k_index.long()), dim=-1).to(dtype)
    return x, gate_up_proj, down_proj, top_k_index, top_k_weights


# ---------------------------------------------------------------------------
# Test 1: Routing metadata invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T, E, K",
    [
        (64, 8, 2),
        (128, 16, 4),
        (100, 8, 2),  # non-power-of-2 T
        (256, 32, 8),
    ],
)
def test_routing_metadata_invariants(T, E, K):
    torch.manual_seed(0)
    logits = torch.randn(T, E, device=device)
    top_k_index = torch.topk(logits, K, dim=-1).indices.to(torch.int32)

    expert_freq, expert_freq_offset, x_gather_idx, s_scatter_idx, s_rev_scatter_idx, _, _ = compute_routing_metadata(
        top_k_index, E
    )

    TK = T * K

    # Inverse permutation: s_rev[s_scatter[i]] == i
    reconstructed = s_rev_scatter_idx[s_scatter_idx.long()]
    assert torch.all(reconstructed == torch.arange(TK, device=device, dtype=torch.int32)), (
        "s_reverse_scatter_idx is not the inverse of s_scatter_idx"
    )

    # expert_freq_offset[e+1] - expert_freq_offset[e] == expert_frequency[e]
    freq_from_offset = expert_freq_offset[1:] - expert_freq_offset[:-1]
    assert torch.all(freq_from_offset == expert_freq), "expert_freq_offset does not match expert_frequency"

    # Total tokens: offset[E] == TK
    assert int(expert_freq_offset[-1]) == TK

    # Expert frequencies sum to TK
    assert int(expert_freq.sum()) == TK

    # x_gather_idx values are in [0, T)
    assert x_gather_idx.min() >= 0 and x_gather_idx.max() < T


# ---------------------------------------------------------------------------
# Test 2: Forward + backward correctness vs. reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T, E, H, intermediate_dim, K",
    [
        (7, 4, 64, 32, 2),  # T < BLOCK_M_TOKEN: tile row-mask is mostly padding (unique sub-tile edge)
        (512, 8, 256, 128, 2),  # multi-tile baseline: T*K/E=128 → 2 tiles/expert
        (512, 8, 97, 47, 2),  # multi-tile + odd H/I: tail masking across tile boundaries
        (512, 7, 128, 64, 3),  # multi-tile + prime E: non-pow2 grid decomposition
        (512, 8, 256, 64, 1),  # multi-tile + K=1: no weighted sum in token aggregation
        (128, 8, 256, 64, 8),  # multi-tile + K=E: maximum routing density
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-4),
        (torch.bfloat16, 1e-1, 1e-2),
    ],
)
def test_correctness(T, E, H, intermediate_dim, K, dtype, atol, rtol):
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)

    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    if dtype == torch.float32:
        x1 = x.clone().requires_grad_(True)
        gup1 = gate_up_proj.clone().requires_grad_(True)
        dn1 = down_proj.clone().requires_grad_(True)
        wts1 = top_k_weights.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)
        gup2 = gate_up_proj.clone().requires_grad_(True)
        dn2 = down_proj.clone().requires_grad_(True)
        wts2 = top_k_weights.clone().requires_grad_(True)

        out_ref = _reference_moe_forward(x1, gup1, dn1, top_k_index, wts1)
        out_ref.sum().backward()
        out_fused = LigerFusedMoEFunction.apply(x2, gup2, dn2, top_k_index, wts2)
        out_fused.sum().backward()

        b_atol, b_rtol = 3e-3, 1e-2
        torch.testing.assert_close(wts2.grad, wts1.grad, atol=b_atol, rtol=b_rtol)
        torch.testing.assert_close(dn2.grad, dn1.grad, atol=b_atol, rtol=b_rtol)
        torch.testing.assert_close(x2.grad, x1.grad, atol=b_atol, rtol=b_rtol)
        torch.testing.assert_close(gup2.grad, gup1.grad, atol=b_atol, rtol=b_rtol)


# ---------------------------------------------------------------------------
# Test 3: Edge cases (forward correctness)
# ---------------------------------------------------------------------------


def test_all_tokens_to_one_expert():
    """All tokens route to expert 0; all others empty."""
    T, E, H, intermediate_dim, K = 32, 8, 64, 32, 2
    dtype = torch.float32
    torch.manual_seed(0)

    x = torch.randn(T, H, dtype=dtype, device=device)
    gate_up_proj = torch.randn(E, 2 * intermediate_dim, H, dtype=dtype, device=device) * 0.02
    down_proj = torch.randn(E, H, intermediate_dim, dtype=dtype, device=device) * 0.02
    # Force all tokens to expert 0 and 1
    top_k_index = torch.zeros(T, K, dtype=torch.int32, device=device)
    top_k_weights = torch.ones(T, K, dtype=dtype, device=device) / K

    # Should not crash
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)


def test_single_token():
    """T=1 edge case."""
    T, E, H, intermediate_dim, K = 1, 4, 32, 16, 2
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)


def test_K_equals_E():
    """K == E: every token routes to every expert."""
    T, E, H, intermediate_dim, K = 16, 4, 32, 16, 4
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3,
    reason="Regression needs >= 16 GB GPU memory: pre_act tensor must exceed 2^31 bytes to trigger int32 stride overflow.",
)
def test_large_tk_pointer_overflow_regression():
    """Regression for #1246: int32 overflow in pre_act / post_act / Y pointer arithmetic.

    When TK * stride > 2^31, the int32 offset wraps and the kernel writes to
    garbage addresses (corruption → NaN/Inf, or unmapped → IMA depending on
    allocator state). Bug is shape-dependent, not hardware-dependent, but
    surfaces at the Qwen3.5-MoE scale reported in #1246
    (T~135k, K=8 → TK ~ 1M; intermediate_dim=1024 → stride_pre_TK=2048 →
    TK*stride ~ 2.21 G > 2^31).

    Repro shape: TK * 2*intermediate_dim must exceed 2^31 = 2,147,483,648 with
    enough margin that many rows of pre_act are corrupted (a tiny overflow can
    false-negative if the allocator happens to hand back zeroed memory for the
    rows the kernel was supposed to write but skipped). Other dims kept small
    so total GPU memory stays around ~8 GB.
    """
    # TK = 140000 * 8 = 1,120,000 → 1,120,000 * 2048 = 2,293,760,000.
    # Overflows int32 max (2,147,483,647) by ~146 M elements → corrupts last
    # ~72k rows of pre_act / post_act stores. E ≥ K required for topk routing.
    T, K = 140000, 8
    E, H, intermediate_dim = 16, 512, 1024
    dtype = torch.bfloat16

    TK = T * K
    stride_pre_TK = 2 * intermediate_dim
    assert TK * stride_pre_TK > 2**31, (
        f"Test shape doesn't actually trigger overflow: TK*stride={TK * stride_pre_TK:,} vs 2^31={2**31:,}"
    )

    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(
        T, E, H, intermediate_dim, K, dtype, device
    )

    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    # Before the fix: stores to the last rows of pre_act/post_act/Y land at
    # wrapped int32 addresses → final output has NaN/Inf in those token rows.
    assert torch.isfinite(out).all(), (
        "Output contains NaN/Inf — int32 overflow in pre_act/post_act/Y pointer arithmetic. "
        f"TK={TK:,}, stride_pre_TK={stride_pre_TK:,}, "
        f"TK*stride={TK * stride_pre_TK:,} (int32 max={2**31 - 1:,})"
    )
