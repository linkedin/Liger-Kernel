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

from liger_kernel.ops.fused_moe import LigerFusedMoEFunction
from liger_kernel.ops.fused_moe import compute_routing_metadata
from liger_kernel.utils import infer_device

device = infer_device()


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
