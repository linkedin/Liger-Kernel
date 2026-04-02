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

# ---------------------------------------------------------------------------
# Skip if no CUDA
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
    device = "cuda"
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
# Test 2: Forward correctness vs. reference
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
def test_forward_correctness(T, E, H, intermediate_dim, K, dtype, atol, rtol):
    device = "cuda"
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)

    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    assert torch.allclose(out, ref, atol=atol, rtol=rtol), (
        f"Forward mismatch (dtype={dtype}): max_diff={torch.abs(out - ref).max():.4e}"
    )


# ---------------------------------------------------------------------------
# Test 3: Backward / gradient correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T, E, H, intermediate_dim, K",
    [
        (7, 4, 32, 16, 2),  # T < BLOCK_M_TOKEN: sub-tile padding edge case
        (512, 8, 128, 64, 2),  # multi-tile baseline: T*K/E=128 → 2 tiles/expert
        (512, 8, 97, 47, 2),  # multi-tile + odd H/I: tail masking in dW1, dW2, dX_expanded
        (512, 7, 64, 32, 3),  # multi-tile + prime E: non-pow2 dW grid decomposition
        (512, 8, 64, 32, 1),  # multi-tile + K=1: dx reduction is trivial gather
    ],
)
def test_backward_correctness(T, E, H, intermediate_dim, K):
    """Compare gradients of fused kernel vs. reference implementation."""
    device = "cuda"
    dtype = torch.float32
    x_ref, gup_ref, dn_ref, idx, wts = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)

    # Clone inputs and enable gradients for both
    x1 = x_ref.clone().requires_grad_(True)
    gup1 = gup_ref.clone().requires_grad_(True)
    dn1 = dn_ref.clone().requires_grad_(True)
    wts1 = wts.clone().requires_grad_(True)

    x2 = x_ref.clone().requires_grad_(True)
    gup2 = gup_ref.clone().requires_grad_(True)
    dn2 = dn_ref.clone().requires_grad_(True)
    wts2 = wts.clone().requires_grad_(True)

    # Reference backward
    out_ref = _reference_moe_forward(x1, gup1, dn1, idx, wts1)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # Fused backward
    out_fused = LigerFusedMoEFunction.apply(x2, gup2, dn2, idx, wts2)
    loss_fused = out_fused.sum()
    loss_fused.backward()

    atol, rtol = 3e-3, 1e-2

    def _mismatch_info(a, b, atol, rtol, name):
        diff = torch.abs(a - b)
        bad = ~torch.isclose(a, b, atol=atol, rtol=rtol)
        bad_count = bad.sum().item()
        return f"{name} mismatch: {bad_count}/{bad.numel()} elements, max_diff={diff.max():.4e}" + (
            f", mean_bad_diff={diff[bad].mean():.4e}" if bad_count > 0 else ""
        )

    assert torch.allclose(wts2.grad, wts1.grad, atol=atol, rtol=rtol), _mismatch_info(
        wts2.grad, wts1.grad, atol, rtol, "dtopk_weights"
    )
    assert torch.allclose(dn2.grad, dn1.grad, atol=atol, rtol=rtol), _mismatch_info(
        dn2.grad, dn1.grad, atol, rtol, "ddown_proj"
    )
    assert torch.allclose(x2.grad, x1.grad, atol=atol, rtol=rtol), _mismatch_info(x2.grad, x1.grad, atol, rtol, "dx")
    assert torch.allclose(gup2.grad, gup1.grad, atol=atol, rtol=rtol), _mismatch_info(
        gup2.grad, gup1.grad, atol, rtol, "dgate_up"
    )


# ---------------------------------------------------------------------------
# Test 4: Edge cases
# ---------------------------------------------------------------------------


def test_all_tokens_to_one_expert():
    """All tokens route to expert 0; all others empty."""
    T, E, H, intermediate_dim, K = 32, 8, 64, 32, 2
    device = "cuda"
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

    assert torch.allclose(out, ref, atol=1e-3), f"all-to-one-expert mismatch: max_diff={torch.abs(out - ref).max():.4e}"


def test_single_token():
    """T=1 edge case."""
    T, E, H, intermediate_dim, K = 1, 4, 32, 16, 2
    device = "cuda"
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    assert torch.allclose(out, ref, atol=1e-3)


def test_K_equals_E():
    """K == E: every token routes to every expert."""
    T, E, H, intermediate_dim, K = 16, 4, 32, 16, 4
    device = "cuda"
    dtype = torch.float32
    x, gate_up_proj, down_proj, top_k_index, top_k_weights = _make_inputs(T, E, H, intermediate_dim, K, dtype, device)
    out = LigerFusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    ref = _reference_moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    assert torch.allclose(out, ref, atol=1e-3)
