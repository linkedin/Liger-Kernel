"""Correctness checks for the sole SM90 fragment forward."""

import pytest
import torch

import liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 as fragment_module

from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import CLUSTER_M
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import EPILOGUE_SLICE_N
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import LOGICAL_N
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import ScaledCEForwardFragmentConfig
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import _resolve_split
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import scaled_ce_forward_fragment
from test.utils import assert_verbose_allclose
from test.utils import set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the fragment forward requires a Hopper (SM90) GPU",
)


def _torch_oracle(x, weight, target, temperature, ignore_index):
    logits = (x.float() @ weight.float().t()) / temperature
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    weighted = (probs * logits).sum(dim=-1)
    safe_target = target.masked_fill(target == ignore_index, 0)
    target_logits = logits.gather(1, safe_target[:, None]).squeeze(1)
    valid = target != ignore_index
    zeros = torch.zeros_like(lse)
    nll = torch.where(valid, lse - target_logits, zeros)
    entropy = torch.where(valid, lse - weighted, zeros)
    return nll, lse, entropy


def _assert_matches_oracle(
    x,
    weight,
    target,
    temperature,
    ignore_index,
    return_entropy,
    config,
):
    actual_nll, actual_entropy, actual_lse = scaled_ce_forward_fragment(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        return_entropy,
        config,
    )
    expected_nll, expected_lse, expected_entropy = _torch_oracle(
        x,
        weight,
        target,
        temperature,
        ignore_index,
    )

    assert actual_nll.dtype == torch.float32
    assert actual_lse.dtype == torch.float32
    assert_verbose_allclose(actual_nll, expected_nll, atol=5e-2, rtol=2e-2)
    assert_verbose_allclose(actual_lse, expected_lse, atol=5e-2, rtol=2e-2)
    if return_entropy:
        assert actual_entropy.dtype == torch.bfloat16
        assert_verbose_allclose(actual_entropy.float(), expected_entropy, atol=5e-2, rtol=5e-2)
    else:
        assert actual_entropy is None
    return actual_nll, actual_entropy, actual_lse


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(2048, (9, 8, 2)), (4096, (5, 4, 2))],
)
def test_fragment_explicit_cluster_target_uses_uneven_split(tokens, expected):
    config = ScaledCEForwardFragmentConfig(target_cluster_pairs=66)
    split_n, base_split_n, extra_m_pairs, num_m_tiles, _ = _resolve_split(config, tokens, 131072)
    num_m_pairs = (num_m_tiles + CLUSTER_M - 1) // CLUSTER_M

    assert (split_n, base_split_n, extra_m_pairs) == expected
    assert num_m_pairs * base_split_n + extra_m_pairs == 66


def test_fragment_auto_split_uses_hardware_cluster_capacity(monkeypatch):
    calls = []

    def fake_max_active_clusters(cluster_m):
        calls.append(cluster_m)
        return 66

    monkeypatch.setattr(fragment_module, "_max_active_clusters", fake_max_active_clusters)
    split_n, base_split_n, extra_m_pairs, _, _ = _resolve_split(
        ScaledCEForwardFragmentConfig(),
        2048,
        131072,
    )

    assert calls == [CLUSTER_M]
    assert (split_n, base_split_n, extra_m_pairs) == (9, 8, 2)


def test_fragment_uneven_split_entropy_matches_torch_oracle():
    set_seed(40)
    tokens = 512
    hidden_size = 64
    vocab_size = 1600
    temperature = 0.7
    ignore_index = -17
    config = ScaledCEForwardFragmentConfig(target_cluster_pairs=7)
    split_n, base_split_n, extra_m_pairs, _, num_logical_n_tiles = _resolve_split(
        config,
        tokens,
        vocab_size,
    )

    assert (split_n, base_split_n, extra_m_pairs) == (4, 3, 1)
    assert num_logical_n_tiles == 5

    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.15
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (tokens,), device="cuda", dtype=torch.long)
    target[0] = vocab_size - 1
    target[1] = ignore_index

    actual_nll, actual_entropy, _ = _assert_matches_oracle(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        True,
        config,
    )
    assert actual_nll[1].item() == 0.0
    assert actual_entropy[1].item() == 0.0


@pytest.mark.parametrize(
    ("tokens", "hidden_size", "vocab_size", "temperature", "ignore_index", "return_entropy"),
    [
        (129, 64, 257, 1.1, -11, True),
        (256, 64, 1600, 1.0, -7, False),
        (256, 64, 1600, 0.8, -7, True),
        (257, 70, 1937, 0.7, -23, False),
        (257, 70, 1937, 1.3, -23, True),
    ],
    ids=["small-v-entropy", "k64-nll", "k64-entropy", "ragged-mhv-nll", "ragged-mhv-entropy"],
)
def test_fragment_matches_independent_torch_oracle(
    tokens,
    hidden_size,
    vocab_size,
    temperature,
    ignore_index,
    return_entropy,
):
    set_seed(41 + hidden_size + int(return_entropy))
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.15
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (tokens,), device="cuda", dtype=torch.long)
    target[0] = vocab_size - 1
    target[1] = ignore_index
    target[-1] = vocab_size - 1

    actual_nll, actual_entropy, _ = _assert_matches_oracle(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        return_entropy,
        ScaledCEForwardFragmentConfig(split_n=min(2, (vocab_size + LOGICAL_N - 1) // LOGICAL_N)),
    )
    assert actual_nll[1].item() == 0.0
    if return_entropy:
        assert actual_entropy[1].item() == 0.0


def test_fragment_two_n160_accumulators_four_n80_slices_k4096():
    set_seed(160)
    tokens = 133
    hidden_size = 4095
    vocab_size = 1277
    temperature = 0.75
    ignore_index = -29
    config = ScaledCEForwardFragmentConfig(split_n=2)
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.15
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (tokens,), device="cuda", dtype=torch.long)
    target[:8] = torch.tensor(
        [79, 80, 159, 160, 239, 240, 319, vocab_size - 1],
        device="cuda",
    )
    target[8] = ignore_index
    target[-1] = vocab_size - 1

    actual_nll, actual_entropy, _ = _assert_matches_oracle(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        True,
        config,
    )

    assert LOGICAL_N == 320
    assert EPILOGUE_SLICE_N == 80
    assert (vocab_size + LOGICAL_N - 1) // LOGICAL_N == 4
    assert (vocab_size - 1) % LOGICAL_N >= 3 * EPILOGUE_SLICE_N
    assert actual_nll[8].item() == 0.0
    assert actual_entropy[8].item() == 0.0


def test_fragment_repeated_calls_preserve_nll_and_refresh_partial_statistics():
    set_seed(161)
    tokens = 129
    hidden_size = 64
    vocab_size = 641
    temperature = 0.9
    ignore_index = -37
    config = ScaledCEForwardFragmentConfig(split_n=2)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (tokens,), device="cuda", dtype=torch.long)
    target[0] = vocab_size - 1
    target[1] = ignore_index
    x0 = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
    x1 = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.2

    nll_only = _assert_matches_oracle(
        x0,
        weight,
        target,
        temperature,
        ignore_index,
        False,
        config,
    )
    first = _assert_matches_oracle(
        x0,
        weight,
        target,
        temperature,
        ignore_index,
        True,
        config,
    )
    second = _assert_matches_oracle(
        x1,
        weight,
        target,
        temperature,
        ignore_index,
        True,
        config,
    )

    assert_verbose_allclose(first[0], nll_only[0], atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(first[2], nll_only[2], atol=1e-5, rtol=1e-5)
    assert not torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])
