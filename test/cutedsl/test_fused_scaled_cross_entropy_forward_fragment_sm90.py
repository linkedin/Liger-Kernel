"""Correctness checks for the fixed SM90 N160 fragment forward."""

import pytest
import torch

import liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 as fragment_module

from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import CLUSTER_M
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import EPILOGUE_SLICE_N
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import LOGICAL_N
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import ScaledCEForwardFragmentConfig
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import _resolve_split
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_fragment_sm90 import scaled_ce_forward_fragment
from liger_kernel.ops.cutedsl.ops._fused_scaled_cross_entropy_forward_sm90 import scaled_ce_forward
from test.utils import assert_verbose_allclose
from test.utils import set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the fragment forward requires a Hopper (SM90) GPU",
)


def _torch_oracle(x, weight, target, temperature, ignore_index):
    logits = (x.float() @ weight.float().t()) / temperature
    lse = torch.logsumexp(logits, dim=-1)
    safe_target = target.masked_fill(target == ignore_index, 0)
    target_logits = logits.gather(1, safe_target[:, None]).squeeze(1)
    nll = torch.where(target == ignore_index, torch.zeros_like(lse), lse - target_logits)
    return nll, lse


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


def test_fragment_uneven_split_matches_torch_oracle():
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

    actual_nll, _, actual_lse = scaled_ce_forward_fragment(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        False,
        config,
    )
    expected_nll, expected_lse = _torch_oracle(x, weight, target, temperature, ignore_index)

    assert_verbose_allclose(actual_nll, expected_nll, atol=5e-2, rtol=2e-2)
    assert_verbose_allclose(actual_lse, expected_lse, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("tokens", "hidden_size", "vocab_size", "temperature", "ignore_index"),
    [
        (256, 64, 1600, 1.0, -7),
        (257, 70, 1937, 0.7, -23),
    ],
    ids=["k64-five-logical-tiles", "ragged-mhv"],
)
def test_fragment_matches_independent_torch_oracle(
    tokens,
    hidden_size,
    vocab_size,
    temperature,
    ignore_index,
):
    set_seed(41 + hidden_size)
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.15
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (tokens,), device="cuda", dtype=torch.long)
    target[0] = vocab_size - 1
    target[1] = ignore_index
    target[-1] = vocab_size - 1

    actual_nll, entropy, actual_lse = scaled_ce_forward_fragment(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        False,
        ScaledCEForwardFragmentConfig(split_n=2),
    )
    expected_nll, expected_lse = _torch_oracle(
        x,
        weight,
        target,
        temperature,
        ignore_index,
    )

    assert entropy is None
    assert actual_nll.dtype == torch.float32
    assert actual_lse.dtype == torch.float32
    assert actual_nll[1].item() == 0.0
    assert_verbose_allclose(actual_nll, expected_nll, atol=5e-2, rtol=2e-2)
    assert_verbose_allclose(actual_lse, expected_lse, atol=5e-2, rtol=2e-2)


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

    actual_nll, entropy, actual_lse = scaled_ce_forward_fragment(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        False,
        config,
    )
    expected_nll, expected_lse = _torch_oracle(
        x,
        weight,
        target,
        temperature,
        ignore_index,
    )

    assert LOGICAL_N == 320
    assert EPILOGUE_SLICE_N == 80
    assert (vocab_size + LOGICAL_N - 1) // LOGICAL_N == 4
    assert (vocab_size - 1) % LOGICAL_N >= 3 * EPILOGUE_SLICE_N
    assert entropy is None
    assert actual_nll[8].item() == 0.0
    assert_verbose_allclose(actual_nll, expected_nll, atol=5e-2, rtol=2e-2)
    assert_verbose_allclose(actual_lse, expected_lse, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("tokens", "hidden_size", "vocab_size"),
    [(256, 64, 1600), (193, 70, 1919)],
    ids=["aligned", "ragged"],
)
def test_fragment_matches_production_baseline(tokens, hidden_size, vocab_size):
    set_seed(97)
    temperature = 0.8
    ignore_index = -13
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.15
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    target = torch.randint(vocab_size, (tokens,), device="cuda", dtype=torch.long)
    target[0] = vocab_size - 1
    target[::17] = ignore_index

    expected_nll, _, expected_lse = scaled_ce_forward(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        False,
    )
    actual_nll, _, actual_lse = scaled_ce_forward_fragment(
        x,
        weight,
        target,
        temperature,
        ignore_index,
        False,
        ScaledCEForwardFragmentConfig(split_n=2),
    )

    assert_verbose_allclose(actual_nll, expected_nll, atol=5e-2, rtol=2e-2)
    assert_verbose_allclose(actual_lse, expected_lse, atol=5e-2, rtol=2e-2)


def test_fragment_entropy_is_explicitly_unsupported():
    x = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1600, 64, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(1600, (128,), device="cuda", dtype=torch.long)

    with pytest.raises(NotImplementedError, match="does not support entropy"):
        scaled_ce_forward_fragment(x, weight, target, return_entropy=True)
