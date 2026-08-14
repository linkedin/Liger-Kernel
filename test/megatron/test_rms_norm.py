"""Unit tests for LigerMegatronRMSNorm.

These tests deliberately do not import megatron-core; the wrapper's contract
is verified using a duck-typed ``SimpleNamespace`` for ``config``. The
parametrization style mirrors ``test/transformers/test_rms_norm.py``.
"""

import os

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm
from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------------
# References + helpers
# ---------------------------------------------------------------------------


def _config(
    normalization: str = "RMSNorm",
    sequence_parallel: bool = False,
    layernorm_zero_centered_gamma: bool = False,
):
    return SimpleNamespace(
        normalization=normalization,
        sequence_parallel=sequence_parallel,
        layernorm_zero_centered_gamma=layernorm_zero_centered_gamma,
    )


class _LlamaRMSNorm(nn.Module):
    """Matches LigerMegatronRMSNorm semantics with zero_centered_gamma=False."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class _ZeroCenteredRMSNorm(nn.Module):
    """Matches LigerMegatronRMSNorm with zero_centered_gamma=True: applies
    ``(1 + w) * x_normalized``, weight initialized to zeros."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (1.0 + self.weight) * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Forward + backward correctness (mirrors test_correctness in
# test/transformers/test_rms_norm.py).
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        # weird shapes
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-4, 1e-5),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "reference_cls, zero_centered_gamma",
    [
        (_LlamaRMSNorm, False),
        (_ZeroCenteredRMSNorm, True),
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol, reference_cls, zero_centered_gamma):
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    do = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    ref_rms = reference_cls(hidden_size=hd).to(device).to(dtype)
    ref_o = ref_rms(h1)
    ref_o.backward(do, retain_graph=True)

    liger_rms = (
        LigerMegatronRMSNorm(
            config=_config(layernorm_zero_centered_gamma=zero_centered_gamma),
            hidden_size=hd,
        )
        .to(device)
        .to(dtype)
    )
    liger_o = liger_rms(h2)
    liger_o.backward(do, retain_graph=True)

    assert_verbose_allclose(ref_o, liger_o, atol=atol, rtol=rtol)
    assert_verbose_allclose(ref_rms.weight.grad, liger_rms.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(h1.grad, h2.grad, atol=atol, rtol=rtol, max_print=20)


# ---------------------------------------------------------------------------
# Wrapper-specific contracts not covered by test_correctness.
# ---------------------------------------------------------------------------


def test_zero_centered_init_and_offset():
    """zero_centered_gamma flips both the weight init and the kernel offset."""
    m = LigerMegatronRMSNorm(config=_config(layernorm_zero_centered_gamma=True), hidden_size=64)
    assert torch.equal(m.weight.detach(), torch.zeros(64))
    assert m._offset == 1.0

    m = LigerMegatronRMSNorm(config=_config(layernorm_zero_centered_gamma=False), hidden_size=64)
    assert torch.equal(m.weight.detach(), torch.ones(64))
    assert m._offset == 0.0


@pytest.mark.parametrize("sp", [True, False])
def test_sequence_parallel_attribute_propagates(sp):
    """Megatron's distributed optimizer reads this attribute to decide
    whether to all-reduce the weight's gradient across TP ranks under SP."""
    m = LigerMegatronRMSNorm(config=_config(sequence_parallel=sp), hidden_size=64)
    assert getattr(m.weight, "sequence_parallel", None) is sp


def test_rejects_non_rmsnorm_config():
    """The wrapper only supports RMSNorm; surface a clear error otherwise."""
    with pytest.raises(ValueError, match="RMSNorm"):
        LigerMegatronRMSNorm(config=_config(normalization="LayerNorm"), hidden_size=64)


def test_forward_does_not_modify_input():
    """``in_place=False`` is hardcoded for Megatron's recompute / CUDA-graph
    safety; verify the input tensor isn't touched by the forward."""
    if device != "cuda":
        pytest.skip("CUDA required for Liger kernels.")
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
    snapshot = x.detach().clone()
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    _ = m(x)
    assert torch.equal(x, snapshot)
