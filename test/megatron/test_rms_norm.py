"""Unit tests for LigerMegatronRMSNorm.

These tests deliberately do not import megatron-core; the wrapper's contract
is verified using a duck-typed ``SimpleNamespace`` for ``config``. The
parametrization style mirrors ``test/transformers/test_rms_norm.py`` —
shape sweeps, dtype sweeps, and zero-centered-gamma variants.
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm
from liger_kernel.utils import infer_device

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


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Liger kernels.")


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
        return (self.weight * x.to(input_dtype))


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
        return ((1.0 + self.weight) * x.to(input_dtype))


# ---------------------------------------------------------------------------
# Forward / backward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        (1, 64, 768),
        (4, 256, 4096),
        # weird shapes
        (5, 123, 257),
        (3, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "reference_cls, zero_centered",
    [
        (_LlamaRMSNorm, False),
        (_ZeroCenteredRMSNorm, True),
    ],
)
def test_forward_matches_reference(
    bs, sl, hd, dtype, atol, rtol, reference_cls, zero_centered
):
    _require_cuda()
    x = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    liger = (
        LigerMegatronRMSNorm(
            config=_config(layernorm_zero_centered_gamma=zero_centered),
            hidden_size=hd,
            eps=1e-6,
        )
        .to(device)
        .to(dtype)
    )
    ref = reference_cls(hidden_size=hd, eps=1e-6).to(device).to(dtype)

    assert_verbose_allclose(liger(x), ref(x), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        (5, 123, 257),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "reference_cls, zero_centered",
    [
        (_LlamaRMSNorm, False),
        (_ZeroCenteredRMSNorm, True),
    ],
)
def test_backward_matches_reference(
    bs, sl, hd, dtype, atol, rtol, reference_cls, zero_centered
):
    _require_cuda()
    x = torch.randn(bs, sl, hd, device=device, dtype=dtype)
    dy = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    x_l = x.detach().clone().requires_grad_(True)
    x_r = x.detach().clone().requires_grad_(True)

    liger = (
        LigerMegatronRMSNorm(
            config=_config(layernorm_zero_centered_gamma=zero_centered),
            hidden_size=hd,
            eps=1e-6,
        )
        .to(device)
        .to(dtype)
    )
    ref = reference_cls(hidden_size=hd, eps=1e-6).to(device).to(dtype)

    liger(x_l).backward(dy)
    ref(x_r).backward(dy)

    assert_verbose_allclose(x_l.grad, x_r.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger.weight.grad, ref.weight.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Construction / config plumbing
# ---------------------------------------------------------------------------


def test_zero_centered_init_and_offset():
    _require_cuda()
    m = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=True), hidden_size=64
    ).to(device)
    assert torch.equal(m.weight.detach(), torch.zeros(64, device=device))
    assert m._offset == 1.0

    m = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=False), hidden_size=64
    ).to(device)
    assert torch.equal(m.weight.detach(), torch.ones(64, device=device))
    assert m._offset == 0.0


def test_zero_centered_forward_equivalent_to_ones_init():
    """Zero-centered with w=0 should produce the same output as standard with w=1."""
    _require_cuda()
    x = torch.randn(2, 8, 128, dtype=torch.float32, device=device)

    m_zc = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=True), hidden_size=128
    ).to(device)
    m_std = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=False), hidden_size=128
    ).to(device)

    assert_verbose_allclose(m_zc(x), m_std(x), atol=1e-5, rtol=1e-5)


def test_constructor_kwarg_overrides_config_for_zero_centered():
    """Explicit zero_centered_gamma=True wins even when config says False."""
    m = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=False),
        hidden_size=32,
        zero_centered_gamma=True,
    )
    assert m.zero_centered_gamma is True
    assert m._offset == 1.0


@pytest.mark.parametrize("sp", [True, False])
def test_sequence_parallel_attribute_propagates(sp):
    m = LigerMegatronRMSNorm(config=_config(sequence_parallel=sp), hidden_size=64)
    assert getattr(m.weight, "sequence_parallel", None) is sp


def test_rejects_non_rmsnorm_config():
    with pytest.raises(ValueError, match="RMSNorm"):
        LigerMegatronRMSNorm(
            config=_config(normalization="LayerNorm"), hidden_size=64
        )


def test_rejects_non_rmsnorm_l2():
    with pytest.raises(ValueError, match="RMSNorm"):
        LigerMegatronRMSNorm(config=_config(normalization="L2Norm"), hidden_size=64)


def test_constructor_accepts_extra_compat_kwargs():
    """Should silently accept the interface-compat kwargs that FusedLayerNorm
    takes — persist_layer_norm and normalization — without raising."""
    m = LigerMegatronRMSNorm(
        config=_config(),
        hidden_size=64,
        eps=1e-6,
        persist_layer_norm=True,
        normalization="RMSNorm",
    )
    assert m.eps == 1e-6


def test_eps_propagates():
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64, eps=3.14e-5)
    assert m.eps == 3.14e-5


@pytest.mark.parametrize("hidden_size", [64, (64,), (128,)])
def test_hidden_size_accepts_int_or_tuple(hidden_size):
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=hidden_size)
    expected = hidden_size if isinstance(hidden_size, tuple) else (hidden_size,)
    assert tuple(m.hidden_size) == expected
    assert m.weight.shape == torch.Size(expected)


# ---------------------------------------------------------------------------
# Module behaviour
# ---------------------------------------------------------------------------


def test_extra_repr_is_informative():
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64, eps=1e-5)
    rep = m.extra_repr()
    assert "hidden_size" in rep
    assert "eps" in rep
    assert "zero_centered_gamma" in rep


def test_forward_does_not_modify_input():
    """in_place=False is hardcoded; verify the input tensor is untouched."""
    _require_cuda()
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
    snapshot = x.detach().clone()
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    _ = m(x)
    assert torch.equal(x, snapshot)


def test_multiple_forwards_are_deterministic():
    """Same input → same output across repeated calls."""
    _require_cuda()
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    y1 = m(x)
    y2 = m(x)
    assert torch.equal(y1, y2)


def test_independent_instances_do_not_share_state():
    """Two wrappers built from the same config must have independent weights."""
    _require_cuda()
    cfg = _config()
    m1 = LigerMegatronRMSNorm(config=cfg, hidden_size=64).to(device)
    m2 = LigerMegatronRMSNorm(config=cfg, hidden_size=64).to(device)

    assert m1.weight.data_ptr() != m2.weight.data_ptr()
    with torch.no_grad():
        m1.weight.fill_(2.0)
    assert not torch.allclose(m1.weight, m2.weight)


def test_reset_parameters_reinitializes_weight():
    _require_cuda()
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    with torch.no_grad():
        m.weight.fill_(7.0)
    m.reset_parameters()
    assert torch.equal(m.weight, torch.ones(64, device=device))

    m_zc = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=True), hidden_size=64
    ).to(device)
    with torch.no_grad():
        m_zc.weight.fill_(7.0)
    m_zc.reset_parameters()
    assert torch.equal(m_zc.weight, torch.zeros(64, device=device))


def test_state_dict_roundtrip_preserves_output():
    """Save weights, load into a fresh instance, output must match."""
    _require_cuda()
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)

    src = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    with torch.no_grad():
        src.weight.uniform_(-0.5, 0.5)

    dst = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    dst.load_state_dict(copy.deepcopy(src.state_dict()))

    assert torch.equal(src(x), dst(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_module_to_dtype_changes_weight_dtype(dtype):
    if dtype == torch.bfloat16 and not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    _require_cuda()
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device).to(dtype)
    assert m.weight.dtype == dtype


def test_weight_gradient_accumulates_across_backwards():
    """Repeated backward calls (without zero_grad) should accumulate."""
    _require_cuda()
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
    dy = torch.randn(2, 8, 64, device=device, dtype=torch.float32)

    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)

    x1 = x.clone().requires_grad_(True)
    m(x1).backward(dy)
    g1 = m.weight.grad.clone()

    x2 = x.clone().requires_grad_(True)
    m(x2).backward(dy)
    g2 = m.weight.grad.clone()

    assert_verbose_allclose(g2, 2.0 * g1, atol=1e-5, rtol=1e-5)


def test_requires_grad_is_respected_when_input_does_not_need_grad():
    """If input doesn't require grad, output still doesn't get a grad_fn that
    blocks .backward() on something else."""
    _require_cuda()
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32, requires_grad=False)
    m = LigerMegatronRMSNorm(config=_config(), hidden_size=64).to(device)
    y = m(x)
    # weight still requires grad → output should require grad
    assert y.requires_grad
