"""Unit tests for LigerMegatronRMSNorm.

These tests deliberately do not import megatron-core. The wrapper's contract
is verified using a duck-typed ``SimpleNamespace`` for ``config``.
"""

from types import SimpleNamespace

import pytest
import torch

from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_matches_torch_rmsnorm(dtype):
    _require_cuda()
    torch.manual_seed(0)
    hidden, eps = 128, 1e-5

    x = torch.randn(4, 16, hidden, dtype=dtype, device="cuda")

    liger = (
        LigerMegatronRMSNorm(config=_config(), hidden_size=hidden, eps=eps)
        .cuda()
        .to(dtype)
    )
    ref = torch.nn.RMSNorm(hidden, eps=eps).cuda().to(dtype)

    atol, rtol = (2e-1, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-6)
    torch.testing.assert_close(liger(x), ref(x), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_matches_torch_rmsnorm(dtype):
    _require_cuda()
    torch.manual_seed(0)
    hidden, eps = 128, 1e-5

    x_l = torch.randn(4, 16, hidden, dtype=dtype, device="cuda", requires_grad=True)
    x_r = x_l.detach().clone().requires_grad_(True)

    liger = (
        LigerMegatronRMSNorm(config=_config(), hidden_size=hidden, eps=eps)
        .cuda()
        .to(dtype)
    )
    ref = torch.nn.RMSNorm(hidden, eps=eps).cuda().to(dtype)
    with torch.no_grad():
        ref.weight.copy_(liger.weight)

    g = torch.randn(4, 16, hidden, dtype=dtype, device="cuda")
    liger(x_l).backward(g)
    ref(x_r).backward(g)

    atol, rtol = (2e-1, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-6)
    torch.testing.assert_close(x_l.grad, x_r.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(liger.weight.grad, ref.weight.grad, atol=atol, rtol=rtol)


def test_zero_centered_init_and_offset():
    _require_cuda()
    m = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=True), hidden_size=64
    ).cuda()
    assert torch.equal(m.weight.detach(), torch.zeros(64, device="cuda"))
    assert m._offset == 1.0

    m = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=False), hidden_size=64
    ).cuda()
    assert torch.equal(m.weight.detach(), torch.ones(64, device="cuda"))
    assert m._offset == 0.0


def test_zero_centered_forward_equivalent_to_ones_init():
    """Zero-centered with w=0 should produce the same output as standard with w=1."""
    _require_cuda()
    torch.manual_seed(0)
    hidden = 128
    x = torch.randn(2, 8, hidden, dtype=torch.float32, device="cuda")

    m_zc = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=True), hidden_size=hidden
    ).cuda()
    m_std = LigerMegatronRMSNorm(
        config=_config(layernorm_zero_centered_gamma=False), hidden_size=hidden
    ).cuda()

    torch.testing.assert_close(m_zc(x), m_std(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sp", [True, False])
def test_sequence_parallel_attribute_propagates(sp):
    m = LigerMegatronRMSNorm(config=_config(sequence_parallel=sp), hidden_size=64)
    assert getattr(m.weight, "sequence_parallel", None) is sp


def test_rejects_non_rmsnorm_config():
    with pytest.raises(ValueError, match="RMSNorm"):
        LigerMegatronRMSNorm(
            config=_config(normalization="LayerNorm"), hidden_size=64
        )


def test_constructor_accepts_extra_compat_kwargs():
    m = LigerMegatronRMSNorm(
        config=_config(),
        hidden_size=64,
        eps=1e-6,
        persist_layer_norm=True,
        normalization="RMSNorm",
    )
    assert m.eps == 1e-6
