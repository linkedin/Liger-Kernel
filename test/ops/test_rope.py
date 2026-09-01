"""Cross-backend correctness tests for RoPE."""

from __future__ import annotations

import pytest
import torch

import liger_kernel.functional  # noqa: F401

from liger_kernel.backends.dispatch import dispatch

from .conftest import get_available_backends_for_op

_REGISTERED_BACKENDS = get_available_backends_for_op("rope")


def _reference(q, k, cos, sin):
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    cos_q = cos[:, None, :, : q1.shape[-1]]
    sin_q = sin[:, None, :, : q1.shape[-1]]
    return (
        torch.cat((q1 * cos_q - q2 * sin_q, q2 * cos_q + q1 * sin_q), dim=-1),
        torch.cat((k1 * cos_q - k2 * sin_q, k2 * cos_q + k1 * sin_q), dim=-1),
    )


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_correctness(backend, dtype):
    if backend == "__none__":
        pytest.skip("No rope backends registered")

    torch.manual_seed(0)
    q = torch.randn(2, 4, 17, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 4, 17, 64, device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn(2, 17, 64, device="cuda", dtype=dtype)
    sin = torch.randn(2, 17, 64, device="cuda", dtype=dtype)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()

    q_out, k_out = dispatch("rope", q, k, cos, sin, None, 1, backend=backend)
    q_expected, k_expected = _reference(q_ref, k_ref, cos, sin)

    atol_fwd = atol_bwd = 1e-5 if dtype == torch.float32 else 1e-2
    rtol_fwd = rtol_bwd = 1e-5 if dtype == torch.float32 else 1e-2
    if backend == "nvidia-cutedsl" and dtype == torch.bfloat16:
        # The delegated fused-TMA backend (ops/cutedsl/ops/rope.py) rounds the
        # rotation's fan-in products in fp32 before the bf16 store, while this
        # test's reference multiplies in bf16; at near-cancellation elements
        # (|products| ~16x the result) the two can land one last bf16 ulp
        # (~9e-3 at |q|~2.3) apart.  Against a float64 ground truth the fused
        # kernel is strictly *closer* than both the old inline backend and
        # Triton (max|q err| 0.0156 vs 0.0267 / 0.0246; elements >0.01 from
        # truth 67 vs 130 / 129 on this case), so this cell carries the
        # backend's registered bf16 tolerances (see ``tolerances=`` in
        # backends/_cutedsl/rope.py) instead of the uniform 1e-2/1e-2.
        atol_fwd, rtol_fwd = 2e-2, 1e-2
        atol_bwd, rtol_bwd = 1e-1, 2e-2
    torch.testing.assert_close(q_out, q_expected, atol=atol_fwd, rtol=rtol_fwd)
    torch.testing.assert_close(k_out, k_expected, atol=atol_fwd, rtol=rtol_fwd)

    dq = torch.randn_like(q_out)
    dk = torch.randn_like(k_out)
    dq_ref = dq.clone()
    dk_ref = dk.clone()
    torch.autograd.backward((q_out, k_out), (dq, dk))
    torch.autograd.backward((q_expected, k_expected), (dq_ref, dk_ref))
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol_bwd, rtol=rtol_bwd)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol_bwd, rtol=rtol_bwd)


@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 requires Ampere or newer")
def test_rope_cutedsl_mixed_qk_dtypes_do_not_reuse_wrong_kernel():
    if "nvidia-cutedsl" not in _REGISTERED_BACKENDS:
        pytest.skip("CuTe DSL RoPE is unavailable")

    torch.manual_seed(1)
    cos = torch.randn(1, 11, 64, device="cuda", dtype=torch.float32)
    sin = torch.randn(1, 11, 64, device="cuda", dtype=torch.float32)

    for q_dtype, k_dtype in (
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ):
        q = torch.randn(1, 2, 11, 64, device="cuda", dtype=q_dtype)
        k = torch.randn(1, 2, 11, 64, device="cuda", dtype=k_dtype)
        q_out, k_out = dispatch(
            "rope",
            q,
            k,
            cos,
            sin,
            None,
            1,
            backend="nvidia-cutedsl",
        )
        assert q_out.dtype == q_dtype
        assert k_out.dtype == k_dtype


def test_rope_cutedsl_rejects_odd_head_dimension():
    if "nvidia-cutedsl" not in _REGISTERED_BACKENDS:
        pytest.skip("CuTe DSL RoPE is unavailable")

    q = torch.randn(1, 2, 7, 41, device="cuda")
    k = torch.randn(1, 2, 7, 41, device="cuda")
    cos = torch.randn(1, 7, 41, device="cuda")
    sin = torch.randn(1, 7, 41, device="cuda")
    with pytest.raises(ValueError, match="even q/k head dimensions"):
        dispatch("rope", q, k, cos, sin, None, 1, backend="nvidia-cutedsl")
