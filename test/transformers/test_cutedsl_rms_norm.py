"""Correctness tests for the CuTe DSL RMSNorm kernel.

Parity oracle is the default Triton ``LigerRMSNormFunction`` (the repo's reference
RMSNorm). We check output, input-grad and weight-grad agreement across dtypes,
casting modes, affine/non-affine, in-place/out-of-place, and both regular and
irregular hidden dimensions.

The suite is skipped (never failed) when CUDA or the ``nvidia-cutlass-dsl``
package is unavailable, so it is safe to collect on CPU-only / non-NVIDIA hosts.
"""

import pytest
import torch

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl RMSNorm requires CUDA")


def _supports_bf16():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


# Per-(dtype, casting_mode) tolerances for the cutedsl-vs-Triton parity checks.
# fp32 llama/gemma are near bit-identical (only reduction-order noise). The cutedsl
# kernel accumulates in fp32 for every mode, so the "none" mode (which the Triton
# kernel keeps in the input dtype) is intentionally the loosest bar in low precision.
_TOL = {
    (torch.float32, "llama"): (2e-4, 1e-5),
    (torch.float32, "gemma"): (2e-4, 1e-5),
    (torch.float32, "none"): (2e-4, 1e-5),
    (torch.bfloat16, "llama"): (5e-2, 2e-2),
    (torch.bfloat16, "gemma"): (5e-2, 2e-2),
    (torch.bfloat16, "none"): (2e-1, 2e-2),
}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Imports (skip — don't fail — when the optional backend isn't installed)
# =============================================================================
def _cutedsl_rmsnorm():
    """cutedsl ``LigerRMSNormFunction``, or skip if CUTLASS isn't installed."""
    try:
        from liger_kernel.ops.cutedsl.ops.rms_norm import LigerRMSNormFunction
    except ImportError as exc:
        pytest.skip(f"cutedsl backend not importable (cutlass.cute missing?): {exc}")
    return LigerRMSNormFunction


def _triton_rmsnorm():
    """Triton reference ``LigerRMSNormFunction`` (the parity oracle)."""
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    return LigerRMSNormFunction


# =============================================================================
# Helpers
# =============================================================================
def _run(fn, x, w, do, eps, offset, casting_mode, in_place):
    """Forward + backward one RMSNorm variant; returns (y, dx, dw)."""
    x = x.clone().detach().requires_grad_(True)
    w_local = None
    if w is not None:
        w_local = w.clone().detach().requires_grad_(True)
    y = fn.apply(x, w_local, eps, offset, casting_mode, in_place, None)
    # Fresh grad clone: in-place backward may overwrite the upstream grad buffer.
    y.backward(do.clone())
    dw = w_local.grad if w_local is not None else None
    return y, x.grad, dw


# =============================================================================
# Correctness: cutedsl vs Triton
# =============================================================================
@cuda_required
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),  # regular
        (2, 64, 4096),  # regular, benchmark-sized hidden dim
        (2, 77, 1023),  # irregular hidden dim (not a multiple of 8 or 4)
        (1, 200, 2050),  # irregular hidden dim (even but not power of 2)
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not _supports_bf16(), reason="bf16 needs SM80+"),
        ),
    ],
)
@pytest.mark.parametrize(
    "offset, casting_mode",
    [
        (0.0, "llama"),
        (1.0, "gemma"),
        (0.0, "none"),
    ],
)
@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("in_place", [True, False])
def test_rms_norm_parity(bs, sl, hd, dtype, offset, casting_mode, elementwise_affine, in_place):
    set_seed(42)
    atol, rtol = _TOL[(dtype, casting_mode)]
    eps = 1e-6

    x = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)
    do = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)
    w = torch.randn(hd, device="cuda", dtype=dtype) if elementwise_affine else None

    y_cd, dx_cd, dw_cd = _run(_cutedsl_rmsnorm(), x, w, do, eps, offset, casting_mode, in_place)
    y_tr, dx_tr, dw_tr = _run(_triton_rmsnorm(), x, w, do, eps, offset, casting_mode, in_place)

    torch.testing.assert_close(y_cd, y_tr, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx_cd, dx_tr, atol=atol, rtol=rtol)
    if elementwise_affine:
        torch.testing.assert_close(dw_cd, dw_tr, atol=atol, rtol=rtol)


@cuda_required
@pytest.mark.skipif(not _supports_bf16(), reason="bf16 needs SM80+")
def test_rms_norm_mixed_weight_dtype_no_cache_collision():
    """Same-process bf16-weight then fp32-weight (both with bf16 activations) must each
    match Triton. Guards the compiled-kernel cache key against dropping W.dtype: a
    kernel baked for a bf16 weight must not be reused for an fp32 weight buffer."""
    set_seed(0)
    fn = _cutedsl_rmsnorm()
    ref = _triton_rmsnorm()
    eps, offset, mode = 1e-6, 0.0, "llama"
    atol, rtol = _TOL[(torch.bfloat16, "llama")]

    x = torch.randn(4, 128, 512, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(4, 128, 512, device="cuda", dtype=torch.bfloat16)

    # Populate the cache with a bf16-weight kernel first, then hit it with an fp32 weight.
    for w_dtype in (torch.bfloat16, torch.float32):
        w = torch.randn(512, device="cuda", dtype=w_dtype)
        y_cd, dx_cd, dw_cd = _run(fn, x, w, do, eps, offset, mode, True)
        y_tr, dx_tr, dw_tr = _run(ref, x, w, do, eps, offset, mode, True)
        torch.testing.assert_close(y_cd, y_tr, atol=atol, rtol=rtol)
        torch.testing.assert_close(dx_cd, dx_tr, atol=atol, rtol=rtol)
        torch.testing.assert_close(dw_cd, dw_tr, atol=atol, rtol=rtol)
