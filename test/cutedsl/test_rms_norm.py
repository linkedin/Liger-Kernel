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


# =============================================================================
# Blackwell fast path (TVM-FFI direct call + packed-f32x2 fused backward)
# =============================================================================
# These exercise machinery the parity suite above cannot reach: it only covers
# hidden widths <= 4096, so it never compiles the packed-f32x2 backward, and it
# always runs on the default stream.
def _rms_norm_mod():
    """The cutedsl rms_norm module, or skip if CUTLASS isn't installed."""
    try:
        from liger_kernel.ops.cutedsl.ops import rms_norm as mod
    except ImportError as exc:
        pytest.skip(f"cutedsl backend not importable (cutlass.cute missing?): {exc}")
    return mod


def _fwd_bwd(mod, X, W, dY, casting_mode="llama", offset=0.0):
    """One forward + backward through the module-level functional API.

    ``dW`` is ``None`` in the non-affine case; callers zip over the names.
    """
    Y, X2, RSTD, BS, NW, cm = mod.rms_norm_forward(X, W, 1e-6, offset, casting_mode, None)
    dX, dW = mod.rms_norm_backward(dY, X2, W, RSTD, offset, cm, BS, NW, False, None)
    return Y, dX, dW


@cuda_required
@pytest.mark.skipif(not _supports_bf16(), reason="bf16 needs SM80+")
@pytest.mark.parametrize("n_cols", [2048, 8192])
@pytest.mark.parametrize("casting_mode", ["llama", "none"])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_rms_norm_ffi_matches_marshalling_path(monkeypatch, n_cols, casting_mode, elementwise_affine):
    """The TVM-FFI direct-call path must be bit-identical to the marshalling path.

    Both compile the same kernel; only the calling convention differs, so any
    difference means the abstract-tensor layouts baked at compile time do not
    describe the tensors actually handed over (wrong stride divisibility, wrong
    dtype for the non-affine dummy W, ...). The non-affine case is the one that
    pins the dummy-W layout: it hands RSTD over in W's slot, so the compiled
    signature must expect an fp32 vector there.
    """
    mod = _rms_norm_mod()
    if not mod._TVM_FFI_PRESENT:
        pytest.skip("apache-tvm-ffi not installed; only the marshalling path exists")

    set_seed(0)
    X = torch.randn(512, n_cols, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(n_cols, device="cuda", dtype=torch.bfloat16) if elementwise_affine else None
    dY = torch.randn(512, n_cols, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setattr(mod, "_FORCE_NO_FFI", True)
    ref = [None if t is None else t.clone() for t in _fwd_bwd(mod, X, W, dY, casting_mode)]
    monkeypatch.setattr(mod, "_FORCE_NO_FFI", False)
    got = _fwd_bwd(mod, X, W, dY, casting_mode)

    # Guard the test itself: both launch paths must really have been exercised,
    # or this degenerates into comparing one path against itself.
    assert any(k[0] == "fwd_vec" for k in mod._compile_cache), "marshalling path never compiled"
    assert any(k[0] == "fwd_vec_ffi" for k in mod._ffi_compile_cache), "FFI path never compiled"

    for name, a, b in zip(("Y", "dX", "dW"), got, ref):
        if a is None or b is None:
            assert a is b is None, f"{name} present on only one launch path"
            continue
        assert torch.equal(a, b), f"{name} differs between the FFI and marshalling launch paths"


@cuda_required
@pytest.mark.skipif(not _supports_bf16(), reason="bf16 needs SM80+")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_packed_backward_matches_scalar(monkeypatch, dtype):
    """The packed-f32x2 fused backward must agree with the scalar one.

    Only the instruction form changes -- the packed variant pairs lanes and
    keeps two fp32 accumulators, so results differ at most by fp32 reassociation
    (and one output rounding step), never structurally. Uses a width above the
    ``_use_packed_math`` threshold so the packed kernel is the one compiled.
    """
    mod = _rms_norm_mod()
    if not mod._is_blackwell(torch.device("cuda")):
        pytest.skip("packed-f32x2 math needs a data-center Blackwell (sm_100/sm_103)")

    set_seed(0)
    n_cols = 8192  # > 4096, the _use_packed_math threshold
    X = torch.randn(1024, n_cols, device="cuda", dtype=dtype)
    W = torch.randn(n_cols, device="cuda", dtype=dtype)
    dY = torch.randn(1024, n_cols, device="cuda", dtype=dtype)

    assert mod._use_packed_math(X.device, n_cols, 16 // X.element_size())
    monkeypatch.setattr(mod, "_FORCE_NO_PACKED", True)
    ref = [t.clone() for t in _fwd_bwd(mod, X, W, dY)]
    monkeypatch.setattr(mod, "_FORCE_NO_PACKED", False)
    got = _fwd_bwd(mod, X, W, dY)

    # Guard the test itself: the packed flag is the last element of the fused
    # backward compile keys, and both specializations must have been built.
    bwd_keys = [k for k in (*mod._compile_cache, *mod._ffi_compile_cache) if k[0].startswith("bwd_fused")]
    assert {k[-1] for k in bwd_keys} == {True, False}, f"only one backward variant compiled: {bwd_keys}"

    tol = 1e-5 if dtype == torch.float32 else 8e-3
    for name, a, b in zip(("Y", "dX", "dW"), got, ref):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol, msg=lambda m, n=name: f"{n}: {m}")


@cuda_required
@pytest.mark.skipif(not _supports_bf16(), reason="bf16 needs SM80+")
def test_rms_norm_runs_on_the_current_stream():
    """Kernels must follow ``torch.cuda.current_stream()``, not the default stream.

    The TVM-FFI path drops the stream argument and reads the caller's
    environment stream instead; if that ever stopped tracking torch, every
    launch under ``torch.cuda.stream(...)`` would silently race with the
    surrounding side-stream work. A CUDA graph is the sharpest available probe:
    capture only records work issued on the capturing stream, so a kernel that
    escaped to the default stream would leave the poisoned output untouched.
    """
    from liger_kernel.ops.cutedsl.ops.rms_norm import LigerRMSNormFunction as fn

    set_seed(0)
    X = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    expected = fn.apply(X, W, 1e-6, 0.0, "llama", False, None).clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn.apply(X, W, 1e-6, 0.0, "llama", False, None)
    captured.zero_()  # poison: only a captured kernel can restore this
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured, expected), "kernel did not run on the capturing stream"
