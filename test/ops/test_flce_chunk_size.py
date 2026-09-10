"""Focused tests for the explicit FLCE ``chunk_size`` controls.

In this layer only the backends that already ship an explicit token ``chunk_size``
override participate: the shared Triton FLCE and the native SM100 CuTe DSL FLCE. The
cuTile backend is intentionally left at its legacy 15-arg contract here (chunking lands
in a later layer), so its only assertion below is the *negative* one — a non-``None``
override is refused, never silently ignored.

Covers:

* the shared validation/clamp helper (``validate_flce_chunk_size``) — pure Python,
* numeric parity of an explicit chunk vs. the default heuristic (Triton + native CuTe),
* clamping (``chunk_size >= BT``) and non-power-of-two tails,
* invalid overrides (zero / negative / bool / float) raising,
* actual projection-GEMM row-geometry traces proving the explicit ``chunk_size`` really
  partitions the token loop (Triton ``aten::mm`` shapes; native CuTe epilogue-GEMM rows),
* the ``reduction="none"`` deferred-backward recomputation honoring the override, incl.
  a non-uniform upstream gradient vs. an fp32 reference (Triton) and the native CuTe
  repeated-backward recompute re-deriving the same chunking,
* the env-facing wrappers (top-level + transformers functional + nn.Module),
* the capability-gated dispatch helper refusing an override on unsupported backends,
* the legacy cuTile Function rejecting an explicit override on this layer.

Optional CUDA backends (native CuTe DSL SM100) are skipped when the current GPU / SDK
does not satisfy them.
"""

import pytest
import torch
import torch.nn.functional as F

from torch.utils._python_dispatch import TorchDispatchMode

from liger_kernel.ops.utils import validate_flce_chunk_size

_HAS_CUDA = torch.cuda.is_available()


def _cc():
    return torch.cuda.get_device_capability() if _HAS_CUDA else (0, 0)


def _native_cutedsl_available():
    # Native CuTe DSL FLCE is SM100-only.
    if not _HAS_CUDA or _cc() != (10, 0):
        return False
    try:
        import cutlass.cute  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # Only a *missing* optional dependency skips the test; a broken but
        # installed SDK must surface, not be silently hidden here.
        return False
    return True


@pytest.fixture(autouse=True)
def _require_cuda_for_ops_tests():
    """Override the directory-level CUDA auto-skip for this mixed CPU/GPU file.

    This module splits into pure-CPU groups (``validate_flce_chunk_size`` and the
    dispatch-helper arity logic) and real-GPU groups. Every GPU group carries its
    own explicit ``@pytest.mark.skipif`` guard, so the CPU groups must run even
    when no CUDA device is visible. Yield without skipping instead of inheriting
    the conftest ``autouse`` skip.
    """
    yield


# ---------------------------------------------------------------------------
# Pure-Python helper (no GPU required for the logic itself)
# ---------------------------------------------------------------------------


class TestValidateHelper:
    def test_none_passthrough(self):
        assert validate_flce_chunk_size(None, 128) is None

    def test_positive_int_unchanged_when_below_total(self):
        assert validate_flce_chunk_size(32, 128) == 32

    def test_clamped_to_total_rows(self):
        assert validate_flce_chunk_size(1024, 128) == 128
        assert validate_flce_chunk_size(128, 128) == 128

    def test_non_power_of_two_preserved_not_rounded(self):
        # explicit request is honored exactly (no power-of-two rounding)
        assert validate_flce_chunk_size(30, 100) == 30
        assert validate_flce_chunk_size(97, 100) == 97

    @pytest.mark.parametrize("bad", [0, -1, -128])
    def test_non_positive_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_flce_chunk_size(bad, 128)

    @pytest.mark.parametrize("bad", [True, False])
    def test_bool_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_flce_chunk_size(bad, 128)

    @pytest.mark.parametrize("bad", [3.5, 16.0, "16", 2.0])
    def test_non_int_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_flce_chunk_size(bad, 128)


# ---------------------------------------------------------------------------
# Shared references / helpers for the GPU parity tests
# ---------------------------------------------------------------------------


def _make_inputs(BT, H, V, dtype, seed=0, bias=False, device="cuda"):
    torch.manual_seed(seed)
    x = torch.randn(BT, H, device=device, dtype=dtype).requires_grad_(True)
    w = (torch.randn(V, H, device=device, dtype=dtype) / (H**0.5)).detach().requires_grad_(True)
    t = torch.randint(0, V, (BT,), device=device)
    b = torch.randn(V, device=device, dtype=dtype).requires_grad_(True) if bias else None
    return x, w, t, b


def _torch_reference(x, w, t, bias=None, ignore_index=-100, reduction="mean"):
    xr = x.detach().clone().float().requires_grad_(True)
    wr = w.detach().clone().float().requires_grad_(True)
    logits = xr @ wr.t()
    if bias is not None:
        logits = logits + bias.detach().float()
    loss = F.cross_entropy(logits, t, ignore_index=ignore_index, reduction=reduction)
    grad = torch.ones_like(loss) if reduction == "none" else None
    loss.backward(grad)
    return loss.detach(), xr.grad, wr.grad


def _triton_flce():
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

    return LigerFusedLinearCrossEntropyFunction


def _run_triton(x, w, t, bias=None, reduction="mean", chunk_size=None, accum_dtype=None):
    Fn = _triton_flce()
    loss, _, _, _ = Fn.apply(
        x,
        w,
        t,
        bias,
        None,
        -100,
        0.0,
        0.0,
        reduction,
        None,
        False,
        accum_dtype,
        False,
        False,
        False,
        None,
        None,
        chunk_size,
    )
    return loss


# ---------------------------------------------------------------------------
# Triton backend (universal — runs on any CUDA GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
class TestTritonChunkSize:
    def test_parity_default_vs_explicit(self):
        x, w, t, _ = _make_inputs(128, 64, 512, torch.float32, seed=1)
        loss_def = _run_triton(x, w, t)
        loss_def.backward()
        gx_def, gw_def = x.grad.clone(), w.grad.clone()
        x.grad = None
        w.grad = None
        loss_ex = _run_triton(x, w, t, chunk_size=16)
        loss_ex.backward()
        assert torch.allclose(loss_def, loss_ex, atol=1e-4, rtol=1e-4)
        assert torch.allclose(gx_def, x.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(gw_def, w.grad, atol=1e-3, rtol=1e-3)

    def test_parity_vs_torch_reference(self):
        x, w, t, _ = _make_inputs(96, 48, 256, torch.float32, seed=2)
        ref_loss, ref_gx, ref_gw = _torch_reference(x, w, t)
        loss = _run_triton(x, w, t, chunk_size=30)  # non-power-of-two tail: 96 = 30*3 + 6
        loss.backward()
        assert torch.allclose(loss, ref_loss, atol=1e-3, rtol=1e-3)
        assert torch.allclose(x.grad, ref_gx, atol=1e-3, rtol=1e-3)
        assert torch.allclose(w.grad, ref_gw, atol=1e-3, rtol=1e-3)

    def test_clamp_above_bt_matches_single_chunk(self):
        x, w, t, _ = _make_inputs(64, 32, 128, torch.float32, seed=3)
        loss_big = _run_triton(x, w, t, chunk_size=100000)  # clamped to BT=64
        loss_big.backward()
        gx_big = x.grad.clone()
        x.grad = None
        w.grad = None
        loss_bt = _run_triton(x, w, t, chunk_size=64)
        loss_bt.backward()
        assert torch.allclose(loss_big, loss_bt, atol=1e-5, rtol=1e-5)
        assert torch.allclose(gx_big, x.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bad", [0, -1, True, 3.5])
    def test_invalid_chunk_rejected(self, bad):
        x, w, t, _ = _make_inputs(32, 16, 64, torch.float32, seed=4)
        with pytest.raises(ValueError):
            _run_triton(x, w, t, chunk_size=bad)

    def test_reduction_none_recompute_honors_override(self):
        # reduction="none" defers grads to backward, recomputing chunked logits.
        # The explicit chunk_size must be persisted into that recomputation.
        x, w, t, _ = _make_inputs(100, 40, 256, torch.float32, seed=5, bias=True)
        ref_loss, ref_gx, ref_gw = _torch_reference(x, w, t, reduction="none")
        Fn = _triton_flce()
        loss, _, _, _ = Fn.apply(
            x, w, t, None, None, -100, 0.0, 0.0, "none", None, False, None, False, False, False, None, None, 24
        )
        loss.sum().backward()  # upstream grad of ones -> matches reference reduction="none"
        assert loss.shape == (100,)
        assert torch.allclose(x.grad, ref_gx, atol=1e-3, rtol=1e-3)
        assert torch.allclose(w.grad, ref_gw, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Native CuTe DSL backend (SM100 only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _native_cutedsl_available(), reason="native CuTe DSL FLCE requires SM100 + cutlass")
class TestNativeCuTeChunkSize:
    def _fn(self):
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

        return LigerFusedLinearCrossEntropyFunction

    def _run(self, x, w, t, chunk_size=None, ce_impl=None, ce_mode=None):
        Fn = self._fn()
        loss, _, _, _ = Fn.apply(
            x,
            w,
            t,
            None,
            None,
            -100,
            0.0,
            0.0,
            "mean",
            None,
            False,
            torch.float32,
            False,
            False,
            False,
            ce_impl,
            ce_mode,
            chunk_size,
        )
        return loss

    def test_supports_flag(self):
        assert self._fn().supports_chunk_size is True

    def test_parity_default_vs_explicit(self):
        x, w, t, _ = _make_inputs(256, 256, 4096, torch.bfloat16, seed=6)
        loss_def = self._run(x, w, t)
        loss_def.backward()
        gx_def, gw_def = x.grad.clone(), w.grad.clone()
        x.grad = None
        w.grad = None
        loss_ex = self._run(x, w, t, chunk_size=64)
        loss_ex.backward()
        assert torch.allclose(loss_def, loss_ex, atol=2e-2, rtol=2e-2)
        assert torch.allclose(gx_def, x.grad, atol=1e-1, rtol=1e-1)
        assert torch.allclose(gw_def, w.grad, atol=1e-1, rtol=1e-1)

    def test_non_power_of_two_tail(self):
        x, w, t, _ = _make_inputs(200, 256, 4096, torch.bfloat16, seed=7)
        loss = self._run(x, w, t, chunk_size=48)  # 200 = 48*4 + 8
        loss.backward()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("bad", [0, -3, True, 2.5])
    def test_invalid_chunk_rejected(self, bad):
        x, w, t, _ = _make_inputs(64, 256, 4096, torch.bfloat16, seed=8)
        with pytest.raises(ValueError):
            self._run(x, w, t, chunk_size=bad)

    def test_ce_impl_mode_placeholders(self):
        x, w, t, _ = _make_inputs(64, 256, 4096, torch.bfloat16, seed=9)
        # self-identity / default accepted
        self._run(x, w, t, ce_impl="nvidia-cutedsl", ce_mode="default", chunk_size=32)
        with pytest.raises(ValueError):
            self._run(x, w, t, ce_impl="nvidia-triton")
        with pytest.raises(ValueError):
            self._run(x, w, t, ce_mode="static_persistent")


# ---------------------------------------------------------------------------
# cuTile backend (SM90 + SM100) — chunking is owned by a later layer. Here we only
# assert the *negative* contract: this layer leaves cuTile at its legacy 15-arg
# Function, so an explicit override must be refused, never silently ignored.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_cc() not in ((9, 0), (10, 0)), reason="cuTile FLCE requires SM90/SM100")
class TestCuTileLegacyRejectsChunk:
    def _fn(self):
        pytest.importorskip("cuda.tile")
        from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

        return LigerFusedLinearCrossEntropyFunction

    def test_does_not_advertise_chunk_support(self):
        # cuTile stays on the legacy contract in this layer.
        assert getattr(self._fn(), "supports_chunk_size", False) is False

    def test_explicit_override_refused_by_dispatch_helper(self):
        from liger_kernel.transformers.functional import build_flce_apply_args

        base = tuple(range(15))
        with pytest.raises(ValueError):
            build_flce_apply_args(self._fn(), base, chunk_size=64)


# ---------------------------------------------------------------------------
# Projection-GEMM row-geometry traces — prove the explicit ``chunk_size`` really
# partitions the token loop (C is used, not ignored). Adapted (minimal) from the
# combined branch's test/ops/test_flce_benchmark.py; no benchmark harness here.
# ---------------------------------------------------------------------------


def _expected_chunk_rows(tokens, chunk):
    """Row count of each projection chunk after clamping ``chunk`` to ``tokens``."""
    chunk = min(chunk, tokens)
    rows, start = [], 0
    while start < tokens:
        rows.append(min(chunk, tokens - start))
        start += chunk
    return rows


def _flce_inputs_cuda(tokens, hidden, vocab, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype, generator=g).requires_grad_(True)
    w = (torch.randn(vocab, hidden, device="cuda", dtype=dtype, generator=g) / (hidden**0.5)).detach()
    w.requires_grad_(True)
    target = torch.randint(0, vocab, (tokens,), device="cuda", generator=g)
    return x, w, target


def _apply_direct(fn, x, w, target, chunk_size, reduction="mean"):
    loss, _, _, _ = fn.apply(
        x,
        w,
        target,
        None,
        None,
        -100,
        0.0,
        0.0,
        reduction,
        None,
        False,
        torch.float32,
        False,
        False,
        False,
        None,
        None,
        chunk_size,
    )
    return loss


class _ProjectionMMRecorder(TorchDispatchMode):
    """Record the LHS/RHS shapes of every ``aten::mm`` the forward issues."""

    def __init__(self):
        self.mms = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if str(func).startswith("aten.mm"):
            a, b = args[0], args[1]
            self.mms.append((tuple(a.shape), tuple(b.shape)))
        return out


_TOKENS, _HIDDEN, _VOCAB, _CHUNK = 101, 256, 4096, 33


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
class TestTritonProjectionGeometry:
    def _fn(self):
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

        return LigerFusedLinearCrossEntropyFunction

    def test_projection_row_geometry_matches_chunking(self):
        # Non-power-of-two override: 101 tokens / chunk 33 -> [33, 33, 33, 2].
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=20)
        with _ProjectionMMRecorder() as rec:
            _apply_direct(self._fn(), x, w, target, _CHUNK)
        # Projection GEMM is x_chunk @ w.T -> lhs [rows, H], rhs [H, V].
        proj = [lhs for lhs, rhs in rec.mms if rhs == (_HIDDEN, _VOCAB)]
        rows = [lhs[0] for lhs in proj]
        assert all(lhs[1] == _HIDDEN for lhs in proj)
        assert rows == _expected_chunk_rows(_TOKENS, _CHUNK)

    def test_clamp_above_tokens_is_single_chunk(self):
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=21)
        with _ProjectionMMRecorder() as rec:
            _apply_direct(self._fn(), x, w, target, _TOKENS * 5)  # clamps to N
        rows = [lhs[0] for lhs, rhs in rec.mms if rhs == (_HIDDEN, _VOCAB)]
        assert rows == [_TOKENS]

    def test_reduction_none_backward_matches_fp32_ref_nonuniform_upstream(self):
        # reduction="none" defers grads to backward, recomputing chunked logits with
        # the persisted override; a non-uniform upstream grad exercises the full path.
        tokens, hidden, vocab = 100, 40, 256
        torch.manual_seed(22)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
        w = (torch.randn(vocab, hidden, device="cuda", dtype=torch.float32) / (hidden**0.5)).detach()
        w.requires_grad_(True)
        target = torch.randint(0, vocab, (tokens,), device="cuda")
        upstream = torch.rand(tokens, device="cuda", dtype=torch.float32) + 0.1  # nonuniform per-token

        xr = x.detach().clone().requires_grad_(True)
        wr = w.detach().clone().requires_grad_(True)
        ref = F.cross_entropy(xr @ wr.t(), target, reduction="none")
        ref.backward(upstream)

        loss = _apply_direct(self._fn(), x, w, target, 24, reduction="none")
        assert loss.shape == (tokens,)
        loss.backward(upstream)
        assert torch.allclose(x.grad, xr.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(w.grad, wr.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not _native_cutedsl_available(), reason="native CuTe DSL FLCE requires SM100 + cutlass")
class TestNativeCuTeProjectionGeometry:
    def _fn(self):
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

        return LigerFusedLinearCrossEntropyFunction

    def test_projection_epilogue_gemm_row_geometry(self, monkeypatch):
        fn = self._fn()
        import liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy as mod

        real = mod.run_epilogue_gemm
        rows = []

        def _spy(Xc, W, dlogits, epilogue, *a, **k):
            rows.append(dlogits.shape[0])
            return real(Xc, W, dlogits, epilogue, *a, **k)

        monkeypatch.setattr(mod, "run_epilogue_gemm", _spy)
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=24)
        _apply_direct(fn, x, w, target, _CHUNK)
        assert rows == _expected_chunk_rows(_TOKENS, _CHUNK)

    def test_repeated_backward_recompute_persists_chunk_override(self, monkeypatch):
        fn = self._fn()
        import liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy as mod

        real = mod.run_epilogue_gemm
        rows = []

        def _spy(Xc, W, dlogits, epilogue, *a, **k):
            rows.append(dlogits.shape[0])
            return real(Xc, W, dlogits, epilogue, *a, **k)

        monkeypatch.setattr(mod, "run_epilogue_gemm", _spy)
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=25)

        expected = _expected_chunk_rows(_TOKENS, _CHUNK)
        loss = _apply_direct(fn, x, w, target, _CHUNK)
        rows.clear()
        loss.backward(retain_graph=True)  # 1st: consumes native fwd grads, no recompute
        assert rows == []
        loss.backward(retain_graph=True)  # 2nd: recompute -> same chunk override
        assert rows == expected
        rows.clear()
        loss.backward(retain_graph=True)  # 3rd: cached -> no recompute
        assert rows == []


# ---------------------------------------------------------------------------
# Capability-gated dispatch helper (pure Python)
# ---------------------------------------------------------------------------


class TestDispatchHelper:
    def _helper(self):
        from liger_kernel.transformers.functional import build_flce_apply_args

        return build_flce_apply_args

    def test_appends_triple_for_chunk_capable(self):
        build = self._helper()

        class Cap:
            supports_chunk_size = True
            supports_inner_impl_dispatch = True

        base = tuple(range(15))
        out = build(Cap, base, chunk_size=7)
        assert out == base + (None, None, 7)

    def test_appends_pair_for_dispatch_only(self):
        build = self._helper()

        class Cap:
            supports_inner_impl_dispatch = True

        base = tuple(range(15))
        assert build(Cap, base) == base + (None, None)

    def test_legacy_class_gets_no_extra_args(self):
        build = self._helper()

        class Legacy:
            pass

        base = tuple(range(15))
        assert build(Legacy, base) == base

    def test_chunk_on_unsupported_raises(self):
        build = self._helper()

        class Legacy:
            supports_inner_impl_dispatch = True  # dispatch but no chunk support

        base = tuple(range(15))
        with pytest.raises(ValueError):
            build(Legacy, base, chunk_size=8)

    def test_ce_impl_on_legacy_raises(self):
        build = self._helper()

        class Legacy:
            pass

        base = tuple(range(15))
        with pytest.raises(ValueError):
            build(Legacy, base, ce_impl="nvidia-triton")


# ---------------------------------------------------------------------------
# Env-facing wrappers (top-level functional, transformers functional + nn.Module)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
class TestEnvWrappers:
    def test_top_level_functional_chunk(self):
        import liger_kernel.functional as LF

        x, w, t, _ = _make_inputs(128, 64, 512, torch.float32, seed=11)
        ref_loss, _, _ = _torch_reference(x, w, t)
        out = LF.fused_linear_cross_entropy(x, w, t, chunk_size=20)
        assert torch.allclose(out[0], ref_loss, atol=1e-3, rtol=1e-3)

    def test_transformers_functional_chunk(self):
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

        x, w, t, _ = _make_inputs(128, 64, 512, torch.float32, seed=12)
        ref_loss, _, _ = _torch_reference(x, w, t)
        loss = liger_fused_linear_cross_entropy(x, w, t, chunk_size=20)
        assert torch.allclose(loss, ref_loss, atol=1e-3, rtol=1e-3)

    def test_nn_module_chunk(self):
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        x, w, t, _ = _make_inputs(128, 64, 512, torch.float32, seed=13)
        ref_loss, ref_gx, ref_gw = _torch_reference(x, w, t)
        loss = LigerFusedLinearCrossEntropyLoss(chunk_size=20)(w, x, t)
        loss.backward()
        assert torch.allclose(loss, ref_loss, atol=1e-3, rtol=1e-3)
        assert torch.allclose(x.grad, ref_gx, atol=1e-3, rtol=1e-3)
        assert torch.allclose(w.grad, ref_gw, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_top_level_invalid_chunk_rejected(self, bad):
        import liger_kernel.functional as LF

        x, w, t, _ = _make_inputs(64, 32, 128, torch.float32, seed=14)
        with pytest.raises(ValueError):
            LF.fused_linear_cross_entropy(x, w, t, chunk_size=bad)
