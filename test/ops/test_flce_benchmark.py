"""Tests for the FLCE backend micro-benchmark and the whole-pipeline contract it
exercises (``benchmark/scripts/benchmark_flce_backends.py``).

Three groups:

* **A. CPU-mocked benchmark plumbing** (no GPU): architecture-driven backend
  selection, capability gating, positive-argument validation, fail-fast error
  propagation, the memory-baseline lifecycle (leaf grads cleared before the
  baseline snapshot), and the probe helper releasing its autograd graph.
* **B. GPU projection-geometry traces**: the projection GEMM is issued once per
  token chunk with the expected clamped row geometry, across whichever direct
  backends the current GPU supports.
* **C. Env-facing wrapper arities**: the transformers functional / nn.Module
  wrappers append exactly the trailing args the (monkeypatched) Function class
  advertises, and reject an override the class cannot honor.
"""

import gc
import importlib.util
import weakref

from pathlib import Path

import pytest
import torch

from torch.utils._python_dispatch import TorchDispatchMode

_HAS_CUDA = torch.cuda.is_available()


def _cc():
    return torch.cuda.get_device_capability() if _HAS_CUDA else (0, 0)


def _load_bench():
    path = Path(__file__).resolve().parents[2] / "benchmark" / "scripts" / "benchmark_flce_backends.py"
    spec = importlib.util.spec_from_file_location("benchmark_flce_backends", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


@pytest.fixture(autouse=True)
def _require_cuda_for_ops_tests():
    """Override the directory-level CUDA auto-skip for this mixed CPU/GPU file.

    This module deliberately splits into pure-CPU groups (benchmark plumbing,
    backend selection, capability gating, arg validation, lifecycle/loss/run
    logic, wrapper arities) and real-GPU groups. Every GPU group carries its own
    explicit ``@pytest.mark.skipif`` guard, so the CPU groups must run even when
    no CUDA device is visible. Yield without skipping instead of inheriting the
    conftest ``autouse`` skip.
    """
    yield


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Stand-in for ``torch.cuda.Event`` with deterministic elapsed time."""

    _clock = 0.0

    def record(self):
        _FakeEvent._clock += 1.0
        self._t = _FakeEvent._clock

    def elapsed_time(self, other):
        return float(other._t - self._t)


class _CpuFLCE(torch.autograd.Function):
    """A real CPU autograd Function honoring the full 18-arg FLCE signature.

    Populates both leaf grads on backward so grad-lifecycle assertions are
    meaningful; returns the standard ``(loss, None, None, None)`` 4-tuple.
    """

    supports_chunk_size = True
    supports_inner_impl_dispatch = True

    @staticmethod
    def forward(ctx, x, w, target, *rest):
        ctx.save_for_backward(x, w)
        logits = x.float() @ w.float().t()
        loss = torch.nn.functional.cross_entropy(logits, target)
        return loss.to(x.dtype), None, None, None

    @staticmethod
    def backward(ctx, grad_loss, *rest):
        x, w = ctx.saved_tensors
        gx = torch.ones_like(x) * grad_loss
        gw = torch.ones_like(w) * grad_loss
        return (gx, gw) + (None,) * 16


def _cpu_inputs(tokens=8, hidden=4, vocab=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g).requires_grad_(True)
    w = torch.randn(vocab, hidden, generator=g).requires_grad_(True)
    target = torch.randint(0, vocab, (tokens,), generator=g)
    return x, w, target


# ===========================================================================
# A. CPU-mocked benchmark plumbing
# ===========================================================================


class TestBackendSelection:
    def test_default_sm100_is_all_three(self):
        assert bench._select_default_backends((10, 0)) == ["triton", "cutedsl", "cutile"]

    def test_default_sm90_is_triton_and_cutile(self):
        assert bench._select_default_backends((9, 0)) == ["triton", "cutile"]

    def test_default_other_arch_is_triton_only(self):
        assert bench._select_default_backends((8, 0)) == ["triton"]

    def test_resolve_none_uses_arch_default(self):
        assert bench._resolve_backends(None, (10, 0)) == ["triton", "cutedsl", "cutile"]
        assert bench._resolve_backends([], (9, 0)) == ["triton", "cutile"]

    def test_resolve_all3_expands(self):
        assert bench._resolve_backends(["all3"], (8, 0)) == ["triton", "cutedsl", "cutile"]

    def test_all3_is_exclusive(self):
        with pytest.raises(ValueError):
            bench._resolve_backends(["all3", "triton"], (10, 0))

    def test_explicit_backends_passthrough(self):
        assert bench._resolve_backends(["triton", "cutedsl"], (10, 0)) == ["triton", "cutedsl"]


class TestCapabilityGating:
    def test_triton_ok_on_any_arch(self):
        for cc in [(7, 0), (8, 0), (9, 0), (10, 0)]:
            bench._check_backend_supported("triton", cc)  # no raise

    def test_cutedsl_requires_exact_sm100(self):
        with pytest.raises(RuntimeError):
            bench._check_backend_supported("cutedsl", (9, 0))
        with pytest.raises(RuntimeError):
            bench._check_backend_supported("cutedsl", (10, 1))

    def test_cutile_rejects_pre_hopper(self):
        with pytest.raises(RuntimeError):
            bench._check_backend_supported("cutile", (8, 0))

    def test_cutile_accepts_sm90_and_sm100(self):
        assert bench._backend_cc_ok("cutile", (9, 0))
        assert bench._backend_cc_ok("cutile", (10, 0))

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            bench._check_backend_supported("nope", (10, 0))

    def test_missing_sdk_propagates(self, monkeypatch):
        # cc is valid, but the optional SDK import fails -> surface it loudly.
        def _boom(name):
            raise ImportError(f"no module {name}")

        monkeypatch.setattr(bench.importlib, "import_module", _boom)
        with pytest.raises(ImportError):
            bench._check_backend_supported("cutedsl", (10, 0))


class TestArgValidation:
    def _parse(self, argv):
        return bench._build_parser().parse_args(argv)

    def test_defaults_are_explicit_matched_chunks(self):
        args = self._parse([])
        assert args.chunk_sizes == [256, 1024]

    @pytest.mark.parametrize("flag", ["--tokens", "--hidden-size", "--vocab-size", "--iters"])
    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_non_positive_scalars_rejected(self, flag, bad):
        with pytest.raises(SystemExit):
            self._parse([flag, bad])

    @pytest.mark.parametrize("bad", ["0", "-4"])
    def test_non_positive_chunk_rejected(self, bad):
        with pytest.raises(SystemExit):
            self._parse(["--chunk-sizes", "128", bad])

    @pytest.mark.parametrize("word", ["none", "default", "auto"])
    def test_symbolic_chunk_syntax_removed(self, word):
        # The None/default/auto sweep syntax is gone; only explicit ints parse.
        with pytest.raises(SystemExit):
            self._parse(["--chunk-sizes", word])

    def test_warmup_zero_allowed_negative_rejected(self):
        assert self._parse(["--warmup", "0"]).warmup == 0
        with pytest.raises(SystemExit):
            self._parse(["--warmup", "-1"])

    def test_accum_dtype_defaults_to_fp32(self):
        # Backward compatible: omitting the flag keeps the FP32-matched accumulator.
        assert self._parse([]).accum_dtype == "fp32"

    @pytest.mark.parametrize("mode", ["fp32", "default"])
    def test_accum_dtype_choices_accepted(self, mode):
        assert self._parse(["--accum-dtype", mode]).accum_dtype == mode

    @pytest.mark.parametrize("bad", ["bf16", "none", "float32", "auto"])
    def test_accum_dtype_rejects_unknown(self, bad):
        with pytest.raises(SystemExit):
            self._parse(["--accum-dtype", bad])


class TestTimeBackendLifecycle:
    def _patch_cuda(self, monkeypatch, on_baseline=None):
        _FakeEvent._clock = 0.0
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: _FakeEvent())

        def _alloc(*a, **k):
            if on_baseline is not None:
                on_baseline()
            return 1000

        monkeypatch.setattr(torch.cuda, "memory_allocated", _alloc)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 5000)

    def test_baseline_snapshot_has_no_leaf_grads(self, monkeypatch):
        x, w, target = _cpu_inputs()
        seen = {}

        def _check():
            # memory_allocated is only invoked for the baseline snapshot, at which
            # point both leaf grads must already be cleared (input-only baseline).
            seen["x_grad"] = x.grad
            seen["w_grad"] = w.grad

        self._patch_cuda(monkeypatch, on_baseline=_check)
        median_ms, peak_mb = bench._time_backend(_CpuFLCE, x, w, target, None, warmup=2, iters=3)

        assert seen["x_grad"] is None and seen["w_grad"] is None
        assert peak_mb == pytest.approx((5000 - 1000) / (1024**2))
        assert median_ms == pytest.approx(1.0)  # every fake event pair is 1 tick apart

    def test_uses_statistics_median(self, monkeypatch):
        # 4 timed iters -> median averages the two middle event deltas (all 1.0 here).
        x, w, target = _cpu_inputs()
        self._patch_cuda(monkeypatch)
        median_ms, _ = bench._time_backend(_CpuFLCE, x, w, target, None, warmup=0, iters=4)
        assert median_ms == pytest.approx(1.0)


class TestLossProbe:
    def test_returns_float_and_releases_graph(self):
        class _ProbeFn:
            holder = {}

            @staticmethod
            def apply(x, w, target, *rest):
                loss = (x.float() @ w.float().t()).sum()
                _ProbeFn.holder["ref"] = weakref.ref(loss)
                return loss, None, None, None

        x, w, target = _cpu_inputs()
        value = bench._loss_value(_ProbeFn, x, w, target, None)

        assert isinstance(value, float)
        assert x.grad is None and w.grad is None
        gc.collect()
        # The probe's loss tensor (and its graph) must be dead before timing starts.
        assert _ProbeFn.holder["ref"]() is None


class TestRunFailFast:
    def _fake_args(self, backends):
        return type(
            "Args",
            (),
            {
                "backends": backends,
                "tokens": 8,
                "hidden_size": 4,
                "vocab_size": 6,
                "chunk_sizes": [4],
                "warmup": 0,
                "iters": 1,
                "seed": 0,
                "accum_dtype": "fp32",
            },
        )()

    def _patch_run(self, monkeypatch, fn):
        monkeypatch.setattr(bench, "_capability", lambda: (10, 0))
        monkeypatch.setattr(bench, "_check_backend_supported", lambda *a, **k: None)
        monkeypatch.setattr(bench, "_load_backend", lambda name: fn)
        monkeypatch.setattr(bench, "_build_inputs", lambda *a, **k: _cpu_inputs())
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "MockGPU")
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
        _FakeEvent._clock = 0.0
        monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: _FakeEvent())

    def test_backend_error_propagates(self, monkeypatch):
        class _BoomFn:
            @staticmethod
            def apply(*a, **k):
                raise RuntimeError("kernel exploded")

        self._patch_run(monkeypatch, _BoomFn)
        # No broad try/except in _run: the failure surfaces (nonzero-exit behavior),
        # it is not swallowed into a printed "FAILED" success-shaped fallback.
        with pytest.raises(RuntimeError, match="kernel exploded"):
            bench._run(self._fake_args(["triton"]))

    def test_successful_run_completes(self, monkeypatch, capsys):
        self._patch_run(monkeypatch, _CpuFLCE)
        bench._run(self._fake_args(["triton"]))
        out = capsys.readouterr().out
        assert "triton" in out
        assert "eff_chunk" in out


class _SlotRecFn(torch.autograd.Function):
    """Records the value passed at FLCE Function slot 11 (``accum_dtype``)."""

    seen = []

    @staticmethod
    def forward(ctx, x, w, target, *rest):
        # rest = (bias, ce_weight, ignore_index, lse_square_scale, label_smoothing,
        #         reduction, softcap, return_z_loss, accum_dtype, ...); slot 11 overall
        # is accum_dtype, i.e. rest[8].
        _SlotRecFn.seen.append(rest[8])
        ctx.save_for_backward(x, w)
        loss = (x.float() @ w.float().t()).sum()
        return loss, None, None, None

    @staticmethod
    def backward(ctx, grad_loss, *rest):
        x, w = ctx.saved_tensors
        return (torch.ones_like(x) * grad_loss, torch.ones_like(w) * grad_loss) + (None,) * 16


class TestAccumDtypeThreading:
    def test_effective_accum_labels(self):
        # fp32 forces FP32 everywhere; default keeps triton/cutedsl BF16 but cutile FP32.
        for name in ("triton", "cutedsl", "cutile"):
            assert bench._effective_accum(name, "fp32") == "fp32"
        assert bench._effective_accum("triton", "default") == "bf16"
        assert bench._effective_accum("cutedsl", "default") == "bf16"
        assert bench._effective_accum("cutile", "default") == "fp32"

    def test_apply_forwards_dtype_to_slot_11(self):
        x, w, target = _cpu_inputs()
        _SlotRecFn.seen = []
        bench._apply(_SlotRecFn, x, w, target, None, accum_dtype=torch.float32)
        bench._apply(_SlotRecFn, x, w, target, None, accum_dtype=None)
        assert _SlotRecFn.seen == [torch.float32, None]

    def test_apply_default_is_fp32(self):
        # Drop-in default preserves the earlier hard-coded FP32 accumulator.
        x, w, target = _cpu_inputs()
        _SlotRecFn.seen = []
        bench._apply(_SlotRecFn, x, w, target, None)
        assert _SlotRecFn.seen == [torch.float32]

    def _patch_cuda(self, monkeypatch):
        _FakeEvent._clock = 0.0
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 1000)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 5000)
        monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: _FakeEvent())

    @pytest.mark.parametrize("accum", [torch.float32, None])
    def test_both_modes_roundtrip_through_iterations(self, monkeypatch, accum):
        # Every timed/warmup/probe iteration threads the same accum dtype to slot 11.
        x, w, target = _cpu_inputs()
        self._patch_cuda(monkeypatch)
        _SlotRecFn.seen = []
        bench._time_backend(_SlotRecFn, x, w, target, None, warmup=2, iters=3, accum_dtype=accum)
        # 1 compile + 2 warmup + 1 memory-probe + 3 timed = 7 applies, all identical.
        assert len(_SlotRecFn.seen) == 7
        assert all(v is accum for v in _SlotRecFn.seen)

    def _run_args(self, backends, accum_dtype):
        return type(
            "Args",
            (),
            {
                "backends": backends,
                "tokens": 8,
                "hidden_size": 4,
                "vocab_size": 6,
                "chunk_sizes": [4],
                "warmup": 0,
                "iters": 1,
                "seed": 0,
                "accum_dtype": accum_dtype,
            },
        )()

    def _patch_run(self, monkeypatch):
        monkeypatch.setattr(bench, "_capability", lambda: (10, 0))
        monkeypatch.setattr(bench, "_check_backend_supported", lambda *a, **k: None)
        monkeypatch.setattr(bench, "_load_backend", lambda name: _CpuFLCE)
        monkeypatch.setattr(bench, "_build_inputs", lambda *a, **k: _cpu_inputs())
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "MockGPU")
        self._patch_cuda(monkeypatch)

    def test_default_mode_prints_requested_and_per_backend_effective(self, monkeypatch, capsys):
        self._patch_run(monkeypatch)
        bench._run(self._run_args(["triton", "cutedsl", "cutile"], "default"))
        out = capsys.readouterr().out
        # Header advertises the requested mode and refuses to call it precision-matched.
        assert "accum_dtype_requested=default" in out
        assert "NOT precision matched" in out
        assert "eff_accum" in out
        rows = {
            line.split()[0]: line
            for line in out.splitlines()
            if line.split() and line.split()[0] in bench._MODULE_PATHS
        }
        # cutile keeps FP32 even under default; triton/cutedsl drop to BF16.
        assert "fp32" in rows["cutile"].split()
        assert "bf16" in rows["triton"].split()
        assert "bf16" in rows["cutedsl"].split()

    def test_fp32_mode_labels_precision_matched(self, monkeypatch, capsys):
        self._patch_run(monkeypatch)
        bench._run(self._run_args(["triton", "cutile"], "fp32"))
        out = capsys.readouterr().out
        assert "accum_dtype_requested=fp32" in out
        assert "precision matched" in out
        assert "NOT precision matched" not in out
        for line in out.splitlines():
            if line.split() and line.split()[0] in bench._MODULE_PATHS:
                assert "fp32" in line.split()

    def test_run_does_not_mutate_module_globals(self, monkeypatch):
        self._patch_run(monkeypatch)
        before = bench._ACCUM_DTYPE
        bench._run(self._run_args(["triton"], "default"))
        bench._run(self._run_args(["triton"], "fp32"))
        # Modes flow purely as parameters; no state leak between runs.
        assert bench._ACCUM_DTYPE is before is torch.float32


# ===========================================================================
# B. GPU projection-geometry traces (whole-pipeline, not CE-only isolation)
# ===========================================================================


def _expected_chunk_rows(tokens, chunk):
    """Row count of each projection chunk after clamping ``chunk`` to ``tokens``."""
    chunk = min(chunk, tokens)
    rows, start = [], 0
    while start < tokens:
        rows.append(min(chunk, tokens - start))
        start += chunk
    return rows


def _flce_inputs_cuda(tokens, hidden, vocab, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_(True)
    w = (torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, generator=g) / (hidden**0.5)).detach()
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
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=20)
        with _ProjectionMMRecorder() as rec:
            _apply_direct(self._fn(), x, w, target, _CHUNK)
        # Projection GEMM is x_chunk @ w.T -> lhs [rows, H], rhs [H, V].
        proj = [lhs for lhs, rhs in rec.mms if rhs == (_HIDDEN, _VOCAB)]
        rows = [lhs[0] for lhs in proj]
        assert all(lhs[1] == _HIDDEN for lhs in proj)
        assert rows == _expected_chunk_rows(_TOKENS, _CHUNK)  # [33, 33, 33, 2]

    def test_clamp_above_tokens_is_single_chunk(self):
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=21)
        with _ProjectionMMRecorder() as rec:
            _apply_direct(self._fn(), x, w, target, _TOKENS * 5)  # clamps to N
        rows = [lhs[0] for lhs, rhs in rec.mms if rhs == (_HIDDEN, _VOCAB)]
        assert rows == [_TOKENS]

    def test_reduction_none_backward_matches_fp32_ref_nonuniform_upstream(self):
        import torch.nn.functional as F

        tokens, hidden, vocab = 100, 40, 256
        torch.manual_seed(22)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
        w = (torch.randn(vocab, hidden, device="cuda", dtype=torch.float32) / (hidden**0.5)).detach()
        w.requires_grad_(True)
        target = torch.randint(0, vocab, (tokens,), device="cuda")
        upstream = torch.rand(tokens, device="cuda", dtype=torch.float32) + 0.1  # nonuniform per-token

        # fp32 reference
        xr = x.detach().clone().requires_grad_(True)
        wr = w.detach().clone().requires_grad_(True)
        ref = F.cross_entropy(xr @ wr.t(), target, reduction="none")
        ref.backward(upstream)

        loss = _apply_direct(self._fn(), x, w, target, 24, reduction="none")
        assert loss.shape == (tokens,)
        loss.backward(upstream)
        assert torch.allclose(x.grad, xr.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(w.grad, wr.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(_cc() not in ((9, 0), (10, 0)), reason="cuTile FLCE requires SM90/SM100")
class TestCuTileProjectionGeometry:
    def _fn(self):
        pytest.importorskip("cuda.tile")
        from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

        return LigerFusedLinearCrossEntropyFunction

    def test_projection_row_geometry_matches_chunking(self):
        fn = self._fn()
        x, w, target = _flce_inputs_cuda(_TOKENS, _HIDDEN, _VOCAB, seed=23)
        with _ProjectionMMRecorder() as rec:
            _apply_direct(fn, x, w, target, _CHUNK)
        # cuTile owns only dW; the projection (x_chunk @ w.T) still runs on torch BLAS.
        proj = [lhs for lhs, rhs in rec.mms if rhs == (_HIDDEN, _VOCAB)]
        rows = [lhs[0] for lhs in proj]
        assert rows == _expected_chunk_rows(_TOKENS, _CHUNK)


@pytest.mark.skipif(_cc() != (10, 0), reason="native CuTe DSL FLCE requires SM100")
class TestNativeCuTeProjectionGeometry:
    def _fn(self):
        pytest.importorskip("cutlass.cute")
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


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_benchmark_run_all3_smoke():
    """Real (correctness-not-timing) end-to-end run of every arch-supported backend."""
    cc = _cc()
    backends = bench._select_default_backends(cc)
    # This test runs the production path (``bench._run``), which fails fast on a
    # missing optional SDK by design. So skip — rather than fail — when a
    # selected backend's optional dependency isn't installed, mirroring how the
    # other backend tests guard their imports. Every arch-supported backend is
    # still exercised whenever its SDK is present; none are silently dropped.
    for backend in backends:
        dep = bench._BACKEND_DEP.get(backend)
        if dep is not None:
            pytest.importorskip(dep, reason=f"{backend} backend requires the {dep!r} SDK")
    args = type(
        "Args",
        (),
        {
            "backends": backends,
            "tokens": 48,
            "hidden_size": 32,
            "vocab_size": 128,
            "chunk_sizes": [16, 1000],  # 1000 clamps to 48
            "warmup": 0,
            "iters": 2,
            "seed": 0,
            "accum_dtype": "fp32",
        },
    )()
    bench._run(args)  # must not raise


# ===========================================================================
# C. Env-facing wrapper arities (CPU autograd stub, monkeypatched class)
# ===========================================================================


class TestWrapperArities:
    def _inputs(self):
        return _cpu_inputs(tokens=8, hidden=4, vocab=6)

    def test_functional_appends_chunk_triple(self, monkeypatch):
        import liger_kernel.transformers.functional as tf

        recorded = {}

        class Rec(torch.autograd.Function):
            supports_chunk_size = True
            supports_inner_impl_dispatch = True

            @staticmethod
            def forward(ctx, *args):
                recorded["args"] = args
                return args[0].sum(), None, None, None

            @staticmethod
            def backward(ctx, *g):
                return (None,) * len(recorded["args"])

        monkeypatch.setattr(tf, "LigerFusedLinearCrossEntropyFunction", Rec)
        x, w, t = self._inputs()
        tf.liger_fused_linear_cross_entropy(x, w, t, chunk_size=7)
        assert len(recorded["args"]) == 18
        assert recorded["args"][-3:] == (None, None, 7)

    def test_functional_default_preserves_none_triple(self, monkeypatch):
        import liger_kernel.transformers.functional as tf

        recorded = {}

        class Rec(torch.autograd.Function):
            supports_chunk_size = True
            supports_inner_impl_dispatch = True

            @staticmethod
            def forward(ctx, *args):
                recorded["args"] = args
                return args[0].sum(), None, None, None

            @staticmethod
            def backward(ctx, *g):
                return (None,) * len(recorded["args"])

        monkeypatch.setattr(tf, "LigerFusedLinearCrossEntropyFunction", Rec)
        x, w, t = self._inputs()
        tf.liger_fused_linear_cross_entropy(x, w, t)
        assert recorded["args"][-3:] == (None, None, None)

    def test_functional_unsupported_chunk_raises(self, monkeypatch):
        import liger_kernel.transformers.functional as tf

        class Legacy(torch.autograd.Function):
            # dispatch-capable but NOT chunk-capable
            supports_inner_impl_dispatch = True

            @staticmethod
            def forward(ctx, *args):
                return args[0].sum(), None, None, None

            @staticmethod
            def backward(ctx, *g):
                return (None,) * 17

        monkeypatch.setattr(tf, "LigerFusedLinearCrossEntropyFunction", Legacy)
        x, w, t = self._inputs()
        with pytest.raises(ValueError):
            tf.liger_fused_linear_cross_entropy(x, w, t, chunk_size=7)

    def test_nn_module_appends_chunk_triple(self, monkeypatch):
        import liger_kernel.transformers.fused_linear_cross_entropy as mod

        recorded = {}

        class Rec(torch.autograd.Function):
            supports_chunk_size = True
            supports_inner_impl_dispatch = True

            @staticmethod
            def forward(ctx, *args):
                recorded["args"] = args
                return args[0].sum(), None, None, None

            @staticmethod
            def backward(ctx, *g):
                return (None,) * len(recorded["args"])

        monkeypatch.setattr(mod, "LigerFusedLinearCrossEntropyFunction", Rec)
        x, w, t = self._inputs()
        mod.LigerFusedLinearCrossEntropyLoss(chunk_size=9)(w, x, t)
        assert recorded["args"][-3:] == (None, None, 9)
