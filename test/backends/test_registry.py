"""Unit tests for ``liger_kernel.backends.registry`` and ``dispatch``.

These tests deliberately do NOT touch CUDA — they exercise the registration,
discovery, and resolution machinery using fake ops in a fresh namespace. The
``_isolated_registry`` fixture snapshots and restores ``_REGISTRY`` /
``_DISCOVERY_MAP`` / ``_DISCOVERED`` so tests don't bleed into each other.
"""

from __future__ import annotations

import os
import sys

from typing import Iterator
from unittest import mock

import pytest

from liger_kernel.backends.capability import Capability
from liger_kernel.backends.dispatch import BackendNotAvailableError
from liger_kernel.backends.dispatch import ImplNotAvailableError
from liger_kernel.backends.dispatch import NoBackendAvailableError
from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import clear_available_cache
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.dispatch import resolve_backend
from liger_kernel.backends.dispatch import set_backend
from liger_kernel.backends.registry import _DISCOVERED
from liger_kernel.backends.registry import _DISCOVERY_EVENTS
from liger_kernel.backends.registry import _DISCOVERY_MAP
from liger_kernel.backends.registry import _REGISTRY
from liger_kernel.backends.registry import OpImpl
from liger_kernel.backends.registry import all_registered_ops
from liger_kernel.backends.registry import declare_op_locations
from liger_kernel.backends.registry import get_registered
from liger_kernel.backends.registry import list_registered
from liger_kernel.backends.registry import register_op

# The dispatch *function* shadows the *module* in ``liger_kernel.backends`` due
# to ``from liger_kernel.backends.dispatch import dispatch`` in
# ``backends/__init__.py``. Pulling from ``sys.modules`` recovers the real
# module object (needed by the isolated_registry fixture below to swap the
# module-level _GLOBAL_IMPL/_GLOBAL_BACKEND globals).
dispatch_mod = sys.modules["liger_kernel.backends.dispatch"]


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    """Snapshot the global registry/discovery state for the duration of one test."""
    saved_registry = dict(_REGISTRY)
    saved_map = dict(_DISCOVERY_MAP)
    saved_discovered = dict(_DISCOVERED)
    saved_events = dict(_DISCOVERY_EVENTS)
    # The dispatcher refactor renamed _GLOBAL_BACKEND → _GLOBAL_IMPL and keeps
    # the legacy attribute as a mirror written by set_impl(). Snapshot and
    # restore the canonical one; the legacy mirror picks up on next write.
    saved_global = getattr(dispatch_mod, "_GLOBAL_IMPL", dispatch_mod._GLOBAL_BACKEND)
    try:
        yield
    finally:
        clear_available_cache()
        _REGISTRY.clear()
        _REGISTRY.update(saved_registry)
        _DISCOVERY_MAP.clear()
        _DISCOVERY_MAP.update(saved_map)
        _DISCOVERED.clear()
        _DISCOVERED.update(saved_discovered)
        # Restore _DISCOVERY_EVENTS too — otherwise stale per-op threading.Event
        # objects survive teardown and short-circuit _discover() in later tests
        # against an empty registry.
        _DISCOVERY_EVENTS.clear()
        _DISCOVERY_EVENTS.update(saved_events)
        dispatch_mod._GLOBAL_IMPL = saved_global
        dispatch_mod._GLOBAL_BACKEND = saved_global


@pytest.fixture
def clean_env() -> Iterator[None]:
    """Drop all LIGER_KERNEL_* env vars for the duration of one test."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("LIGER_KERNEL")}
    for k in list(saved):
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ[k] = v
        # Drop anything the test set.
        for k in list(os.environ):
            if k.startswith("LIGER_KERNEL") and k not in saved:
                os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# register_op
# ---------------------------------------------------------------------------


class TestRegisterOp:
    def test_registers_callable_with_correct_fields(self, isolated_registry):
        @register_op(
            "fake_op_callable",
            backend="bk_a",
            capability=Capability(),
            preference_rank=42,
            notes="hello",
        )
        def impl(x):
            return x + 1

        entry = _REGISTRY[("fake_op_callable", "bk_a")]
        assert isinstance(entry, OpImpl)
        assert entry.op_name == "fake_op_callable"
        assert entry.backend == "bk_a"
        assert entry.preference_rank == 42
        assert entry.notes == "hello"
        assert entry.forward is impl
        assert entry.autograd_function is None

    def test_registers_autograd_function_class(self, isolated_registry):
        import torch

        @register_op("fake_op_class", backend="bk_a", capability=Capability())
        class FakeFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, dy):
                return dy * 2

        entry = _REGISTRY[("fake_op_class", "bk_a")]
        assert entry.autograd_function is FakeFn
        assert entry.forward is None

    def test_overwrites_previous_entry_on_re_register(self, isolated_registry):
        @register_op("fake_op_overwrite", backend="bk_a", capability=Capability())
        def first(x):
            return x

        first_impl = _REGISTRY[("fake_op_overwrite", "bk_a")]

        @register_op("fake_op_overwrite", backend="bk_a", capability=Capability())
        def second(x):
            return x + 99

        second_impl = _REGISTRY[("fake_op_overwrite", "bk_a")]
        assert second_impl is not first_impl
        assert second_impl.forward is second

    def test_rejects_non_callable(self, isolated_registry):
        with pytest.raises(TypeError, match="expects a class or callable"):
            register_op("fake_op", backend="bk_a")(42)


# ---------------------------------------------------------------------------
# OpImpl.call
# ---------------------------------------------------------------------------


class TestOpImplCall:
    def test_calls_callable_forward_directly(self):
        impl = OpImpl(op_name="x", impl_name="bk", forward=lambda a, b: a + b)
        assert impl.call(3, 4) == 7

    def test_calls_apply_on_autograd_function(self):
        import torch

        class Doubler(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, dy):
                return dy * 2

        impl = OpImpl(op_name="x", impl_name="bk", autograd_function=Doubler)
        t = torch.tensor([1.0, 2.0])
        out = impl.call(t)
        assert torch.allclose(out, torch.tensor([2.0, 4.0]))

    def test_raises_when_no_callable_set(self):
        impl = OpImpl(op_name="x", impl_name="bk")
        with pytest.raises(RuntimeError, match="has no callable"):
            impl.call(0)

    def test_autograd_function_strips_mode_kwarg(self):
        """``.apply()`` doesn't accept kwargs; the ``mode`` dispatcher meta-kwarg
        must be silently stripped so class-registered impls work."""
        import torch

        class Doubler(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, dy):
                return dy * 2

        impl = OpImpl(op_name="x", impl_name="bk", autograd_function=Doubler)
        t = torch.tensor([1.0, 2.0])
        # Should NOT raise — ``mode=`` is stripped before ``.apply()``.
        out = impl.call(t, mode="default")
        assert torch.allclose(out, torch.tensor([2.0, 4.0]))

    def test_autograd_function_rejects_other_kwargs(self):
        """Any non-``mode`` kwarg passed to a class-registered impl must raise a
        clear TypeError rather than the obscure ``Function.apply`` error."""
        import torch

        class Doubler(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, dy):
                return dy * 2

        impl = OpImpl(op_name="x", impl_name="bk", autograd_function=Doubler)
        t = torch.tensor([1.0])
        with pytest.raises(TypeError, match="autograd_function registration"):
            impl.call(t, casting_mode="llama")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_declare_and_discover_is_idempotent(self, isolated_registry):
        calls = {"n": 0}

        # Build a fake importable module path by inserting into sys.modules.
        import sys
        import types

        mod = types.ModuleType("liger_kernel_test_fake_mod_one")

        def _on_import():
            calls["n"] += 1

        mod._on_import = _on_import

        # The discovery just imports the module; "side effects" happen at import time.
        # Use a real registration triggered on first import.
        @register_op("discoverable_op", backend="probe", capability=Capability())
        def _stub(x):  # pragma: no cover — registered lazily; not invoked
            return x

        # Remove the registration the decorator just made, then re-create it
        # only when our fake module is imported.
        del _REGISTRY[("discoverable_op", "probe")]

        def _register_on_import():
            @register_op("discoverable_op", backend="probe", capability=Capability())
            def _impl(x):
                return x

            calls["n"] += 1

        mod.__dict__["_register"] = _register_on_import
        # Side-effect at the bottom of the module:
        _register_on_import()
        sys.modules["liger_kernel_test_fake_mod_one"] = mod

        declare_op_locations("discoverable_op", ("liger_kernel_test_fake_mod_one",))
        # First lookup discovers.
        assert get_registered("discoverable_op", "probe") is not None
        # Subsequent lookups must not re-import (the import-once cache).
        before = calls["n"]
        list_registered("discoverable_op")
        get_registered("discoverable_op", "probe")
        assert calls["n"] == before

        del sys.modules["liger_kernel_test_fake_mod_one"]

    def test_failed_import_does_not_raise(self, isolated_registry):
        declare_op_locations("ghost_op", ("liger_kernel.this_module_does_not_exist",))
        # Should NOT raise — missing impl modules are expected when a DSL isn't installed.
        impls = list_registered("ghost_op")
        assert impls == ()


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


class TestListingHelpers:
    def test_list_registered_returns_only_matching_op(self, isolated_registry):
        @register_op("op_alpha", backend="bk1", capability=Capability())
        def _a(x):
            return x

        @register_op("op_alpha", backend="bk2", capability=Capability())
        def _b(x):
            return x

        @register_op("op_beta", backend="bk1", capability=Capability())
        def _c(x):
            return x

        impls = list_registered("op_alpha")
        backends = sorted(i.backend for i in impls)
        assert backends == ["bk1", "bk2"]
        assert all(i.op_name == "op_alpha" for i in impls)

    def test_get_registered_returns_specific_entry(self, isolated_registry):
        @register_op("op_x", backend="bkA", capability=Capability(), preference_rank=1)
        def _impl(x):
            return x

        impl = get_registered("op_x", "bkA")
        assert impl is not None
        assert impl.backend == "bkA"
        assert impl.preference_rank == 1
        assert get_registered("op_x", "missing") is None

    def test_all_registered_ops_returns_sorted_unique_names(self, isolated_registry):
        @register_op("zeta", backend="bk", capability=Capability())
        def _a(x):
            return x

        @register_op("alpha", backend="bk", capability=Capability())
        def _b(x):
            return x

        ops = all_registered_ops()
        # Pre-existing real ops (e.g., "rms_norm") may also be present; just check ours sort correctly.
        assert "alpha" in ops
        assert "zeta" in ops
        assert ops.index("alpha") < ops.index("zeta")


# ---------------------------------------------------------------------------
# Dispatch / resolve_backend
# ---------------------------------------------------------------------------


class TestResolveBackend:
    def _register_two(self):
        @register_op("disp_op", backend="fast", capability=Capability(), preference_rank=10)
        def _fast(x):
            return ("fast", x)

        @register_op("disp_op", backend="slow", capability=Capability(), preference_rank=100)
        def _slow(x):
            return ("slow", x)

    def test_explicit_kwarg_beats_env_and_global(self, isolated_registry, clean_env):
        self._register_two()
        os.environ["LIGER_KERNEL_BACKEND"] = "slow"
        set_backend("slow")
        impl = resolve_backend("disp_op", requested="fast")
        assert impl.backend == "fast"

    def test_per_op_env_beats_global_env(self, isolated_registry, clean_env):
        self._register_two()
        os.environ["LIGER_KERNEL_BACKEND_DISP_OP"] = "fast"
        os.environ["LIGER_KERNEL_BACKEND"] = "slow"
        impl = resolve_backend("disp_op")
        assert impl.backend == "fast"

    def test_global_env_beats_set_backend(self, isolated_registry, clean_env):
        self._register_two()
        set_backend("slow")
        os.environ["LIGER_KERNEL_BACKEND"] = "fast"
        impl = resolve_backend("disp_op")
        assert impl.backend == "fast"

    def test_set_backend_used_when_no_env(self, isolated_registry, clean_env):
        self._register_two()
        set_backend("slow")
        impl = resolve_backend("disp_op")
        assert impl.backend == "slow"

    def test_legacy_global_backend_direct_mutation_resolves(self, isolated_registry, clean_env):
        """Legacy callers that assigned ``dispatch._GLOBAL_BACKEND`` directly
        (rather than via ``set_backend()``) must still influence resolution."""
        self._register_two()
        # Bypass set_impl/set_backend — touch the legacy alias directly.
        dispatch_mod._GLOBAL_BACKEND = "slow"
        dispatch_mod._GLOBAL_IMPL = None  # the legacy direct-mutation surface
        impl = resolve_backend("disp_op")
        assert impl.backend == "slow"

    def test_auto_picks_lowest_preference_rank(self, isolated_registry, clean_env):
        self._register_two()
        impl = resolve_backend("disp_op")
        assert impl.backend == "fast"  # rank=10 < rank=100

    def test_unregistered_backend_message_lists_registered(self, isolated_registry, clean_env):
        self._register_two()
        with pytest.raises(BackendNotAvailableError) as ei:
            resolve_backend("disp_op", requested="cutile")
        msg = str(ei.value)
        assert "cutile" in msg
        assert "Registered" in msg  # dispatcher refactor: "Registered backends: …" → "Registered: …"
        assert "fast" in msg
        assert "slow" in msg

    def test_unsatisfied_capability_cites_reason(self, isolated_registry, clean_env):
        @register_op(
            "disp_op2",
            backend="needs_mod",
            capability=Capability(modules=["nonexistent_module_qzxy"]),
        )
        def _impl(x):
            return x

        with pytest.raises(BackendNotAvailableError) as ei:
            resolve_backend("disp_op2", requested="needs_mod")
        msg = str(ei.value)
        assert "needs_mod" in msg
        assert "nonexistent_module_qzxy" in msg

    def test_strict_mode_raises_when_no_backend(self, isolated_registry, clean_env):
        @register_op(
            "disp_op_strict",
            backend="bk",
            capability=Capability(modules=["nonexistent_module_xyz123"]),
        )
        def _impl(x):
            return x

        os.environ["LIGER_KERNEL_STRICT"] = "1"
        with pytest.raises(NoBackendAvailableError):
            resolve_backend("disp_op_strict")

    def test_no_backend_message_when_empty_registry(self, isolated_registry, clean_env):
        os.environ["LIGER_KERNEL_STRICT"] = "1"
        with pytest.raises(NoBackendAvailableError) as ei:
            resolve_backend("op_with_nothing_registered")
        assert "op_with_nothing_registered" in str(ei.value)


class TestDispatch:
    def test_dispatch_invokes_chosen_impl(self, isolated_registry, clean_env):
        @register_op("disp_call", backend="bkA", capability=Capability(), preference_rank=1)
        def _impl(x, y):
            return x * y

        assert dispatch("disp_call", 3, 4) == 12

    def test_dispatch_honors_backend_kwarg(self, isolated_registry, clean_env):
        @register_op("disp_pick", backend="alpha", capability=Capability(), preference_rank=10)
        def _a(x):
            return ("a", x)

        @register_op("disp_pick", backend="beta", capability=Capability(), preference_rank=20)
        def _b(x):
            return ("b", x)

        assert dispatch("disp_pick", 7, backend="beta") == ("b", 7)


class TestAvailableBackends:
    def test_returns_only_satisfied_sorted_by_rank(self, isolated_registry):
        @register_op("avail_op", backend="usable_lo", capability=Capability(), preference_rank=10)
        def _a(x):
            return x

        @register_op("avail_op", backend="usable_hi", capability=Capability(), preference_rank=200)
        def _b(x):
            return x

        @register_op(
            "avail_op",
            backend="unusable",
            capability=Capability(modules=["nonexistent_module_p13"]),
            preference_rank=1,
        )
        def _c(x):
            return x

        order = available_backends("avail_op")
        assert order == ["usable_lo", "usable_hi"]


# ---------------------------------------------------------------------------
# H100/H200 fallback simulation: cuTile gated on sm_100, Triton ungated.
# ---------------------------------------------------------------------------


class TestArchFallback:
    def test_auto_select_falls_back_to_triton_on_hopper(self, isolated_registry, clean_env):
        @register_op(
            "arch_op",
            backend="fake_cutile",
            capability=Capability(min_cc=(10, 0)),
            preference_rank=10,
        )
        def _cutile_impl(x):
            return ("cutile", x)

        @register_op(
            "arch_op",
            backend="fake_triton",
            capability=Capability(),  # always satisfied
            preference_rank=50,
        )
        def _triton_impl(x):
            return ("triton", x)

        # Pretend we're on H100 (sm_90).
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            order = available_backends("arch_op")
            assert order == ["fake_triton"]
            impl = resolve_backend("arch_op")
            assert impl.backend == "fake_triton"

    def test_explicit_cutile_on_hopper_raises_with_cc_reason(self, isolated_registry, clean_env):
        @register_op(
            "arch_op_explicit",
            backend="fake_cutile",
            capability=Capability(min_cc=(10, 0)),
            preference_rank=10,
        )
        def _cutile_impl(x):
            return x

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with pytest.raises(BackendNotAvailableError) as ei:
                resolve_backend("arch_op_explicit", requested="fake_cutile")
            msg = str(ei.value)
            assert "sm_90" in msg
            assert "sm_100" in msg


# ---------------------------------------------------------------------------
# Review-fix regression tests
# ---------------------------------------------------------------------------


class TestReviewFixes:
    """Targeted tests for review issues C-2, M-1, M-2, m-2, m-3, m-5."""

    # -- C-2: list_registered / get_registered / all_registered_ops must be
    #         safe under concurrent @register_op writes (vLLM, TGI scenario).
    def test_list_registered_is_thread_safe_during_register(self, isolated_registry):
        import threading
        import time

        n_threads = 8
        # n_threads writers + 1 reader + main thread that releases = +2
        barrier = threading.Barrier(n_threads + 2)
        errors = []

        def writer(i):
            barrier.wait()
            for k in range(50):

                @register_op(
                    f"ts_op_{i}_{k}",
                    backend="t",
                    capability=Capability(),
                )
                def _impl(x):
                    return x

        def reader():
            barrier.wait()
            for _ in range(50):
                try:
                    # Iterate everything just like a server's "list available"
                    # endpoint would.
                    for op in all_registered_ops():
                        list_registered(op)
                    time.sleep(0)
                except RuntimeError as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        barrier.wait()  # release everyone simultaneously
        for t in threads:
            t.join()
        assert not errors, f"concurrent iteration failed: {errors[:3]}"

    # -- M-2: a module that imports cleanly but does NOT call @register_op
    #         should produce a clear warning (conditional decorator guard,
    #         feature flag, etc).
    def test_discover_warns_when_module_imports_but_does_not_register(self, isolated_registry, caplog):
        import sys
        import types

        # Build a synthetic module whose import succeeds but registers nothing.
        modname = "liger_kernel._tests_silent_backend"
        mod = types.ModuleType(modname)
        # Deliberately nothing here — no @register_op.
        sys.modules[modname] = mod
        try:
            declare_op_locations("silent_op", (modname,))
            from liger_kernel.backends.registry import _discover

            # Make sure _discover actually runs (a prior test may have set the flag).
            _DISCOVERED.pop("silent_op", None)
            with caplog.at_level("WARNING", logger="liger_kernel.backends.registry"):
                _discover("silent_op")

            messages = [r.getMessage() for r in caplog.records]
            assert any("no @register_op ran" in m or "all declared paths imported" in m for m in messages), (
                f"expected silent-registration warning, got: {messages}"
            )
        finally:
            sys.modules.pop(modname, None)

    # -- m-5: _DISCOVERED set should be claimed before imports so two threads
    #         entering _discover for the same op don't both run the import pass.
    def test_discover_is_idempotent_under_concurrent_callers(self, isolated_registry):
        import sys
        import threading
        import types

        modname = "liger_kernel._tests_count_imports_backend"
        # Track how many times the module body executes. The contract under
        # test: even if 4 threads call _discover("race_op") concurrently, the
        # import pass should run exactly once (because _DISCOVERED is claimed
        # under the lock before imports start).
        load_count = {"n": 0}

        def make_module():
            load_count["n"] += 1
            m = types.ModuleType(modname)

            # Register inside the module body so the "did anyone register?"
            # warning path in M-2 stays quiet.
            @register_op(
                "race_op",
                backend="t",
                capability=Capability(),
            )
            def _impl(x):
                return x

            return m

        try:
            declare_op_locations("race_op", (modname,))
            from liger_kernel.backends.registry import _DISCOVERED
            from liger_kernel.backends.registry import _discover

            # Each call to fake_import_module rebuilds the module so the body
            # runs exactly load_count times; sys.modules caches the result so
            # subsequent importlib.import_module calls would normally hit the
            # cache, but our mock always rebuilds — perfect for counting.
            def fake_import_module(name):
                if name == modname:
                    m = make_module()
                    sys.modules[name] = m
                    return m
                return __import__(name)

            # 4 worker threads + main thread that releases = 5 waiters
            barrier = threading.Barrier(5)

            def worker():
                barrier.wait()
                _discover("race_op")

            # Reset state so the test starts clean.
            _DISCOVERED.pop("race_op", None)
            load_count["n"] = 0

            with mock.patch(
                "liger_kernel.backends.registry.importlib.import_module",
                side_effect=fake_import_module,
            ):
                threads = [threading.Thread(target=worker) for _ in range(4)]
                for t in threads:
                    t.start()
                barrier.wait()
                for t in threads:
                    t.join()

            # Exactly one import pass (== one module body execution) is what we want.
            assert load_count["n"] == 1, (
                f"_discover did not deduplicate under concurrent callers: module body ran {load_count['n']} times"
            )
        finally:
            sys.modules.pop(modname, None)

    # -- m-3: emit_fallback_warning must dedup under threads.
    def test_emit_fallback_warning_emits_exactly_once_under_threads(self, isolated_registry):
        import threading
        import warnings

        # Reset state — touching the private attr is acceptable in tests.
        notified = dispatch_mod._FALLBACK_NOTIFIED
        notified.clear()

        barrier = threading.Barrier(8)
        warning_count = {"n": 0}
        warning_lock = threading.Lock()

        def worker():
            with warnings.catch_warnings(record=True) as w_list:
                warnings.simplefilter("always")
                barrier.wait()
                dispatch_mod.emit_fallback_warning("op_x", requested="cutile", actual="triton", reason="test reason")
            with warning_lock:
                warning_count["n"] += sum(
                    1 for x in w_list if issubclass(x.category, dispatch_mod.LigerBackendFallbackWarning)
                )

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cleanup so we don't leak into other tests.
        notified.discard(("op_x", "cutile", "triton"))

        assert warning_count["n"] == 1, (
            f"expected exactly one fallback warning under 8 threads, got {warning_count['n']}"
        )

    def test_emit_fallback_warning_raises_in_strict_mode(self, clean_env):
        os.environ["LIGER_KERNEL_STRICT"] = "1"

        with pytest.raises(ImplNotAvailableError, match="op_x.*cutile.*triton"):
            dispatch_mod.emit_fallback_warning("op_x", requested="cutile", actual="triton", reason="unsupported shape")


# ---------------------------------------------------------------------------
# M-1: tolerance widening should not KeyError on partial tolerance dicts
# ---------------------------------------------------------------------------


class TestToleranceWidening:
    def test_partial_tolerance_keys_dont_crash_on_none_casting(self, isolated_registry):
        import torch

        from liger_kernel.testing._op_helpers import _tolerances_for

        @register_op(
            "partial_tol_op",
            backend="t",
            capability=Capability(),
            tolerances={
                # Only atol — the impl forgot to declare rtol. Pre-fix this
                # would KeyError when casting_mode="none" tried to widen
                # rtol_fwd / rtol_bwd.
                torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2},
            },
        )
        def _impl(x):
            return x

        # Should not raise.
        tols = _tolerances_for("partial_tol_op", "t", torch.float16, casting_mode="none")
        # Widened by 4x on the keys that did exist (in this case atol from
        # the impl, plus rtol filled in from helper defaults — both should be widened).
        assert tols["atol_fwd"] == pytest.approx(5e-3 * 4.0)
        # rtol comes from the helper defaults map for fp16; should also be widened.
        assert "rtol_fwd" in tols
        assert tols["rtol_fwd"] == pytest.approx(1e-2 * 4.0)


# ---------------------------------------------------------------------------
# Legacy HuggingFace RMSNorm subclasses preserve the stable constructor.
# ---------------------------------------------------------------------------


class TestLegacyRMSNormCompatibilityBoundary:
    @pytest.mark.parametrize(
        "subclass_name",
        [
            "LigerRMSNormForGemma",
            "LigerRMSNormForGemma2",
            "LigerRMSNormForGemma3",
            "LigerRMSNormForGemma4",
            "LigerRMSNormForOlmo2",
            "LigerRMSNormForGlm4",
            "LigerRMSNormForQwen3Next",
        ],
    )
    @pytest.mark.parametrize("dispatch_kwarg", ["impl", "backend", "mode"])
    def test_subclass_rejects_dispatch_kwargs(self, subclass_name, dispatch_kwarg):
        try:
            import liger_kernel.transformers.rms_norm as rms_mod
        except Exception as e:
            pytest.skip(f"liger_kernel.transformers.rms_norm unimportable on this host: {e}")

        cls = getattr(rms_mod, subclass_name)
        with pytest.raises(TypeError):
            cls(64, **{dispatch_kwarg: "nvidia-triton"})
