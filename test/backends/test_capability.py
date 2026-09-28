"""Unit tests for :class:`liger_kernel.backends.capability.Capability`.

These tests run without CUDA — they monkeypatch ``torch.cuda`` to simulate
arbitrary device topologies.
"""

from __future__ import annotations

from unittest import mock

from liger_kernel.backends.capability import Capability


class TestEmptyCapability:
    def test_empty_capability_is_satisfied(self):
        cap = Capability()
        assert cap.is_satisfied() is True
        assert cap.reason_if_unsatisfied() is None


class TestModuleGate:
    def test_missing_module_is_unsatisfied(self):
        cap = Capability(modules=["nonexistent_module_xyz"])
        assert cap.is_satisfied() is False
        reason = cap.reason_if_unsatisfied()
        assert reason is not None
        assert "nonexistent_module_xyz" in reason

    def test_present_module_is_satisfied(self):
        # ``sys`` is always importable.
        cap = Capability(modules=["sys"])
        assert cap.is_satisfied() is True


class TestComputeCapabilityGate:
    def test_min_cc_unsatisfied_on_lower_arch(self):
        cap = Capability(min_cc=(10, 0))
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            assert cap.is_satisfied() is False
            reason = cap.reason_if_unsatisfied()
            assert reason is not None
            assert "sm_90" in reason
            assert "sm_100" in reason

    def test_min_cc_satisfied_on_equal_arch(self):
        cap = Capability(min_cc=(9, 0))
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            assert cap.is_satisfied() is True

    def test_max_cc_unsatisfied_on_higher_arch(self):
        cap = Capability(max_cc=(9, 0))
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            assert cap.is_satisfied() is False
            reason = cap.reason_if_unsatisfied()
            assert reason is not None
            assert "sm_100" in reason
            assert "sm_90" in reason

    def test_requires_cuda_without_cuda(self):
        cap = Capability(requires_cuda=True)
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert cap.is_satisfied() is False
            reason = cap.reason_if_unsatisfied()
            assert reason is not None
            assert "CUDA" in reason


class TestPredicate:
    def test_predicate_runs_only_after_other_gates_pass(self):
        called = {"n": 0}

        def _pred():
            called["n"] += 1
            return True

        cap = Capability(modules=["nonexistent_module_abc"], predicate=_pred)
        # Module gate fails first; predicate must NOT be called.
        assert cap.is_satisfied() is False
        assert called["n"] == 0

    def test_predicate_returning_false_unsatisfied(self):
        cap = Capability(predicate=lambda: False)
        assert cap.is_satisfied() is False
        reason = cap.reason_if_unsatisfied()
        assert reason is not None
        assert "predicate" in reason

    def test_predicate_raising_treated_as_unsatisfied(self):
        def _boom():
            raise RuntimeError("something exploded")

        cap = Capability(predicate=_boom)
        assert cap.is_satisfied() is False
        reason = cap.reason_if_unsatisfied()
        assert reason is not None
        assert "predicate" in reason
        assert "something exploded" in reason


class TestCachingBehavior:
    """The current Capability implementation does NOT cache results — each
    ``is_satisfied()`` call re-evaluates. This test pins that behavior so it
    is an intentional choice rather than a regression.

    If caching is added later, this test should be updated and the rationale
    documented (e.g., what invalidates the cache when modules are imported
    after process start).
    """

    def test_is_satisfied_re_evaluates_each_call(self):
        calls = {"n": 0}

        def _pred():
            calls["n"] += 1
            return True

        cap = Capability(predicate=_pred)
        cap.is_satisfied()
        cap.is_satisfied()
        cap.is_satisfied()
        # Three independent evaluations.
        assert calls["n"] == 3
