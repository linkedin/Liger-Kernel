"""Unit tests for :class:`liger_kernel.backends.capability.Capability`.

These tests run without CUDA — they monkeypatch ``torch.cuda`` to simulate
arbitrary device topologies.
"""

from __future__ import annotations

from unittest import mock

import pytest

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
    """``min_cc``/``max_cc`` are CUDA-only, so these pin an NVIDIA platform."""

    def test_min_cc_unsatisfied_on_lower_arch(self):
        cap = Capability(min_cc=(10, 0))
        with (
            mock.patch("torch.version.hip", None),
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
            mock.patch("torch.version.hip", None),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            assert cap.is_satisfied() is True

    def test_max_cc_unsatisfied_on_higher_arch(self):
        cap = Capability(max_cc=(9, 0))
        with (
            mock.patch("torch.version.hip", None),
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


class TestComputeCapabilityGateOnRocm:
    """On ROCm the reported tuple is the gfx ISA version (gfx90a -> (9, 0)); cc gates must not hold."""

    @pytest.mark.parametrize(
        "cap, reported_cc",
        [
            (Capability(min_cc=(9, 0)), (9, 0)),  # gfx90a: numerically >= an sm_90 gate
            (Capability(max_cc=(10, 0)), (9, 4)),  # gfx942: numerically within a <= sm_100 gate
        ],
    )
    def test_cc_gate_unsatisfied_on_rocm_despite_numeric_match(self, cap, reported_cc):
        with (
            mock.patch("torch.version.hip", "6.2.41134"),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=reported_cc),
        ):
            assert cap.is_satisfied() is False
            reason = cap.reason_if_unsatisfied()
            assert reason is not None
            assert "ROCm" in reason

    def test_rocm_short_circuit_does_not_affect_other_gates(self):
        """A gate without a cc bound is untouched and never reads the capability."""
        cap = Capability(modules=["sys"], requires_cuda=True)
        with (
            mock.patch("torch.version.hip", "6.2.41134"),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torch.cuda.get_device_capability",
                side_effect=AssertionError("must not be called without a cc gate"),
            ),
        ):
            assert cap.is_satisfied() is True


class TestArchGate:
    """``archs`` gates on the family from ``infer_device_arch()``."""

    @pytest.mark.parametrize(
        "archs, device_arch, expected",
        [
            (("cdna3",), "cdna3", True),
            (("cdna3",), "cdna2", False),
            (("cdna2", "cdna3"), "cdna2", True),
        ],
    )
    def test_arch_match(self, archs, device_arch, expected):
        cap = Capability(archs=archs)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("liger_kernel.utils.infer_device_arch", return_value=device_arch),
        ):
            assert cap.is_satisfied() is expected
            reason = cap.reason_if_unsatisfied()
            if expected:
                assert reason is None
            else:
                assert reason is not None and device_arch in reason

    @pytest.mark.parametrize(
        "device, expected_index",
        [(1, 1), ("cuda:3", 3), ("cuda", 2), (None, 2)],
    )
    def test_device_selector_resolves_to_its_index(self, device, expected_index):
        cap = Capability(archs=("cdna3",))
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=2),
            mock.patch("liger_kernel.utils.infer_device_arch", return_value="cdna3") as arch,
        ):
            assert cap.is_satisfied(device=device) is True
            arch.assert_called_once_with(expected_index)

    def test_cpu_selector_does_not_resolve_to_accelerator_zero(self):
        """A CPU selector has no GPU arch and must not fall back to accelerator 0."""
        cap = Capability(archs=("cdna3",))
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("liger_kernel.utils.infer_device_arch", return_value="cdna3") as arch,
        ):
            assert cap.is_satisfied(device="cpu") is False
            assert "CPU" in cap.reason_if_unsatisfied(device="cpu")
            arch.assert_not_called()


class TestArchsAndComputeCapabilityAreExclusive:
    """``archs`` with ``min_cc``/``max_cc`` can never be satisfied; reject it at construction."""

    @pytest.mark.parametrize("kwargs", [{"min_cc": (9, 0)}, {"max_cc": (10, 0)}])
    def test_construction_rejects_the_combination(self, kwargs):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Capability(archs=("cdna3",), **kwargs)


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
