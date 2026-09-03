"""Regression tests: capability/dispatch must resolve GPU capability from the
*input tensor's* device, not the current process device.

These tests run on CPU. They monkeypatch ``torch.cuda`` so that device index 0
looks like Hopper (sm_90) and device index 1 looks like Blackwell (sm_100),
then verify that a Blackwell-only impl is auto-selected only when the input
tensor lives on ``cuda:1`` — even though the current device (``None``) reports
sm_90.
"""

from __future__ import annotations

import sys

from typing import Iterator
from unittest import mock

import pytest
import torch

from liger_kernel.backends.capability import Capability
from liger_kernel.backends.dispatch import available_impls
from liger_kernel.backends.dispatch import clear_available_cache
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.dispatch import resolve_impl
from liger_kernel.backends.registry import _DISCOVERED
from liger_kernel.backends.registry import _DISCOVERY_EVENTS
from liger_kernel.backends.registry import _DISCOVERY_MAP
from liger_kernel.backends.registry import _REGISTRY
from liger_kernel.backends.registry import register_op

# The ``dispatch`` *function* shadows the *module* in ``liger_kernel.backends``
# (``from liger_kernel.backends.dispatch import dispatch`` in the package init).
# Recover the real module object from ``sys.modules`` to swap its module-level
# ``_GLOBAL_IMPL`` / ``_GLOBAL_BACKEND`` globals.
dispatch_mod = sys.modules["liger_kernel.backends.dispatch"]


# Per-index simulated compute capability: cuda:0 = Hopper, cuda:1 = Blackwell.
_INDEX_CC = {0: (9, 0), 1: (10, 0)}


def _index_of(device) -> int:
    """Resolve a device (None/int/str/torch.device) to a CUDA index.

    ``None`` = current process device, which we pin to index 0 (Hopper).
    """
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    dev = torch.device(device)
    return dev.index if dev.index is not None else 0


def _fake_get_device_capability(device=None):
    return _INDEX_CC[_index_of(device)]


def _cuda_tensor_on(index: int):
    """A stand-in for a CUDA tensor on ``cuda:<index>`` usable on a CPU host.

    ``MagicMock(spec=torch.Tensor)`` passes ``isinstance(x, torch.Tensor)`` and
    lets us pin ``.is_cuda`` / ``.device`` without a real GPU.
    """
    fake = mock.MagicMock(spec=torch.Tensor)
    fake.is_cuda = True
    fake.device = torch.device("cuda", index)
    return fake


@pytest.fixture
def isolated_registry(monkeypatch) -> Iterator[None]:
    """Snapshot registry/discovery/memo + fake a two-GPU CUDA environment."""
    saved_registry = dict(_REGISTRY)
    saved_map = dict(_DISCOVERY_MAP)
    saved_discovered = dict(_DISCOVERED)
    saved_events = dict(_DISCOVERY_EVENTS)
    saved_global_impl = dispatch_mod._GLOBAL_IMPL
    saved_global_backend = dispatch_mod._GLOBAL_BACKEND

    # Drop any LIGER_KERNEL_* overrides so auto-select is what we exercise.
    for name in ("LIGER_KERNEL_IMPL", "LIGER_KERNEL_BACKEND"):
        monkeypatch.delenv(name, raising=False)

    clear_available_cache()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _fake_get_device_capability)
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
        _DISCOVERY_EVENTS.clear()
        _DISCOVERY_EVENTS.update(saved_events)
        dispatch_mod._GLOBAL_IMPL = saved_global_impl
        dispatch_mod._GLOBAL_BACKEND = saved_global_backend


def _register_blackwell_and_triton(op_name: str) -> None:
    @register_op(
        op_name,
        backend="nvidia-blackwell",
        capability=Capability(min_cc=(10, 0)),
        preference_rank=10,
    )
    def _blackwell_impl(x):
        return ("blackwell", x)

    @register_op(
        op_name,
        backend="nvidia-triton",
        capability=Capability(),  # trivially satisfied — the fallback
        preference_rank=50,
    )
    def _triton_impl(x):
        return ("triton", x)


class TestAvailableImplsFollowsDevice:
    def test_blackwell_available_only_on_cuda1(self, isolated_registry):
        _register_blackwell_and_triton("devcap_avail")

        # Current device (None) == cuda:0 == Hopper → Blackwell gated out.
        assert available_impls("devcap_avail") == ["nvidia-triton"]
        assert available_impls("devcap_avail", device="cuda:0") == ["nvidia-triton"]

        # cuda:1 == Blackwell → Blackwell wins on preference_rank.
        assert available_impls("devcap_avail", device="cuda:1") == [
            "nvidia-blackwell",
            "nvidia-triton",
        ]

    def test_memo_does_not_alias_across_devices(self, isolated_registry):
        _register_blackwell_and_triton("devcap_memo")

        # Prime the memo for cuda:0, then query cuda:1 — must not return the
        # cuda:0 cached result.
        assert available_impls("devcap_memo", device="cuda:0") == ["nvidia-triton"]
        assert available_impls("devcap_memo", device="cuda:1") == [
            "nvidia-blackwell",
            "nvidia-triton",
        ]


class TestResolveImplFollowsDevice:
    def test_resolve_picks_blackwell_on_cuda1(self, isolated_registry):
        _register_blackwell_and_triton("devcap_resolve")
        assert resolve_impl("devcap_resolve", device="cuda:1").impl_name == "nvidia-blackwell"
        assert resolve_impl("devcap_resolve", device="cuda:0").impl_name == "nvidia-triton"
        # Legacy no-device behavior == current device (cuda:0).
        assert resolve_impl("devcap_resolve").impl_name == "nvidia-triton"


class TestDispatchFollowsInputTensorDevice:
    def test_dispatch_selects_by_input_device(self, isolated_registry):
        _register_blackwell_and_triton("devcap_dispatch")

        on_cuda1 = _cuda_tensor_on(1)
        on_cuda0 = _cuda_tensor_on(0)

        # Input on cuda:1 (Blackwell) → Blackwell impl runs.
        impl_name, tensor = dispatch("devcap_dispatch", on_cuda1)
        assert impl_name == "blackwell"
        assert tensor is on_cuda1

        # Input on cuda:0 (Hopper) → Triton fallback runs, even though a
        # Blackwell impl exists and cuda:1 is present in the process.
        impl_name, tensor = dispatch("devcap_dispatch", on_cuda0)
        assert impl_name == "triton"
        assert tensor is on_cuda0
