from types import SimpleNamespace

import pytest
import torch

import liger_kernel.utils as utils

from liger_kernel.utils import _infer_amd_arch
from liger_kernel.utils import _infer_npu_arch
from liger_kernel.utils import _infer_nvidia_arch
from liger_kernel.utils import _infer_xpu_arch
from liger_kernel.utils import infer_device
from liger_kernel.utils import infer_device_arch


@pytest.fixture(autouse=True)
def clear_arch_cache():
    # infer_device_arch is lru_cached; isolate every test from cached results.
    infer_device_arch.cache_clear()
    yield
    infer_device_arch.cache_clear()


# -----------------------------------------------------------------------------
# Per-architecture mapping (helpers, with the underlying torch calls mocked)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "capability, expected",
    [
        ((7, 0), "volta_turing"),
        ((7, 5), "volta_turing"),
        ((8, 0), "ampere_ada"),
        ((8, 6), "ampere_ada"),
        ((8, 9), "ampere_ada"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((10, 3), "blackwell_ultra"),
        ((12, 0), "blackwell_consumer"),
        ((8, 7), "sm_87"),  # known major, unknown minor -> fallback keeps the minor
        ((11, 0), "sm_110"),  # unknown cc -> sm_<major><minor> fallback
    ],
)
def test_infer_nvidia_arch(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device_id=0: capability)
    assert _infer_nvidia_arch(0) == expected


@pytest.mark.parametrize(
    "gcn_arch_name, expected",
    [
        ("gfx908", "cdna"),
        ("gfx90a", "cdna2"),
        ("gfx942:sramecc+:xnack-", "cdna3"),  # decorated name -> gfx target only
        ("gfx1100", "rdna3"),
        ("gfx9999", "gfx9999"),  # unknown gfx -> raw target
        ("", "cuda"),  # missing arch name -> device-type fallback
    ],
)
def test_infer_amd_arch(monkeypatch, gcn_arch_name, expected):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device_id=0: SimpleNamespace(gcnArchName=gcn_arch_name),
    )
    assert _infer_amd_arch(0) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Intel(R) Data Center GPU Max 1550", "pvc"),
        ("Intel(R) Arc(TM) A770 Graphics", "arc"),
        ("Some Future Intel GPU", "xpu"),  # unrecognized -> device-type fallback
    ],
)
def test_infer_xpu_arch(monkeypatch, name, expected):
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(get_device_properties=lambda device_id=0: SimpleNamespace(name=name)),
        raising=False,
    )
    assert _infer_xpu_arch(0) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Ascend910B", "ascend910"),
        ("Ascend910B3", "ascend910"),
        ("Ascend310P3", "ascend310"),
        ("Future NPU", "npu"),  # unrecognized -> device-type fallback
    ],
)
def test_infer_npu_arch(monkeypatch, name, expected):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_properties=lambda device_id=0: SimpleNamespace(name=name)),
        raising=False,
    )
    assert _infer_npu_arch(0) == expected


# -----------------------------------------------------------------------------
# Dispatch in infer_device_arch
# -----------------------------------------------------------------------------
def test_dispatch_nvidia(monkeypatch):
    monkeypatch.setattr(utils, "infer_device", lambda: "cuda")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(utils, "_infer_nvidia_arch", lambda device_id: "blackwell")
    monkeypatch.setattr(utils, "_infer_amd_arch", lambda device_id: pytest.fail("AMD path taken on NVIDIA"))
    assert infer_device_arch() == "blackwell"


def test_dispatch_amd(monkeypatch):
    # ROCm reports as "cuda"; torch.version.hip routes to the AMD helper.
    monkeypatch.setattr(utils, "infer_device", lambda: "cuda")
    monkeypatch.setattr(torch.version, "hip", "6.0.0", raising=False)
    monkeypatch.setattr(utils, "_infer_amd_arch", lambda device_id: "cdna3")
    monkeypatch.setattr(utils, "_infer_nvidia_arch", lambda device_id: pytest.fail("NVIDIA path taken on AMD"))
    assert infer_device_arch() == "cdna3"


@pytest.mark.parametrize("device", ["xpu", "npu"])
def test_dispatch_xpu_npu(monkeypatch, device):
    monkeypatch.setattr(utils, "infer_device", lambda: device)
    monkeypatch.setattr(utils, f"_infer_{device}_arch", lambda device_id: f"{device}-arch")
    assert infer_device_arch() == f"{device}-arch"


def test_falls_back_to_device_type_on_error(monkeypatch):
    monkeypatch.setattr(utils, "infer_device", lambda: "cuda")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    def boom(device_id):
        raise RuntimeError("driver not initialized")

    monkeypatch.setattr(utils, "_infer_nvidia_arch", boom)
    assert infer_device_arch() == "cuda"


def test_unaccelerated_device_returns_device_type(monkeypatch):
    monkeypatch.setattr(utils, "infer_device", lambda: "cpu")
    assert infer_device_arch() == "cpu"


# -----------------------------------------------------------------------------
# Real-environment behavior + caching
# -----------------------------------------------------------------------------
def test_returns_nonempty_string_for_current_device():
    arch = infer_device_arch()
    assert isinstance(arch, str) and arch
    # On an unaccelerated host this collapses to the device type.
    if infer_device() == "cpu":
        assert arch == "cpu"


def test_result_is_cached():
    infer_device_arch.cache_clear()
    first = infer_device_arch()
    second = infer_device_arch()
    assert first == second
    info = infer_device_arch.cache_info()
    assert info.hits >= 1 and info.misses == 1
