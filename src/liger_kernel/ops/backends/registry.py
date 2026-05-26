"""
Implementation registry for Liger-Kernel.

An "implementation" here is a named alternative kernel set. It may correspond
to a different hardware device (e.g., Ascend on NPU, in ``backends/_ascend/``)
or a different DSL on the same device (e.g., cuTile on CUDA, in ``ops/cutile/``).
It may support one or more devices.

Each implementation declares:
  - the set of devices it supports
  - the subset of those devices on which it is the *default* (auto-applied on
    import). On any other supported device the implementation is opt-in only
    and must be requested explicitly via the LIGER_KERNEL_IMPL environment
    variable.
  - the Python module path where its operators live.

Each implementation registers itself by calling register_impl() in its
__init__.py.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Tuple

# Environment variable users set to explicitly select an opt-in implementation.
LIGER_KERNEL_IMPL_ENV = "LIGER_KERNEL_IMPL"


@dataclass(frozen=True)
class ImplInfo:
    """
    Information about a kernel implementation.

    Attributes:
        name: Implementation identifier (e.g., "ascend", "cutile"). Also the
            value users pass via ``LIGER_KERNEL_IMPL=<name>``.
        devices: Tuple of device types this implementation supports
            (e.g., ``("npu",)``, ``("cuda",)``, ``("cuda", "xpu")``).
        default_devices: Subset of ``devices`` on which this implementation
            is automatically applied at import time. On supported devices not
            listed here, it is opt-in only via ``LIGER_KERNEL_IMPL``. Empty
            tuple (the default) means opt-in only on every supported device.
        module_path: Python module path where the operator implementations
            live (e.g., ``"liger_kernel.ops.cutile.ops"``). Required.
    """

    name: str
    devices: Tuple[str, ...]
    default_devices: Tuple[str, ...] = field(default_factory=tuple)
    module_path: str = ""

    def __post_init__(self):
        if not self.devices:
            raise ValueError(f"Implementation {self.name!r} must declare at least one supported device.")
        if not self.module_path:
            raise ValueError(f"Implementation {self.name!r} must declare a module_path.")
        extra = set(self.default_devices) - set(self.devices)
        if extra:
            raise ValueError(
                f"Implementation {self.name!r}: default_devices {sorted(extra)} not in devices {list(self.devices)}."
            )


# Registry mapping implementation names to their info.
IMPL_REGISTRY: dict[str, ImplInfo] = {}


def register_impl(info: ImplInfo) -> None:
    """Register an implementation's info in the global registry."""
    IMPL_REGISTRY[info.name] = info


def select_impl(device: str, explicit: Optional[str] = None) -> Optional[ImplInfo]:
    """
    Select the implementation for the current device.

    Args:
        device: Device type from ``infer_device()`` (e.g., "cuda", "npu").
        explicit: If set, force selection of this named implementation. The
            supported devices are validated against the runtime.

    Returns:
        ``ImplInfo`` if an implementation should replace the defaults,
        ``None`` to keep defaults.

    Raises:
        RuntimeError: If ``explicit`` names an unknown implementation or one
            incompatible with the current device.
    """
    if explicit:
        info = IMPL_REGISTRY.get(explicit)
        if info is None:
            known = ", ".join(sorted(IMPL_REGISTRY)) or "<none registered>"
            raise RuntimeError(
                f"Unknown {LIGER_KERNEL_IMPL_ENV}={explicit!r}. Registered implementations: {known}."
            )
        if device not in info.devices:
            supported = ", ".join(info.devices)
            raise RuntimeError(
                f"{LIGER_KERNEL_IMPL_ENV}={info.name!r} supports devices ({supported}), "
                f"but the current device is {device!r}."
            )
        return info

    # Auto-select: pick an implementation that lists the current device in its defaults.
    for info in IMPL_REGISTRY.values():
        if device in info.default_devices:
            return info
    return None
