"""
Backend registry for Liger-Kernel multi-backend support.

A "backend" here is a named implementation of Liger's operators. It may correspond
to a different hardware device (e.g., Ascend on NPU) or a different DSL on the
same device (e.g., cuTile on CUDA), and it may support one or more devices.

Each backend declares:
  - the set of devices it supports
  - the subset of those devices on which it is the *default* (auto-applied on
    import). On any other supported device the backend is opt-in only and must
    be requested explicitly via the LIGER_KERNEL_BACKEND environment variable.

Each backend registers itself by calling register_backend() in its __init__.py.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Tuple

# Dynamically get backends package path to avoid hardcoding
_BACKENDS_PACKAGE = __name__.rsplit(".", 1)[0]  # "liger_kernel.ops.backends"

# Environment variable users set to explicitly select an opt-in backend.
LIGER_KERNEL_BACKEND_ENV = "LIGER_KERNEL_BACKEND"


@dataclass(frozen=True)
class BackendInfo:
    """
    Information about a backend implementation.

    Attributes:
        name: Backend identifier (e.g., "ascend", "cutile"). The on-disk
            directory must be ``backends/_<name>/``.
        devices: Tuple of device types this backend supports
            (e.g., ``("npu",)``, ``("cuda",)``, ``("cuda", "xpu")``).
        default_devices: Subset of ``devices`` on which this backend is
            automatically applied at import time. On supported devices not
            listed here, the backend is opt-in only via ``LIGER_KERNEL_BACKEND``.
            Empty tuple (the default) means the backend is opt-in only on every
            device it supports.
    """

    name: str
    devices: Tuple[str, ...]
    default_devices: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.devices:
            raise ValueError(f"Backend {self.name!r} must declare at least one supported device.")
        extra = set(self.default_devices) - set(self.devices)
        if extra:
            raise ValueError(
                f"Backend {self.name!r}: default_devices {sorted(extra)} not in devices {list(self.devices)}."
            )

    @property
    def module_path(self) -> str:
        """Auto-generated module path based on backend name."""
        return f"{_BACKENDS_PACKAGE}._{self.name}.ops"


# Registry mapping backend names to their info.
BACKEND_REGISTRY: dict[str, BackendInfo] = {}


def register_backend(info: BackendInfo) -> None:
    """Register a backend's info in the global registry."""
    BACKEND_REGISTRY[info.name] = info


def select_backend(device: str, explicit: Optional[str] = None) -> Optional[BackendInfo]:
    """
    Select the backend implementation for the current device.

    Args:
        device: Device type from ``infer_device()`` (e.g., "cuda", "npu").
        explicit: If set, force selection of this named backend. The backend's
            supported devices are validated against the runtime.

    Returns:
        ``BackendInfo`` if a backend should replace the defaults, ``None`` to keep defaults.

    Raises:
        RuntimeError: If ``explicit`` names an unknown backend or is incompatible
            with the current device.
    """
    if explicit:
        info = BACKEND_REGISTRY.get(explicit)
        if info is None:
            known = ", ".join(sorted(BACKEND_REGISTRY)) or "<none registered>"
            raise RuntimeError(
                f"Unknown {LIGER_KERNEL_BACKEND_ENV}={explicit!r}. Registered backends: {known}."
            )
        if device not in info.devices:
            supported = ", ".join(info.devices)
            raise RuntimeError(
                f"{LIGER_KERNEL_BACKEND_ENV}={info.name!r} supports devices ({supported}), "
                f"but the current device is {device!r}."
            )
        return info

    # Auto-select: pick a backend that declares the current device as one of its defaults.
    for info in BACKEND_REGISTRY.values():
        if device in info.default_devices:
            return info
    return None
