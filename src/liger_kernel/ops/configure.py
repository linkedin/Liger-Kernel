"""Optional native runtime setup, separate from ordinary operator imports.

Importing this module and constructing configuration data do not require LCK.
When LCK is absent or the device is unsupported, ``configure`` warns and
returns ``False`` without changing the runtime.
"""

from __future__ import annotations

import importlib
import warnings

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device as Device
    from torch.distributed import ProcessGroup

__all__ = ["configure", "FusedLinearCrossEntropyConfig", "MoEConfig"]


@dataclass(frozen=True)
class FusedLinearCrossEntropyConfig:
    """FLSCE capacity and optional name from ``process_groups``."""

    max_tokens: int
    hidden_size: int
    local_vocab_size: int
    group: str | None = None
    tiles_per_reduce: int = 1


@dataclass(frozen=True)
class MoEConfig:
    """MoE capacity, EP-team topology, and optional process-group name."""

    max_tokens: int
    hidden_size: int
    num_experts: int
    top_k: int
    num_hosts: int
    gpus_per_host: int
    group: str | None = None
    max_inflight: int = 1


def _load_lck() -> ModuleType | None:
    try:
        return importlib.import_module("liger_cute_kernels")
    except ModuleNotFoundError as exc:
        if exc.name != "liger_cute_kernels":
            raise
        return None


def configure(
    *,
    process_groups: Mapping[str, ProcessGroup] | None = None,
    bootstrap_group: ProcessGroup | None = None,
    device: Device | str | None = None,
    flsce: FusedLinearCrossEntropyConfig | None = None,
    moe: MoEConfig | None = None,
) -> bool:
    """Collectively configure native resources using LCK's shared runtime API.

    Returns ``False`` with a warning when optional LCK is absent or its native
    kernels do not support the device, leaving non-LCK kernels available.
    Converts LCK's ``UnsupportedDeviceError`` into a warning and ``False``.
    Returns ``True`` after native setup succeeds. Other failures inside an
    installed LCK package propagate rather than silently disabling it.
    All bootstrap ranks must use a consistent installation.
    See ``liger_cute_kernels.configure`` for collective setup, named process
    groups, capacity, and lifetime requirements.
    """
    lck = _load_lck()
    if lck is None:
        warnings.warn(
            "liger-cute-kernels is not installed; native FLSCE/MoE configuration was skipped. "
            "Liger's non-LCK kernels remain available.",
            UserWarning,
            stacklevel=2,
        )
        return False
    # Older native packages have no typed eligibility error; their errors still propagate.
    unsupported_device_errors = (lck.UnsupportedDeviceError,) if hasattr(lck, "UnsupportedDeviceError") else ()
    try:
        lck.configure(
            process_groups=process_groups,
            bootstrap_group=bootstrap_group,
            device=device,
            flsce=lck.FusedLinearCrossEntropyConfig(**asdict(flsce)) if flsce is not None else None,
            moe=lck.MoEConfig(**asdict(moe)) if moe is not None else None,
        )
    except unsupported_device_errors as exc:
        warnings.warn(
            f"{exc}; native configuration was skipped. Liger's non-LCK kernels remain available.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True
