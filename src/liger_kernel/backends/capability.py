"""Declarative capability gates for backend implementations.

A :class:`Capability` describes what a kernel impl needs in order to run on the
current process and device. Capabilities are evaluated every time ``is_satisfied()``
is called so that environment-dependent gates (compiler-on-path, runtime feature
flags) stay live. The cost is negligible: module imports hit ``sys.modules`` after
the first call, and the CUDA capability lookup is O(1).

Design notes:

- Compute-capability gates are expressed as ``(major, minor)`` tuples (e.g.,
  ``(10, 0)`` = sm_100 = Blackwell). This matches
  :func:`torch.cuda.get_device_capability`. They are **CUDA-only**: on ROCm
  that call returns the AMD GPU ISA version, which collides with NVIDIA sm
  numbers (gfx90a reports ``(9, 0)``, same as an H100).
- Arch gates (``archs``) take family names from
  :func:`liger_kernel.utils.infer_device_arch` (``"hopper"``, ``"cdna3"``,
  ``"ascend910"``, ...). Vendor-neutral; mutually exclusive with
  ``min_cc``/``max_cc``.
- Module gates accept a list of importable module names. We try ``importlib``
  and treat ``ImportError`` as unsatisfied. ``ModuleNotFoundError`` is a
  subclass — handled the same way.
- The optional ``predicate`` is the escape hatch for runtime-detected features
  ("cuda.tile claims it supports Hopper"). It is called only if all other
  gates pass.
- A capability with no fields satisfies everything — useful for PyTorch-only
  reference impls.

We do NOT block on CUDA version by default. CUDA mismatches surface as import
errors on the underlying DSL package, which we already catch.
"""

from __future__ import annotations

import importlib
import logging

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

logger = logging.getLogger(__name__)


def _device_index(device) -> Optional[int]:
    """Accelerator index for ``device`` (``None`` = current); ``None`` for a CPU."""
    import torch

    if isinstance(device, int):
        return device
    if device is not None:
        dev = torch.device(device)
        if dev.type == "cpu":
            return None
        if dev.index is not None:
            return dev.index
    try:
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    except Exception:  # pragma: no cover - broken driver
        return 0


@dataclass(frozen=True)
class Capability:
    """Predicate describing when a backend impl can run."""

    min_cc: Optional[Tuple[int, int]] = None
    """Minimum CUDA compute capability, inclusive. ``None`` = no lower bound. CUDA-only."""

    max_cc: Optional[Tuple[int, int]] = None
    """Maximum CUDA compute capability, inclusive. ``None`` = no upper bound. CUDA-only."""

    modules: List[str] = field(default_factory=list)
    """Python module names that must import successfully."""

    requires_cuda: bool = False
    """If True, requires a CUDA device to be visible."""

    archs: Tuple[str, ...] = ()
    """Accepted arch families from ``infer_device_arch()``, e.g. ``("cdna3",)``. Empty = any."""

    predicate: Optional[Callable[[], bool]] = None
    """Optional ad-hoc check. Called only if the other gates pass."""

    def __post_init__(self):
        if self.archs and (self.min_cc is not None or self.max_cc is not None):
            raise ValueError("archs is mutually exclusive with min_cc/max_cc (cc gates are CUDA-only)")

    def is_satisfied(self, device=None) -> bool:
        ok, _ = self._evaluate(device)
        return ok

    def reason_if_unsatisfied(self, device=None) -> Optional[str]:
        """Return a human-readable reason if the capability does not hold.

        ``device`` (an int index, ``str``, ``torch.device``, or ``None``) selects
        which GPU's compute capability is checked. ``None`` preserves the legacy
        behavior of querying the current process device.
        """
        ok, reason = self._evaluate(device)
        return None if ok else reason

    def _evaluate(self, device=None) -> Tuple[bool, Optional[str]]:
        # Module imports first (fast, deterministic, doesn't need a GPU).
        # GPU libraries can fail with non-ImportError exceptions (OSError from
        # missing .so, RuntimeError from a broken CUDA runtime, etc.); treat
        # any failure here as "module unavailable" so the capability gates the
        # impl out instead of crashing dispatch.
        for mod in self.modules:
            try:
                importlib.import_module(mod)
            except ImportError as e:
                return False, f"module '{mod}' not importable: {e}"
            except Exception as e:
                return False, f"module '{mod}' import raised {type(e).__name__}: {e}"

        needs_cc = self.min_cc is not None or self.max_cc is not None
        if self.requires_cuda or needs_cc:
            try:
                import torch
            except ImportError as e:  # pragma: no cover — torch is a hard dep
                return False, f"torch unavailable: {e}"
            if not torch.cuda.is_available():
                return False, "CUDA device not available"

            if needs_cc:
                # On ROCm this returns the gfx ISA version, which collides with
                # NVIDIA sm values (gfx90a -> (9, 0), same as an H100).
                if torch.version.hip is not None:
                    return False, "compute-capability gates are CUDA-only; this is a ROCm build"

                cc = torch.cuda.get_device_capability(device)
                if self.min_cc is not None and cc < self.min_cc:
                    return False, f"device sm_{cc[0]}{cc[1]} < required sm_{self.min_cc[0]}{self.min_cc[1]}"
                if self.max_cc is not None and cc > self.max_cc:
                    return False, f"device sm_{cc[0]}{cc[1]} > supported sm_{self.max_cc[0]}{self.max_cc[1]}"

        if self.archs:
            # Lazy import keeps this module torch-free. Outside the CUDA block
            # so NPU/XPU impls can gate themselves.
            from liger_kernel.utils import infer_device_arch

            index = _device_index(device)
            if index is None:
                return False, "arch gate needs an accelerator; selector names a CPU"
            arch = infer_device_arch(index)
            if arch not in self.archs:
                return False, f"device arch '{arch}' not in {sorted(self.archs)}"

        if self.predicate is not None:
            try:
                if not self.predicate():
                    return False, "custom predicate returned False"
            except Exception as e:
                return False, f"custom predicate raised: {e}"

        return True, None


PYTORCH_FALLBACK = Capability()
"""A trivially-satisfied capability — useful for reference impls."""
