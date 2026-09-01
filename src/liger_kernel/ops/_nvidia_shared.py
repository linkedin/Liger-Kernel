"""Helpers shared across NVIDIA DSL implementations (cuTile + CuTeDSL).

A single source of truth for utilities that were previously copy-pasted in
each ``ops/backends/_cutile/*.py`` and ``ops/backends/_cutedsl/*.py`` file:

- :func:`next_pow2` — smallest power of two >= n.
- :func:`num_sms` — multiprocessor count of the current CUDA device.
- Casting-mode constants and :data:`STR_TO_CASTING_MODE` — Llama / Gemma /
  None reduction-precision modes shared by RMSNorm forward kernels.

These helpers are *NVIDIA-wide* (CUDA-only). Per-impl heuristics like
``_select_mode`` stay in their respective files because the mode space
differs by DSL (cuTile has three persistent variants, CuTeDSL has one,
Triton dispatches by ``BLOCK_SIZE`` / ``num_warps``).
"""

from __future__ import annotations

import logging

import torch

_log = logging.getLogger(__name__)

__all__ = [
    "CASTING_MODE_GEMMA",
    "CASTING_MODE_LLAMA",
    "CASTING_MODE_NONE",
    "STR_TO_CASTING_MODE",
    "cutile_compiler_available",
    "is_dtensor",
    "next_pow2",
    "num_sms",
    "to_local_if_dtensor",
]


# Casting-mode integer constants. Mirror Liger's Triton enum exactly so
# ctx-saved values from one impl's forward could in principle be replayed by
# another's backward. We don't rely on this — but keeping them aligned avoids
# surprises during cross-impl validation.
CASTING_MODE_NONE: int = -1
CASTING_MODE_LLAMA: int = 0
CASTING_MODE_GEMMA: int = 1

STR_TO_CASTING_MODE: dict[str, int] = {
    "none": CASTING_MODE_NONE,
    "llama": CASTING_MODE_LLAMA,
    "gemma": CASTING_MODE_GEMMA,
}


def next_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (``n >= 1``)."""
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def num_sms(device: int | torch.device | None = 0) -> int:
    """Multiprocessor count of the named CUDA device (default: ``cuda:0``)."""
    return torch.cuda.get_device_properties(device).multi_processor_count


# DTensor lives at ``torch.distributed.tensor.DTensor`` from torch 2.5+ but is
# not bound as an attribute of ``torch.distributed`` until imported (lazy
# submodule). Resolve once at module load so per-kernel hot paths just do an
# attribute check / isinstance, and so we don't blow up on torch builds without
# the distributed package.
try:
    from torch.distributed.tensor import DTensor as _DTensor  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover — torch built without distributed.
    _DTensor = None  # type: ignore[assignment]


def is_dtensor(t) -> bool:
    """Return True iff ``t`` is a ``torch.distributed.tensor.DTensor``.

    Cheap and safe on torch builds that don't ship ``torch.distributed.tensor``
    (returns ``False`` without raising).
    """
    return _DTensor is not None and isinstance(t, _DTensor)


def to_local_if_dtensor(t):
    """Materialize a DTensor's full tensor; pass-through for plain tensors.

    Used by every backend ``Function.forward`` / ``.backward`` to flatten a
    sharded autograd input before launching a CUDA kernel. (We don't yet have
    sharded kernel paths; the materialize-on-entry fallback keeps correctness
    while still letting callers use Liger from inside a DTensor pipeline.)
    """
    return t.full_tensor() if is_dtensor(t) else t


def cutile_compiler_available() -> bool:
    """Return True when the ``tileiras`` compiler used by ``cuda.tile`` is reachable.

    The ``cuda.tile`` Python package and the ``tileiras`` compiler binary ship
    separately. On some hosts the runtime is installed (``import cuda.tile``
    succeeds) but the compiler isn't (no ``cuda-tile[tileiras]`` extras and no
    CTK 13.1+ ``tileiras`` on ``PATH``), so a JIT compile only fails at first
    kernel launch. Use this as a ``Capability(predicate=...)`` to gate the cuTile
    impl out at registration time and fall back to Triton gracefully.
    """
    try:
        from cuda.tile import _compile as _ct_compile
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception as e:  # broken cuda.tile install — log so it's diagnosable
        _log.debug("cuTile compiler probe: unexpected error importing cuda.tile._compile: %r", e)
        return False
    try:
        _ct_compile._find_compiler_bin()
        return True
    except (FileNotFoundError, OSError):
        # Expected when tileiras binary is absent / not on PATH.
        return False
    except Exception as e:
        _log.debug("cuTile compiler probe: _find_compiler_bin raised: %r", e)
        return False
