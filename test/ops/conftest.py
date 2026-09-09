"""Pytest fixtures for op-level cross-backend tests.

The ``available_backends_for`` fixture exposes the list of registered, currently
*satisfied* backends for a given op so individual test files can parametrize
without depending on whether cuTile/CuTe DSL is installed in the test env.

All tests in this directory require a CUDA device — we install a session-level
``autouse`` marker that skips when CUDA is unavailable, so individual test
files don't need to repeat the check.
"""

from __future__ import annotations

from typing import Callable
from typing import List

import pytest
import torch

# Importing the functional module triggers ``declare_op_locations`` so the
# dispatcher knows where to probe for impls — without this, every
# ``available_backends("rms_norm")`` call returns ``[]`` until the user
# imports ``liger_kernel.functional`` somewhere.
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends.dispatch import available_backends


def _available_backends_for(op_name: str) -> List[str]:
    """Return the list of backend names registered AND satisfied for ``op_name``.

    Used at collection time so the parametrize machinery can pre-compute
    backend lists for each op.
    """
    return available_backends(op_name)


# Module-level helper (used directly by ``test_rms_norm.py`` to parametrize).
def get_available_backends_for_op(op_name: str) -> List[str]:
    return _available_backends_for(op_name)


@pytest.fixture
def available_backends_for() -> Callable[[str], List[str]]:
    """Fixture returning a callable that, given an op name, returns the list
    of currently usable backends.
    """
    return _available_backends_for


@pytest.fixture(autouse=True)
def _require_cuda_for_ops_tests():
    """Auto-skip every test in this directory when CUDA isn't visible.

    Op-level tests dispatch into real Triton/cuTile kernels that need a GPU.
    The backend abstraction tests under ``test/backends/`` exercise the same
    machinery without GPUs.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for op-level kernel tests")
    yield
