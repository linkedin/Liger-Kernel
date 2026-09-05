"""CPU-only checks for CuTe DSL RMSNorm fast-path launch policy."""

import importlib.util

from pathlib import Path

import pytest

_HELPERS_PATH = Path(__file__).parents[2] / "src/liger_kernel/ops/cutedsl/ops/rms_norm_fastpath.py"
_spec = importlib.util.spec_from_file_location("rms_norm_fastpath_test_helpers", _HELPERS_PATH)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
fast_path_vector_width = _helpers.fast_path_vector_width
backward_warp_count = _helpers.backward_warp_count


def test_fast_path_vector_width_uses_largest_participating_element():
    assert fast_path_vector_width(2, 2) == 8
    assert fast_path_vector_width(2, 4, 2) == 4
    assert fast_path_vector_width(4, 4) == 4


def test_fast_path_vector_width_rejects_non_vectorizable_size():
    with pytest.raises(ValueError):
        fast_path_vector_width(3)


@pytest.mark.parametrize(
    ("n_cols", "expected"),
    [
        (512, 4),
        (1024, 4),
        (1025, 8),
        (2048, 8),
        (4096, 8),
        (4097, 16),
        (8192, 16),
    ],
)
def test_backward_warp_count(n_cols, expected):
    assert backward_warp_count(n_cols) == expected
