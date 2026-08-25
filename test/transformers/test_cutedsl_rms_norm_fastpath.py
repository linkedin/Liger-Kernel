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
fwd_warp_count = _helpers.fwd_warp_count


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


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "vec", "expected"),
    [
        (1024, 1024, 8, 4),
        (1024, 2048, 8, 8),
        (2048, 1024, 8, 4),
        (2048, 4096, 8, 4),
        (4096, 1024, 4, 2),
        (4096, 2048, 8, 4),
        (8192, 2048, 8, 2),
        (8192, 2048, 4, 4),
        (8192, 4096, 4, 8),
    ],
)
def test_hopper_forward_warp_count_uses_row_parallelism(n_rows, n_cols, vec, expected):
    assert fwd_warp_count(n_cols, vec, n_rows, sm90=True) == expected


def test_forward_warp_count_preserves_width_policy_off_hopper():
    assert fwd_warp_count(2048, 8, 8192, sm90=False) == 4
    assert fwd_warp_count(4096, 8, 8192, sm90=False) == 8
