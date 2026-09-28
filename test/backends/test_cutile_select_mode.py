"""cuTile ``_select_mode`` shape/width validation.

These are pure host-side selector tests (no kernel launch), so they run on any
box where the ``_cutile`` module *imports* — CUDA is not required. Only a
genuine import failure (e.g. the ``cuda.tile`` package is absent) is skipped;
real regressions in the selector surface as failures rather than skips.
"""

import pytest
import torch


def _import(module, name):
    try:
        mod = __import__(
            f"liger_kernel.ops.backends._cutile.{module}",
            fromlist=[name],
        )
    except ImportError as e:
        pytest.skip(f"cuTile backend unimportable on this host: {e}")
    return getattr(mod, name)


# A device descriptor is required by the new signature but is only *used* on the
# auto path (mode=None) for the SM-count heuristic. The zero/negative-dim and
# explicit-mode cases below return/raise before touching it, so a descriptor for
# a possibly-absent GPU is fine.
_DEV = torch.device("cuda:0")


class TestRmsNormSelectMode:
    @pytest.mark.parametrize("n_rows,n_cols", [(0, 256), (256, 0), (-1, 256), (256, -1), (0, 0)])
    def test_zero_or_negative_dims_raise(self, n_rows, n_cols):
        _select_mode = _import("rms_norm", "_select_mode")
        with pytest.raises(ValueError) as ei:
            _select_mode(None, n_rows, n_cols, _DEV)
        assert "invalid shape" in str(ei.value)

    def test_explicit_mode_returned_as_is(self):
        _select_mode = _import("rms_norm", "_select_mode")
        # Explicit mode is returned verbatim (caller owns dim validation) and is
        # a plain string — the obsolete (mode, tile_m) tuple contract is gone.
        assert _select_mode("standard", 0, 0, _DEV) == "standard"


class TestLayerNormSelectMode:
    @pytest.mark.parametrize("n_rows,n_cols", [(0, 256), (256, 0), (0, 0)])
    def test_zero_or_negative_dims_raise(self, n_rows, n_cols):
        _select_mode = _import("layer_norm", "_select_mode")
        with pytest.raises(ValueError) as ei:
            _select_mode(None, n_rows, n_cols, _DEV)
        assert "invalid shape" in str(ei.value)

    def test_explicit_mode_returned_as_is(self):
        _select_mode = _import("layer_norm", "_select_mode")
        assert _select_mode("static_persistent", 0, 0, _DEV) == "static_persistent"


class TestSoftmaxSelectMode:
    @pytest.mark.parametrize("n_rows,n_cols", [(0, 256), (256, 0), (0, 0)])
    def test_zero_or_negative_dims_raise(self, n_rows, n_cols):
        _select_mode = _import("softmax", "_select_mode")
        with pytest.raises(ValueError) as ei:
            _select_mode(None, n_rows, n_cols, _DEV)
        assert "invalid shape" in str(ei.value)

    def test_explicit_single_tile_mode_rejects_wide_row(self):
        _select_mode = _import("softmax", "_select_mode")
        # next_pow2(9000) = 16384 > 8192 → the single-tile modes must reject it
        # instead of compiling/launching an oversized tile.
        for mode in ("standard", "static_persistent"):
            with pytest.raises(ValueError) as ei:
                _select_mode(mode, 8, 9000, _DEV)
            assert "next_pow2" in str(ei.value)

    def test_explicit_chunked_mode_accepts_wide_row(self):
        _select_mode = _import("softmax", "_select_mode")
        # chunked streams arbitrarily wide rows, so it is not rejected.
        assert _select_mode("chunked", 8, 9000, _DEV) == "chunked"

    def test_explicit_single_tile_mode_accepts_supported_width(self):
        _select_mode = _import("softmax", "_select_mode")
        assert _select_mode("standard", 8, 8192, _DEV) == "standard"
