"""cuTile ``_select_mode`` shape validation.

Moved here from ``test/backends/test_registry.py`` (PR-A) so the test lives in
the same PR that first ships the ``_cutile`` backend it imports.
"""

import pytest


class TestCutileSelectModeValidation:
    def _import_select_mode(self):
        try:
            from liger_kernel.ops.backends._cutile.rms_norm import _select_mode
        except Exception as e:
            pytest.skip(f"cuTile backend unimportable on this host: {e}")
        return _select_mode

    @pytest.mark.parametrize("n_rows,n_cols", [(0, 256), (256, 0), (-1, 256), (256, -1), (0, 0)])
    def test_zero_or_negative_dims_raise_valueerror(self, n_rows, n_cols):
        _select_mode = self._import_select_mode()
        with pytest.raises(ValueError) as ei:
            _select_mode(None, n_rows, n_cols)
        assert "invalid shape" in str(ei.value)

    def test_explicit_mode_skips_dim_check(self):
        _select_mode = self._import_select_mode()
        # Explicit mode is returned as-is, even with zero dims (caller's
        # responsibility). Since the multi-row perf push, the function
        # returns a (mode, tile_m) tuple — tile_m is 1 for non-persistent
        # modes and for explicit-mode calls (caller decides whether to use
        # the multi-row path).
        result = _select_mode("standard", 0, 0)
        if isinstance(result, tuple):
            assert result[0] == "standard"
        else:
            # Older cuTile RMSNorm impl (pre-PR-4) returned just the string.
            assert result == "standard"
