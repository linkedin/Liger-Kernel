"""Unit tests for ``liger_kernel.ops.utils.device_context``.

CPU-runnable: no kernel is launched and no real device is touched. The tests
drive the context manager directly, including the torch.compile skip added for
#1590, by patching the module-level ``_is_compiling`` seam instead of reaching
into ``torch.compiler`` internals.

Kept outside ``test/ops/`` on purpose: that directory has an autouse fixture
that skips every test when CUDA is unavailable, and these tests are meant to
run on CPU-only machines.
"""

from __future__ import annotations

from unittest import mock

from liger_kernel.ops import utils
from liger_kernel.ops.utils import device_context


class TestIsCompiling:
    def test_false_in_eager_mode(self):
        """Outside a compile region the helper reports False."""
        assert utils._is_compiling() is False


class TestDeviceContext:
    def test_cpu_device_is_noop(self):
        """A CPU device has no backend.device guard, so this must not raise."""
        with device_context("cpu"):
            pass

    def test_compile_skip_skips_backend_device(self):
        """Under torch.compile, backend.device() must NOT be entered (#1590)."""
        fake_cuda = mock.MagicMock()
        with mock.patch("torch.cuda", fake_cuda), mock.patch.object(utils, "_is_compiling", lambda: True):
            with device_context("cuda:0"):
                pass
        fake_cuda.device.assert_not_called()

    def test_not_compile_uses_backend_device(self):
        """Outside torch.compile, backend.device() is entered exactly once."""
        fake_cuda = mock.MagicMock()
        with mock.patch("torch.cuda", fake_cuda), mock.patch.object(utils, "_is_compiling", lambda: False):
            with device_context("cuda:0"):
                pass
        fake_cuda.device.assert_called_once()

    def test_backend_without_device_is_noop(self):
        """A backend object with no ``device`` attribute yields straight through."""
        with mock.patch.object(utils, "_is_compiling", lambda: False):
            with device_context("meta:0"):
                pass
