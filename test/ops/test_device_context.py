"""Unit tests for ``liger_kernel.ops.utils.device_context``.

CPU-runnable: does not launch any kernel and never touches a real CUDA device.
It only exercises the context-manager guard logic, including the torch.compile
skip introduced for #1590.

Requires torch (installable CPU-only). triton is imported by the module under
test but no kernel is launched here.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from liger_kernel.ops.utils import device_context


class TestDeviceContext:
    def test_cpu_device_is_noop(self):
        """A CPU device has no backend.device guard, so this must not raise."""
        with device_context("cpu"):
            pass

    def test_compile_skip_skips_backend_device(self):
        """Under torch.compile, backend.device() must NOT be entered (#1590)."""
        fake_cuda = mock.MagicMock()
        fake_cuda.device = mock.MagicMock()
        with mock.patch("torch.cuda", fake_cuda), mock.patch.object(
            torch, "is_compiling", lambda: True, create=True
        ):
            with device_context("cuda:0"):
                pass
        fake_cuda.device.assert_not_called()

    def test_not_compile_uses_backend_device(self):
        """Outside torch.compile, backend.device() is entered exactly once."""
        fake_cuda = mock.MagicMock()
        fake_cuda.device = mock.MagicMock()
        with mock.patch("torch.cuda", fake_cuda), mock.patch.object(
            torch, "is_compiling", lambda: False, create=True
        ):
            with device_context("cuda:0"):
                pass
        fake_cuda.device.assert_called_once()

    def test_nonexistent_backend_is_noop(self):
        """A device type without a backend.device yields straight through."""
        with mock.patch.object(
            torch, "is_compiling", lambda: False, create=True
        ):
            with device_context("cpu"):
                pass
