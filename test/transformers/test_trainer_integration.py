import pytest


def test_import():
    try:
        from liger_kernel.transformers.trainer_integration import _apply_liger_kernel  # noqa: F401
    except Exception:
        pytest.fail("Import _apply_liger_kernel fails")
