import pytest


def test_import():
    try:
        from liger_kernel.transformers.trainer_integration import (  # noqa: F401
            _apply_liger_kernel,
        )
    except Exception:
        pytest.fail("Import _apply_liger_kernel fails")
