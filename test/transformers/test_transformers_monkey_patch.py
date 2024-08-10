import pytest


def test_import_from_root():
    try:
        from liger_kernel.transformers import (  # noqa: F401
            apply_liger_kernel_to_gemma,
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_mistral,
            apply_liger_kernel_to_mixtral,
        )
    except Exception:
        pytest.fail("Import kernel patch from root fails")
