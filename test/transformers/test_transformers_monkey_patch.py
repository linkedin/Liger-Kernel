from unittest.mock import Mock, patch

import pytest

from liger_kernel.transformers import apply_liger_kernel
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN


def test_import_from_root():
    try:
        from liger_kernel.transformers import (  # noqa: F401
            apply_liger_kernel,
            apply_liger_kernel_to_gemma,
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_mistral,
            apply_liger_kernel_to_mixtral,
        )
    except Exception:
        pytest.fail("Import kernel patch from root fails")


def test_apply_liger_kernel_no_supported_model_type():
    # Test that calling apply_liger_kernel with an unsupported model type is a no-op
    mock_mistral = Mock()

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"mistral": mock_mistral}):
        apply_liger_kernel("foobar")
        MODEL_TYPE_TO_APPLY_LIGER_FN["mistral"].assert_not_called()


def test_apply_liger_kernel_only_supported_model_type_called():
    # Test that liger kernel is applied only to the specified model
    mock_gemma = Mock()
    mock_llama = Mock()
    mock_mistral = Mock()

    with patch.dict(
        MODEL_TYPE_TO_APPLY_LIGER_FN,
        {"gemma": mock_gemma, "llama": mock_llama, "mistral": mock_mistral},
    ):
        apply_liger_kernel("llama")
        mock_llama.assert_called_once()
        mock_gemma.assert_not_called()
        mock_mistral.assert_not_called()


def test_apply_liger_kernel_passes_kwargs():
    # Test that keyword args are correctly passed to apply_liger_* function
    mock_llama = Mock()

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}):
        apply_liger_kernel(
            "llama", rope=False, cross_entropy=True, fused_linear_cross_entropy=False
        ),
        mock_llama.assert_called_once_with(
            rope=False, cross_entropy=True, fused_linear_cross_entropy=False
        )
