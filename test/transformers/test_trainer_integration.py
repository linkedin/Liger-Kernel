from unittest.mock import Mock, patch

from liger_kernel.transformers.trainer_integration import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    _apply_liger_kernel,
)


def test_apply_liger_kernel_no_supported_model_type():
    # Test that calling _apply_liger_kernel with an unsupported model type is a no-op
    mock_mistral = Mock()

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"mistral": mock_mistral}):
        _apply_liger_kernel("foobar")
        MODEL_TYPE_TO_APPLY_LIGER_FN["mistral"].assert_not_called()


def test_apply_liger_kernel_only_supported_model_type_called():
    # Test that liger kernel is applied only to the specified model
    mock_gemma = Mock()
    mock_llama = Mock()
    mock_mistral = Mock()
    mock_mixtral = Mock()
    mock_phi3 = Mock()

    with patch.dict(
        MODEL_TYPE_TO_APPLY_LIGER_FN,
        {"gemma": mock_gemma, "llama": mock_llama, "mistral": mock_mistral, "mixtral": mock_mixtral, "phi3": mock_phi3},
    ):
        _apply_liger_kernel("llama")
        mock_llama.assert_called_once()
        mock_gemma.assert_not_called()
        mock_mistral.assert_not_called()
        mock_mixtral.assert_not_called()
        mock_phi3.assert_not_called()


def test_apply_liger_kernel_passes_kwargs():
    # Test that keyword args are correctly passed to apply_liger_* function
    mock_llama = Mock()

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}):
        _apply_liger_kernel(
            "llama", rope=False, cross_entropy=True, fused_linear_cross_entropy=False
        ),
        mock_llama.assert_called_once_with(
            rope=False, cross_entropy=True, fused_linear_cross_entropy=False
        )
