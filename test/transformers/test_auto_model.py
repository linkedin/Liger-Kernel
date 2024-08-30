from inspect import signature
from unittest import mock
from unittest.mock import MagicMock, patch

from transformers import AutoConfig, AutoModelForCausalLM

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.monkey_patch import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    apply_liger_kernel_to_llama,
)


def test_auto_liger_kernel_for_causal_lm_from_pretrained():
    pretrained_model_name_or_path = "/path/to/llama/model"
    model_args = ("model_arg1", "model_arg2")

    valid_kwargs = {
        "valid_arg_1": "some_value_1",
        "valid_arg_2": 10,
    }

    # This arg should be filtered out as it is not part of the model config
    invalid_kwargs = {
        "invalid_arg": "another_value",
    }

    # These args should be passed through to apply_liger_kernel_to_llama fn
    apply_liger_kernel_kwargs = {
        "rope": False,
        "swiglu": True,
    }

    kwargs = {**valid_kwargs, **invalid_kwargs, **apply_liger_kernel_kwargs}

    # Mock the model config instance returned from AutoConfig.from_pretrained()
    mock_model_config = MagicMock()
    mock_model_config.__dict__ = {
        "model_type": "llama",
        "valid_arg_1": "",
        "valid_arg_2": 0,
    }
    mock_llama = mock.Mock()

    with patch.dict(
        MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}
    ), mock.patch.object(
        AutoConfig, "from_pretrained", return_value=mock_model_config
    ), mock.patch.object(
        AutoModelForCausalLM, "from_pretrained", return_value="mock_model"
    ) as mock_super_from_pretrained:

        # Mock the function signature of apply_liger_kernel_to_llama
        mock_llama.__signature__ = signature(apply_liger_kernel_to_llama)

        model = AutoLigerKernelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Check that the apply_liger_kernel_to_llama mock was called with the correct kwargs
        mock_llama.assert_called_once_with(rope=False, swiglu=True)
        # Check that only valid kwargs are passed to super().from_pretrained
        mock_super_from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path, *model_args, **valid_kwargs
        )
        assert model == "mock_model"
