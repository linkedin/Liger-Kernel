from inspect import signature
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import patch

from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama


def test_auto_liger_kernel_for_causal_lm_from_pretrained():
    pretrained_model_name_or_path = "/path/to/llama/model"
    model_args = ("model_arg1", "model_arg2")

    original_kwargs = {
        "valid_arg_1": "some_value_1",
        "valid_arg_2": 10,
    }

    # These args should be passed through to apply_liger_kernel_to_llama fn
    apply_liger_kernel_kwargs = {
        "rope": False,
        "swiglu": True,
    }

    kwargs = {**original_kwargs, **apply_liger_kernel_kwargs}

    # Mock the model config instance returned from AutoConfig.from_pretrained()
    mock_model_config = MagicMock()
    mock_model_config.model_type = "llama"
    mock_llama = mock.Mock()

    with (
        patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}),
        mock.patch.object(AutoConfig, "from_pretrained", return_value=mock_model_config),
        mock.patch.object(
            AutoModelForCausalLM, "from_pretrained", return_value="mock_model"
        ) as mock_super_from_pretrained,
    ):
        # Mock the function signature of apply_liger_kernel_to_llama
        mock_llama.__signature__ = signature(apply_liger_kernel_to_llama)

        model = AutoLigerKernelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Check that the apply_liger_kernel_to_llama mock was called with the correct kwargs
        mock_llama.assert_called_once_with(rope=False, swiglu=True)
        # Check that the original kwargs are passed to super().from_pretrained
        mock_super_from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path, *model_args, **original_kwargs
        )
        assert model == "mock_model"


def test_auto_liger_kernel_for_causal_lm_from_config():
    original_kwargs = {
        "valid_arg_1": "some_value_1",
        "valid_arg_2": 10,
    }

    # These args should be passed through to apply_liger_kernel_to_llama fn
    apply_liger_kernel_kwargs = {
        "rope": False,
        "swiglu": True,
    }

    kwargs = {**original_kwargs, **apply_liger_kernel_kwargs}

    # Mock the model config instance returned from AutoConfig.from_pretrained()
    mock_model_config = MagicMock()
    mock_model_config.model_type = "llama"
    mock_llama = mock.Mock()

    with (
        patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}),
        mock.patch.object(AutoModelForCausalLM, "from_config", return_value="mock_model") as mock_super_from_config,
    ):
        # Mock the function signature of apply_liger_kernel_to_llama
        mock_llama.__signature__ = signature(apply_liger_kernel_to_llama)

        model = AutoLigerKernelForCausalLM.from_config(mock_model_config, **kwargs)

        # Check that the apply_liger_kernel_to_llama mock was called with the correct kwargs
        mock_llama.assert_called_once_with(rope=False, swiglu=True)
        # Check that the original kwargs are passed to super().from_pretrained
        mock_super_from_config.assert_called_once_with(mock_model_config, **original_kwargs)
        assert model == "mock_model"


def test_auto_liger_kernel_for_causal_lm_from_pretrained_unsupported_model_type():
    # Model types without a Liger patching fn must still load, just unpatched.
    pretrained_model_name_or_path = "/path/to/unsupported/model"
    model_args = ("model_arg1", "model_arg2")
    kwargs = {"valid_arg_1": "some_value_1", "valid_arg_2": 10}

    mock_model_config = MagicMock()
    mock_model_config.model_type = "unsupported_model_type"

    with (
        mock.patch.object(AutoConfig, "from_pretrained", return_value=mock_model_config),
        mock.patch.object(
            AutoModelForCausalLM, "from_pretrained", return_value="mock_model"
        ) as mock_super_from_pretrained,
    ):
        assert "unsupported_model_type" not in MODEL_TYPE_TO_APPLY_LIGER_FN

        model = AutoLigerKernelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # No Liger kwargs to strip, so everything is forwarded untouched
        mock_super_from_pretrained.assert_called_once_with(pretrained_model_name_or_path, *model_args, **kwargs)
        assert model == "mock_model"


def test_auto_liger_kernel_for_causal_lm_from_config_unsupported_model_type():
    kwargs = {"valid_arg_1": "some_value_1", "valid_arg_2": 10}

    mock_model_config = MagicMock()
    mock_model_config.model_type = "unsupported_model_type"

    with mock.patch.object(AutoModelForCausalLM, "from_config", return_value="mock_model") as mock_super_from_config:
        assert "unsupported_model_type" not in MODEL_TYPE_TO_APPLY_LIGER_FN

        model = AutoLigerKernelForCausalLM.from_config(mock_model_config, **kwargs)

        mock_super_from_config.assert_called_once_with(mock_model_config, **kwargs)
        assert model == "mock_model"


def test_auto_liger_kernel_for_causal_lm_from_config_without_model_type():
    # An undeterminable model type means "no Liger kernels", not "no model".
    kwargs = {"valid_arg_1": "some_value_1"}

    mock_model_config = MagicMock()
    mock_model_config.model_type = ""

    with mock.patch.object(AutoModelForCausalLM, "from_config", return_value="mock_model") as mock_super_from_config:
        model = AutoLigerKernelForCausalLM.from_config(mock_model_config, **kwargs)

        mock_super_from_config.assert_called_once_with(mock_model_config, **kwargs)
        assert model == "mock_model"
