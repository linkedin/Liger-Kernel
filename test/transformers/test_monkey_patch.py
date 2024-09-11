import inspect
from inspect import signature
from unittest.mock import Mock, patch, MagicMock
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM
from transformers.models.llama import modeling_llama, configuration_llama
from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP, liger_rotary_pos_emb
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
import torch
import pytest

from liger_kernel.transformers import monkey_patch
from liger_kernel.transformers.monkey_patch import (
    MODEL_TYPE_TO_APPLY_LIGER_FN,
    _apply_liger_kernel,
    _apply_liger_kernel_to_instance,
)


def test_import_from_root():
    try:
        from liger_kernel.transformers import (  # noqa: F401
            AutoLigerKernelForCausalLM,
            apply_liger_kernel_to_gemma,
            apply_liger_kernel_to_gemma2,
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_mistral,
            apply_liger_kernel_to_mixtral,
            apply_liger_kernel_to_phi3,
            apply_liger_kernel_to_qwen2,
            apply_liger_kernel_to_qwen2_vl,
        )
    except Exception:
        pytest.fail("Import kernel patch from root fails")


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

    with patch.dict(
        MODEL_TYPE_TO_APPLY_LIGER_FN,
        {"gemma": mock_gemma, "llama": mock_llama, "mistral": mock_mistral},
    ):
        _apply_liger_kernel("llama")
        mock_llama.assert_called_once()
        mock_gemma.assert_not_called()
        mock_mistral.assert_not_called()


def test_apply_liger_kernel_only_passes_valid_kwargs():
    # Test that keyword args that are not valid for the apply_liger_* function are not passed
    mock_llama = Mock()

    def dummy_apply_liger_kernal_to_llama(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
    ):
        pass

    apply_liger_kernal_to_llama_sig = signature(dummy_apply_liger_kernal_to_llama)

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}):
        mock_llama.__signature__ = apply_liger_kernal_to_llama_sig
        _apply_liger_kernel(
            "llama",
            rope=False,
            fused_linear_cross_entropy=False,
            cross_entropy=True,
            foobar=True,
            barbaz=False,
        ),
        mock_llama.assert_called_once()
        mock_llama.assert_called_once_with(
            rope=False,
            fused_linear_cross_entropy=False,
            cross_entropy=True,
        )


def test_patching_apis_match_auto_mapping():
    # Test that all of the patching APIs present also have a corresponding entry in the auto mapping
    patching_functions = [
        func
        for name, func in inspect.getmembers(monkey_patch, inspect.isfunction)
        if name.startswith("apply_liger_kernel_to_")
    ]

    assert set(patching_functions) == set(MODEL_TYPE_TO_APPLY_LIGER_FN.values())


def test_apply_liger_kernel_to_instance_for_llama():
    with patch("transformers.models.llama.modeling_llama") as mock_modeling_llama:

        # Instantiate a dummy model with a llama configuration
        config = configuration_llama.LlamaConfig(
            model_type="llama",
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert not isinstance(dummy_model_instance.model.norm, LigerRMSNorm)
        for layer in dummy_model_instance.model.layers:
            assert not isinstance(layer.mlp, LigerSwiGLUMLP)
            assert not isinstance(layer.input_layernorm, LigerRMSNorm)
            assert not isinstance(layer.post_attention_layernorm, LigerRMSNorm)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert isinstance(dummy_model_instance.model.norm, LigerRMSNorm)
        for layer in dummy_model_instance.model.layers:
            assert isinstance(layer.mlp, LigerSwiGLUMLP)
            assert isinstance(layer.input_layernorm, LigerRMSNorm)
            assert isinstance(layer.post_attention_layernorm, LigerRMSNorm)




