import inspect

from inspect import signature
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
import transformers

from packaging import version
from transformers import AutoModelForCausalLM
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from liger_kernel.transformers import LigerBlockSparseTop2MLP
from liger_kernel.transformers import LigerGEGLUMLP
from liger_kernel.transformers import LigerPhi3SwiGLUMLP
from liger_kernel.transformers import LigerQwen3MoeSwiGLUMLP
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers import LigerSwiGLUMLP
from liger_kernel.transformers import monkey_patch
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

# Import transformer version check
transformer_version = version.parse(transformers.__version__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"

# Import forward functions based on transformer version
if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
    from liger_kernel.transformers.model.gemma import lce_forward as gemma_lce_forward
    from liger_kernel.transformers.model.gemma2 import lce_forward as gemma2_lce_forward
    from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
    from liger_kernel.transformers.model.mistral import lce_forward as mistral_lce_forward
    from liger_kernel.transformers.model.mixtral import lce_forward as mixtral_lce_forward
    from liger_kernel.transformers.model.mllama import lce_forward as mllama_lce_forward
    from liger_kernel.transformers.model.paligemma import lce_forward as paligemma_lce_forward
    from liger_kernel.transformers.model.phi3 import lce_forward as phi3_lce_forward
    from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
    from liger_kernel.transformers.model.smollm3 import lce_forward as smolllm3_lce_forward
else:
    from liger_kernel.transformers.model.gemma import lce_forward_deprecated as gemma_lce_forward
    from liger_kernel.transformers.model.gemma2 import lce_forward_deprecated as gemma2_lce_forward
    from liger_kernel.transformers.model.llama import lce_forward_deprecated as llama_lce_forward
    from liger_kernel.transformers.model.mistral import (
        lce_forward as mistral_lce_forward,  # mistral doesn't have deprecated version
    )
    from liger_kernel.transformers.model.mixtral import lce_forward_deprecated as mixtral_lce_forward
    from liger_kernel.transformers.model.mllama import lce_forward_deprecated as mllama_lce_forward
    from liger_kernel.transformers.model.paligemma import lce_forward_deprecated as paligemma_lce_forward
    from liger_kernel.transformers.model.phi3 import lce_forward_deprecated as phi3_lce_forward
    from liger_kernel.transformers.model.qwen2 import lce_forward_deprecated as qwen2_lce_forward


# Check if optional modules are available
def is_mllama_available():
    try:
        import transformers.models.mllama  # noqa: F401

        return True
    except ImportError:
        return False


def is_llama4_available():
    try:
        import transformers.models.llama4  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen3_available():
    try:
        import transformers.models.qwen3  # noqa: F401

        return True
    except ImportError:
        return False


def is_smollm3_available():
    try:
        import transformers.models.smollm3  # noqa: F401

        return True
    except ImportError:
        return False


def is_olmo2_available():
    try:
        import transformers.models.olmo2  # noqa: F401

        return True
    except ImportError:
        return False


def is_glm4_available():
    try:
        import transformers.models.glm4  # noqa: F401

        return True
    except ImportError:
        return False


def is_glm4v_available():
    try:
        import transformers.models.glm4v  # noqa: F401

        return True
    except ImportError:
        return False


def is_gemma3_available():
    try:
        import transformers.models.gemma3  # noqa: F401

        return True
    except ImportError:
        return False


def is_paligemma_available():
    try:
        import transformers.models.paligemma  # noqa: F401

        return True
    except ImportError:
        return False


def test_import_from_root():
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_gemma  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_glm4  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_llama  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_mistral  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_mllama  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_phi3  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_smollm3  # noqa: F401
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
        (
            _apply_liger_kernel(
                "llama",
                rope=False,
                fused_linear_cross_entropy=False,
                cross_entropy=True,
                foobar=True,
                barbaz=False,
            ),
        )
        mock_llama.assert_called_once()
        mock_llama.assert_called_once_with(
            rope=False,
            fused_linear_cross_entropy=False,
            cross_entropy=True,
        )


def test_apply_liger_kernel_to_instance_no_supported_model_type():
    # Test that calling _apply_liger_kernel_to_instance with an unsupported model type is a no-op
    mock_mistral = Mock()
    mock_unknown_model = MagicMock(spec=PreTrainedModel)
    mock_unknown_model.config = {"model_type": "foobar"}

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"mistral": mock_mistral}):
        _apply_liger_kernel_to_instance(model=mock_unknown_model)
        MODEL_TYPE_TO_APPLY_LIGER_FN["mistral"].assert_not_called()


def test_apply_liger_kernel_to_instance_only_supported_model_type_called():
    # Test that liger kernel is applied only to the specified model
    mock_gemma = Mock()
    mock_llama = Mock()
    mock_mistral = Mock()

    mock_llama_model_instance = MagicMock(spec=PreTrainedModel)
    mock_llama_model_instance.config = MagicMock(spec=PretrainedConfig)
    mock_llama_model_instance.config.model_type = "llama"

    with patch.dict(
        MODEL_TYPE_TO_APPLY_LIGER_FN,
        {"gemma": mock_gemma, "llama": mock_llama, "mistral": mock_mistral},
    ):
        _apply_liger_kernel_to_instance(model=mock_llama_model_instance)
        mock_llama.assert_called_once()
        mock_gemma.assert_not_called()
        mock_mistral.assert_not_called()


def test_apply_liger_kernel_to_instance_only_passes_valid_kwargs():
    # Test that keyword args that are not valid for the apply_liger_* function are not passed
    mock_llama = Mock()

    mock_llama_model_instance = MagicMock(spec=PreTrainedModel)
    mock_llama_model_instance.config = MagicMock(spec=PretrainedConfig)
    mock_llama_model_instance.config.model_type = "llama"

    def dummy_apply_liger_kernel_to_llama(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        pass

    apply_liger_kernel_to_llama_sig = signature(dummy_apply_liger_kernel_to_llama)

    with patch.dict(MODEL_TYPE_TO_APPLY_LIGER_FN, {"llama": mock_llama}):
        mock_llama.__signature__ = apply_liger_kernel_to_llama_sig
        (
            _apply_liger_kernel_to_instance(
                model=mock_llama_model_instance,
                rope=False,
                fused_linear_cross_entropy=False,
                cross_entropy=True,
                foobar=True,
                barbaz=False,
            ),
        )
        mock_llama.assert_called_once()
        mock_llama.assert_called_once_with(
            model=mock_llama_model_instance,
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


def test_patching_apis_support_patching_model_instance():
    # Test that all the patching APIs present support passing in
    # model (PreTrainedModel) as an argument indicating that it supports
    # patching post-model creation
    patching_functions = [
        func
        for name, func in inspect.getmembers(monkey_patch, inspect.isfunction)
        if name.startswith("apply_liger_kernel_to_")
    ]

    for func in patching_functions:
        sig = inspect.signature(func)
        # Ensure 'model' is in the parameters
        assert "model" in sig.parameters, (
            f"{func.__name__} does not have 'model' as an argument. All patching methods must support patching an existing model instance."
        )


def test_apply_liger_kernel_to_instance_for_llama():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.llama.modeling_llama"):
        # Instantiate a dummy model
        config = transformers.models.llama.configuration_llama.LlamaConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(llama_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(llama_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        # Ensure that the model patched with Liger modules can work properly
        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_mllama_available(), reason="mllama module not available")
def test_apply_liger_kernel_to_instance_for_mllama_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.mllama.modeling_mllama"):
        from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
        from transformers.models.mllama.modeling_mllama import MllamaTextModel

        # Instantiate a dummy model
        config = transformers.models.mllama.configuration_mllama.MllamaConfig(
            torch_dtype=torch.bfloat16,
            text_config=transformers.models.mllama.configuration_mllama.MllamaTextConfig(
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="silu",
                num_hidden_layers=2,
                rope_scaling=dict(
                    factor=8.0,
                    high_freq_factor=4.0,
                    low_freq_factor=1.0,
                    original_max_position_embeddings=8192,
                    rope_type="llama3",
                ),
            ),
            vision_config=transformers.models.mllama.configuration_mllama.MllamaVisionConfig(
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="gelu",
                num_hidden_layers=2,
                vision_output_dim=64,
            ),
        )
        dummy_model_instance = MllamaForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, MllamaForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(mllama_lce_forward)

        if isinstance(dummy_model_instance.language_model, MllamaTextModel):
            language_model = dummy_model_instance.language_model
        else:
            language_model = dummy_model_instance.language_model.model

        assert inspect.getsource(language_model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_pre.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_post.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        for layer in dummy_model_instance.vision_model.transformer.layers:
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerLayerNorm.forward
            )
        for layer in dummy_model_instance.vision_model.global_transformer.layers:
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerLayerNorm.forward
            )

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(mllama_lce_forward)
        assert inspect.getsource(language_model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_pre.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_post.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        for layer in dummy_model_instance.vision_model.transformer.layers:
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerLayerNorm.forward
            )
        for layer in dummy_model_instance.vision_model.global_transformer.layers:
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerLayerNorm.forward
            )
        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_mllama_available(), reason="mllama module not available")
def test_apply_liger_kernel_to_instance_for_mllama_for_causal_lm():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.mllama.modeling_mllama"):
        from transformers.models.mllama.modeling_mllama import MllamaForCausalLM

        # Instantiate a dummy model
        config = transformers.models.mllama.configuration_mllama.MllamaTextConfig(
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            rope_scaling=dict(
                factor=8.0,
                high_freq_factor=4.0,
                low_freq_factor=1.0,
                original_max_position_embeddings=8192,
                rope_type="llama3",
            ),
        )

        dummy_model_instance = MllamaForCausalLM._from_config(config)

        assert isinstance(dummy_model_instance, MllamaForCausalLM)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(mllama_lce_forward)
        assert not isinstance(dummy_model_instance.model.norm, LigerRMSNorm)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(mllama_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_llama4_available(), reason="llama4 module not available")
def test_apply_liger_kernel_to_instance_for_llama4_for_causal_lm():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.llama4.modeling_llama4"):
        from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

        # Instantiate a dummy model
        config = transformers.models.llama4.configuration_llama4.Llama4TextConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = Llama4ForCausalLM._from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.feed_forward.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.feed_forward.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_llama4_available(), reason="llama4 module not available")
def test_apply_liger_kernel_to_instance_for_llama4_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.llama4.modeling_llama4"):
        from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration

        # Instantiate a dummy model
        config = transformers.models.llama4.configuration_llama4.Llama4Config(
            torch_dtype=torch.bfloat16,
            text_config=transformers.models.llama4.configuration_llama4.Llama4TextConfig(
                torch_dtype=torch.bfloat16,
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="silu",
                num_hidden_layers=2,
            ),
            vision_config=transformers.models.llama4.configuration_llama4.Llama4VisionConfig(
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="gelu",
                num_hidden_layers=2,
                vision_output_dim=64,
            ),
        )
        dummy_model_instance = Llama4ForConditionalGeneration._from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert isinstance(dummy_model_instance, Llama4ForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.language_model.model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.model.layers:
            assert inspect.getsource(layer.feed_forward.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_pre.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_post.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        for layer in dummy_model_instance.vision_model.model.layers:
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerLayerNorm.forward
            )

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.language_model.model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.model.layers:
            assert inspect.getsource(layer.feed_forward.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_pre.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        assert inspect.getsource(dummy_model_instance.vision_model.layernorm_post.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        for layer in dummy_model_instance.vision_model.model.layers:
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerLayerNorm.forward
            )

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.49.0"),
    reason="fused linear cross entropy patch doesn't work on mistral in transformers<4.49.0",
)
def test_apply_liger_kernel_to_instance_for_mistral():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.mistral.modeling_mistral"):
        # Instantiate a dummy model
        config = transformers.models.mistral.configuration_mistral.MistralConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(mistral_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(mistral_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_mixtral():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.mixtral.modeling_mixtral"):
        # Instantiate a dummy model
        config = transformers.models.mixtral.configuration_mixtral.MixtralConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            num_local_experts=3,
            num_experts_per_tok=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(mixtral_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            for expert in layer.block_sparse_moe.experts:
                assert inspect.getsource(expert.forward) != inspect.getsource(LigerBlockSparseTop2MLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(mixtral_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            for expert in layer.block_sparse_moe.experts:
                assert inspect.getsource(expert.forward) == inspect.getsource(LigerBlockSparseTop2MLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_gemma():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.gemma.modeling_gemma"):
        # Instantiate a dummy model
        config = transformers.models.gemma.configuration_gemma.GemmaConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(gemma_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_gemma2():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.gemma2.modeling_gemma2"):
        # Instantiate a dummy model
        config = transformers.models.gemma2.configuration_gemma2.Gemma2Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma2_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(gemma2_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_paligemma_available(), reason="paligemma module not available")
def test_apply_liger_kernel_to_instance_for_paligemma():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.paligemma.modeling_paligemma"):
        from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

        # Instantiate a dummy model
        config = transformers.models.paligemma.configuration_paligemma.PaliGemmaConfig(
            torch_dtype=torch.bfloat16,
            text_config={
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "hidden_size": 32,
                "intermediate_size": 64,
                "hidden_act": "silu",
            },
            vision_config={
                "num_hidden_layers": 2,
                "layer_norm_eps": 1e-5,
                "hidden_size": 48,
                "intermediate_size": 64,
            },
        )

        dummy_model_instance = PaliGemmaForConditionalGeneration(config)
        assert isinstance(dummy_model_instance, PaliGemmaForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(paligemma_lce_forward)
        assert inspect.getsource(
            dummy_model_instance.vision_tower.vision_model.post_layernorm.forward
        ) != inspect.getsource(LigerLayerNorm.forward)

        for layer in dummy_model_instance.vision_tower.vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(paligemma_lce_forward)
        assert inspect.getsource(
            dummy_model_instance.vision_tower.vision_model.post_layernorm.forward
        ) == inspect.getsource(LigerLayerNorm.forward)

        for layer in dummy_model_instance.vision_tower.vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_gemma3_available(), reason="gemma3 module not available")
def test_apply_liger_kernel_to_instance_for_gemma3_text():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.gemma3.modeling_gemma3"):
        from liger_kernel.transformers.model.gemma3 import causal_forward as gemma3_causal_forward

        # Instantiate a dummy model
        config = transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma3_causal_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(gemma3_causal_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_gemma3_available(), reason="gemma3 module not available")
def test_apply_liger_kernel_to_instance_for_gemma3_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests

    with patch("transformers.models.gemma3.modeling_gemma3"):
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

        from liger_kernel.transformers.model.gemma3 import multimodal_forward as gemma3_multimodal_forward

        # Instantiate a dummy model
        text_config = transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
        )
        vision_config = transformers.models.siglip.configuration_siglip.SiglipVisionConfig(
            layer_norm_eps=1e-5,
            hidden_size=48,
            intermediate_size=64,
        )
        config = transformers.models.gemma3.configuration_gemma3.Gemma3Config(text_config, vision_config)

        dummy_model_instance = Gemma3ForConditionalGeneration._from_config(config)
        assert isinstance(dummy_model_instance, Gemma3ForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma3_multimodal_forward)
        assert inspect.getsource(
            dummy_model_instance.vision_tower.vision_model.post_layernorm.forward
        ) != inspect.getsource(LigerLayerNorm.forward)

        for layer in dummy_model_instance.vision_tower.vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        assert inspect.getsource(
            dummy_model_instance.multi_modal_projector.mm_soft_emb_norm.forward
        ) != inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )

        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(gemma3_multimodal_forward)
        assert inspect.getsource(
            dummy_model_instance.vision_tower.vision_model.post_layernorm.forward
        ) == inspect.getsource(LigerLayerNorm.forward)

        for layer in dummy_model_instance.vision_tower.vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        assert inspect.getsource(
            dummy_model_instance.multi_modal_projector.mm_soft_emb_norm.forward
        ) == inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_qwen2():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2.modeling_qwen2"):
        # Instantiate a dummy model
        config = transformers.models.qwen2.configuration_qwen2.Qwen2Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen2_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_available(), reason="qwen3 module not available")
def test_apply_liger_kernel_to_instance_for_qwen3():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3.modeling_qwen3"):
        from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3.configuration_qwen3.Qwen3Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_available(), reason="qwen3 module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_moe():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_moe.modeling_qwen3_moe"):
        from liger_kernel.transformers.model.qwen3_moe import lce_forward as qwen3_moe_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            for mlp_expert in layer.mlp.experts:
                assert inspect.getsource(mlp_expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.52.4"),
    reason="Qwen2-VL support is only compatible with transformers >= 4.52.4",
)
def test_apply_liger_kernel_to_instance_for_qwen2_vl_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2_vl.modeling_qwen2_vl"):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

        from liger_kernel.transformers.model.qwen2_vl import lce_forward as qwen2_vl_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=48,
            embed_dim=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            vocab_size=1000,
            vision_config={
                "depth": 4,
                "embed_dim": 128,
                "num_heads": 8,
                "hidden_size": 1024,
            },
        )
        dummy_model_instance = Qwen2VLForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, Qwen2VLForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen2_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.52.4"),
    reason="Qwen2-VL support is only compatible with transformers >= 4.52.4",
)
def test_apply_liger_kernel_to_instance_for_qwen2_vl():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2_vl.modeling_qwen2_vl"):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

        from liger_kernel.transformers.model.qwen2_vl import lce_forward as qwen2_vl_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=48,
            embed_dim=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            vocab_size=1000,
            vision_config={
                "depth": 4,
                "embed_dim": 128,
                "num_heads": 8,
                "hidden_size": 1024,
            },
        )
        dummy_model_instance = Qwen2VLModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen2VLModel)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen2_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.52.4"),
    reason="Qwen2-VL support is only compatible with transformers >= 4.52.4",
)
def test_apply_liger_kernel_to_instance_for_qwen2_vl_text():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2_vl.modeling_qwen2_vl"):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel

        # Instantiate a dummy model
        config = transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLTextConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=48,
            embed_dim=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            vocab_size=1000,
        )
        dummy_model_instance = Qwen2VLTextModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen2VLTextModel)

        # Check that model instance variables are not yet patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.52.4"),
    reason="Qwen2.5-VL support is only compatible with transformers >= 4.52.4",
)
def test_apply_liger_kernel_to_instance_for_qwen2_5_vl():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

        from liger_kernel.transformers.model.qwen2_5_vl import lce_forward as qwen2_5_vl_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=48,
            embed_dim=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            vocab_size=1000,
            vision_config={
                "depth": 4,
                "embed_dim": 128,
                "num_heads": 8,
                "hidden_size": 1024,
            },
        )
        dummy_model_instance = Qwen2_5_VLModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen2_5_VLModel)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen2_5_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_5_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.52.4"),
    reason="Qwen2.5-VL support is only compatible with transformers >= 4.52.4",
)
def test_apply_liger_kernel_to_instance_for_qwen2_5_vl_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        from liger_kernel.transformers.model.qwen2_5_vl import lce_forward as qwen2_5_vl_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=48,
            embed_dim=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            vocab_size=1000,
            vision_config={
                "depth": 4,
                "embed_dim": 128,
                "num_heads": 8,
                "hidden_size": 1024,
            },
        )
        dummy_model_instance = Qwen2_5_VLForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, Qwen2_5_VLForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen2_5_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_5_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(
    transformer_version < version.parse("4.52.4"),
    reason="Qwen2.5-VL support is only compatible with transformers >= 4.52.4",
)
def test_apply_liger_kernel_to_instance_for_qwen2_5_vl_text():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

        # Instantiate a dummy model
        config = transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLTextConfig(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=48,
            embed_dim=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            vocab_size=1000,
        )
        dummy_model_instance = Qwen2_5_VLTextModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen2_5_VLTextModel)

        # Check that model instance variables are not yet patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_phi3():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.phi3.modeling_phi3"):
        # Instantiate a dummy model
        config = transformers.models.phi3.configuration_phi3.Phi3Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(phi3_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(phi3_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_olmo2_available(), reason="olmo2 module not available")
def test_apply_liger_kernel_to_instance_for_olmo2():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.olmo2.modeling_olmo2"):
        from liger_kernel.transformers.model.olmo2 import lce_forward as olmo2_lce_forward

        # Instantiate a dummy model
        config = transformers.models.olmo2.configuration_olmo2.Olmo2Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(olmo2_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(olmo2_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_glm4_available(), reason="glm4 module not available")
def test_apply_liger_kernel_to_instance_for_glm4():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.glm4.modeling_glm4"):
        from liger_kernel.transformers.model.glm4 import lce_forward as glm4_lce_forward

        # Instantiate a dummy model
        config = transformers.models.glm4.configuration_glm4.Glm4Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(glm4_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_self_attn_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_mlp_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(glm4_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_self_attn_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_mlp_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_glm4v_available(), reason="glm4v module not available")
def test_apply_liger_kernel_to_instance_for_glm4v():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.glm4v.modeling_glm4v"):
        from transformers.models.glm4v.modeling_glm4v import Glm4vForConditionalGeneration

        from liger_kernel.transformers.model.glm4v import lce_forward as glm4v_lce_forward

        # Instantiate a dummy model
        config = transformers.models.glm4v.configuration_glm4v.Glm4vConfig(
            torch_dtype=torch.bfloat16,
            text_config={
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "hidden_size": 32,
                "intermediate_size": 64,
                "hidden_act": "silu",
            },
            vision_config={
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "hidden_size": 48,
                "intermediate_size": 64,
            },
        )
        dummy_model_instance = Glm4vForConditionalGeneration(config)
        assert isinstance(dummy_model_instance, Glm4vForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(glm4v_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_self_attn_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_mlp_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(glm4v_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_self_attn_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_mlp_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_smollm3_available(), reason="smollm3 module not available")
def test_apply_liger_kernel_to_instance_for_smollm3():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.smollm3.modeling_smollm3"):
        # Instantiate a dummy model
        config = transformers.models.smollm3.configuration_smollm3.SmolLM3Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(smolllm3_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(smolllm3_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        # Ensure that the model patched with Liger modules can work properly
        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")
