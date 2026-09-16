import copy
import importlib
import inspect
import subprocess
import sys

from inspect import signature
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
import transformers

from packaging import version
from test.utils import get_mllama_rope_config
from test.utils import get_qwen3_vl_rope_config
from transformers import AutoModelForCausalLM
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from liger_kernel.transformers import LigerBlockSparseTop2MLP
from liger_kernel.transformers import LigerExperts
from liger_kernel.transformers import LigerGEGLUMLP
from liger_kernel.transformers import LigerPhi3SwiGLUMLP
from liger_kernel.transformers import LigerQwen3MoeSwiGLUMLP
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers import LigerSwiGLUMLP
from liger_kernel.transformers import monkey_patch
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.model.falcon_h1 import lce_forward as falcon_h1_lce_forward
from liger_kernel.transformers.model.gemma import lce_forward as gemma_lce_forward
from liger_kernel.transformers.model.gemma2 import lce_forward as gemma2_lce_forward
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
from liger_kernel.transformers.model.ministral import lce_forward as ministral_lce_forward
from liger_kernel.transformers.model.mistral import lce_forward as mistral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward as mixtral_lce_forward
from liger_kernel.transformers.model.mllama import lce_forward as mllama_lce_forward
from liger_kernel.transformers.model.paligemma import lce_forward as paligemma_lce_forward
from liger_kernel.transformers.model.phi3 import lce_forward as phi3_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.model.qwen3_5 import lce_forward as qwen3_5_lce_forward
from liger_kernel.transformers.model.qwen3_5 import lce_forward_for_multimodal as qwen3_5_lce_forward_for_multimodal
from liger_kernel.transformers.model.qwen3_next import lce_forward as qwen3_next_lce_forward
from liger_kernel.transformers.model.smollm3 import lce_forward as smolllm3_lce_forward
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

# We only support transformers >= 4.52.0
transformer_version = version.parse(transformers.__version__)
MIN_SUPPORTED_TRANSFORMERS_VERSION = version.parse("4.52.0")
if transformer_version < MIN_SUPPORTED_TRANSFORMERS_VERSION:
    pytest.skip(
        f"tests require transformers >= {MIN_SUPPORTED_TRANSFORMERS_VERSION}, got {transformers.__version__}",
        allow_module_level=True,
    )

IS_TRANSFORMERS_V5_OR_LATER = transformer_version >= version.parse("5.0.0")


def test_instance_norm_patches_initialize_dispatch_attributes():
    rms_norm = torch.nn.RMSNorm(16)
    layer_norm = torch.nn.LayerNorm(16)

    monkey_patch._patch_rms_norm_module(rms_norm)
    monkey_patch._patch_layer_norm_module(layer_norm)

    assert rms_norm.impl is None
    assert rms_norm.mode is None
    assert layer_norm.impl is None
    assert layer_norm.mode is None


# Check if optional modules are available
def is_mllama_available():
    try:
        import transformers.models.mllama  # noqa: F401

        return True
    except ImportError:
        return False


def is_internvl_available():
    try:
        import transformers.models.internvl  # noqa: F401

        return True
    except ImportError:
        return False


def is_smolvlm_available():
    try:
        import transformers.models.smolvlm  # noqa: F401

        return True
    except ImportError:
        return False


def is_llama4_available():
    try:
        import transformers.models.llama4  # noqa: F401

        return True
    except ImportError:
        return False


def is_ministral_available():
    try:
        import transformers.models.ministral  # noqa: F401

        return True
    except ImportError:
        return False


def is_muse_glimmer_available():
    try:
        import transformers.models.muse_glimmer  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen3_available():
    try:
        import transformers.models.qwen3  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen3_vl_available():
    try:
        import transformers.models.qwen3_vl  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen3_vl_moe_available():
    try:
        import transformers.models.qwen3_vl_moe  # noqa: F401

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


def is_olmo3_available():
    try:
        import transformers.models.olmo3  # noqa: F401

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


def is_exaone4_available():
    try:
        import transformers.models.exaone4  # noqa: F401

        return True
    except ImportError:
        return False


def is_glm4v_moe_available():
    try:
        import transformers.models.glm4v_moe  # noqa: F401

        return True
    except ImportError:
        return False


def is_gemma3_available():
    try:
        import transformers.models.gemma3  # noqa: F401

        return True
    except ImportError:
        return False


def is_gemma4_available():
    try:
        import transformers.models.gemma4  # noqa: F401

        return True
    except ImportError:
        return False


def is_paligemma_available():
    try:
        import transformers.models.paligemma  # noqa: F401

        return True
    except ImportError:
        return False


def is_deepseek_v4_available():
    try:
        import transformers.models.deepseek_v4  # noqa: F401

        return True
    except ImportError:
        return False


def is_falcon_h1_available():
    try:
        import transformers.models.falcon_h1  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen3_next_available():
    try:
        import transformers.models.qwen3_next  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen4_exp_available():
    try:
        import transformers.models.qwen4_exp  # noqa: F401

        return True
    except ImportError:
        return False


def _qwen4_exp_compat_config(*, hidden_act="silu", experts_implementation="eager"):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        dtype=torch.float32,
        rms_norm_eps=1e-5,
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=1,
        num_experts=4,
        hc_count=4,
        hc_lowrank=8,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[],
        ple_embed_dim=32,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["qwen_sparse_attention"],
        hidden_act=hidden_act,
        output_gate_type="silu",
        experts_implementation=experts_implementation,
    )


def test_monkey_patch_import_without_qwen4_exp():
    script = """
import importlib.abc
import sys


class BlockQwen4Exp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transformers.models.qwen4_exp" or fullname.startswith("transformers.models.qwen4_exp."):
            raise ModuleNotFoundError("simulated Transformers without qwen4_exp", name=fullname)
        return None


sys.meta_path.insert(0, BlockQwen4Exp())
from liger_kernel.transformers import apply_liger_kernel_to_llama  # noqa: F401
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr


def is_qwen3_5_available():
    try:
        import transformers.models.qwen3_5  # noqa: F401

        return True
    except ImportError:
        return False


def is_qwen3_5_moe_available():
    try:
        import transformers.models.qwen3_5_moe  # noqa: F401

        return True
    except ImportError:
        return False


def is_pixtral_available():
    try:
        import transformers.models.pixtral  # noqa: F401

        return True
    except ImportError:
        return False


def is_hunyuan_v1_available():
    try:
        import transformers.models.hunyuan_v1_dense  # noqa: F401

        return True
    except ImportError:
        return False


def is_nemotron_available():
    try:
        import transformers.models.nemotron  # noqa: F401

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
        from liger_kernel.transformers import apply_liger_kernel_to_gemma4_text  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_glm4  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v_moe  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_internvl  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_llama  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_ministral  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_mistral  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_mllama  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_phi3  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe  # noqa: F401
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_next  # noqa: F401
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
            dtype=torch.bfloat16,
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


@pytest.mark.skipif(not is_muse_glimmer_available(), reason="muse_glimmer module not available")
def test_apply_liger_kernel_to_instance_for_muse_glimmer():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.muse_glimmer.modeling_muse_glimmer"):
        from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

        from liger_kernel.transformers.model.muse_glimmer import lce_forward as muse_glimmer_lce_forward
        from liger_kernel.transformers.rms_norm import LigerRMSNormForMuseGlimmer

        # Instantiate a dummy model
        config = transformers.models.muse_glimmer.configuration_muse_glimmer.MuseGlimmerConfig(
            attn_implementation="sdpa",
            out_hidden_size=128,
            projector_hidden_size=64,
            vision_config=transformers.models.muse_glimmer.configuration_muse_glimmer.MuseGlimmerVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                patch_size=14,
                pos_emb_height=4,
                pos_emb_width=4,
                max_position_embeddings=16,
            ),
            text_config=transformers.models.muse_glimmer.configuration_muse_glimmer.MuseGlimmerTextConfig(
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=128,
                sliding_window=16,
            ),
        )
        dummy_model_instance = MuseGlimmerForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, MuseGlimmerForConditionalGeneration)

        text_model = dummy_model_instance.model.language_model
        vision_model = dummy_model_instance.model.vision_tower

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(muse_glimmer_lce_forward)
        assert inspect.getsource(text_model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        assert inspect.getsource(text_model.embed_tokens.embed_norm.forward) != inspect.getsource(
            LigerRMSNormForMuseGlimmer.forward
        )
        assert inspect.getsource(dummy_model_instance.model.perception_emb_norm.forward) != inspect.getsource(
            LigerRMSNormForMuseGlimmer.forward
        )
        for decoder_layer in text_model.layers:
            assert inspect.getsource(decoder_layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(decoder_layer.pre_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(decoder_layer.post_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(decoder_layer.self_attn.qk_norm.forward) != inspect.getsource(
                LigerRMSNormForMuseGlimmer.forward
            )
        assert inspect.getsource(vision_model.ln_pre.forward) != inspect.getsource(LigerLayerNorm.forward)
        assert inspect.getsource(vision_model.ln_post.forward) != inspect.getsource(LigerLayerNorm.forward)
        for vision_layer in vision_model.layers:
            assert inspect.getsource(vision_layer.norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_layer.norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(muse_glimmer_lce_forward)
        assert inspect.getsource(text_model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        assert inspect.getsource(text_model.embed_tokens.embed_norm.forward) == inspect.getsource(
            LigerRMSNormForMuseGlimmer.forward
        )
        assert inspect.getsource(dummy_model_instance.model.perception_emb_norm.forward) == inspect.getsource(
            LigerRMSNormForMuseGlimmer.forward
        )
        for decoder_layer in text_model.layers:
            assert inspect.getsource(decoder_layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(decoder_layer.pre_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(decoder_layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(decoder_layer.self_attn.qk_norm.forward) == inspect.getsource(
                LigerRMSNormForMuseGlimmer.forward
            )
            # MuseGlimmerTextCenteredRMSNorm scales by (1 + weight) and the post-norms use
            # a distinct, much smaller epsilon -- both must survive patching.
            assert decoder_layer.input_layernorm.offset == 1.0
            assert decoder_layer.input_layernorm.casting_mode == "gemma"
            assert decoder_layer.input_layernorm.in_place is False
            assert decoder_layer.post_attention_layernorm.variance_epsilon == config.text_config.post_norm_eps
            assert decoder_layer.input_layernorm.variance_epsilon == config.text_config.rms_norm_eps
            # The scale-free QK norm has no weight to scale by
            assert decoder_layer.self_attn.qk_norm.with_scale is False
        # The final norm scales by weight directly (no +1 offset)
        assert text_model.norm.offset == 0.0
        assert inspect.getsource(vision_model.ln_pre.forward) == inspect.getsource(LigerLayerNorm.forward)
        assert inspect.getsource(vision_model.ln_post.forward) == inspect.getsource(LigerLayerNorm.forward)
        for vision_layer in vision_model.layers:
            assert inspect.getsource(vision_layer.norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_layer.norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_available(), reason="qwen3_vl module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_vl_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl.modeling_qwen3_vl"):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        from liger_kernel.transformers.model.qwen3_vl import lce_forward as qwen3_vl_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig(
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLVisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                deepstack_visual_indexes=[1, 2, 3],
                initializer_range=0.02,
            ).to_dict(),
            text_config=transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLTextConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                tie_word_embeddings=True,
                attention_dropout=0.0,
                attention_bias=False,
                **get_qwen3_vl_rope_config(),  # Version-aware rope configuration
            ).to_dict(),
        )
        dummy_model_instance = Qwen3VLForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3VLForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_available(), reason="qwen3_vl module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_vl():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl.modeling_qwen3_vl"):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        from liger_kernel.transformers.model.qwen3_vl import lce_forward as qwen3_vl_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig(
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLVisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                deepstack_visual_indexes=[1, 2, 3],
                initializer_range=0.02,
            ).to_dict(),
            text_config=transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLTextConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                tie_word_embeddings=True,
                attention_dropout=0.0,
                attention_bias=False,
                **get_qwen3_vl_rope_config(),  # Version-aware rope configuration
            ).to_dict(),
        )
        dummy_model_instance = Qwen3VLModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3VLModel)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_available(), reason="qwen3_vl module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_vl_text():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl.modeling_qwen3_vl"):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        # Instantiate a dummy model
        config = transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLTextConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=False,
            tie_word_embeddings=True,
            attention_dropout=0.0,
            attention_bias=False,
            **get_qwen3_vl_rope_config(),  # Version-aware rope configuration
        )
        dummy_model_instance = Qwen3VLTextModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3VLTextModel)

        # Check that model instance variables are not yet patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for decoder_layer in dummy_model_instance.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for decoder_layer in dummy_model_instance.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_moe_available(), reason="qwen3_vl_moe module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_vl_moe_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"):
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

        from liger_kernel.transformers.model.qwen3_vl_moe import lce_forward as qwen3_vl_moe_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeConfig(
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeVisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                deepstack_visual_indexes=[1, 2, 3],
                initializer_range=0.02,
            ).to_dict(),
            text_config=transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeTextConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                tie_word_embeddings=True,
                attention_dropout=0.0,
                attention_bias=False,
                decoder_sparse_step=1,
                moe_intermediate_size=1024,
                num_experts_per_tok=2,
                num_experts=4,
                mlp_only_layers=[],
                pad_token_id=None,
                **get_qwen3_vl_rope_config(),  # Version-aware rope configuration
            ).to_dict(),
        )
        dummy_model_instance = Qwen3VLMoeForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3VLMoeForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_vl_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None and IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(experts.forward) != inspect.getsource(LigerExperts.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_vl_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None and IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(experts.forward) == inspect.getsource(LigerExperts.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_moe_available(), reason="qwen3_vl_moe module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_vl_moe():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"):
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel

        from liger_kernel.transformers.model.qwen3_vl_moe import lce_forward as qwen3_vl_moe_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeConfig(
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeVisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                deepstack_visual_indexes=[1, 2, 3],
                initializer_range=0.02,
            ).to_dict(),
            text_config=transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeTextConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                tie_word_embeddings=True,
                attention_dropout=0.0,
                attention_bias=False,
                decoder_sparse_step=1,
                moe_intermediate_size=1024,
                num_experts_per_tok=2,
                num_experts=4,
                mlp_only_layers=[],
                pad_token_id=None,
                **get_qwen3_vl_rope_config(),  # Version-aware rope configuration
            ).to_dict(),
        )
        dummy_model_instance = Qwen3VLMoeModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3VLMoeModel)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_vl_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None and IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(experts.forward) != inspect.getsource(LigerExperts.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_vl_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for decoder_layer in dummy_model_instance.language_model.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None and IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(experts.forward) == inspect.getsource(LigerExperts.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_moe_available(), reason="qwen3_vl_moe module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_vl_moe_text():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"):
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel

        # Instantiate a dummy model
        config = transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeTextConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=False,
            tie_word_embeddings=True,
            attention_dropout=0.0,
            attention_bias=False,
            decoder_sparse_step=1,
            moe_intermediate_size=1024,
            num_experts_per_tok=2,
            num_experts=4,
            mlp_only_layers=[],
            pad_token_id=None,
            **get_qwen3_vl_rope_config(),  # Version-aware rope configuration
        )
        dummy_model_instance = Qwen3VLMoeTextModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3VLMoeTextModel)

        # Check that model instance variables are not yet patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for decoder_layer in dummy_model_instance.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None and IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(experts.forward) != inspect.getsource(LigerExperts.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        # Note: Text models don't have forward method patching, so skip this check
        assert inspect.getsource(dummy_model_instance.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for decoder_layer in dummy_model_instance.layers:
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                    assert inspect.getsource(self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
                if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
                    assert inspect.getsource(self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None and IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(experts.forward) == inspect.getsource(LigerExperts.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_vl_available(), reason="qwen3_vl module not available")
def test_qwen3_vl_rope_hooks_applied():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl.modeling_qwen3_vl") as modeling_mod:
        from liger_kernel.transformers.monkey_patch import liger_rotary_pos_emb
        from liger_kernel.transformers.monkey_patch import liger_rotary_pos_emb_vision

        # Before applying, make sure attributes exist but are not the liger implementations
        setattr(modeling_mod, "apply_rotary_pos_emb", object())
        setattr(modeling_mod, "apply_rotary_pos_emb_vision", object())

        _apply_liger_kernel("qwen3_vl")

        assert modeling_mod.apply_rotary_pos_emb is liger_rotary_pos_emb
        assert modeling_mod.apply_rotary_pos_emb_vision is liger_rotary_pos_emb_vision


@pytest.mark.skipif(not is_qwen3_vl_moe_available(), reason="qwen3_vl_moe module not available")
def test_qwen3_vl_moe_rope_hooks_applied():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe") as modeling_mod:
        from liger_kernel.transformers.monkey_patch import liger_rotary_pos_emb
        from liger_kernel.transformers.monkey_patch import liger_rotary_pos_emb_vision

        # Before applying, make sure attributes exist but are not the liger implementations
        setattr(modeling_mod, "apply_rotary_pos_emb", object())
        setattr(modeling_mod, "apply_rotary_pos_emb_vision", object())

        _apply_liger_kernel("qwen3_vl_moe")

        assert modeling_mod.apply_rotary_pos_emb is liger_rotary_pos_emb
        assert modeling_mod.apply_rotary_pos_emb_vision is liger_rotary_pos_emb_vision


@pytest.mark.skipif(not is_falcon_h1_available(), reason="falcon_h1 module not available")
def test_apply_liger_kernel_to_falcon_h1_for_causal_lm():
    with patch("transformers.models.falcon_h1.modeling_falcon_h1"):
        from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

        # Instantiate a dummy model
        config = transformers.models.falcon_h1.configuration_falcon_h1.FalconH1Config(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=1024,
        )
        dummy_model_instance = FalconH1ForCausalLM(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(falcon_h1_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.final_layernorm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_ff_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(falcon_h1_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.final_layernorm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_ff_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
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
            dtype=torch.bfloat16,
            text_config=transformers.models.mllama.configuration_mllama.MllamaTextConfig(
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="silu",
                num_hidden_layers=2,
                **get_mllama_rope_config(),  # Version-aware rope configuration
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

        if isinstance(dummy_model_instance.model.language_model, MllamaTextModel):
            language_model = dummy_model_instance.model.language_model
        else:
            language_model = dummy_model_instance.model.language_model.model

        assert inspect.getsource(language_model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.model.vision_model.layernorm_pre.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        assert inspect.getsource(dummy_model_instance.model.vision_model.layernorm_post.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        for layer in dummy_model_instance.model.vision_model.transformer.layers:
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerLayerNorm.forward
            )
        for layer in dummy_model_instance.model.vision_model.global_transformer.layers:
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

        assert inspect.getsource(dummy_model_instance.model.vision_model.layernorm_pre.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        assert inspect.getsource(dummy_model_instance.model.vision_model.layernorm_post.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        for layer in dummy_model_instance.model.vision_model.transformer.layers:
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerLayerNorm.forward
            )
        for layer in dummy_model_instance.model.vision_model.global_transformer.layers:
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
            **get_mllama_rope_config(),  # Version-aware rope configuration
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


@pytest.mark.skipif(not is_pixtral_available(), reason="pixtral module not available")
def test_apply_liger_kernel_to_instance_for_pixtral_vision_model():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.pixtral.modeling_pixtral"):
        from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

        # Instantiate a dummy model
        config = transformers.models.pixtral.configuration_pixtral.PixtralVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=64,
            patch_size=16,
            hidden_act="silu",
            attention_dropout=0.0,
            rope_theta=10000.0,
        )
        dummy_model_instance = PixtralVisionModel._from_config(config)

        assert isinstance(dummy_model_instance, PixtralVisionModel)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.ln_pre.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.transformer.layers:
            assert inspect.getsource(layer.feed_forward.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.attention_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.ffn_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.ln_pre.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.transformer.layers:
            assert inspect.getsource(layer.feed_forward.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.attention_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.ffn_norm.forward) == inspect.getsource(LigerRMSNorm.forward)

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
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            moe_layers=[1],
        )
        dummy_model_instance = Llama4ForCausalLM._from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if layer.is_moe_layer:
                assert inspect.getsource(layer.feed_forward.shared_expert.forward) != inspect.getsource(
                    LigerSwiGLUMLP.forward
                )
            else:
                assert inspect.getsource(layer.feed_forward.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if layer.is_moe_layer:
                assert inspect.getsource(layer.feed_forward.shared_expert.forward) == inspect.getsource(
                    LigerSwiGLUMLP.forward
                )
            else:
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
            dtype=torch.bfloat16,
            text_config=transformers.models.llama4.configuration_llama4.Llama4TextConfig(
                dtype=torch.bfloat16,
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="silu",
                num_hidden_layers=2,
                moe_layers=[1],
            ),
            vision_config=transformers.models.llama4.configuration_llama4.Llama4VisionConfig(
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="gelu",
                num_hidden_layers=2,
                vision_output_dim=64,
            ),
            pad_token_id=None,
        )
        dummy_model_instance = Llama4ForConditionalGeneration._from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert isinstance(dummy_model_instance, Llama4ForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.language_model.model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.model.layers:
            if layer.is_moe_layer:
                assert inspect.getsource(layer.feed_forward.shared_expert.forward) != inspect.getsource(
                    LigerSwiGLUMLP.forward
                )
            else:
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
            if layer.is_moe_layer:
                assert inspect.getsource(layer.feed_forward.shared_expert.forward) == inspect.getsource(
                    LigerSwiGLUMLP.forward
                )
            else:
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


def test_apply_liger_kernel_to_instance_for_mistral():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.mistral.modeling_mistral"):
        # Instantiate a dummy model
        config = transformers.models.mistral.configuration_mistral.MistralConfig(
            dtype=torch.bfloat16,
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


@pytest.mark.skipif(not is_ministral_available(), reason="ministral module not available")
def test_apply_liger_kernel_to_instance_for_ministral():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.ministral.modeling_ministral"):
        # Instantiate a dummy model
        config = transformers.models.ministral.configuration_ministral.MinistralConfig(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            head_dim=16,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(ministral_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(ministral_lce_forward)
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
            dtype=torch.bfloat16,
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
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
            else:
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
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.block_sparse_moe.experts:
                    assert inspect.getsource(expert.forward) == inspect.getsource(LigerBlockSparseTop2MLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_deepseek_v4_available(), reason="deepseek_v4 module not available")
def test_apply_liger_kernel_to_instance_for_deepseek_v4():
    with patch("transformers.models.deepseek_v4.modeling_deepseek_v4"):
        from liger_kernel.transformers.model.deepseek_v4 import lce_forward as deepseek_v4_lce_forward

        config = transformers.models.deepseek_v4.configuration_deepseek_v4.DeepseekV4Config(
            vocab_size=1024,
            hidden_size=32,
            moe_intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            q_lora_rank=8,
            num_experts_per_tok=2,
            n_routed_experts=4,
            hc_mult=2,
            sliding_window=8,
            o_groups=2,
            o_lora_rank=8,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            layer_types=[
                "heavily_compressed_attention",
                "compressed_sparse_attention",
                "sliding_attention",
                "sliding_attention",
            ],
            mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe"],
            max_position_embeddings=128,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(deepseek_v4_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(deepseek_v4_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
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
            dtype=torch.bfloat16,
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
            assert layer.mlp._get_name() == LigerGEGLUMLP.__name__
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
            dtype=torch.bfloat16,
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
            assert layer.mlp._get_name() == LigerGEGLUMLP.__name__
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
            dtype=torch.bfloat16,
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
        siglip_vision_model = getattr(
            dummy_model_instance.model.vision_tower, "vision_model", dummy_model_instance.model.vision_tower
        )

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(paligemma_lce_forward)
        assert inspect.getsource(siglip_vision_model.post_layernorm.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )

        for layer in siglip_vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(paligemma_lce_forward)
        assert inspect.getsource(siglip_vision_model.post_layernorm.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )

        for layer in siglip_vision_model.encoder.layers:
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
            dtype=torch.bfloat16,
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
            assert layer.mlp._get_name() == LigerGEGLUMLP.__name__
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
            dtype=torch.bfloat16,
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
        config = transformers.models.gemma3.configuration_gemma3.Gemma3Config(
            text_config=text_config, vision_config=vision_config
        )

        dummy_model_instance = Gemma3ForConditionalGeneration._from_config(config)
        assert isinstance(dummy_model_instance, Gemma3ForConditionalGeneration)
        siglip_vision_model = getattr(
            dummy_model_instance.model.vision_tower, "vision_model", dummy_model_instance.model.vision_tower
        )

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma3_multimodal_forward)
        assert inspect.getsource(siglip_vision_model.post_layernorm.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )

        for layer in siglip_vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        assert inspect.getsource(
            dummy_model_instance.model.multi_modal_projector.mm_soft_emb_norm.forward
        ) != inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )

        for layer in dummy_model_instance.model.language_model.layers:
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
        assert inspect.getsource(siglip_vision_model.post_layernorm.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )

        for layer in siglip_vision_model.encoder.layers:
            assert inspect.getsource(layer.layer_norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(layer.layer_norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        assert inspect.getsource(
            dummy_model_instance.model.multi_modal_projector.mm_soft_emb_norm.forward
        ) == inspect.getsource(LigerRMSNorm.forward)

        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert layer.mlp._get_name() == LigerGEGLUMLP.__name__
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


@pytest.mark.skipif(not is_gemma4_available(), reason="gemma4 module not available")
def test_apply_liger_kernel_to_instance_for_gemma4_text():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.gemma4.modeling_gemma4"):
        from liger_kernel.transformers.model.gemma4 import causal_forward as gemma4_causal_forward

        # Instantiate a dummy model
        config = transformers.models.gemma4.configuration_gemma4.Gemma4TextConfig(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            # Pin every novel Gemma 4 knob off so the test exercises the dense path.
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            enable_moe_block=False,
            hidden_size_per_layer_input=0,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Pre-patch assertions
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma4_causal_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        # q_norm / k_norm are only present on non-KV-shared layers; we pin
        # num_kv_shared_layers=0 in the config above so every layer has them.
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

        # Apply kernels to the instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Post-patch assertions
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(gemma4_causal_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert layer.mlp._get_name() == LigerGEGLUMLP.__name__
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            # v_norm is scale-free (with_scale=False); _maybe_patch_scaled_norm
            # intentionally skips it, so the instance must retain the HF forward.
            v_norm = getattr(layer.self_attn, "v_norm", None)
            if v_norm is not None:
                assert inspect.getsource(v_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_gemma4_available(), reason="gemma4 module not available")
def test_apply_liger_kernel_to_instance_for_gemma4_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.gemma4.modeling_gemma4"):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

        from liger_kernel.transformers.model.gemma4 import multimodal_forward as gemma4_multimodal_forward

        # Minimal dense-path text config — same knobs pinned off as the
        # text-only test below (no PLE, MoE, KV-share, double-wide MLP).
        text_config = transformers.models.gemma4.configuration_gemma4.Gemma4TextConfig(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            enable_moe_block=False,
            hidden_size_per_layer_input=0,
        )
        # Vision/audio configs left as None — Gemma4Model wraps both towers in
        # `if config.<m>_config is not None`, so a None-towers model still
        # constructs as Gemma4ForConditionalGeneration and exercises the
        # multimodal forward we're patching. The towers themselves are
        # polymorphic (AutoModel.from_config) and not in this PR's scope.
        config = transformers.models.gemma4.configuration_gemma4.Gemma4Config(
            text_config=text_config,
            vision_config=None,
            audio_config=None,
        )

        dummy_model_instance = Gemma4ForConditionalGeneration._from_config(config)
        assert isinstance(dummy_model_instance, Gemma4ForConditionalGeneration)

        # Pre-patch: forward and language-model norms must NOT be Liger.
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(gemma4_multimodal_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerGEGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) != inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Post-patch: top-level forward is multimodal_forward, language_model
        # norms / MLPs are Liger.
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(gemma4_multimodal_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerGEGLUMLP.forward)
            assert layer.mlp._get_name() == LigerGEGLUMLP.__name__
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.pre_feedforward_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )
            assert inspect.getsource(layer.self_attn.q_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
            v_norm = getattr(layer.self_attn, "v_norm", None)
            if v_norm is not None:
                # with_scale=False → intentionally not patched.
                assert inspect.getsource(v_norm.forward) != inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_qwen2():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen2.modeling_qwen2"):
        # Instantiate a dummy model
        config = transformers.models.qwen2.configuration_qwen2.Qwen2Config(
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
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
            dtype=torch.bfloat16,
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
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
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
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen2_5_vl_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
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
            dtype=torch.bfloat16,
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


@pytest.mark.skipif(not is_internvl_available(), reason="internvl module not available")
def test_apply_liger_kernel_to_instance_for_internvl():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.internvl.modeling_internvl"):
        from transformers.models.internvl.modeling_internvl import InternVLForConditionalGeneration

        # Instantiate a dummy model
        config = transformers.models.internvl.configuration_internvl.InternVLConfig(
            dtype=torch.bfloat16,
            text_config={
                "rms_norm_eps": 1e-5,
                "hidden_size": 256,  # 1024
                "intermediate_size": 1024,  # 4096
                "hidden_act": "silu",
                "num_hidden_layers": 4,  # 24
                "num_attention_heads": 4,  # 16
                "num_key_value_heads": 2,  # 16
                "max_position_embeddings": 4096,  # 8192
                "vocab_size": 32000,  # 151936
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 2,
                "tie_word_embeddings": False,
            },
            vision_config={
                "hidden_size": 256,  # 1024
                "intermediate_size": 1024,  # 4096
                "num_hidden_layers": 4,  # 24
                "num_attention_heads": 4,  # 16
            },
            image_token_id=10,
            attn_implementation="sdpa",  # default value, pytorch native attention
        )
        dummy_model_instance = InternVLForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, InternVLForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_smolvlm_available(), reason="smolvlm module not available")
def test_apply_liger_kernel_to_instance_for_smolvlm2():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.smolvlm.modeling_smolvlm"):
        from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

        # Instantiate a dummy model
        config = transformers.models.smolvlm.configuration_smolvlm.SmolVLMConfig(
            dtype=torch.bfloat16,
            text_config={
                "rms_norm_eps": 1e-5,
                "hidden_size": 576,
                "intermediate_size": 1536,
                "hidden_act": "silu",
                "num_hidden_layers": 2,
                "num_attention_heads": 9,
                "num_key_value_heads": 3,
                "max_position_embeddings": 128,
                "vocab_size": 1000,
            },
            vision_config={
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 2,
                "num_attention_heads": 12,
            },
        )
        dummy_model_instance = SmolVLMForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, SmolVLMForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        # Text model checks
        assert inspect.getsource(dummy_model_instance.model.text_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.text_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Vision model checks
        assert inspect.getsource(dummy_model_instance.model.vision_model.post_layernorm.forward) != inspect.getsource(
            LigerLayerNorm.forward
        )
        for encoder_layer in dummy_model_instance.model.vision_model.encoder.layers:
            assert inspect.getsource(encoder_layer.layer_norm1.forward) != inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(encoder_layer.layer_norm2.forward) != inspect.getsource(LigerLayerNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        # Text model checks
        assert inspect.getsource(dummy_model_instance.model.text_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.text_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        # Vision model checks
        assert inspect.getsource(dummy_model_instance.model.vision_model.post_layernorm.forward) == inspect.getsource(
            LigerLayerNorm.forward
        )
        for encoder_layer in dummy_model_instance.model.vision_model.encoder.layers:
            assert inspect.getsource(encoder_layer.layer_norm1.forward) == inspect.getsource(LigerLayerNorm.forward)
            assert inspect.getsource(encoder_layer.layer_norm2.forward) == inspect.getsource(LigerLayerNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def test_apply_liger_kernel_to_instance_for_phi3():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.phi3.modeling_phi3"):
        # Instantiate a dummy model
        config = transformers.models.phi3.configuration_phi3.Phi3Config(
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
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


@pytest.mark.skipif(not is_olmo3_available(), reason="olmo3 module not available")
def test_apply_liger_kernel_to_instance_for_olmo3():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.olmo3.modeling_olmo3"):
        from liger_kernel.transformers.model.olmo3 import lce_forward as olmo3_lce_forward

        # Instantiate a dummy model
        config = transformers.models.olmo3.configuration_olmo3.Olmo3Config(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(olmo3_lce_forward)
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
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(olmo3_lce_forward)
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
            dtype=torch.bfloat16,
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
            dtype=torch.bfloat16,
            text_config={
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "hidden_size": 32,
                "intermediate_size": 64,
                "hidden_act": "silu",
                "pad_token_id": None,
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
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_self_attn_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_mlp_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(glm4v_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerPhi3SwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_self_attn_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_mlp_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(vision_block.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_glm4v_moe_available(), reason="glm4v_moe module not available")
def test_apply_liger_kernel_to_instance_for_glm4v_moe():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.glm4v_moe.modeling_glm4v_moe"):
        from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeForConditionalGeneration

        from liger_kernel.transformers.model.glm4v_moe import lce_forward as glm4v_moe_lce_forward
        from liger_kernel.transformers.rms_norm import LigerRMSNormForGlm4

        # Instantiate a dummy model
        config = transformers.models.glm4v_moe.configuration_glm4v_moe.Glm4vMoeConfig(
            dtype=torch.bfloat16,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            text_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "hidden_act": "silu",
                "n_routed_experts": 1,
            },
            vision_config={
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "hidden_size": 48,
                "intermediate_size": 64,
            },
        )
        dummy_model_instance = Glm4vMoeForConditionalGeneration(config)
        assert isinstance(dummy_model_instance, Glm4vMoeForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(glm4v_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNormForGlm4.forward
        )
        assert inspect.getsource(dummy_model_instance.model.visual.post_conv_layernorm.forward) != inspect.getsource(
            LigerRMSNormForGlm4.forward
        )
        assert inspect.getsource(dummy_model_instance.model.visual.post_layernorm.forward) != inspect.getsource(
            LigerRMSNormForGlm4.forward
        )

        for decoder_layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) != inspect.getsource(
                LigerRMSNormForGlm4.forward
            )
            assert inspect.getsource(decoder_layer.input_layernorm.forward) != inspect.getsource(
                LigerRMSNormForGlm4.forward
            )
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None:
                if IS_TRANSFORMERS_V5_OR_LATER:
                    assert inspect.getsource(experts.forward) != inspect.getsource(LigerExperts.forward)
                else:
                    for expert in experts:
                        assert inspect.getsource(expert.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
                shared_experts = getattr(decoder_layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    assert inspect.getsource(shared_experts.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            else:
                assert inspect.getsource(decoder_layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) != inspect.getsource(LigerRMSNormForGlm4.forward)
            assert inspect.getsource(vision_block.norm2.forward) != inspect.getsource(LigerRMSNormForGlm4.forward)
            assert inspect.getsource(vision_block.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(glm4v_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNormForGlm4.forward
        )
        assert inspect.getsource(dummy_model_instance.model.visual.post_conv_layernorm.forward) == inspect.getsource(
            LigerRMSNormForGlm4.forward
        )
        assert inspect.getsource(dummy_model_instance.model.visual.post_layernorm.forward) == inspect.getsource(
            LigerRMSNormForGlm4.forward
        )

        for decoder_layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(decoder_layer.post_attention_layernorm.forward) == inspect.getsource(
                LigerRMSNormForGlm4.forward
            )
            assert inspect.getsource(decoder_layer.input_layernorm.forward) == inspect.getsource(
                LigerRMSNormForGlm4.forward
            )
            experts = getattr(decoder_layer.mlp, "experts", None)
            if experts is not None:
                if IS_TRANSFORMERS_V5_OR_LATER:
                    assert inspect.getsource(experts.forward) == inspect.getsource(LigerExperts.forward)
                else:
                    for expert in experts:
                        assert inspect.getsource(expert.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
                shared_experts = getattr(decoder_layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    assert inspect.getsource(shared_experts.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            else:
                assert inspect.getsource(decoder_layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
        for vision_block in dummy_model_instance.model.visual.blocks:
            assert inspect.getsource(vision_block.norm1.forward) == inspect.getsource(LigerRMSNormForGlm4.forward)
            assert inspect.getsource(vision_block.norm2.forward) == inspect.getsource(LigerRMSNormForGlm4.forward)
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
            dtype=torch.bfloat16,
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


@pytest.mark.skipif(not is_qwen3_next_available(), reason="qwen3_next module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_next():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_next.modeling_qwen3_next"):
        # Instantiate a dummy model
        config = transformers.models.qwen3_next.configuration_qwen3_next.Qwen3NextConfig(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_experts=2,
            num_experts_per_tok=1,
            mlp_only_layers=[1],
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_next_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                if IS_TRANSFORMERS_V5_OR_LATER:
                    assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
                else:
                    for expert in layer.mlp.experts:
                        assert inspect.getsource(expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
                if hasattr(layer.mlp, "shared_expert"):
                    assert inspect.getsource(layer.mlp.shared_expert.forward) != inspect.getsource(
                        LigerSwiGLUMLP.forward
                    )
            else:
                assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)

            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_next_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                if IS_TRANSFORMERS_V5_OR_LATER:
                    assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
                else:
                    for expert in layer.mlp.experts:
                        assert inspect.getsource(expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
                if hasattr(layer.mlp, "shared_expert"):
                    assert inspect.getsource(layer.mlp.shared_expert.forward) == inspect.getsource(
                        LigerSwiGLUMLP.forward
                    )
            else:
                assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)

            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_apply_liger_kernel_to_instance_for_qwen4_exp():
    modeling_qwen4_exp = importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
    Qwen4ExpTextModel = modeling_qwen4_exp.Qwen4ExpTextModel

    from liger_kernel.transformers.model.qwen4_exp import lce_forward as qwen4_exp_lce_forward
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_decoder_layer_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_experts_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_gated_residual_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_mlp_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_ngram_embedding_forward
    from liger_kernel.transformers.swiglu import LigerExperts
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    # Instantiate a dummy model (tiny hybrid GatedDeltaNet + QSA + MoE config)
    config = transformers.models.qwen4_exp.configuration_qwen4_exp.Qwen4ExpTextConfig(
        dtype=torch.bfloat16,
        rms_norm_eps=1e-5,
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=1,
        num_experts=4,
        hc_count=4,
        hc_lowrank=8,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        experts_implementation="eager",
    )
    dummy_model_instance = AutoModelForCausalLM.from_config(config)

    # Reject unsupported (including multimodal composite) instances before any process-global
    # Qwen4Exp text class is mutated.
    original_classes = (
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm,
        modeling_qwen4_exp.Qwen4ExpTextMLP,
        modeling_qwen4_exp.Qwen4ExpTextExperts,
    )
    original_for_causal_lm_forward = modeling_qwen4_exp.Qwen4ExpForCausalLM.forward
    with pytest.raises(TypeError, match="Unsupported qwen4_exp model type"):
        apply_liger_kernel_to_qwen4_exp(model=object())
    assert (
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm,
        modeling_qwen4_exp.Qwen4ExpTextMLP,
        modeling_qwen4_exp.Qwen4ExpTextExperts,
    ) == original_classes
    assert modeling_qwen4_exp.Qwen4ExpForCausalLM.forward is original_for_causal_lm_forward

    with pytest.raises(ValueError, match="cannot both be True"):
        apply_liger_kernel_to_qwen4_exp(cross_entropy=True, fused_linear_cross_entropy=True)
    assert (
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm,
        modeling_qwen4_exp.Qwen4ExpTextMLP,
        modeling_qwen4_exp.Qwen4ExpTextExperts,
    ) == original_classes

    unsupported_activation_config = copy.deepcopy(config)
    unsupported_activation_config.hidden_act = "gelu"
    unsupported_activation_config.output_gate_type = "silu"
    unsupported_activation_model = AutoModelForCausalLM.from_config(unsupported_activation_config)
    apply_liger_kernel_to_qwen4_exp(
        fused_linear_cross_entropy=False,
        rms_norm=False,
        engram=False,
        hyper_connection=False,
        model=unsupported_activation_model,
    )
    assert (
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm,
        modeling_qwen4_exp.Qwen4ExpTextMLP,
        modeling_qwen4_exp.Qwen4ExpTextExperts,
    ) == original_classes
    unsupported_mlp = unsupported_activation_model.model.layers[0].mlp
    assert inspect.getsource(unsupported_mlp.shared_expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
    assert inspect.getsource(unsupported_mlp.experts.forward) != inspect.getsource(LigerExperts.forward)

    def all_rms_norms(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextRMSNorm"]

    def all_mlps(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextMLP"]

    def all_experts(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextExperts"]

    def all_ngram_embeddings(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextNGramEmbedding"]

    def all_ple_layers(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextPLELayer"]

    native_ple_forward = all_ple_layers(dummy_model_instance)[0].forward.__func__
    native_ngram_forward = all_ngram_embeddings(dummy_model_instance)[0].forward.__func__

    def all_gated_residuals(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextGatedResidual"]

    def all_decoder_layers(model):
        return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextDecoderLayer"]

    def all_attention_modules(model):
        return [
            m for m in model.modules() if type(m).__name__ in ("Qwen4ExpTextGatedDeltaNet", "Qwen4ExpTextAttention")
        ]

    # Explicitly disabled kernels must leave an existing model instance unchanged.
    disabled_model_instance = AutoModelForCausalLM.from_config(config)
    _apply_liger_kernel_to_instance(
        model=disabled_model_instance,
        fused_linear_cross_entropy=False,
        rms_norm=False,
        swiglu=False,
        engram=False,
        hyper_connection=False,
    )
    assert inspect.getsource(disabled_model_instance.forward) != inspect.getsource(qwen4_exp_lce_forward)
    for norm in all_rms_norms(disabled_model_instance):
        assert inspect.getsource(norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        assert not getattr(norm, "_liger_rms_norm_patched", False)
    for mlp in all_mlps(disabled_model_instance):
        assert inspect.getsource(mlp.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
    for experts in all_experts(disabled_model_instance):
        assert inspect.getsource(experts.forward) != inspect.getsource(LigerExperts.forward)
    for ngram_embedding in all_ngram_embeddings(disabled_model_instance):
        assert inspect.getsource(ngram_embedding.forward) != inspect.getsource(liger_qwen4_exp_ngram_embedding_forward)
    for ple in all_ple_layers(disabled_model_instance):
        assert ple.forward.__func__ is native_ple_forward
    for gated_residual in all_gated_residuals(disabled_model_instance):
        assert inspect.getsource(gated_residual.forward) != inspect.getsource(liger_qwen4_exp_gated_residual_forward)
    for decoder_layer in all_decoder_layers(disabled_model_instance):
        assert inspect.getsource(decoder_layer.forward) != inspect.getsource(liger_qwen4_exp_decoder_layer_forward)

    assert len(all_rms_norms(dummy_model_instance)) > 0
    assert len(all_mlps(dummy_model_instance)) > 0
    assert len(all_experts(dummy_model_instance)) > 0
    assert len(all_ngram_embeddings(dummy_model_instance)) > 0
    assert len(all_ple_layers(dummy_model_instance)) > 0
    assert len(all_gated_residuals(dummy_model_instance)) > 0
    assert len(all_decoder_layers(dummy_model_instance)) > 0
    assert {type(module).__name__ for module in all_attention_modules(dummy_model_instance)} == {
        "Qwen4ExpTextGatedDeltaNet",
        "Qwen4ExpTextAttention",
    }
    attention_forward_sources = {
        id(module): inspect.getsource(module.forward) for module in all_attention_modules(dummy_model_instance)
    }
    native_state = {name: tensor.clone() for name, tensor in dummy_model_instance.state_dict().items()}
    native_norm_paths = {
        name for name, module in dummy_model_instance.named_modules() if type(module).__name__ == "Qwen4ExpTextRMSNorm"
    }
    assert native_norm_paths == {
        "model.layers.0.ple.norm_key",
        "model.layers.0.ple.norm_query",
        "model.layers.0.ple.norm_conv",
        "model.layers.0.attn_hyper_connection.hc_norm",
        "model.layers.0.mlp_hyper_connection.hc_norm",
        "model.layers.1.self_attn.q_norm",
        "model.layers.1.self_attn.k_norm",
        "model.layers.1.self_attn.indexer.q_layernorm",
        "model.layers.1.self_attn.indexer.k_layernorm",
        "model.layers.1.attn_hyper_connection.hc_norm",
        "model.layers.1.mlp_hyper_connection.hc_norm",
        "model.hyper_connection_mixer.hc_norm",
    }
    native_unpatched_rms_paths = {
        name: inspect.getsource(module.forward)
        for name, module in dummy_model_instance.named_modules()
        if type(module).__name__ == "Qwen4ExpTextRMSNormGated"
    }
    assert native_unpatched_rms_paths.keys() == {"model.layers.0.linear_attn.norm"}

    native_state_keys = set(native_state)
    assert {
        "model.layers.0.ple.ple_embedding.layer_multipliers",
        "model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes",
        "model.layers.0.ple.ple_embedding.ngram_heads_offsets",
    } <= native_state_keys
    assert any(".ple." in name for name in native_state_keys)
    assert any("_hyper_connection." in name for name in native_state_keys)
    assert all(f"{name}.weight" in native_state_keys for name in native_norm_paths)

    # Check that model instance variables are not yet patched with Liger modules
    assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen4_exp_lce_forward)
    for norm in all_rms_norms(dummy_model_instance):
        assert inspect.getsource(norm.forward) != inspect.getsource(LigerRMSNorm.forward)
    for mlp in all_mlps(dummy_model_instance):
        assert inspect.getsource(mlp.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
    for experts in all_experts(dummy_model_instance):
        assert inspect.getsource(experts.forward) != inspect.getsource(LigerExperts.forward)
    for ngram_embedding in all_ngram_embeddings(dummy_model_instance):
        assert inspect.getsource(ngram_embedding.forward) != inspect.getsource(liger_qwen4_exp_ngram_embedding_forward)
    for ple in all_ple_layers(dummy_model_instance):
        assert ple.forward.__func__ is native_ple_forward
    for gated_residual in all_gated_residuals(dummy_model_instance):
        assert inspect.getsource(gated_residual.forward) != inspect.getsource(liger_qwen4_exp_gated_residual_forward)
    for decoder_layer in all_decoder_layers(dummy_model_instance):
        assert inspect.getsource(decoder_layer.forward) != inspect.getsource(liger_qwen4_exp_decoder_layer_forward)

    # Test applying kernels to the model instance (auto-detected from model_type qwen4_exp_text)
    _apply_liger_kernel_to_instance(model=dummy_model_instance)

    # Check that the model's instance variables were correctly patched with Liger modules
    assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen4_exp_lce_forward)
    for norm in all_rms_norms(dummy_model_instance):
        assert inspect.getsource(norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        assert norm._liger_rms_norm_patched is True
        assert norm.eps == norm.variance_epsilon == config.rms_norm_eps
        assert norm.casting_mode == "gemma"
        assert norm.offset == 1.0
        assert norm.in_place is False
    for mlp in all_mlps(dummy_model_instance):
        assert inspect.getsource(mlp.forward) == inspect.getsource(liger_qwen4_exp_mlp_forward)
    for experts in all_experts(dummy_model_instance):
        assert inspect.getsource(experts.forward) == inspect.getsource(liger_qwen4_exp_experts_forward)
    for ngram_embedding in all_ngram_embeddings(dummy_model_instance):
        assert inspect.getsource(ngram_embedding.forward) == inspect.getsource(liger_qwen4_exp_ngram_embedding_forward)
        input_ids = torch.tensor([[11, 12, 2, 21]], dtype=torch.long)
        expected = native_ngram_forward(ngram_embedding, input_ids, None)
        assert torch.equal(ngram_embedding(input_ids, None), expected)
    for ple in all_ple_layers(dummy_model_instance):
        assert ple.forward.__func__ is native_ple_forward
    for gated_residual in all_gated_residuals(dummy_model_instance):
        assert inspect.getsource(gated_residual.forward) == inspect.getsource(liger_qwen4_exp_gated_residual_forward)
    for decoder_layer in all_decoder_layers(dummy_model_instance):
        assert inspect.getsource(decoder_layer.forward) == inspect.getsource(liger_qwen4_exp_decoder_layer_forward)
    for attention_module in all_attention_modules(dummy_model_instance):
        assert inspect.getsource(attention_module.forward) == attention_forward_sources[id(attention_module)]

    patched_state = dummy_model_instance.state_dict()
    assert patched_state.keys() == native_state.keys()
    for name, tensor in patched_state.items():
        assert torch.equal(tensor, native_state[name]), f"patch changed state for {name}"
    patched_modules = dict(dummy_model_instance.named_modules())
    assert all(
        inspect.getsource(patched_modules[name].forward) == inspect.getsource(LigerRMSNorm.forward)
        for name in native_norm_paths
    )
    assert all(
        inspect.getsource(patched_modules[name].forward) == source
        for name, source in native_unpatched_rms_paths.items()
    )

    # The grouped RMSNorm variant used by Gated Residual hyper-connections must keep its group size
    grouped_norms = [m for m in all_rms_norms(dummy_model_instance) if getattr(m, "group_size", None)]
    assert len(grouped_norms) > 0

    try:
        print(dummy_model_instance)
    except Exception as e:
        pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")

    # The base text model has no LM head. Default patching must skip fused LCE while still
    # applying all text-stack kernels instead of rejecting the supported model type.
    text_model_instance = Qwen4ExpTextModel(config)
    _apply_liger_kernel_to_instance(model=text_model_instance)
    assert inspect.getsource(text_model_instance.forward) != inspect.getsource(qwen4_exp_lce_forward)
    for norm in all_rms_norms(text_model_instance):
        assert inspect.getsource(norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    for mlp in all_mlps(text_model_instance):
        assert inspect.getsource(mlp.forward) == inspect.getsource(liger_qwen4_exp_mlp_forward)
    for experts in all_experts(text_model_instance):
        assert inspect.getsource(experts.forward) == inspect.getsource(liger_qwen4_exp_experts_forward)
    for ngram_embedding in all_ngram_embeddings(text_model_instance):
        assert inspect.getsource(ngram_embedding.forward) == inspect.getsource(liger_qwen4_exp_ngram_embedding_forward)
    for ple in all_ple_layers(text_model_instance):
        assert ple.forward.__func__ is native_ple_forward
    for gated_residual in all_gated_residuals(text_model_instance):
        assert inspect.getsource(gated_residual.forward) == inspect.getsource(liger_qwen4_exp_gated_residual_forward)
    for decoder_layer in all_decoder_layers(text_model_instance):
        assert inspect.getsource(decoder_layer.forward) == inspect.getsource(liger_qwen4_exp_decoder_layer_forward)


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.parametrize("patch_before_construction", [False, True], ids=["instance", "global"])
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_qwen4_exp_grouped_rms_norm_falls_back_for_historical_backend_abi(monkeypatch, patch_before_construction):
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    import liger_kernel.transformers.rms_norm as rms_norm_transformers

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    native_rms_norm_class = modeling_qwen4_exp.Qwen4ExpTextRMSNorm
    native_rms_norm_forward = native_rms_norm_class.forward
    backend_calls = []

    class HistoricalBackendRMSNormFunction:
        @staticmethod
        def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
            raise AssertionError("the test backend forward should only be reached through apply")

        @staticmethod
        def apply(X, W, eps, offset, casting_mode, in_place, row_mode):
            backend_calls.append((X, W, eps, offset, casting_mode, in_place, row_mode))
            return X

    monkeypatch.setattr(rms_norm_transformers, "LigerRMSNormFunction", HistoricalBackendRMSNormFunction)

    config = _qwen4_exp_compat_config()
    if patch_before_construction:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            swiglu=False,
            engram=False,
            hyper_connection=False,
        )
        model = modeling_qwen4_exp.Qwen4ExpTextModel(config)
    else:
        model = modeling_qwen4_exp.Qwen4ExpTextModel(config)
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            swiglu=False,
            engram=False,
            hyper_connection=False,
            model=model,
        )

    ordinary = next(
        module
        for module in model.modules()
        if getattr(module, "group_size", None) is None and isinstance(module, native_rms_norm_class)
    )
    grouped = next(module for module in model.modules() if getattr(module, "group_size", None) is not None)
    ordinary_input = torch.randn(2, ordinary.weight.numel())
    grouped_input = torch.randn(2, grouped.weight.numel())
    expected_grouped = native_rms_norm_forward(grouped, grouped_input)

    assert ordinary(ordinary_input) is ordinary_input
    calls_before_grouped = len(backend_calls)
    torch.testing.assert_close(grouped(grouped_input), expected_grouped)
    assert len(backend_calls) == calls_before_grouped
    assert all(len(call) == 7 for call in backend_calls)
    assert not getattr(grouped, "_liger_rms_norm_patched", False)
    if patch_before_construction:
        assert isinstance(grouped, native_rms_norm_class)
    else:
        assert grouped.forward.__func__ is native_rms_norm_forward
        assert ordinary.forward.__func__ is LigerRMSNorm.forward


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.parametrize("patch_before_construction", [False, True], ids=["instance", "global"])
@pytest.mark.parametrize(
    "hidden_act, experts_implementation, expect_liger_mlp, expect_liger_experts",
    [
        pytest.param("silu", "eager", True, True, id="silu-eager"),
        pytest.param("gelu", "eager", False, False, id="gelu-eager"),
        pytest.param("silu", "batched_mm", True, False, id="silu-batched-mm"),
        pytest.param("gelu", "batched_mm", False, False, id="gelu-batched-mm"),
    ],
)
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_qwen4_exp_swiglu_and_experts_dispatch_matrix(
    monkeypatch,
    patch_before_construction,
    hidden_act,
    experts_implementation,
    expect_liger_mlp,
    expect_liger_experts,
):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.swiglu import LigerExperts
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP
    from liger_kernel.utils import infer_device

    config = _qwen4_exp_compat_config(
        hidden_act=hidden_act,
        experts_implementation=experts_implementation,
    )
    test_device = infer_device()
    if test_device == "cpu" and (expect_liger_mlp or expect_liger_experts):
        pytest.skip("Liger SwiGLU execution requires an available accelerator backend")
    torch.manual_seed(11)
    reference = modeling_qwen4_exp.Qwen4ExpTextModel(config).to(test_device)
    hidden_states = torch.randn(5, config.hidden_size, device=test_device)
    top_k_index = torch.tensor([[0], [1], [2], [3], [0]], device=test_device)
    top_k_weights = torch.ones(5, 1, device=test_device)
    reference_sparse_moe = reference.layers[0].mlp
    expected_mlp = reference_sparse_moe.shared_expert(hidden_states)
    expected_experts = reference_sparse_moe.experts(hidden_states, top_k_index, top_k_weights)
    reference_state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}

    calls = {"liger_mlp": 0, "liger_experts": 0, "batched_mm": 0}
    original_liger_mlp_forward = LigerQwen3MoeSwiGLUMLP.forward
    original_liger_experts_forward = LigerExperts.forward
    original_batched_mm_forward = ALL_EXPERTS_FUNCTIONS["batched_mm"]

    def spy_liger_mlp_forward(self, *args, **kwargs):
        calls["liger_mlp"] += 1
        return original_liger_mlp_forward(self, *args, **kwargs)

    def spy_liger_experts_forward(self, *args, **kwargs):
        calls["liger_experts"] += 1
        return original_liger_experts_forward(self, *args, **kwargs)

    def spy_batched_mm_forward(self, *args, **kwargs):
        calls["batched_mm"] += 1
        return original_batched_mm_forward(self, *args, **kwargs)

    monkeypatch.setattr(LigerQwen3MoeSwiGLUMLP, "forward", spy_liger_mlp_forward)
    monkeypatch.setattr(LigerExperts, "forward", spy_liger_experts_forward)
    monkeypatch.setitem(ALL_EXPERTS_FUNCTIONS, "batched_mm", spy_batched_mm_forward)

    if patch_before_construction:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            engram=False,
            hyper_connection=False,
        )
        torch.manual_seed(11)
        candidate = modeling_qwen4_exp.Qwen4ExpTextModel(copy.deepcopy(config)).to(test_device)
    else:
        torch.manual_seed(11)
        candidate = modeling_qwen4_exp.Qwen4ExpTextModel(copy.deepcopy(config)).to(test_device)
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            engram=False,
            hyper_connection=False,
            model=candidate,
        )
    candidate.load_state_dict(reference_state)
    assert candidate.state_dict().keys() == reference_state.keys()
    candidate_sparse_moe = candidate.layers[0].mlp
    actual_mlp = candidate_sparse_moe.shared_expert(hidden_states)
    actual_experts = candidate_sparse_moe.experts(hidden_states, top_k_index, top_k_weights)

    torch.testing.assert_close(actual_mlp, expected_mlp)
    torch.testing.assert_close(actual_experts, expected_experts, atol=2e-5, rtol=2e-5)
    assert calls["liger_mlp"] == int(expect_liger_mlp)
    assert calls["liger_experts"] == int(expect_liger_experts)
    assert calls["batched_mm"] == int(experts_implementation == "batched_mm")


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_apply_liger_kernel_to_qwen4_exp_before_construction_feature_flag_independence():
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from liger_kernel.transformers.model.qwen4_exp import lce_forward
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_decoder_layer_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_gated_residual_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_ngram_embedding_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen4Exp

    config = Qwen4ExpTextConfig(
        dtype=torch.bfloat16,
        rms_norm_eps=1e-5,
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=1,
        num_experts=4,
        hc_count=4,
        hc_lowrank=8,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )
    saved_attributes = {
        "Qwen4ExpTextRMSNorm": modeling_qwen4_exp.Qwen4ExpTextRMSNorm,
        "Qwen4ExpTextMLP": modeling_qwen4_exp.Qwen4ExpTextMLP,
        "Qwen4ExpTextExperts": modeling_qwen4_exp.Qwen4ExpTextExperts,
    }
    saved_forwards = {
        "Qwen4ExpTextNGramEmbedding": modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding.forward,
        "Qwen4ExpTextPLELayer": modeling_qwen4_exp.Qwen4ExpTextPLELayer.forward,
        "Qwen4ExpTextGatedResidual": modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward,
        "Qwen4ExpTextDecoderLayer": modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward,
        "Qwen4ExpForCausalLM": modeling_qwen4_exp.Qwen4ExpForCausalLM.forward,
    }

    def restore_native_classes():
        for name, value in saved_attributes.items():
            setattr(modeling_qwen4_exp, name, value)
        for name, value in saved_forwards.items():
            getattr(modeling_qwen4_exp, name).forward = value

    def construct_with_flags(**flags):
        restore_native_classes()
        torch.manual_seed(7)
        reference = modeling_qwen4_exp.Qwen4ExpForCausalLM(config)
        apply_liger_kernel_to_qwen4_exp(
            cross_entropy=False,
            swiglu=False,
            **flags,
        )
        torch.manual_seed(7)
        candidate = modeling_qwen4_exp.Qwen4ExpForCausalLM(config)
        reference_state = reference.state_dict()
        candidate_state = candidate.state_dict()
        assert candidate_state.keys() == reference_state.keys()
        for name in reference_state:
            assert torch.equal(candidate_state[name], reference_state[name]), f"initialization mismatch for {name}"
        return candidate

    try:
        engram_only = construct_with_flags(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            engram=True,
            hyper_connection=False,
        )
        assert any(type(module) is saved_attributes["Qwen4ExpTextRMSNorm"] for module in engram_only.modules())
        engram_embeddings = [
            module for module in engram_only.modules() if type(module).__name__ == "Qwen4ExpTextNGramEmbedding"
        ]
        ple_layers = [module for module in engram_only.modules() if type(module).__name__ == "Qwen4ExpTextPLELayer"]
        assert engram_embeddings and ple_layers
        if torch.cuda.is_available() and torch.version.hip is None:
            assert all(
                module.forward.__func__ is liger_qwen4_exp_ngram_embedding_forward for module in engram_embeddings
            )
        else:
            assert all(
                module.forward.__func__ is saved_forwards["Qwen4ExpTextNGramEmbedding"] for module in engram_embeddings
            )
        assert all(module.forward.__func__ is saved_forwards["Qwen4ExpTextPLELayer"] for module in ple_layers)

        rms_only = construct_with_flags(
            fused_linear_cross_entropy=False,
            rms_norm=True,
            engram=False,
            hyper_connection=False,
        )
        liger_norms = [module for module in rms_only.modules() if isinstance(module, LigerRMSNormForQwen4Exp)]
        assert liger_norms
        assert all(module._liger_rms_norm_patched is True for module in liger_norms)
        assert all(module.variance_epsilon == config.rms_norm_eps for module in liger_norms)
        assert any(module.group_size is None for module in liger_norms)
        assert any(module.group_size == config.hidden_size for module in liger_norms)
        assert all(
            module.forward.__func__ is saved_forwards["Qwen4ExpTextPLELayer"]
            for module in rms_only.modules()
            if type(module).__name__ == "Qwen4ExpTextPLELayer"
        )

        no_hyper_connection = construct_with_flags(
            fused_linear_cross_entropy=False,
            rms_norm=True,
            engram=True,
            hyper_connection=False,
        )
        gated_residuals = [
            module for module in no_hyper_connection.modules() if type(module).__name__ == "Qwen4ExpTextGatedResidual"
        ]
        decoder_layers = [
            module for module in no_hyper_connection.modules() if type(module).__name__ == "Qwen4ExpTextDecoderLayer"
        ]
        assert gated_residuals and decoder_layers
        assert all(module.forward.__func__ is saved_forwards["Qwen4ExpTextGatedResidual"] for module in gated_residuals)
        assert all(module.forward.__func__ is saved_forwards["Qwen4ExpTextDecoderLayer"] for module in decoder_layers)

        flce_off = construct_with_flags(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            engram=False,
            hyper_connection=False,
        )
        assert flce_off.forward.__func__ is saved_forwards["Qwen4ExpForCausalLM"]
        flce_on = construct_with_flags(
            fused_linear_cross_entropy=True,
            rms_norm=False,
            engram=False,
            hyper_connection=False,
        )
        assert flce_on.forward.__func__ is lce_forward

        hyper_enabled = construct_with_flags(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            engram=False,
            hyper_connection=True,
        )
        gated_residuals = [
            module for module in hyper_enabled.modules() if type(module).__name__ == "Qwen4ExpTextGatedResidual"
        ]
        decoder_layers = [
            module for module in hyper_enabled.modules() if type(module).__name__ == "Qwen4ExpTextDecoderLayer"
        ]
        assert gated_residuals and decoder_layers
        assert all(module.forward.__func__ is liger_qwen4_exp_gated_residual_forward for module in gated_residuals)
        assert all(module.forward.__func__ is liger_qwen4_exp_decoder_layer_forward for module in decoder_layers)

        restore_native_classes()
        apply_liger_kernel_to_qwen4_exp()
        global_ngram = modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, 0, 0)
        input_ids = torch.tensor([[11, 12, 2, 21]], dtype=torch.long)
        expected = saved_forwards["Qwen4ExpTextNGramEmbedding"](global_ngram, input_ids, None)
        assert torch.equal(global_ngram(input_ids, None), expected)
    finally:
        restore_native_classes()


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.parametrize(
    "detected_device, is_rocm",
    [
        pytest.param("cpu", False, id="cpu"),
        pytest.param("xpu", False, id="xpu"),
        pytest.param("cuda", True, id="rocm"),
    ],
)
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_apply_liger_kernel_to_qwen4_exp_keeps_native_qwen_specific_paths_off_nvidia_cuda(
    monkeypatch, detected_device, is_rocm
):
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_experts_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_mlp_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen4Exp

    native_rms_norm_class = modeling_qwen4_exp.Qwen4ExpTextRMSNorm
    native_mlp_class = modeling_qwen4_exp.Qwen4ExpTextMLP
    native_experts_class = modeling_qwen4_exp.Qwen4ExpTextExperts
    native_mlp_forward = native_mlp_class.forward
    native_experts_forward = native_experts_class.forward
    native_ngram_forward = modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding.forward
    native_gated_residual_forward = modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward
    native_decoder_layer_forward = modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward
    monkeypatch.setattr("liger_kernel.utils.infer_device", lambda: detected_device)
    monkeypatch.setattr("liger_kernel.ops.utils.is_hip", lambda: is_rocm)

    # Exercise the already-constructed instance path without allocating a full model. These are real
    # HF module classes, and Module.modules() traverses the same objects the high-level patch sees.
    instance = object.__new__(modeling_qwen4_exp.Qwen4ExpTextModel)
    torch.nn.Module.__init__(instance)
    mock_config = MagicMock(hidden_act="silu")
    mock_config.get_text_config.return_value = mock_config
    instance.config = mock_config
    instance.gated_residual = object.__new__(modeling_qwen4_exp.Qwen4ExpTextGatedResidual)
    torch.nn.Module.__init__(instance.gated_residual)
    instance.decoder_layer = object.__new__(modeling_qwen4_exp.Qwen4ExpTextDecoderLayer)
    torch.nn.Module.__init__(instance.decoder_layer)

    try:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            model=instance,
        )
        # Shared primitives retain their existing backend policy. Only the Qwen-specific
        # n-gram and HyperConnection/decoder paths are gated to NVIDIA CUDA.
        assert modeling_qwen4_exp.Qwen4ExpTextRMSNorm is LigerRMSNormForQwen4Exp
        assert modeling_qwen4_exp.Qwen4ExpTextMLP is native_mlp_class
        assert modeling_qwen4_exp.Qwen4ExpTextMLP.forward is liger_qwen4_exp_mlp_forward
        assert modeling_qwen4_exp.Qwen4ExpTextExperts is native_experts_class
        assert modeling_qwen4_exp.Qwen4ExpTextExperts.forward is liger_qwen4_exp_experts_forward
        assert modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding.forward is native_ngram_forward
        assert modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward is native_gated_residual_forward
        assert modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward is native_decoder_layer_forward
        assert instance.gated_residual.forward.__func__ is native_gated_residual_forward
        assert instance.decoder_layer.forward.__func__ is native_decoder_layer_forward
    finally:
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm = native_rms_norm_class
        native_mlp_class.forward = native_mlp_forward
        native_experts_class.forward = native_experts_forward
        modeling_qwen4_exp.Qwen4ExpTextMLP = native_mlp_class
        modeling_qwen4_exp.Qwen4ExpTextExperts = native_experts_class
        modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding.forward = native_ngram_forward
        modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward = native_gated_residual_forward
        modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward = native_decoder_layer_forward


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_qwen4_exp_hyper_connection_global_and_instance_patch_fall_back_for_cpu_tensors(monkeypatch):
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_decoder_layer_forward
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_gated_residual_forward

    gated_class = modeling_qwen4_exp.Qwen4ExpTextGatedResidual
    decoder_class = modeling_qwen4_exp.Qwen4ExpTextDecoderLayer
    native_gated_forward = gated_class.forward
    native_decoder_forward = decoder_class.forward
    native_attributes = {
        gated_class: "_liger_qwen4_exp_native_gated_residual_forward",
        decoder_class: "_liger_qwen4_exp_native_decoder_forward",
    }
    saved_native_attributes = {
        module_class: module_class.__dict__.get(attribute) for module_class, attribute in native_attributes.items()
    }
    had_native_attributes = {
        module_class: attribute in module_class.__dict__ for module_class, attribute in native_attributes.items()
    }

    class NativeHyperConnection(torch.nn.Module):
        def forward(self, hidden_states):
            grouped = hidden_states.unflatten(-1, (4, hidden_states.shape[-1] // 4))
            mixed = grouped.mean(dim=-2)
            injection_weights = hidden_states.new_ones((*hidden_states.shape[:-1], 4))
            return mixed, hidden_states, injection_weights

    class NativeAttention(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings, **kwargs):
            return hidden_states * 0.5, None

    def make_decoder():
        decoder = object.__new__(decoder_class)
        torch.nn.Module.__init__(decoder)
        decoder.ple = None
        decoder.attn_hyper_connection = NativeHyperConnection()
        decoder.layer_type = "full_attention"
        decoder.self_attn = NativeAttention()
        decoder.mlp_hyper_connection = NativeHyperConnection()
        decoder.mlp = torch.nn.Identity()
        return decoder

    config = Qwen4ExpTextConfig(hidden_size=8, hc_count=4, hc_lowrank=4, rms_norm_eps=1e-5)
    existing_gated = gated_class(config, use_combine=True)
    existing_decoder = make_decoder()
    instance = object.__new__(modeling_qwen4_exp.Qwen4ExpTextModel)
    torch.nn.Module.__init__(instance)
    mock_config = MagicMock(hidden_act="silu")
    mock_config.get_text_config.return_value = mock_config
    instance.config = mock_config
    instance.gated_residual = existing_gated
    instance.decoder_layer = existing_decoder

    monkeypatch.setattr("liger_kernel.utils.infer_device", lambda: "cuda")
    monkeypatch.setattr("liger_kernel.ops.utils.is_hip", lambda: False)
    monkeypatch.setattr("liger_kernel.transformers.qwen4_exp.is_hip", lambda: False)

    try:
        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            swiglu=False,
            engram=False,
            model=instance,
        )
        assert gated_class.forward is liger_qwen4_exp_gated_residual_forward
        assert decoder_class.forward is liger_qwen4_exp_decoder_layer_forward
        assert existing_gated.forward.__func__ is liger_qwen4_exp_gated_residual_forward
        assert existing_decoder.forward.__func__ is liger_qwen4_exp_decoder_layer_forward

        hyper_input = torch.randn(2, 3, config.hc_count * config.hidden_size)
        expected_gated = native_gated_forward(existing_gated, hyper_input)
        actual_gated = existing_gated(hyper_input)
        for actual, expected in zip(actual_gated, expected_gated):
            assert torch.equal(actual, expected)

        position_embeddings = (torch.empty(0), torch.empty(0))
        expected_decoder = native_decoder_forward(existing_decoder, hyper_input, position_embeddings)
        actual_decoder = existing_decoder(hyper_input, position_embeddings)
        assert torch.equal(actual_decoder, expected_decoder)

        # Modules constructed after the global class patch use the same saved HF forwards
        # when their runtime tensors are placed on CPU.
        global_gated = gated_class(config, use_combine=True)
        global_decoder = make_decoder()
        expected_global_gated = native_gated_forward(global_gated, hyper_input)
        actual_global_gated = global_gated(hyper_input)
        for actual, expected in zip(actual_global_gated, expected_global_gated):
            assert torch.equal(actual, expected)
        assert torch.equal(
            global_decoder(hyper_input, position_embeddings),
            native_decoder_forward(global_decoder, hyper_input, position_embeddings),
        )
    finally:
        gated_class.forward = native_gated_forward
        decoder_class.forward = native_decoder_forward
        for module_class, attribute in native_attributes.items():
            if had_native_attributes[module_class]:
                setattr(module_class, attribute, saved_native_attributes[module_class])
            elif attribute in module_class.__dict__:
                delattr(module_class, attribute)


@pytest.mark.skipif(not is_qwen4_exp_available(), reason="qwen4_exp module not available")
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Qwen4Exp Liger n-gram patch requires NVIDIA CUDA",
)
@pytest.mark.usefixtures("qwen4_exp_globals")
def test_qwen4_exp_instance_ngram_fallback_is_isolated_from_class_patch():
    from types import MethodType

    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen4_exp
    from liger_kernel.transformers.qwen4_exp import liger_qwen4_exp_ngram_embedding_forward
    from liger_kernel.transformers.qwen4_exp import patch_qwen4_exp_text_module_for_ngram

    ngram_class = modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding
    native_forward = ngram_class.forward
    native_attr = "_liger_qwen4_exp_native_ngram_forward"
    saved_native_attr = ngram_class.__dict__.get(native_attr)
    had_native_attr = native_attr in ngram_class.__dict__
    if had_native_attr:
        delattr(ngram_class, native_attr)

    config = Qwen4ExpTextConfig(
        vocab_size=101,
        hidden_size=32,
        hc_count=4,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
    )
    patched = ngram_class(config, config.ple_embed_dim, 0, 0)
    untouched = ngram_class(config, config.ple_embed_dim, 0, 0)
    input_ids = torch.tensor([[11, 12, 2, 21]], dtype=torch.long)
    custom_calls = 0

    def custom_forward(self, input_ids, past_key_values):
        nonlocal custom_calls
        custom_calls += 1
        return torch.full((*input_ids.shape, config.ple_embed_dim), 17.0)

    patched.forward = MethodType(custom_forward, patched)

    try:
        patch_qwen4_exp_text_module_for_ngram(patched, ngram_class)
        assert patched.forward.__func__ is liger_qwen4_exp_ngram_embedding_forward
        assert patched.__dict__[native_attr] is custom_forward
        assert untouched.forward.__func__ is native_forward
        assert ngram_class.forward is native_forward
        assert native_attr not in ngram_class.__dict__
        expected_custom = torch.full((*input_ids.shape, config.ple_embed_dim), 17.0)
        assert torch.equal(patched(input_ids, None), expected_custom)

        apply_liger_kernel_to_qwen4_exp(
            fused_linear_cross_entropy=False,
            rms_norm=False,
            swiglu=False,
            hyper_connection=False,
        )
        assert ngram_class.__dict__[native_attr] is native_forward
        assert torch.equal(patched(input_ids, None), expected_custom)

        constructed_after_global_patch = ngram_class(config, config.ple_embed_dim, 0, 0)
        expected_native = native_forward(constructed_after_global_patch, input_ids, None)
        assert torch.equal(constructed_after_global_patch(input_ids, None), expected_native)
        assert custom_calls == 2
    finally:
        ngram_class.forward = native_forward
        if had_native_attr:
            setattr(ngram_class, native_attr, saved_native_attr)
        elif native_attr in ngram_class.__dict__:
            delattr(ngram_class, native_attr)


@pytest.mark.skipif(not is_qwen3_5_moe_available(), reason="qwen3_5_moe module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_5_moe():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
        from liger_kernel.transformers.model.qwen3_5_moe import lce_forward as qwen3_5_moe_lce_forward

        # Instantiate a dummy model
        config = transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeTextConfig(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_5_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.mlp.shared_expert.forward) != inspect.getsource(
                LigerQwen3MoeSwiGLUMLP.forward
            )
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_5_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.mlp.shared_expert.forward) == inspect.getsource(
                LigerQwen3MoeSwiGLUMLP.forward
            )
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


def _build_qwen3_5_moe_multimodal_config():
    text_config = transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeTextConfig(
        dtype=torch.bfloat16,
        rms_norm_eps=1e-5,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hidden_act="silu",
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
    )
    vision_config = transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeVisionConfig(
        depth=2,
        hidden_size=32,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=64,
        num_heads=2,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=64,
        initializer_range=0.02,
    )
    return transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeConfig(
        attn_implementation="sdpa",
        image_token_id=4,
        video_token_id=5,
        vision_start_token_id=1,
        vision_end_token_id=2,
        tie_word_embeddings=True,
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
    )


@pytest.mark.skipif(not is_qwen3_5_moe_available(), reason="qwen3_5_moe module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_5_moe_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

        from liger_kernel.transformers.model.qwen3_5_moe import (
            lce_forward_conditional_generation as qwen3_5_moe_conditional_generation_lce_forward,
        )

        config = _build_qwen3_5_moe_multimodal_config()
        dummy_model_instance = Qwen3_5MoeForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3_5MoeForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(
            qwen3_5_moe_conditional_generation_lce_forward
        )
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.mlp.shared_expert.forward) != inspect.getsource(
                LigerQwen3MoeSwiGLUMLP.forward
            )
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(
            qwen3_5_moe_conditional_generation_lce_forward
        )
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.mlp.shared_expert.forward) == inspect.getsource(
                LigerQwen3MoeSwiGLUMLP.forward
            )
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_5_moe_available(), reason="qwen3_5_moe module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_5_moe_model():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

        config = _build_qwen3_5_moe_multimodal_config()
        dummy_model_instance = Qwen3_5MoeModel._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3_5MoeModel)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.mlp.shared_expert.forward) != inspect.getsource(
                LigerQwen3MoeSwiGLUMLP.forward
            )
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.language_model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.mlp.shared_expert.forward) == inspect.getsource(
                LigerQwen3MoeSwiGLUMLP.forward
            )
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_5_available(), reason="qwen3_5 module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_5():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_5.modeling_qwen3_5"):
        # Instantiate a dummy model
        config = transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_5_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_5_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_qwen3_5_available(), reason="qwen3_5 module not available")
def test_apply_liger_kernel_to_instance_for_qwen3_5_for_conditional_generation():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.qwen3_5.modeling_qwen3_5"):
        # Instantiate a dummy model
        config = transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5Config(
            attn_implementation="sdpa",
            image_token_id=4,
            video_token_id=5,
            vision_start_token_id=1,
            vision_end_token_id=2,
            tie_word_embeddings=True,
            vision_config=transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5VisionConfig(
                depth=4,
                hidden_size=256,
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=512,
                num_heads=4,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=512,
                num_position_embeddings=256,
                initializer_range=0.02,
            ).to_dict(),
            text_config=transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig(
                dtype=torch.bfloat16,
                rms_norm_eps=1e-5,
                hidden_size=32,
                intermediate_size=64,
                hidden_act="silu",
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=16,
                linear_conv_kernel_dim=4,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            ).to_dict(),
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        dummy_model_instance = Qwen3_5ForConditionalGeneration._from_config(config)

        assert isinstance(dummy_model_instance, Qwen3_5ForConditionalGeneration)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(qwen3_5_lce_forward_for_multimodal)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) != inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(qwen3_5_lce_forward_for_multimodal)
        assert inspect.getsource(dummy_model_instance.model.language_model.norm.forward) == inspect.getsource(
            LigerRMSNorm.forward
        )
        for layer in dummy_model_instance.model.language_model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_hunyuan_v1_available(), reason="hunyuan_v1 module not available")
def test_apply_liger_kernel_to_instance_for_hunyuan_v1_moe():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe"):
        from liger_kernel.transformers.model.hunyuan_v1 import lce_forward as hunyuan_v1_moe_lce_forward

        # Instantiate a dummy model
        config = transformers.models.hunyuan_v1_moe.configuration_hunyuan_v1_moe.HunYuanMoEV1Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            head_dim=1,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(hunyuan_v1_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) != inspect.getsource(LigerExperts.forward)
            else:
                for expert in layer.mlp.experts:
                    assert inspect.getsource(expert.forward) != inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(hunyuan_v1_moe_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            if IS_TRANSFORMERS_V5_OR_LATER:
                assert inspect.getsource(layer.mlp.experts.forward) == inspect.getsource(LigerExperts.forward)
            else:
                for mlp_expert in layer.mlp.experts:
                    assert inspect.getsource(mlp_expert.forward) == inspect.getsource(LigerQwen3MoeSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_hunyuan_v1_available(), reason="hunyuan_v1_dense module not available")
def test_apply_liger_kernel_to_instance_for_hunyuan_v1_dense():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense"):
        from liger_kernel.transformers.model.hunyuan_v1 import lce_forward as hunyuan_v1_dense_lce_forward

        # Instantiate a dummy model
        config = transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense.HunYuanDenseV1Config(
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            head_dim=1,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(hunyuan_v1_dense_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) != inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's instance variables were correctly patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(hunyuan_v1_dense_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_nemotron_available(), reason="nemotron not available")
def test_apply_liger_kernel_to_instance_for_nemotron():
    from liger_kernel.transformers.model.nemotron import lce_forward as nemotron_lce_forward
    from liger_kernel.transformers.relu_squared import LigerReLUSquared

    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.nemotron.modeling_nemotron"):
        # Instantiate a dummy model
        config = transformers.models.nemotron.configuration_nemotron.NemotronConfig(
            hidden_size=32,
            intermediate_size=64,
            hidden_act="relu2",
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            norm_eps=1e-5,
        )
        dummy_model_instance = AutoModelForCausalLM.from_config(config)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(nemotron_lce_forward)
        for decoder_layer in dummy_model_instance.model.layers:
            assert not isinstance(decoder_layer.mlp.act_fn, LigerReLUSquared)

        # Test applying kernels to the model instance
        _apply_liger_kernel_to_instance(model=dummy_model_instance)

        # Check that the model's forward was correctly patched
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(nemotron_lce_forward)

        # Check that the activation function was correctly patched
        for decoder_layer in dummy_model_instance.model.layers:
            assert isinstance(decoder_layer.mlp.act_fn, LigerReLUSquared)

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")


@pytest.mark.skipif(not is_exaone4_available(), reason="exaone4 module not available")
def test_apply_liger_kernel_to_instance_for_exaone4():
    # Ensure any monkey patching is cleaned up for subsequent tests
    with patch("transformers.models.exaone4.modeling_exaone4"):
        from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM

        from liger_kernel.transformers.model.exaone4 import lce_forward as exaone4_lce_forward

        # Instantiate a dummy model
        config = transformers.models.exaone4.configuration_exaone4.Exaone4Config(
            dtype=torch.bfloat16,
            rms_norm_eps=1e-5,
            hidden_size=32,
            intermediate_size=64,
            hidden_act="silu",
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        dummy_model_instance = Exaone4ForCausalLM._from_config(config)
        assert isinstance(dummy_model_instance, Exaone4ForCausalLM)

        # Check that model instance variables are not yet patched with Liger modules
        assert inspect.getsource(dummy_model_instance.forward) != inspect.getsource(exaone4_lce_forward)
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
        assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource(exaone4_lce_forward)
        assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        for layer in dummy_model_instance.model.layers:
            assert inspect.getsource(layer.mlp.forward) == inspect.getsource(LigerSwiGLUMLP.forward)
            assert layer.mlp._get_name() == LigerSwiGLUMLP.__name__
            assert inspect.getsource(layer.post_attention_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.post_feedforward_layernorm.forward) == inspect.getsource(
                LigerRMSNorm.forward
            )

        try:
            print(dummy_model_instance)
        except Exception as e:
            pytest.fail(f"An exception occured in extra_expr: {type(e).__name__} - {e}")
