import inspect

import pytest

from liger_kernel.transformers.auto_model import AutoLigerKernelForCausalLM
from liger_kernel.transformers.lfm2_moe_router import liger_lfm2_moe_route_tokens_to_experts
from liger_kernel.transformers.lfm2_short_conv import liger_lfm2_short_conv_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as lfm2_lce_forward
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerLfm2MoeExperts
from liger_kernel.transformers.swiglu import LigerLfm2SwiGLUMLP


def _has_module(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_LFM2 = _has_module("transformers.models.lfm2.modeling_lfm2")
HAS_LFM2_MOE = _has_module("transformers.models.lfm2_moe.modeling_lfm2_moe")
HAS_LFM2_VL = _has_module("transformers.models.lfm2_vl.modeling_lfm2_vl")


def _lfm2_config(**overrides):
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

    config = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "block_multiple_of": 8,
        "block_auto_adjust_ff_dim": False,
        "layer_types": ["conv", "full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "use_cache": False,
    }
    config.update(overrides)
    return Lfm2Config(**config)


@pytest.mark.skipif(not HAS_LFM2, reason="lfm2 module not available")
def test_apply_liger_kernel_to_lfm2_instance():
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    model = Lfm2ForCausalLM(_lfm2_config())
    _apply_liger_kernel_to_instance(model)

    assert inspect.getsource(model.forward) == inspect.getsource(lfm2_lce_forward)
    assert inspect.getsource(model.model.embedding_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    for layer in model.model.layers:
        assert inspect.getsource(layer.feed_forward.forward) == inspect.getsource(LigerLfm2SwiGLUMLP.forward)
        assert inspect.getsource(layer.operator_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        assert inspect.getsource(layer.ffn_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
        if hasattr(layer, "conv"):
            assert inspect.getsource(layer.conv.forward) == inspect.getsource(liger_lfm2_short_conv_forward)
        if hasattr(layer, "self_attn"):
            assert inspect.getsource(layer.self_attn.q_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
            assert inspect.getsource(layer.self_attn.k_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)


@pytest.mark.skipif(not HAS_LFM2_MOE, reason="lfm2_moe module not available")
def test_apply_liger_kernel_to_lfm2_moe_instance():
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM

    config = Lfm2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_dense_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["conv", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        use_cache=False,
    )
    model = Lfm2MoeForCausalLM(config)
    _apply_liger_kernel_to_instance(model)

    assert inspect.getsource(model.forward) == inspect.getsource(lfm2_lce_forward)
    assert inspect.getsource(model.model.embedding_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    dense_layer, sparse_layer = model.model.layers
    assert inspect.getsource(dense_layer.feed_forward.forward) == inspect.getsource(LigerLfm2SwiGLUMLP.forward)
    assert inspect.getsource(sparse_layer.feed_forward.experts.forward) == inspect.getsource(
        LigerLfm2MoeExperts.forward
    )
    assert inspect.getsource(sparse_layer.feed_forward.route_tokens_to_experts) == inspect.getsource(
        liger_lfm2_moe_route_tokens_to_experts
    )

    assert inspect.getsource(dense_layer.conv.forward) == inspect.getsource(liger_lfm2_short_conv_forward)


@pytest.mark.skipif(not HAS_LFM2_VL, reason="lfm2_vl module not available")
def test_apply_liger_kernel_to_lfm2_vl_instance():
    from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration

    from liger_kernel.transformers.model.lfm2_vl import lce_forward as lfm2_vl_lce_forward

    text_config = _lfm2_config().to_dict()
    vision_config = {
        "model_type": "siglip2_vision_model",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "num_patches": 16,
        "patch_size": 2,
        "vision_use_head": False,
    }
    config = Lfm2VlConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=127,
        projector_hidden_size=32,
        downsample_factor=2,
    )
    model = Lfm2VlForConditionalGeneration(config)
    _apply_liger_kernel_to_instance(model)

    assert inspect.getsource(model.forward) == inspect.getsource(lfm2_vl_lce_forward)
    language_model = model.model.language_model
    assert inspect.getsource(language_model.embedding_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    for layer in language_model.layers:
        assert inspect.getsource(layer.feed_forward.forward) == inspect.getsource(LigerLfm2SwiGLUMLP.forward)

    vision_model = getattr(model.model.vision_tower, "vision_model", model.model.vision_tower)
    assert vision_model.post_layernorm._get_name() == "LigerLayerNorm"
    for layer in vision_model.encoder.layers:
        assert layer.layer_norm1._get_name() == "LigerLayerNorm"
        assert layer.layer_norm2._get_name() == "LigerLayerNorm"
    assert model.model.multi_modal_projector.layer_norm._get_name() == "LigerLayerNorm"


@pytest.mark.skipif(not HAS_LFM2, reason="lfm2 module not available")
def test_auto_liger_kernel_for_lfm2_from_config():
    model = AutoLigerKernelForCausalLM.from_config(_lfm2_config())

    assert isinstance(model.model.layers[0].feed_forward, LigerLfm2SwiGLUMLP)
    assert inspect.getsource(model.model.layers[0].conv.forward) == inspect.getsource(liger_lfm2_short_conv_forward)


@pytest.mark.skipif(not HAS_LFM2_MOE, reason="lfm2_moe module not available")
def test_auto_liger_kernel_for_lfm2_moe_from_config():
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig

    config = Lfm2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_dense_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["conv", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        use_cache=False,
    )
    model = AutoLigerKernelForCausalLM.from_config(config)

    dense_layer, sparse_layer = model.model.layers
    assert isinstance(dense_layer.feed_forward, LigerLfm2SwiGLUMLP)
    assert isinstance(sparse_layer.feed_forward.experts, LigerLfm2MoeExperts)
    assert inspect.getsource(sparse_layer.feed_forward.route_tokens_to_experts) == inspect.getsource(
        liger_lfm2_moe_route_tokens_to_experts
    )
