import inspect

import pytest
import torch

from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.auto_model import AutoLigerKernelForCausalLM
from liger_kernel.transformers.lfm2_moe_router import liger_lfm2_moe_route_tokens_to_experts
from liger_kernel.transformers.lfm2_short_conv import liger_lfm2_short_conv_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as lfm2_lce_forward
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerLfm2MoeExperts
from liger_kernel.transformers.swiglu import LigerLfm2SwiGLUMLP
from liger_kernel.utils import infer_device


def _has_module(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_LFM2 = _has_module("transformers.models.lfm2.modeling_lfm2")
HAS_LFM2_MOE = _has_module("transformers.models.lfm2_moe.modeling_lfm2_moe")
HAS_LFM2_VL = _has_module("transformers.models.lfm2_vl.modeling_lfm2_vl")
device = infer_device()


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
    original_forward = inspect.getsource(model.forward)
    _apply_liger_kernel_to_instance(model)

    expected_forward = inspect.getsource(lfm2_lce_forward) if is_hip() else original_forward
    assert inspect.getsource(model.forward) == expected_forward
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
    original_forward = inspect.getsource(model.forward)
    _apply_liger_kernel_to_instance(model)

    expected_forward = inspect.getsource(lfm2_lce_forward) if is_hip() else original_forward
    assert inspect.getsource(model.forward) == expected_forward
    assert inspect.getsource(model.model.embedding_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    dense_layer, sparse_layer = model.model.layers
    assert inspect.getsource(dense_layer.feed_forward.forward) == inspect.getsource(LigerLfm2SwiGLUMLP.forward)
    assert inspect.getsource(sparse_layer.feed_forward.experts.forward) == inspect.getsource(
        LigerLfm2MoeExperts.forward
    )
    assert sparse_layer.feed_forward.experts.has_gate
    assert not sparse_layer.feed_forward.experts.has_bias
    assert not sparse_layer.feed_forward.experts.is_transposed
    assert inspect.getsource(sparse_layer.feed_forward.experts._apply_gate) == inspect.getsource(
        LigerLfm2MoeExperts._apply_gate
    )
    assert inspect.getsource(sparse_layer.feed_forward.route_tokens_to_experts) == inspect.getsource(
        liger_lfm2_moe_route_tokens_to_experts
    )

    assert inspect.getsource(dense_layer.conv.forward) == inspect.getsource(liger_lfm2_short_conv_forward)


@pytest.mark.skipif(not HAS_LFM2_VL, reason="lfm2_vl module not available")
@pytest.mark.parametrize(
    ("layer_norm", "expected_liger_layer_norm"),
    [
        (None, False),
        (True, True),
    ],
)
def test_apply_liger_kernel_to_lfm2_vl_instance(layer_norm, expected_liger_layer_norm):
    from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration

    from liger_kernel.transformers import monkey_patch
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
    original_forward = inspect.getsource(model.forward)
    monkey_patch.apply_liger_kernel_to_lfm2_vl(model=model, layer_norm=layer_norm)

    expected_forward = inspect.getsource(lfm2_vl_lce_forward) if is_hip() else original_forward
    assert inspect.getsource(model.forward) == expected_forward
    language_model = model.model.language_model
    assert inspect.getsource(language_model.embedding_norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    for layer in language_model.layers:
        assert inspect.getsource(layer.feed_forward.forward) == inspect.getsource(LigerLfm2SwiGLUMLP.forward)

    vision_model = getattr(model.model.vision_tower, "vision_model", model.model.vision_tower)
    expected_layer_norm_name = "LigerLayerNorm" if expected_liger_layer_norm else "LayerNorm"
    assert vision_model.post_layernorm._get_name() == expected_layer_norm_name
    for layer in vision_model.encoder.layers:
        assert layer.layer_norm1._get_name() == expected_layer_norm_name
        assert layer.layer_norm2._get_name() == expected_layer_norm_name
    assert model.model.multi_modal_projector.layer_norm._get_name() == expected_layer_norm_name


@pytest.mark.skipif(not HAS_LFM2, reason="lfm2 module not available")
@pytest.mark.parametrize(
    ("hip", "fused_linear_cross_entropy", "expect_liger"),
    [
        (False, None, False),
        (True, None, True),
        (False, True, True),
        (True, False, False),
    ],
)
def test_lfm2_fused_linear_cross_entropy_backend_default(monkeypatch, hip, fused_linear_cross_entropy, expect_liger):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    from liger_kernel.transformers import monkey_patch

    model = Lfm2ForCausalLM(_lfm2_config())
    original_forward = inspect.getsource(model.forward)
    monkeypatch.setattr(monkey_patch, "is_hip", lambda: hip)
    monkey_patch.apply_liger_kernel_to_lfm2(
        model=model,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
    )

    expected_forward = inspect.getsource(lfm2_lce_forward) if expect_liger else original_forward
    assert inspect.getsource(model.forward) == expected_forward


@pytest.mark.skipif(not HAS_LFM2 or device == "cpu", reason="requires LFM2 and an accelerator")
def test_lfm2_explicit_fused_linear_cross_entropy_forward_backward():
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    from liger_kernel.transformers import monkey_patch

    model = Lfm2ForCausalLM(_lfm2_config()).to(device)
    monkey_patch.apply_liger_kernel_to_lfm2(model=model, fused_linear_cross_entropy=True)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device=device)

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


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
