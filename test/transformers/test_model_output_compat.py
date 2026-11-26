import importlib
import importlib.util
import sys

from dataclasses import dataclass
from dataclasses import fields
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.modeling_outputs import MoeModelOutputWithPast

from liger_kernel.transformers.model import output_classes

_MODELS = [
    ("glm4v_moe", "Glm4vMoeCausalLMOutputWithPast", MoeCausalLMOutputWithPast),
    ("qwen2_5_vl", "Qwen2_5_VLCausalLMOutputWithPast", CausalLMOutputWithPast),
    ("qwen2_vl", "Qwen2VLCausalLMOutputWithPast", CausalLMOutputWithPast),
    ("qwen3_5", "Qwen3_5CausalLMOutputWithPast", CausalLMOutputWithPast),
    ("qwen3_5_moe", "Qwen3_5MoeCausalLMOutputWithPast", MoeCausalLMOutputWithPast),
    ("qwen3_vl", "Qwen3VLCausalLMOutputWithPast", CausalLMOutputWithPast),
    ("qwen3_vl_moe", "Qwen3VLMoeCausalLMOutputWithPast", MoeCausalLMOutputWithPast),
]


@pytest.mark.parametrize("model_name, output_name, standard_output", _MODELS)
@pytest.mark.parametrize("legacy_output", [False, True])
def test_multimodal_output_supports_legacy_and_standard_bases(
    monkeypatch, model_name, output_name, standard_output, legacy_output
):
    modeling = pytest.importorskip(f"transformers.models.{model_name}.modeling_{model_name}")

    @dataclass
    class LegacyOutput(standard_output):
        rope_deltas: Optional[torch.LongTensor] = None

    if legacy_output:
        monkeypatch.setattr(modeling, output_name, LegacyOutput, raising=False)
    else:
        monkeypatch.delattr(modeling, output_name, raising=False)

    # Load a separate module so this test cannot replace classes used by other tests.
    spec = importlib.util.spec_from_file_location("_liger_output_classes_test", output_classes.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    output_cls = getattr(module, f"Liger{output_name}")
    expected_base = LegacyOutput if legacy_output else standard_output
    assert issubclass(output_cls, expected_base)
    assert {field.name for field in fields(expected_base)} <= {field.name for field in fields(output_cls)}

    values = {
        "loss": torch.tensor(1.0),
        "logits": torch.randn(1, 2, 4),
        "rope_deltas": torch.tensor([[2]]),
        "token_accuracy": torch.tensor(0.5),
        "predicted_tokens": torch.tensor([[1, 2]]),
    }
    if standard_output is MoeCausalLMOutputWithPast:
        values["aux_loss"] = torch.tensor(0.1)
        values["router_logits"] = (torch.randn(2, 3),)
    output = output_cls(**values)
    for name, value in values.items():
        assert getattr(output, name) is value
        assert output[name] is value
    assert len(output.to_tuple()) == len(values)


@pytest.mark.parametrize("model_name, output_name, standard_output", _MODELS)
@pytest.mark.parametrize("rope_source", ["outputs", "model", "none"])
@pytest.mark.parametrize("return_dict", [False, True])
def test_multimodal_forward_handles_removed_rope_output(
    model_name, output_name, standard_output, rope_source, return_dict
):
    modeling = pytest.importorskip(f"transformers.models.{model_name}.modeling_{model_name}")
    module = importlib.import_module(f"liger_kernel.transformers.model.{model_name}")
    forward_name = {
        "qwen3_5": "lce_forward_for_multimodal",
        "qwen3_5_moe": "lce_forward_conditional_generation",
    }.get(model_name, "lce_forward")
    forward = getattr(module, forward_name)
    hidden_states = torch.randn(1, 2, 4)
    outputs = MoeModelOutputWithPast(last_hidden_state=hidden_states)
    rope_deltas = torch.tensor([[2]])
    base_model = Mock(return_value=outputs)
    base_model.rope_deltas = rope_deltas if rope_source == "model" else None
    if rope_source == "outputs":
        outputs.rope_deltas = rope_deltas
        base_model.rope_deltas = torch.tensor([[9]])
    model = SimpleNamespace(
        config=PretrainedConfig(return_dict=return_dict),
        model=base_model,
        lm_head=torch.nn.Linear(4, 8, bias=False),
        training=False,
    )

    output = forward(model, input_ids=torch.tensor([[0, 1]]), return_dict=return_dict)
    expected_logits = model.lm_head(hidden_states)
    if return_dict:
        expected_base = getattr(modeling, output_name, standard_output)
        assert isinstance(output, expected_base)
        assert output.rope_deltas is (None if rope_source == "none" else rope_deltas)
        assert output.token_accuracy is None
        assert output.predicted_tokens is None
        torch.testing.assert_close(output.logits, expected_logits)
    else:
        assert isinstance(output, tuple)
        torch.testing.assert_close(output[0], expected_logits)


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_muse_glimmer_forward_builds_output_across_contracts(multimodal, return_dict):
    """The patched MuseGlimmer forward must construct its output using only the keyword
    arguments the installed transformers dataclass accepts. Newer versions renamed
    ``past_key_values`` to ``cache`` and turned ``image_hidden_states`` into a read-only
    property (replaced by native ``image_outputs``/``video_outputs``/``audio_outputs``),
    so passing ``image_hidden_states=`` raised ``TypeError`` for both text-only and
    multimodal outputs. This covers the construction path without a GPU.
    """
    pytest.importorskip("transformers.models.muse_glimmer.modeling_muse_glimmer")
    from liger_kernel.transformers.model.muse_glimmer import lce_forward
    from liger_kernel.transformers.model.output_classes import LigerMuseGlimmerCausalLMOutputWithPast

    hidden_states = torch.randn(1, 2, 4)
    sentinel = SimpleNamespace(pooler_output=torch.randn(1, 4)) if multimodal else None
    outputs = SimpleNamespace(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        image_outputs=sentinel,
        video_outputs=None,
        audio_outputs=None,
        # Only read on older transformers where it is a real field.
        image_hidden_states=sentinel,
    )
    text_config = SimpleNamespace(
        output_multiplier=1.0,
        final_logit_softcapping=30.0,
        hidden_size=4,
        vocab_size=8,
    )
    config = PretrainedConfig(return_dict=return_dict)
    config.text_config = text_config
    model = SimpleNamespace(
        config=config,
        model=Mock(return_value=outputs),
        lm_head=torch.nn.Linear(4, 8, bias=False),
        training=False,
    )

    output = lce_forward(model, input_ids=torch.tensor([[0, 1]]), return_dict=return_dict)

    logits = model.lm_head(hidden_states)
    logits = logits * text_config.output_multiplier
    logits = logits / text_config.final_logit_softcapping
    logits = torch.tanh(logits)
    expected_logits = logits * text_config.final_logit_softcapping

    if return_dict:
        assert isinstance(output, LigerMuseGlimmerCausalLMOutputWithPast)
        assert output.loss is None
        assert output.token_accuracy is None
        assert output.predicted_tokens is None
        torch.testing.assert_close(output.logits, expected_logits)

        valid_fields = {field.name for field in fields(LigerMuseGlimmerCausalLMOutputWithPast)}
        if "image_outputs" in valid_fields:
            assert output.image_outputs is sentinel
        else:
            assert output.image_hidden_states is sentinel
    else:
        assert isinstance(output, tuple)
        torch.testing.assert_close(output[0], expected_logits)
