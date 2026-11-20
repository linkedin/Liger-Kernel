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
    forward_name = "lce_forward_conditional_generation" if model_name == "qwen3_5_moe" else "lce_forward"
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
