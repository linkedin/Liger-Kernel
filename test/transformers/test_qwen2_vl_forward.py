import ast
import importlib

from pathlib import Path
from unittest.mock import patch

import pytest

MODEL_FILES = (
    Path("src/liger_kernel/transformers/model/qwen2_vl.py"),
    Path("src/liger_kernel/transformers/model/qwen2_5_vl.py"),
)


def _forward_function(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "lce_forward")


def _make_dummy_model(torch):
    class DummyOutputs(tuple):
        def __new__(cls, hidden_states):
            output = super().__new__(cls, (hidden_states,))
            output.past_key_values = None
            output.hidden_states = None
            output.attentions = None
            output.rope_deltas = None
            return output

    class DummyBaseModel:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return DummyOutputs(self.hidden_states)

    class DummyModel:
        def __init__(self, hidden_states):
            text_config = type("TextConfig", (), {"hidden_size": hidden_states.shape[-1], "vocab_size": 2})()
            self.config = type(
                "Config",
                (),
                {
                    "hidden_size": hidden_states.shape[-1],
                    "vocab_size": 2,
                    "text_config": text_config,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "use_return_dict": False,
                },
            )()
            self.model = DummyBaseModel(hidden_states)
            self.lm_head = torch.nn.Linear(hidden_states.shape[-1], 2, bias=False)
            self.training = False

    hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    return DummyModel(hidden_states), hidden_states


def test_qwen_vl_forward_declares_logits_to_keep():
    for path in MODEL_FILES:
        function = _forward_function(path)
        parameter_names = [argument.arg for argument in function.args.args + function.args.kwonlyargs]
        assert "logits_to_keep" in parameter_names, path


def test_qwen_vl_forward_slices_hidden_states_before_lm_head():
    for path in MODEL_FILES:
        function = _forward_function(path)
        source = ast.get_source_segment(path.read_text(encoding="utf-8"), function)
        assert source is not None
        assert "slice(-logits_to_keep, None)" in source, path
        lm_head_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "lm_head"
        ]
        assert len(lm_head_calls) == 1, path
        lm_head_argument = lm_head_calls[0].args[0]
        assert isinstance(lm_head_argument, ast.Subscript), path
        assert isinstance(lm_head_argument.value, ast.Name) and lm_head_argument.value.id == "hidden_states", path


def test_qwen_vl_forward_does_not_forward_logits_to_keep_to_base_model():
    for path in MODEL_FILES:
        function = _forward_function(path)
        base_model_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model"
        ]
        assert len(base_model_calls) == 1, path
        forwarded_names = {keyword.arg for keyword in base_model_calls[0].keywords if keyword.arg is not None}
        assert "logits_to_keep" not in forwarded_names, path


@pytest.mark.parametrize(
    "module_name",
    (
        "liger_kernel.transformers.model.qwen2_vl",
        "liger_kernel.transformers.model.qwen2_5_vl",
    ),
)
@pytest.mark.parametrize("selector_kind", ("all", "last_two", "tensor"))
def test_qwen_vl_forward_applies_logits_to_keep_on_cpu(module_name: str, selector_kind: str):
    torch = pytest.importorskip("torch")
    model, hidden_states = _make_dummy_model(torch)
    selector = {"all": 0, "last_two": 2, "tensor": torch.tensor([1, 3])}[selector_kind]
    expected_indices = {
        "all": slice(None),
        "last_two": slice(-2, None),
        "tensor": selector,
    }[selector_kind]
    expected_logits = model.lm_head(hidden_states[:, expected_indices, :])
    forward = importlib.import_module(module_name).lce_forward.__wrapped__

    if selector_kind == "all":
        outputs = forward(model, return_dict=False)
    else:
        outputs = forward(model, logits_to_keep=selector, return_dict=False)

    torch.testing.assert_close(outputs[0], expected_logits)
    assert "logits_to_keep" not in model.model.kwargs


@pytest.mark.parametrize(
    "module_name",
    (
        "liger_kernel.transformers.model.qwen2_vl",
        "liger_kernel.transformers.model.qwen2_5_vl",
    ),
)
def test_qwen_vl_forward_keeps_fused_loss_on_full_hidden_states(module_name: str):
    torch = pytest.importorskip("torch")
    model, hidden_states = _make_dummy_model(torch)
    labels = torch.zeros((1, hidden_states.shape[1]), dtype=torch.long)
    module = importlib.import_module(module_name)
    forward = module.lce_forward.__wrapped__

    with patch.object(module, "LigerForCausalLMLoss", return_value=torch.tensor(0.0)) as fused_loss:
        forward(model, labels=labels, logits_to_keep=2, skip_logits=True, return_dict=False)

    assert fused_loss.call_args.kwargs["hidden_states"] is hidden_states
