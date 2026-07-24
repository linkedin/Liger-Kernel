import ast
from pathlib import Path


MODEL_FILES = (
    Path("src/liger_kernel/transformers/model/qwen2_vl.py"),
    Path("src/liger_kernel/transformers/model/qwen2_5_vl.py"),
)


def _forward_function(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "lce_forward")


def _all_names(node: ast.AST, name: str) -> list[ast.AST]:
    return [candidate for candidate in ast.walk(node) if isinstance(candidate, ast.Name) and candidate.id == name]


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
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "lm_head"
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


def test_qwen_vl_forward_keeps_fused_loss_on_full_hidden_states():
    for path in MODEL_FILES:
        function = _forward_function(path)
        source = ast.get_source_segment(path.read_text(encoding="utf-8"), function)
        assert source is not None
        fused_start = source.index("if skip_logits:")
        logits_start = source.index("else:", fused_start)
        fused_source = source[fused_start:logits_start]
        assert "hidden_states=hidden_states" in fused_source, path
        assert "hidden_states[:, slice_indices, :]" not in fused_source, path
