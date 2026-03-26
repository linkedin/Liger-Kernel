# Template: Instance Patching Test

Reference: `test_apply_liger_kernel_to_instance_for_llama` in `test/transformers/test_monkey_patch.py`.

## Add Availability Checker (at top of file, near other `is_*_available` functions ~lines 59-240, in alphabetical order)

```python
def is_{model_type}_available():
    try:
        import transformers.models.{model_type}
        return True
    except ImportError:
        return False
```

## Add Test Function (after existing `test_apply_liger_kernel_to_instance_for_*` functions, in alphabetical order)

If `min_transformers_version` from the profile is > "4.52.0", add a version skipif **in addition to** the availability skipif:

```python
@pytest.mark.skipif(
    transformer_version < version.parse("{min_transformers_version}"),
    reason="{model_type} requires transformers >= {min_transformers_version}",
)
```

If `min_transformers_version` is "4.52.0" or earlier, only use the availability skipif (the global minimum already enforces >= 4.52.0).

```python
@pytest.mark.skipif(not is_{model_type}_available(), reason="{model_type} not available")
def test_apply_liger_kernel_to_instance_for_{model_type}():
    from liger_kernel.transformers.model.{model_type} import lce_forward as {model_type}_lce_forward
    from transformers.models.{model_type}.configuration_{model_type} import {ModelConfig}

    config = {ModelConfig}(hidden_size=32, intermediate_size=64, num_hidden_layers=2)
    dummy_model_instance = AutoModelForCausalLM.from_config(config)

    # Verify NOT patched
    assert inspect.getsource(dummy_model_instance.model.norm.forward) != inspect.getsource(LigerRMSNorm.forward)
    for layer in dummy_model_instance.model.layers:
        assert inspect.getsource(layer.mlp.forward) != inspect.getsource({LigerMLPClass}.forward)
        assert inspect.getsource(layer.input_layernorm.forward) != inspect.getsource(LigerRMSNorm.forward)

    # Apply patch
    _apply_liger_kernel_to_instance(model=dummy_model_instance)

    # Verify patched
    assert inspect.getsource(dummy_model_instance.forward) == inspect.getsource({model_type}_lce_forward)
    assert inspect.getsource(dummy_model_instance.model.norm.forward) == inspect.getsource(LigerRMSNorm.forward)
    for layer in dummy_model_instance.model.layers:
        assert inspect.getsource(layer.mlp.forward) == inspect.getsource({LigerMLPClass}.forward)
        assert inspect.getsource(layer.input_layernorm.forward) == inspect.getsource(LigerRMSNorm.forward)
```

## Adapt For

- **GeGLU models**: Use `LigerGEGLUMLP` in assertions
- **MoE models**: Check `decoder_layer.mlp.experts.forward` against `LigerExperts`
- **Extra norms**: Add assertions for `pre_feedforward_layernorm` etc.
- **Vision**: Add assertions for `LigerLayerNorm` on vision layers
