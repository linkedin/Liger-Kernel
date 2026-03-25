# Template: Instance Patching Test

Reference: `test_apply_liger_kernel_to_instance_for_llama` in `test/transformers/test_monkey_patch.py`.

## Add at Module Level

```python
def is_{model_type}_available():
    try:
        import transformers.models.{model_type}
        return True
    except ImportError:
        return False
```

## Test Function

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
