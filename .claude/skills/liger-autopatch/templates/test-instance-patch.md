# Template: Instance Patching Test

Use this to add a test function to `test/transformers/test_monkey_patch.py`.

## Reference

Read existing tests like `test_apply_liger_kernel_to_instance_for_llama` in the same file.

## Availability Checker

Add this at module level (near other `is_*_available` functions):

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
@pytest.mark.skipif(
    not is_{model_type}_available(),
    reason="{model_type} module not available",
)
def test_apply_liger_kernel_to_instance_for_{model_type}():
    from liger_kernel.transformers.model.{model_type} import lce_forward as {model_type}_lce_forward

    # Import model-specific classes
    from transformers.models.{model_type}.configuration_{model_type} import {ModelConfig}

    # Create mini config
    config = {ModelConfig}(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        # Add model-specific required fields
    )

    # Instantiate model
    dummy_model_instance = AutoModelForCausalLM.from_config(config)

    # Verify NOT patched initially
    assert inspect.getsource(
        dummy_model_instance.model.norm.forward
    ) != inspect.getsource(LigerRMSNorm.forward)

    for decoder_layer in dummy_model_instance.model.layers:
        assert inspect.getsource(
            decoder_layer.mlp.forward
        ) != inspect.getsource({LigerMLPClass}.forward)
        assert inspect.getsource(
            decoder_layer.input_layernorm.forward
        ) != inspect.getsource(LigerRMSNorm.forward)
        assert inspect.getsource(
            decoder_layer.post_attention_layernorm.forward
        ) != inspect.getsource(LigerRMSNorm.forward)

    # Apply patching
    _apply_liger_kernel_to_instance(model=dummy_model_instance)

    # Verify IS patched
    assert inspect.getsource(
        dummy_model_instance.forward
    ) == inspect.getsource({model_type}_lce_forward)

    assert inspect.getsource(
        dummy_model_instance.model.norm.forward
    ) == inspect.getsource(LigerRMSNorm.forward)

    for decoder_layer in dummy_model_instance.model.layers:
        assert inspect.getsource(
            decoder_layer.mlp.forward
        ) == inspect.getsource({LigerMLPClass}.forward)
        assert inspect.getsource(
            decoder_layer.input_layernorm.forward
        ) == inspect.getsource(LigerRMSNorm.forward)
        assert inspect.getsource(
            decoder_layer.post_attention_layernorm.forward
        ) == inspect.getsource(LigerRMSNorm.forward)
```

## Variations

### GeGLU models (Gemma family)
Replace `LigerSwiGLUMLP` with `LigerGEGLUMLP` in assertions.

### MoE models
Check expert modules instead of simple MLP:
```python
# v5 pattern
assert inspect.getsource(
    decoder_layer.mlp.experts.forward
) == inspect.getsource(LigerExperts.forward)
```

### Extra norm layers
Add assertions for additional norms:
```python
assert inspect.getsource(
    decoder_layer.pre_feedforward_layernorm.forward
) == inspect.getsource(LigerRMSNorm.forward)
```

### LayerNorm (for vision components)
```python
assert inspect.getsource(
    vision_layer.layer_norm1.forward
) == inspect.getsource(LigerLayerNorm.forward)
```
