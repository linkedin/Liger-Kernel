# Template: Convergence Test Entry

Use this to add a convergence test to `test/convergence/bf16/test_mini_models.py`.

## Reference

Read existing entries in the same file (e.g., Llama, Gemma, Qwen2).

## Step 1: Add Imports

Add to the import block at the top of the file:

```python
from liger_kernel.transformers import apply_liger_kernel_to_{model_type}
```

And in the `test/utils.py` imports:
```python
from test.utils import revert_liger_kernel_to_{model_type}
```

## Step 2: Add Availability Guard

```python
try:
    from transformers.models.{model_type}.configuration_{model_type} import {ModelConfig}
    from transformers.models.{model_type}.modeling_{model_type} import {ModelForCausalLM}

    {MODEL_TYPE_UPPER}_AVAILABLE = True
except ImportError:
    {MODEL_TYPE_UPPER}_AVAILABLE = False
```

## Step 3: Add MiniModelConfig Entry

Add to the `MINI_MODEL_CONFIGS` list:

```python
pytest.param(
    MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_{model_type},
        liger_kernel_patch_revert_func=revert_liger_kernel_to_{model_type},
        model_class={ModelForCausalLM},
        mini_model_config={ModelConfig}(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=1024,
            # Add any model-specific required fields
        ),
    ),
    marks=[
        pytest.mark.skipif(
            not {MODEL_TYPE_UPPER}_AVAILABLE,
            reason="{model_type} not available",
        ),
    ],
    id="mini_{model_type}",
),
```

## Step 4: Add Revert Function in test/utils.py

```python
def revert_liger_kernel_to_{model_type}(model_config: MiniModelConfig):
    """Revert Liger kernel patches for {ModelName}."""
    from transformers.models.{model_type} import modeling_{model_type}

    importlib.reload(modeling_{model_type})
    model_config.model_class = modeling_{model_type}.{ModelForCausalLM}
    print("Liger kernel patches have been reverted.")
```

## Mini Config Guidelines

- `hidden_size`: 32 (minimum for most models)
- `intermediate_size`: 64 (2x hidden_size)
- `num_hidden_layers`: 2 (enough to test patching)
- `num_attention_heads`: 2 or 4 (must divide hidden_size)
- `num_key_value_heads`: 2 (for GQA) or same as num_attention_heads
- `vocab_size`: 1024 (small but realistic)
- `max_position_embeddings`: 128 (keeps memory low)
- `head_dim`: hidden_size // num_attention_heads (must be valid)

If the model requires additional config fields (e.g., `num_local_experts` for MoE), add them with minimal values.
