# Template: Convergence Test Entry

Reference: existing entries in `test/convergence/bf16/test_mini_models.py`.

## Add Imports

```python
from liger_kernel.transformers import apply_liger_kernel_to_{model_type}
from test.utils import revert_liger_kernel_to_{model_type}
```

## Add Availability Guard

```python
try:
    from transformers.models.{model_type}.configuration_{model_type} import {ModelConfig}
    from transformers.models.{model_type}.modeling_{model_type} import {ModelForCausalLM}
    {MODEL_UPPER}_AVAILABLE = True
except ImportError:
    {MODEL_UPPER}_AVAILABLE = False
```

## Add MiniModelConfig Entry

```python
pytest.param(
    MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_{model_type},
        liger_kernel_patch_revert_func=revert_liger_kernel_to_{model_type},
        model_class={ModelForCausalLM},
        mini_model_config={ModelConfig}(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=1024,
        ),
    ),
    marks=[pytest.mark.skipif(not {MODEL_UPPER}_AVAILABLE, reason="{model_type} not available")],
    id="mini_{model_type}",
),
```

## Add Revert Function in test/utils.py

```python
def revert_liger_kernel_to_{model_type}(model_config):
    from transformers.models.{model_type} import modeling_{model_type}
    importlib.reload(modeling_{model_type})
    model_config.model_class = modeling_{model_type}.{ModelForCausalLM}
```

## Mini Config Guidelines

Use minimal values: hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, vocab_size=1024. Add model-specific required fields (e.g., `num_local_experts` for MoE).
