# Template: Convergence Tests

Reference: existing entries across `test/convergence/` files. Read the Llama entries as baseline.

## Which Files to Modify

Every model needs entries in **at least 4 files**:

| File | What it tests | dtype |
|------|-------------|-------|
| `test/convergence/bf16/test_mini_models.py` | FLCE path (fused linear cross entropy) | bfloat16 |
| `test/convergence/bf16/test_mini_models_with_logits.py` | Non-FLCE path (tests RMSNorm/SwiGLU/RoPE only, verifies logits) | bfloat16 |
| `test/convergence/fp32/test_mini_models.py` | FLCE path | float32 |
| `test/convergence/fp32/test_mini_models_with_logits.py` | Non-FLCE path | float32 |

**Vision-language models** also need:
- `test/convergence/bf16/test_mini_models_multimodal.py`
- `test/convergence/fp32/test_mini_models_multimodal.py`

## Placement Rule

When adding entries to existing files, always insert in **alphabetical order** alongside similar existing entries. Never append to the end of a section. Specifically:
- Imports: alphabetical among other `apply_liger_kernel_to_*` / `revert_liger_kernel_to_*` imports
- Availability guards: alphabetical among other try/except model import blocks
- `MINI_MODEL_SETUPS` entries: alphabetical among other model setup blocks
- `pytest.param` entries: alphabetical among other model param entries

## What to Add in Each File

Each file has a `MINI_MODEL_SETUPS` dict and a `pytest.param` block. Add to both.

### 1. Imports (same across all files, insert in alphabetical order)

```python
from liger_kernel.transformers import apply_liger_kernel_to_{model_type}
from test.utils import revert_liger_kernel_to_{model_type}
```

### 2. Availability Guard (same across all files)

If `min_transformers_version` from the profile is > "4.52.0", include a version check inside the try block. This handles two cases: (a) model doesn't exist at all in older transformers → `ImportError`, and (b) model exists but is buggy/incomplete → version check fails.

```python
try:
    from transformers.models.{model_type}.configuration_{model_type} import {ModelConfig}
    from transformers.models.{model_type}.modeling_{model_type} import {ModelForCausalLM}
    {MODEL_UPPER}_AVAILABLE = version.parse(transformers.__version__) >= version.parse("{min_transformers_version}")
except ImportError:
    {MODEL_UPPER}_AVAILABLE = False
```

If `min_transformers_version` is "4.52.0" or earlier, omit the version check (the global minimum already covers it):

```python
try:
    from transformers.models.{model_type}.configuration_{model_type} import {ModelConfig}
    from transformers.models.{model_type}.modeling_{model_type} import {ModelForCausalLM}
    {MODEL_UPPER}_AVAILABLE = True
except ImportError:
    {MODEL_UPPER}_AVAILABLE = False
```

### 3. MINI_MODEL_SETUPS Entry (identical across all files)

```python
if {MODEL_UPPER}_AVAILABLE:
    MINI_MODEL_SETUPS["mini_{model_type}"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_{model_type},
        liger_kernel_patch_revert_func=revert_liger_kernel_to_{model_type},
        model_class={ModelForCausalLM},
        mini_model_config={ModelConfig}(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=1024,
        ),
    )
```

### 4. pytest.param Entry (tolerances differ by dtype)

**bf16 files** — looser tolerances (copy from similar existing model):
```python
pytest.param(
    "mini_{model_type}", 32, 1e-5, torch.bfloat16,
    1e-2, 5e-2, 1e-1, 1e-2, 1e-2, 1e-2,
    marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
),
```

**fp32 files** — tighter tolerances:
```python
("mini_{model_type}", 32, 1e-4, torch.float32, 1e-8, 2e-5, 5e-3, 1e-5, 5e-3, 1e-5),
```

### 5. Revert Function in test/utils.py (shared by all files)

```python
def revert_liger_kernel_to_{model_type}(model_config):
    from transformers.models.{model_type} import modeling_{model_type}
    importlib.reload(modeling_{model_type})
    model_config.model_class = modeling_{model_type}.{ModelForCausalLM}
```

## Key Differences Between Test Variants

**`with_logits`**: Sets `fused_linear_cross_entropy=False, cross_entropy=False` and checks `output.logits` match. Tests that non-loss kernels are numerically correct on their own. Same `MINI_MODEL_SETUPS` entry, just add a `pytest.param`.

**`multimodal`**: Uses `multimodal_collate_fn`, image datasets, and model configs with vision sub-config. Only for models with `has_vision: true`.

## Mini Config Guidelines

Use minimal values: hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, vocab_size=1024. Add model-specific required fields (e.g., `num_local_experts` for MoE). Copy config structure from the most similar existing model.
