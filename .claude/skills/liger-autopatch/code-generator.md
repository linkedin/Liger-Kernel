# Code Generator Agent

You are a code generation agent for the Liger Kernel auto-patch system. You receive a confirmed model profile and generate all required files to add Liger Kernel support.

## Input

A model profile markdown document (produced by the Model Analyzer).

## Pre-Requisites

Before generating code:
1. Read the model profile carefully — every field matters.
2. Read the reference implementation that most closely matches this model:
   - Dense model → read `src/liger_kernel/transformers/model/llama.py`
   - MoE model → read `src/liger_kernel/transformers/model/mixtral.py`
   - Vision-Language → read `src/liger_kernel/transformers/model/qwen2_vl.py`
   - Gemma-family (GeGLU, offset) → read `src/liger_kernel/transformers/model/gemma.py`
3. Read the corresponding patching function in `monkey_patch.py` for that reference.
4. Read the templates in `.claude/skills/liger-autopatch/templates/`.

## Files to Generate

### File 1: `src/liger_kernel/transformers/model/{model_type}.py` (NEW)

This contains the `lce_forward` function. Follow the template in `templates/lce-forward-dense.md` or `templates/lce-forward-moe.md`.

Key rules:
- Match the exact forward signature from the HF model's `ForCausalLM.forward`
- Use `LigerForCausalLMLoss` from `liger_kernel.transformers.model.loss_utils`
- Use `unpack_cross_entropy_result` for unpacking loss results
- Use the correct output class from `output_classes.py`
- Import `lce_maybe_trainable_lm_head` from `liger_kernel.transformers.model.llama` (shared utility)
- Handle `logits_to_keep` slicing correctly
- Handle `skip_logits` logic correctly
- Pass `final_logit_softcapping` if the model uses softcapping

### File 2: `src/liger_kernel/transformers/monkey_patch.py` (MODIFY)

Three changes needed:

**A. Add import at top of file** (near other lce_forward imports, ~line 18-28):
```python
from liger_kernel.transformers.model.{model_type} import lce_forward as {model_type}_lce_forward
```

**B. Add the `apply_liger_kernel_to_{model_type}` function.**

Follow the template in `templates/monkey-patch-fn.md`. The function must handle:
- Class-level patching (when `model=None`)
- Instance-level patching (when `model` is provided)
- All boolean flags (rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu/geglu, layer_norm)
- The `assert not (cross_entropy and fused_linear_cross_entropy)` guard
- Correct RMSNorm parameters (offset, casting_mode, in_place)
- Correct MLP class replacement
- Version-aware MoE handling if applicable

**C. Add entry to `MODEL_TYPE_TO_APPLY_LIGER_FN` dict** (~line 3067):
```python
"model_type": apply_liger_kernel_to_model_type,
```
If the model has a `_text` variant (for multimodal), add both:
```python
"model_type": apply_liger_kernel_to_model_type,
"model_type_text": apply_liger_kernel_to_model_type,
```

### File 3: `src/liger_kernel/transformers/__init__.py` (MODIFY)

Add the new function in three places:

**A. TYPE_CHECKING block** (~line 34-74):
```python
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_{model_type}
```

**B. `__getattr__` monkey_patch_symbols set** (~line 109-149):
```python
"apply_liger_kernel_to_{model_type}",
```

**C. `__all__` list extension** (~line 189-232):
```python
"apply_liger_kernel_to_{model_type}",
```

Maintain alphabetical order in all three locations.

### File 4: `src/liger_kernel/transformers/model/output_classes.py` (MODIFY — if needed)

Only modify if the model needs a custom output class (MoE with aux_loss, or VL with extra fields like rope_deltas).

Pattern:
```python
try:
    from transformers.models.{model_type}.modeling_{model_type} import (
        {ModelOutputClass} as _{ModelOutputClass},
    )
except Exception:
    _{ModelOutputClass} = None

if _{ModelOutputClass} is not None:
    @dataclass
    class Liger{ModelOutputClass}(_{ModelOutputClass}):
        token_accuracy: Optional[torch.FloatTensor] = None
        predicted_tokens: Optional[torch.LongTensor] = None
```

### File 5: `test/transformers/test_monkey_patch.py` (MODIFY)

Follow the template in `templates/test-instance-patch.md`.

Add:
- An availability checker function at module level
- A `@pytest.mark.skipif` decorated test function
- Create a mini config, instantiate with `from_config`
- Verify unpatched state, apply patch, verify patched state
- Use `inspect.getsource()` for all assertions

### File 6: `test/convergence/bf16/test_mini_models.py` (MODIFY)

Follow the template in `templates/test-convergence.md`.

Add:
- Import for the model's Config and ForCausalLM classes (with try/except)
- Import for `apply_liger_kernel_to_{model_type}` and `revert_liger_kernel_to_{model_type}`
- A `MiniModelConfig` entry in the `MINI_MODEL_CONFIGS` list
- The mini config must use small values (2 layers, 32 hidden, 64 intermediate)

### File 7: `test/utils.py` (MODIFY)

Add a `revert_liger_kernel_to_{model_type}` function:
```python
def revert_liger_kernel_to_{model_type}(model_config: MiniModelConfig):
    from transformers.models.{model_type} import modeling_{model_type}
    importlib.reload(modeling_{model_type})
    model_config.model_class = modeling_{model_type}.{CausalLMClass}
    print("Liger kernel patches have been reverted.")
```

### File 8: `README.md` (MODIFY)

Add a row to the Patching table under "## High-level APIs > ### Patching":
```
| {ModelName} | `liger_kernel.transformers.apply_liger_kernel_to_{model_type}` | {Supported Operations} |
```

## Code Style Rules

- Line length: 120 characters max
- Double quotes for strings
- Single imports (one per line), sorted with isort
- Follow the exact patterns from existing code — do not innovate on style
- Use `from __future__ import annotations` only if the reference file uses it
- Ensure the project is installed (`pip install -e ".[dev]"`) before running any Python
