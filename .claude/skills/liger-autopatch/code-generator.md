# Code Generator Agent

Takes a confirmed model profile and generates all files to add Liger Kernel support.

## Pre-Requisites

Before generating, read the reference implementation closest to this model:
- Dense → `src/liger_kernel/transformers/model/llama.py`
- MoE → `src/liger_kernel/transformers/model/mixtral.py`
- Vision-Language → `src/liger_kernel/transformers/model/qwen2_vl.py`
- Gemma-family → `src/liger_kernel/transformers/model/gemma.py`

Also read the corresponding patching function in `monkey_patch.py` and the templates in [templates/](templates/).

## Files to Generate

### 1. `src/liger_kernel/transformers/model/{model_type}.py` (NEW)

The `lce_forward` function. See [templates/lce-forward-dense.md](templates/lce-forward-dense.md) or [templates/lce-forward-moe.md](templates/lce-forward-moe.md).

Key rules:
- Match the exact forward signature from HF's `ForCausalLM.forward`
- Use `lce_maybe_trainable_lm_head` from `llama.py` (shared PEFT/FSDP utility)
- If model needs custom loss args (e.g., softcapping), write a local helper instead

### 2. `src/liger_kernel/transformers/monkey_patch.py` (MODIFY)

Three changes — see [templates/monkey-patch-fn.md](templates/monkey-patch-fn.md):

**A.** Add lce_forward import (~line 18-28):
```python
from liger_kernel.transformers.model.{model_type} import lce_forward as {model_type}_lce_forward
```

**B.** Add `apply_liger_kernel_to_{model_type}` function with both class-level and instance-level patching paths.

**C.** Add entry to `MODEL_TYPE_TO_APPLY_LIGER_FN` dict (~line 3067).

### 3. `src/liger_kernel/transformers/__init__.py` (MODIFY)

Add the function in three locations (maintain alphabetical order):
- `TYPE_CHECKING` block
- `__getattr__` monkey_patch_symbols set
- `__all__` list extension

### 4. `src/liger_kernel/transformers/model/output_classes.py` (MODIFY if needed)

Only for models needing custom output (MoE with `aux_loss`, VL with `rope_deltas`). Follow the existing guarded-import pattern in the file.

### 5. `test/transformers/test_monkey_patch.py` (MODIFY)

See [templates/test-instance-patch.md](templates/test-instance-patch.md). Add availability checker + skipif-decorated test function using `inspect.getsource()` assertions.

### 6. Convergence tests (MODIFY multiple files)

See [templates/test-convergence.md](templates/test-convergence.md). Every model needs entries in multiple convergence test files:

**All text models (dense + MoE)** — add to these 4 files:
- `test/convergence/bf16/test_mini_models.py` — FLCE path, bf16
- `test/convergence/bf16/test_mini_models_with_logits.py` — non-FLCE path (tests RMSNorm/SwiGLU/RoPE only), bf16
- `test/convergence/fp32/test_mini_models.py` — FLCE path, fp32
- `test/convergence/fp32/test_mini_models_with_logits.py` — non-FLCE path, fp32

**Vision-language models** — also add to these 2:
- `test/convergence/bf16/test_mini_models_multimodal.py`
- `test/convergence/fp32/test_mini_models_multimodal.py`

Each file needs: imports, availability guard, `MiniModelConfig` entry in `MINI_MODEL_SETUPS` dict, and a `pytest.param` entry in the parametrize block. The `MiniModelConfig` entry is identical across all files for the same model. The `pytest.param` tolerances differ — use bf16 tolerances (looser) for bf16 files and fp32 tolerances (tighter) for fp32 files. Copy tolerance values from a similar existing model (e.g., Llama for dense, Mixtral for MoE).

### 7. `test/utils.py` (MODIFY)

Add `revert_liger_kernel_to_{model_type}` function that reloads the modeling module.

### 8. `README.md` (MODIFY)

Add row to the Patching table under "### Patching":
```
| {ModelName} | `liger_kernel.transformers.apply_liger_kernel_to_{model_type}` | {Supported Operations} |
```

## Code Style

- Line length 120, double quotes, single imports sorted with isort
- Follow exact patterns from existing code — do not innovate on style
- When modifying existing files, insert new entries in **alphabetical order** alongside similar existing entries. Never append to the end of a section — find the correct alphabetical position.
- After generating all files, run `make checkstyle` to verify formatting. If it fails, run `ruff check . --fix && ruff format .` to auto-fix, then verify with `make checkstyle` again.
