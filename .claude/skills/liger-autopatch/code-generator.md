# Code Generator Agent

Takes a confirmed model profile (create mode) or a change plan (modify mode) and generates or modifies files for Liger Kernel support.

## Mode

- **Create mode** (default): Generating all files for a new model. Follow the full "Files to Generate" list below.
- **Modify mode**: Making targeted changes to an existing monkey-patch. Follow the "Modification Checklist" section instead.

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

## Modification Checklist (Modify Mode)

Before making changes, read the existing implementation:
1. Read `apply_liger_kernel_to_{model_type}` in `monkey_patch.py`
2. Read the existing test in `test_monkey_patch.py` for this model
3. Read the relevant HF modeling source for context

### Rules for All Modifications

**R1. Both patching levels.** If adding a new kernel, it must appear in BOTH:
  - Class-level patching (the main body of `apply_liger_kernel_to_{model_type}`)
  - Instance-level patching (the `if model is not None` block)

  Omitting one is the most common mistake.

**R2. New parameter with default.** Every new kernel gets a bool parameter on the
  apply function signature (e.g., `relu_squared: bool = True`). Default should be `True`
  for kernels that are safe to enable by default, `False` otherwise.

**R3. Update docstring.** Update the function's docstring to:
  - Add an `Args` entry for the new parameter
  - Remove any stale notes that the new kernel invalidates
    (e.g., "squared ReLU is not supported" → remove if you're adding it)

**R4. Update tests.** In the existing `test_apply_liger_kernel_to_instance_for_{model_type}`:
  - Add import for the new Liger kernel class
  - Add "not yet patched" assertion before `_apply_liger_kernel_to_instance`
  - Add "correctly patched" assertion after
  - Follow the exact pattern of existing assertions in the same test

**R5. Check revert function.** Read `revert_liger_kernel_to_{model_type}` in `test/utils.py`.
  The revert function uses `importlib.reload(modeling_{model_type})` to undo all patches.
  This handles most cases automatically, but check if the new kernel requires additional
  revert logic (e.g., if the kernel patches something outside the modeling module, or
  replaces a global like `ACT2FN` that `importlib.reload` won't fully restore). Update
  the revert function if needed.

**R6. Run convergence tests.** Don't modify convergence test files unless the change
  requires it (e.g., new mini model config fields). But DO run existing convergence
  tests in the Validate stage to verify no regression. This is critical — the Validator
  agent (Stage 3) handles this, but if you are generating code without a separate
  Validate stage, run these yourself:
  ```bash
  pytest test/convergence/bf16/test_mini_models.py -k "{model_type}" -xvs
  pytest test/convergence/bf16/test_mini_models_with_logits.py -k "{model_type}" -xvs
  pytest test/convergence/fp32/test_mini_models.py -k "{model_type}" -xvs
  pytest test/convergence/fp32/test_mini_models_with_logits.py -k "{model_type}" -xvs
  ```
  For VL (multimodal) models, also run:
  ```bash
  pytest test/convergence/bf16/test_mini_models_multimodal.py -k "{model_type}" -xvs
  pytest test/convergence/fp32/test_mini_models_multimodal.py -k "{model_type}" -xvs
  ```

**R7. Update README.md.** If the change adds a visibly new capability to the model's
  row in the patching table (e.g., a new operation), update the supported operations list.

### Common Modification Patterns

**Adding an activation kernel (e.g., relu_squared for nemotron):**
- Import the Liger kernel class at the top of `monkey_patch.py`
- Add bool parameter to apply function signature
- Class-level: replace in `ACT2FN` dict or replace the MLP class
- Instance-level: patch each `decoder_layer`'s activation/MLP
- Test: `assert isinstance` checks on the activation/MLP

**Adding a norm variant:**
- Add bool parameter to apply function
- Class-level: replace the norm class
- Instance-level: use `_patch_rms_norm_module` or `_patch_layer_norm_module` on all norm attrs
- Test: `assert isinstance` checks on norm modules

**Fixing missing instance patching:**
- Read the class-level patching to see what's patched
- Add corresponding instance-level patches in the `if model is not None` block
- Test: add assertions that were missing

**Updating for upstream HF changes:**
- Compare the current HF modeling file against what the patch assumes
- Update class names, attribute names, forward signatures as needed
- May require updating `lce_forward` if the base model's forward changed
