# Validator Agent

Runs tests on generated code and fixes issues iteratively.

## Pre-Requisites

Verify the project is installed in development mode: `python -c "import liger_kernel"` must succeed. If not, ask the user to run `pip install -e ".[dev]"`.

## Validation Steps

### Step 1: Syntax Check

```bash
python -c "from liger_kernel.transformers import apply_liger_kernel_to_{model_type}"
```

### Step 2: Instance Patching Test

```bash
pytest test/transformers/test_monkey_patch.py -k "{model_type}" -xvs 2>&1 | tail -80
```

Common failures:

| Error | Fix |
|-------|-----|
| `ImportError: cannot import name` | Check exact class name in HF source |
| `AttributeError: has no attribute` | Read decoder layer `__init__` for correct attr names |
| `AssertionError: getsource mismatch` | Check class being replaced in monkey_patch |
| `ModuleNotFoundError` | Add `@pytest.mark.skipif` with availability check |

### Step 3: Convergence Test (requires GPU)

```bash
pytest test/convergence/bf16/test_mini_models.py -k "{model_type}" -xvs 2>&1 | tail -80
```

Common failures:

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce hidden_size/num_layers in mini config |
| `RuntimeError: shape mismatch` | Check `self.config.hidden_size` usage |
| `AssertionError: allclose` | Check casting_mode, offset, in_place settings |
| `KeyError in config` | Add missing field to mini model config |

### Step 4: Lint Check

```bash
ruff check src/liger_kernel/transformers/model/{model_type}.py
ruff check src/liger_kernel/transformers/monkey_patch.py
```

## Retry Logic

On failure: read traceback, identify root cause, fix, re-run. Max 3 retries per step. After 3 failures, report the exact error and diagnosis to the user.

## Success Report

```
| Check              | Status |
|--------------------|--------|
| Import             | PASS   |
| Instance Patching  | PASS   |
| Convergence (bf16) | PASS   |
| Lint               | PASS   |
```
