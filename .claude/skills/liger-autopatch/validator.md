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

### Step 3: Convergence Tests (requires GPU)

Run **all applicable** convergence tests. For text models, there are 4 files. For VL models, 6.

**FLCE path (fused linear cross entropy):**
```bash
pytest test/convergence/bf16/test_mini_models.py -k "{model_type}" -xvs 2>&1 | tail -80
pytest test/convergence/fp32/test_mini_models.py -k "{model_type}" -xvs 2>&1 | tail -80
```

**Non-FLCE path (logits verification — tests RMSNorm/SwiGLU/RoPE without loss fusion):**
```bash
pytest test/convergence/bf16/test_mini_models_with_logits.py -k "{model_type}" -xvs 2>&1 | tail -80
pytest test/convergence/fp32/test_mini_models_with_logits.py -k "{model_type}" -xvs 2>&1 | tail -80
```

**Multimodal (only for VL models):**
```bash
pytest test/convergence/bf16/test_mini_models_multimodal.py -k "{model_type}" -xvs 2>&1 | tail -80
pytest test/convergence/fp32/test_mini_models_multimodal.py -k "{model_type}" -xvs 2>&1 | tail -80
```

Common failures:

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce hidden_size/num_layers in mini config |
| `RuntimeError: shape mismatch` | Check `self.config.hidden_size` usage |
| `AssertionError: allclose` | Check casting_mode, offset, in_place settings |
| `KeyError in config` | Add missing field to mini model config |

### Step 4: Checkstyle

Run the project's full lint and format check:

```bash
make checkstyle
```

If it fails, auto-fix with `ruff check . --fix && ruff format .`, then re-run `make checkstyle`.

## Retry Logic

**Hard failures** (Steps 1, 2, 4 — import, patching, checkstyle): These indicate code bugs. Read traceback, identify root cause, fix, re-run. Max 3 retries per step. After 3 failures, report the exact error and stop.

**Soft failures** (Step 3 — convergence): If a convergence test fails after 3 retries with `allclose` tolerance errors, mark it as `SKIP (tolerance tuning needed)` in the report rather than `FAIL`. This means the patching is correct but tolerances need manual adjustment. Note this in the PR description so reviewers are aware.

Convergence failures from other errors (shape mismatch, CUDA OOM, KeyError) are still hard failures — fix and retry.

## Success Report

```
| Check                        | Status |
|------------------------------|--------|
| Import                       | PASS   |
| Instance Patching            | PASS   |
| Convergence bf16 FLCE        | PASS   |
| Convergence bf16 with_logits | PASS   |
| Convergence fp32 FLCE        | PASS   |
| Convergence fp32 with_logits | PASS   |
| Convergence multimodal       | SKIP (text-only model) |
| Checkstyle                   | PASS   |
```

Include any `SKIP (tolerance tuning needed)` entries in the PR description under a "Known issues" section.
