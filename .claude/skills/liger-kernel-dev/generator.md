# Generator Agent

Takes a confirmed kernel profile and PyTorch reference, generates all files for a production-ready Liger kernel.

## Pre-Requisites

Before generating, read the existing implementation closest to this kernel's tier:
- Tier 1 → `src/liger_kernel/ops/swiglu.py` + `src/liger_kernel/transformers/swiglu.py`
- Tier 2 → `src/liger_kernel/ops/rms_norm.py` + `src/liger_kernel/transformers/rms_norm.py`
- Tier 3 → `src/liger_kernel/ops/cross_entropy.py` + `src/liger_kernel/transformers/cross_entropy.py`

Also read the corresponding test and benchmark files for the reference kernel.

## Files to Generate

### 1. `src/liger_kernel/ops/{kernel}.py` (NEW)

The core Triton kernel. See [templates/ops-kernel.md](templates/ops-kernel.md).

Key rules:
- `@triton.jit` for forward and backward kernels
- `torch.autograd.Function` subclass with `@ensure_contiguous` decorator
- Use `calculate_settings(n_cols)` for BLOCK_SIZE and num_warps
- Cast to `tl.float32` for precision-sensitive ops (sigmoid, rsqrt, exp, log, tanh)
- Use `tl.program_id(0).to(tl.int64)` to avoid overflow
- Use stride parameters for memory access, not hardcoded offsets
- Prefer recomputation over saving for backward when computation is cheap

### 2. `src/liger_kernel/transformers/{kernel}.py` (NEW)

The nn.Module wrapper. See [templates/module-wrapper.md](templates/module-wrapper.md).

Key rules:
- Import from `liger_kernel.ops` (not from the submodule directly)
- Include `extra_repr` method
- Match the PyTorch API as closely as possible

### 3. `src/liger_kernel/transformers/functional.py` (MODIFY)

Add functional API. See [templates/functional-api.md](templates/functional-api.md).

Two changes:
- Add import of the Function class at the top (alphabetical order)
- Add `liger_{kernel}()` function (alphabetical order among existing functions)

### 4. `src/liger_kernel/ops/__init__.py` (MODIFY)

Add import of the Function class. Insert in alphabetical order among existing imports. Use `# noqa: F401` suffix.

### 5. `src/liger_kernel/transformers/__init__.py` (MODIFY)

Add the Module class in three locations (alphabetical order in each):
- Direct import at top (among "Always-safe imports")
- `__all__` list
- If the Module has a `TYPE_CHECKING` import or `__getattr__` entry needed, add there too

### 6. `test/transformers/test_{kernel}.py` (NEW)

Unit tests. See [templates/unit-test.md](templates/unit-test.md).

Key rules:
- Include a PyTorch reference implementation (copy from Analyzer output)
- Test both forward and backward correctness
- Parametrize over shapes (include non-power-of-2 and small/large sizes)
- Parametrize over dtypes with appropriate tolerances
- Use `set_seed()`, `assert_verbose_allclose()`, `infer_device()`, `supports_bfloat16()`
- Import from `test.utils`, NOT `liger_kernel.test.utils`

### 7. `benchmark/scripts/benchmark_{kernel}.py` (NEW)

Benchmarks. See [templates/benchmark.md](templates/benchmark.md).

Key rules:
- Define `_setup_{kernel}()` function returning `(input_tensor, layer)`
- Define `bench_speed_{kernel}()` using `run_speed_benchmark`
- Define `bench_memory_{kernel}()` using `run_memory_benchmark`
- Compare "liger" vs "torch" (and optionally "torch_compile") providers
- Use `get_benchmark_model_config`, `estimate_kernel_peak_memory`, and sweep config utilities
- Run both speed and memory benchmarks for forward, backward, and full modes

## Modify Mode

When modifying an existing kernel:
1. Read the existing kernel files thoroughly
2. Understand the current implementation patterns
3. Make targeted changes — do not rewrite entire files
4. Add new test cases for new functionality
5. Ensure existing tests still cover original behavior

## Code Style

- Line length 120, double quotes, single imports sorted with isort
- Follow exact patterns from existing code — do not innovate on style
- When modifying existing files, insert new entries in **alphabetical order**
- Import style: one import per line, no multi-imports
