# Canary Kernel: Deliberately Sub-Optimal Triton Kernel for Skill Validation

This file contains a deliberately bad Triton kernel designed to test the liger-kernel-perf skill. The kernel implements a fused GELU + Scale operation with **six intentional performance problems**. Any reasonable optimization pass should be able to identify and fix most or all of them, yielding a large speedup (10x+ expected).

## Purpose

When validating the liger-kernel-perf skill, install the canary kernel into the standard Liger paths so it passes pre-flight validation:

```bash
# Copy the canary kernel into standard locations:
cp test/perf_canary/canary_gelu_scale.py src/liger_kernel/ops/canary_gelu_scale.py
cp test/perf_canary/test_canary_gelu_scale.py test/transformers/test_canary_gelu_scale.py
cp test/perf_canary/benchmark_canary_gelu_scale.py benchmark/scripts/benchmark_canary_gelu_scale.py
```

Then run the skill targeting `canary_gelu_scale`. The skill should:
1. Profile the kernel and identify it as memory-bound (element-wise, Tier 1)
2. Diagnose all six problems
3. Generate variants that fix them
4. Achieve a significant speedup

If the skill fails to find obvious improvements on this kernel, something is wrong with the optimization pipeline.

---

## The Deliberately Bad Kernel

Place the following in `test/perf_canary/canary_gelu_scale.py`:

```python
"""
Canary kernel: Fused GELU + Scale with INTENTIONAL performance problems.
DO NOT use this in production. This exists solely to test the liger-kernel-perf skill.

Operation: y = GELU(x) * scale
Where GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Intentional problems:
  1. BLOCK_SIZE = 32 (way too small -- should be 1024+ for most inputs)
  2. num_warps = 1 (should be 4-8 for typical BLOCK_SIZE)
  3. No cache modifiers on loads/stores
  4. Loads X twice from global memory (redundant load)
  5. Uses float64 intermediate computation unnecessarily
  6. Grid is (1,) -- all rows processed sequentially in a single program
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _canary_gelu_scale_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    scale,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU + Scale kernel with intentional performance problems.
    """
    # PROBLEM 6: Grid is (1,) so this single program must loop over ALL rows.
    # A correct kernel would use grid=(n_rows,) with row_idx = tl.program_id(0).
    for row_idx in range(n_rows):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x_ptr_row = X_ptr + row_idx * X_row_stride

        # PROBLEM 4: Load X from global memory TWICE (identical loads).
        # The second load is completely redundant.
        X_row_1 = tl.load(x_ptr_row + col_offsets, mask=mask, other=0.0)
        X_row_2 = tl.load(x_ptr_row + col_offsets, mask=mask, other=0.0)

        # PROBLEM 5: Cast to float64 for no reason. GELU is perfectly fine in
        # float32 (or even bfloat16 for the forward pass). float64 halves
        # throughput on consumer/datacenter GPUs and doubles register pressure.
        X_fp64 = X_row_1.to(tl.float64)
        X_fp64_dup = X_row_2.to(tl.float64)

        # Compute GELU using the tanh approximation, in float64 (unnecessary)
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
        x_cubed = X_fp64 * X_fp64 * X_fp64_dup  # Uses the redundant load
        inner = sqrt_2_over_pi * (X_fp64 + 0.044715 * x_cubed)
        tanh_val = tl.extra.libdevice.tanh(inner)
        gelu_out = 0.5 * X_fp64 * (1.0 + tanh_val)

        # Apply scale
        Y_row = (gelu_out * scale).to(X_row_1.dtype)

        y_ptr_row = Y_ptr + row_idx * Y_row_stride
        tl.store(y_ptr_row + col_offsets, Y_row, mask=mask)


def canary_gelu_scale(X: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Fused GELU + Scale using the deliberately bad canary kernel.

    Args:
        X: Input tensor of shape (M, N) or (B, T, H)
        scale: Scalar multiplier applied after GELU

    Returns:
        Y: GELU(X) * scale, same shape and dtype as X
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    Y = torch.empty_like(X)

    # PROBLEM 1: BLOCK_SIZE = 32 (way too small)
    # For n_cols=4096 with bf16, this means 32 * 2 = 64 bytes per load.
    # Optimal would be 1024-4096, giving 2KB-8KB per load and much better
    # memory bandwidth utilization.
    BLOCK_SIZE = 32

    # PROBLEM 2: num_warps = 1 (way too few)
    # A single warp (32 threads) cannot hide memory latency. For BLOCK_SIZE
    # of 1024+, num_warps should be 4-8.
    num_warps = 1

    # PROBLEM 6 (continued): Grid is (1,) -- only one program instance.
    # On an H100 with 132 SMs, this uses exactly 1 SM and leaves 131 idle.
    grid = (1,)

    _canary_gelu_scale_kernel[grid](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        scale,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return Y.view(*shape)
```

---

## PyTorch Reference Implementation

```python
def torch_gelu_scale(X: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Reference implementation: GELU(X) * scale using PyTorch.
    Uses the tanh approximation to match the Triton kernel exactly.
    """
    return torch.nn.functional.gelu(X, approximate="tanh") * scale
```

---

## Correctness Test

Place the following in `test/perf_canary/test_canary_gelu_scale.py`:

```python
"""
Tests for the canary GELU + Scale kernel.
Validates that the deliberately bad kernel still produces correct results.
"""

import pytest
import torch

from test.perf_canary.canary_gelu_scale import canary_gelu_scale


def torch_gelu_scale(X: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.nn.functional.gelu(X, approximate="tanh") * scale


@pytest.mark.parametrize("shape", [(64, 128), (512, 1024), (2048, 4096)])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_canary_gelu_scale_correctness(shape, scale, dtype):
    """Verify the canary kernel produces correct output despite being slow."""
    torch.manual_seed(42)
    device = "cuda"

    X = torch.randn(shape, dtype=dtype, device=device)
    Y_triton = canary_gelu_scale(X, scale)
    Y_ref = torch_gelu_scale(X, scale)

    # float64 intermediates actually make the canary kernel MORE accurate
    # than needed, so tolerance is generous
    if dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    torch.testing.assert_close(Y_triton, Y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(1, 32), (4, 64), (1, 4096)])
def test_canary_gelu_scale_edge_cases(shape, scale=1.0):
    """Test edge cases: very small inputs, single row."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    X = torch.randn(shape, dtype=dtype, device=device)
    Y_triton = canary_gelu_scale(X, scale)
    Y_ref = torch_gelu_scale(X, scale)

    torch.testing.assert_close(Y_triton, Y_ref, atol=1e-5, rtol=1e-5)


def test_canary_gelu_scale_3d_input():
    """Test that 3D inputs (B, T, H) are handled correctly."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    shape = (2, 128, 256)
    scale = 1.5

    X = torch.randn(shape, dtype=dtype, device=device)
    Y_triton = canary_gelu_scale(X, scale)
    Y_ref = torch_gelu_scale(X, scale)

    assert Y_triton.shape == shape, f"Expected shape {shape}, got {Y_triton.shape}"
    torch.testing.assert_close(Y_triton, Y_ref, atol=1e-5, rtol=1e-5)
```

---

## What Is Wrong (Diagnostic Checklist)

The following table summarizes every intentional problem and what the skill should diagnose:

| # | Problem | Impact | Expected Fix |
|---|---------|--------|--------------|
| 1 | `BLOCK_SIZE = 32` | Extremely low memory bandwidth utilization. Each load is 64 bytes (bf16) instead of 2-8 KB. Triton cannot coalesce memory accesses efficiently. | Increase to 1024-4096 (next power of 2 of n_cols, or use autotune) |
| 2 | `num_warps = 1` | Only 32 threads per program. Cannot hide memory latency, leaves execution units idle. | Increase to 4-8 (or autotune alongside BLOCK_SIZE) |
| 3 | No cache modifiers | All loads/stores use default caching. For a streaming element-wise kernel, `evict_first` on X and Y would reduce L2 pollution. | Add `eviction_policy="evict_first"` on X load and Y store |
| 4 | Redundant global load | X is loaded twice via `X_row_1` and `X_row_2`. This doubles memory traffic for the input tensor. | Remove the duplicate load. Use `X_row_1` everywhere. |
| 5 | `float64` intermediates | Doubles register usage and halves arithmetic throughput. GELU is numerically stable in float32. GPUs have very low float64 throughput (1/64th of float32 on H100 non-FP64 variant). | Use `tl.float32` for intermediates instead of `tl.float64` |
| 6 | `grid = (1,)` with sequential row loop | Only 1 SM is active out of 132 (H100). All other SMs are idle. This is the single biggest performance killer. | Use `grid = (n_rows,)` with `row_idx = tl.program_id(0)`, eliminating the Python-visible loop |

### Expected Impact of Fixes

**Fix 6 alone** (parallelize across rows): ~50-100x speedup (utilizes all 132 SMs instead of 1)

**Fix 1+2** (BLOCK_SIZE + num_warps): ~4-10x speedup on top of fix 6 (better memory coalescing and latency hiding)

**Fix 4** (remove redundant load): ~1.3-1.8x speedup (halves input memory traffic)

**Fix 5** (float32 instead of float64): ~1.5-2x speedup (fp64 is extremely slow on datacenter GPUs)

**Fix 3** (cache modifiers): ~1.05-1.15x speedup (marginal for element-wise kernels)

**All fixes combined:** Expected **10-50x total speedup** depending on input size and GPU. On H100 with shape (2048, 4096) in bfloat16, the canary kernel might take ~5-15ms while an optimized version should take ~0.02-0.05ms.

---

## How to Use This Canary

### Setup

```bash
# From repo root
mkdir -p test/perf_canary
# Copy the kernel code to test/perf_canary/canary_gelu_scale.py
# Copy the test code to test/perf_canary/test_canary_gelu_scale.py
# Create test/perf_canary/__init__.py (empty)

# Verify correctness first
python -m pytest test/perf_canary/test_canary_gelu_scale.py -xvs
```

### Running the Skill

Tell the liger-kernel-perf skill to optimize it:

> "Optimize the canary_gelu_scale kernel in test/perf_canary for speed"

The skill should:
1. **Profile:** Classify as Tier 1 (element-wise), memory-bound, and immediately flag the sequential grid, tiny BLOCK_SIZE, single warp, redundant load, and float64 usage
2. **Optimize:** Generate variants that fix the problems (likely in 1-2 variants since the issues are so obvious)
3. **Finalize:** Apply the optimized kernel, verify tests still pass

### Success Criteria

The skill passes the canary test if:
- It identifies at least 4 of the 6 problems
- The winning variant achieves at least 10x speedup over baseline
- All correctness tests still pass after optimization
- The optimization report accurately describes what was changed and why

### Failure Modes to Watch For

- Skill does not notice `grid = (1,)` -- this is the most critical problem and should be caught by any code analysis
- Skill tries to autotune BLOCK_SIZE but keeps the sequential grid -- autotuning alone cannot fix problem 6
- Skill proposes float16 intermediates instead of float32 -- this could cause correctness issues with tanh
- Skill removes the `for row_idx in range(n_rows)` loop but does not update the grid -- would cause only the first row to be processed
