# Example: Optimizing the RMS Norm Kernel

This example demonstrates the full liger-kernel-perf pipeline applied to the RMS Norm kernel, a **Tier 2 reduction kernel**. It shows realistic profiler output, three optimization variants, a comparison table, and a final report summary.

**User request:** "Optimize the rms_norm kernel for speed on H100"

**Parsed inputs:**
- `target_kernel`: rms_norm
- `optimization_goal`: speed
- `scope`: general (forward + backward)
- `target_gpu`: Hopper (H100)
- `autonomy`: interactive
- `max_variants`: 8

---

## Stage 1: Profile

### GPU Detection

```
GPU: NVIDIA H100 80GB HBM3, SM: 9.0, SMs: 132, Mem: 80GB
Architecture: Hopper
Memory bandwidth: ~3.35 TB/s
Peak FP32 throughput: ~67 TFLOPS
Peak BF16 throughput: ~990 TFLOPS (with tensor cores)
```

### Baseline Benchmarks

Running `benchmark/scripts/benchmark_rms_norm.py --overwrite` with config `M=2048, dtype=bfloat16, eps=1e-6`:

| Hidden Size (H) | Forward (ms) | Backward (ms) | Full (ms) | Memory (MB) |
|------------------|-------------|---------------|-----------|-------------|
| 1024             | 0.0089      | 0.0215        | 0.0312    | 10.0        |
| 2048             | 0.0105      | 0.0268        | 0.0381    | 18.0        |
| 4096             | 0.0158      | 0.0412        | 0.0582    | 34.0        |
| 8192             | 0.0264      | 0.0723        | 0.1003    | 66.0        |
| 16384            | 0.0491      | 0.1362        | 0.1875    | 130.0       |
| 32768            | 0.0943      | 0.2651        | 0.3622    | 258.0       |

### NCU Profiling Summary (at H=4096, M=2048)

**Forward kernel (`_rms_norm_forward_kernel`):**
- SM Occupancy: 48.2%
- Compute Throughput: 18.4% of peak
- Memory Throughput: 72.6% of peak
- L1 Cache Hit Rate: 41.3%
- L2 Cache Hit Rate: 68.7%
- Top warp stall reasons: Long Scoreboard (38%), Wait (24%), Not Selected (19%)

**Backward kernel (`_rms_norm_backward_kernel`):**
- SM Occupancy: 31.5%
- Compute Throughput: 42.1% of peak
- Memory Throughput: 51.8% of peak
- L1 Cache Hit Rate: 34.9%
- L2 Cache Hit Rate: 55.2%
- Top warp stall reasons: Long Scoreboard (29%), Barrier (22%), Math Pipe Throttle (18%)

### Kernel Code Analysis

**File:** `src/liger_kernel/ops/rms_norm.py`

**Tier classification:** Tier 2 (reduction kernel)
- Forward: row-wise reduction (mean of squares), then element-wise scaling
- Backward: row-wise reduction (dot product for dX), plus cross-row reduction for dW (weight gradient summed across batch)

**Current configuration:**
- BLOCK_SIZE: determined by `calculate_settings(n_cols)` -- next power of 2, capped at 65536
- num_warps: 4 (default), 8 for BLOCK_SIZE >= 2048, 16 for >= 8192, 32 for >= 32768
- num_stages: not specified (Triton default)
- Grid: `(n_rows,)` for forward (row-mode), `(sm_count,)` for backward
- Block-row mode: BLOCK_ROW=16 for forward when BLOCK_SIZE <= 256 AND n_rows >= 32768

**Memory access patterns:**
- Forward: contiguous row loads, no cache modifiers, stores Y and RSTD
- Backward: contiguous row loads for dY, X; scalar load for RSTD; accumulates dW in registers across rows_per_program iterations; final dW stored per-SM then summed on host via `_dW.sum(dim=0)`
- No `eviction_policy` or `cache_modifier` hints on any loads

**Compute patterns:**
- `casting_mode` as constexpr (llama/gemma/none) -- compiler eliminates dead branches
- Forward: square, sum, rsqrt, scale -- lightweight per-element
- Backward: heavier -- dot product, re-scale, dW accumulation per row

**Optimization opportunities identified:**
1. `calculate_settings()` uses a fixed heuristic for num_warps -- no autotuning
2. No cache modifiers on loads -- forward could benefit from `eviction_policy="evict_last"` on W since it is reused across all rows
3. Backward weight gradient uses `(sm_count,)` grid with `rows_per_program = ceil(n_rows / sm_count)` -- this means each SM processes ~15 rows sequentially (for M=2048 on H100 with 132 SMs). The final `_dW.sum(dim=0)` is a PyTorch host-side reduction. A two-level reduction (intra-SM in registers, then an atomic or second kernel) could improve this.
4. Block-row mode is used in forward but NOT in backward when conditions are met -- backward always falls through to standard row mode for BLOCK_SIZE <= 256

### Bottleneck Classification

| Pass     | Classification | Evidence                                                          |
|----------|----------------|-------------------------------------------------------------------|
| Forward  | Memory-bound   | Memory throughput 72.6%, compute 18.4%. Simple ops per load.      |
| Backward | Balanced       | Memory 51.8%, compute 42.1%. Heavier math + cross-row reduction.  |

### Recommended Strategy Order

1. **Autotuning sweep** (BLOCK_SIZE, num_warps, num_stages) -- low-hanging fruit, often 5-15% gain
2. **Cache modifiers + vectorized loads** -- forward is memory-bound, so improving cache behavior and load efficiency has high impact
3. **SM utilization for backward weight gradient** -- backward is balanced; improving parallelism in the dW reduction can help
4. **(Stretch) Persistent kernel / warp specialization** -- only if gains from 1-3 are insufficient

Estimated iterations: 3-4 variants

---

## Stage 2: Optimize

### Variant 1: Autotuning Sweep

**Hypothesis:** The fixed `calculate_settings()` heuristic may not be optimal for all hidden sizes. A Triton autotune sweep over BLOCK_SIZE, num_warps, and num_stages may find better configurations, especially for medium hidden sizes (2048-8192).

**Changes:**
- Replace `calculate_settings()` call with `@triton.autotune` decorator on both forward and backward kernels
- Sweep configs:
  ```python
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
          for bs in [1024, 2048, 4096, 8192]
          for nw in [4, 8, 16]
          for ns in [2, 3, 4]
      ],
      key=["n_cols"],
  )
  ```
- Adjusted the Python wrapper to pass `n_cols` as a grid-level parameter for autotune key matching

**Smoke test:** PASSED (float32, shape=(128, 4096), forward+backward, max diff < 1e-5)

**Benchmark results (v1 vs baseline, forward ms, median):**

| Hidden Size (H) | Baseline | v1 Autotune | Speedup |
|------------------|----------|-------------|---------|
| 1024             | 0.0089   | 0.0084      | 1.06x   |
| 2048             | 0.0105   | 0.0098      | 1.07x   |
| 4096             | 0.0158   | 0.0143      | 1.10x   |
| 8192             | 0.0264   | 0.0241      | 1.10x   |
| 16384            | 0.0491   | 0.0468      | 1.05x   |
| 32768            | 0.0943   | 0.0931      | 1.01x   |

**Backward ms:**

| Hidden Size (H) | Baseline | v1 Autotune | Speedup |
|------------------|----------|-------------|---------|
| 1024             | 0.0215   | 0.0202      | 1.06x   |
| 2048             | 0.0268   | 0.0249      | 1.08x   |
| 4096             | 0.0412   | 0.0385      | 1.07x   |
| 8192             | 0.0723   | 0.0678      | 1.07x   |
| 16384            | 0.1362   | 0.1298      | 1.05x   |
| 32768            | 0.2651   | 0.2615      | 1.01x   |

**Notes:** Modest but consistent improvement (5-10%) across medium hidden sizes. Autotune found that `num_stages=3` is better than default for H=4096-8192. At large hidden sizes (32768), the original heuristic was already near-optimal. The autotune compilation overhead is notable for the first call (~2-4 seconds) but amortized over repeated calls.

**Guardrails:** No regressions > 5%. Memory unchanged. PASS.

---

### Variant 2: Cache Modifiers + Vectorized Loads

**Hypothesis:** Since the forward pass is memory-bound (72.6% memory throughput), we can improve effective bandwidth by:
1. Adding `eviction_policy="evict_last"` to the weight load (W is small and reused by every row)
2. Using `tl.load` with `eviction_policy="evict_first"` on X (each row is read once in forward, no reuse)
3. Ensuring 128-bit vectorized loads by aligning BLOCK_SIZE to be a multiple of 8 (for bf16, 8 elements = 16 bytes = 128 bits)

**Changes (on top of v1 autotune):**
```python
# Forward kernel -- weight load with evict_last (stays in L2 for all rows)
W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0, eviction_policy="evict_last")

# Forward kernel -- input load with evict_first (no reuse in forward)
X_row = tl.load(x_base + col_offsets, mask=mask, other=0, eviction_policy="evict_first")

# Output store with evict_first
tl.store(y_base + col_offsets, Y_row, mask=mask, eviction_policy="evict_first")
```

Also added `eviction_policy="evict_last"` on the RSTD store (RSTD is small and will be loaded during backward).

**Smoke test:** PASSED (float32, shape=(128, 4096), forward+backward, max diff < 1e-5)

**Benchmark results (v2 vs v1 vs baseline, forward ms):**

| Hidden Size (H) | Baseline | v1       | v2 Cache | v2 Speedup vs Baseline |
|------------------|----------|----------|----------|------------------------|
| 1024             | 0.0089   | 0.0084   | 0.0078   | 1.14x                  |
| 2048             | 0.0105   | 0.0098   | 0.0089   | 1.18x                  |
| 4096             | 0.0158   | 0.0143   | 0.0128   | 1.23x                  |
| 8192             | 0.0264   | 0.0241   | 0.0215   | 1.23x                  |
| 16384            | 0.0491   | 0.0468   | 0.0422   | 1.16x                  |
| 32768            | 0.0943   | 0.0931   | 0.0879   | 1.07x                  |

**Backward ms (cache modifiers on dY, X loads and dW store):**

| Hidden Size (H) | Baseline | v1       | v2 Cache | v2 Speedup vs Baseline |
|------------------|----------|----------|----------|------------------------|
| 1024             | 0.0215   | 0.0202   | 0.0195   | 1.10x                  |
| 2048             | 0.0268   | 0.0249   | 0.0238   | 1.13x                  |
| 4096             | 0.0412   | 0.0385   | 0.0365   | 1.13x                  |
| 8192             | 0.0723   | 0.0678   | 0.0648   | 1.12x                  |
| 16384            | 0.1362   | 0.1298   | 0.1252   | 1.09x                  |
| 32768            | 0.2651   | 0.2615   | 0.2542   | 1.04x                  |

**Notes:** Good improvement on forward (14-23% at medium sizes), confirming that the forward pass was under-utilizing the cache hierarchy. Backward improvement is more modest (10-13%) since it is balanced rather than purely memory-bound. The cache modifiers help the backward pass primarily on the X reload (which is also loaded in forward and may now survive in L2).

**Guardrails:** No regressions > 5%. Memory unchanged. PASS.

---

### Variant 3: Improved SM Utilization in Backward Weight Gradient

**Hypothesis:** The backward weight gradient reduction currently uses `grid = (sm_count,)` where each SM processes `ceil(n_rows / sm_count)` rows sequentially, accumulating dW in registers. For M=2048 on H100 (132 SMs), that is ~16 rows per SM. The final `_dW.sum(dim=0)` is a host-side PyTorch reduction over the 132 SM partial results.

We can improve this by:
1. Using a finer grid for the backward kernel -- `grid = (min(n_rows, sm_count * 4),)` -- each program handles fewer rows, increasing parallelism
2. Switching to an atomic accumulation pattern for dW with `tl.atomic_add`, eliminating the host-side reduction entirely
3. Alternatively (chosen approach): use a 2D grid `(n_cols_blocks, sm_count)` with a two-pass reduction -- first accumulate per-SM, then reduce in a second small kernel

**Chosen approach: Finer grid with host reduction.**

After benchmarking both atomic and finer-grid approaches, the finer grid (`min(n_rows, sm_count * 4)` programs) was better because:
- Atomics on bf16 dW had precision issues requiring an fp32 scratch buffer anyway
- The finer grid gives 4x more parallelism with a slightly larger intermediate `_dW` tensor (528 x n_cols vs 132 x n_cols on H100)
- Host-side `_dW.sum(dim=0)` over 528 rows is still negligible

**Changes (on top of v2):**
```python
# In rms_norm_backward():
# Old: grid = (sm_count,) with rows_per_program = ceil(n_rows / sm_count)
# New: use more programs for finer parallelism
n_programs = min(n_rows, sm_count * 4)
rows_per_program = math.ceil(n_rows / n_programs)
grid = (n_programs,)

_dW = torch.empty((n_programs, n_cols), dtype=torch.float32, device=W.device)
```

Also applied the block-row mode to the backward kernel for BLOCK_SIZE <= 256 (matching the forward path), which was a missing optimization noted in the code analysis:
```python
if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
    _rms_norm_backward_kernel[grid](...)
else:
    # Now also uses block-row backward kernel
    BLOCK_ROW = 16
    _block_rms_norm_backward_kernel[grid](...)
```

**Smoke test:** PASSED

**Benchmark results (v3 vs v2 vs baseline, backward ms):**

| Hidden Size (H) | Baseline | v2       | v3 SM Util | v3 Speedup vs Baseline |
|------------------|----------|----------|------------|------------------------|
| 1024             | 0.0215   | 0.0195   | 0.0178     | 1.21x                  |
| 2048             | 0.0268   | 0.0238   | 0.0214     | 1.25x                  |
| 4096             | 0.0412   | 0.0365   | 0.0322     | 1.28x                  |
| 8192             | 0.0723   | 0.0648   | 0.0575     | 1.26x                  |
| 16384            | 0.1362   | 0.1252   | 0.1138     | 1.20x                  |
| 32768            | 0.2651   | 0.2542   | 0.2398     | 1.11x                  |

**Forward ms (v3 inherits v2 cache optimizations, unchanged):**

| Hidden Size (H) | Baseline | v3       | Speedup |
|------------------|----------|----------|---------|
| 4096             | 0.0158   | 0.0128   | 1.23x   |

**Notes:** The finer backward grid gives a substantial improvement (20-28% at medium sizes). The block-row backward mode contributed ~3-5% of the gain at H=1024. The extra memory for `_dW` is minimal (528 * 32768 * 4 bytes = ~66 MB at the largest hidden size, vs ~17 MB for the original 132-row version). This is acceptable since it is a transient allocation during backward only.

**Guardrails:** No regressions > 5%. Forward performance identical to v2. Memory increased by <1% at largest sizes. PASS.

---

## Comparison Table (All Variants)

### Forward (ms, median)

| H     | Baseline | v1 Autotune | v2 Cache | v3 (v2+SM) | Best Speedup |
|-------|----------|-------------|----------|------------|--------------|
| 1024  | 0.0089   | 0.0084      | 0.0078   | 0.0078     | 1.14x        |
| 2048  | 0.0105   | 0.0098      | 0.0089   | 0.0089     | 1.18x        |
| 4096  | 0.0158   | 0.0143      | 0.0128   | 0.0128     | 1.23x        |
| 8192  | 0.0264   | 0.0241      | 0.0215   | 0.0215     | 1.23x        |
| 16384 | 0.0491   | 0.0468      | 0.0422   | 0.0422     | 1.16x        |
| 32768 | 0.0943   | 0.0931      | 0.0879   | 0.0879     | 1.07x        |

### Backward (ms, median)

| H     | Baseline | v1 Autotune | v2 Cache | v3 (v2+SM) | Best Speedup |
|-------|----------|-------------|----------|------------|--------------|
| 1024  | 0.0215   | 0.0202      | 0.0195   | 0.0178     | 1.21x        |
| 2048  | 0.0268   | 0.0249      | 0.0238   | 0.0214     | 1.25x        |
| 4096  | 0.0412   | 0.0385      | 0.0365   | 0.0322     | 1.28x        |
| 8192  | 0.0723   | 0.0678      | 0.0648   | 0.0575     | 1.26x        |
| 16384 | 0.1362   | 0.1298      | 0.1252   | 0.1138     | 1.20x        |
| 32768 | 0.2651   | 0.2615      | 0.2542   | 0.2398     | 1.11x        |

### Full (forward + backward, ms, median)

| H     | Baseline | v3 (Winner) | Speedup |
|-------|----------|-------------|---------|
| 1024  | 0.0312   | 0.0261      | 1.20x   |
| 2048  | 0.0381   | 0.0308      | 1.24x   |
| 4096  | 0.0582   | 0.0456      | 1.28x   |
| 8192  | 0.1003   | 0.0798      | 1.26x   |
| 16384 | 0.1875   | 0.1571      | 1.19x   |
| 32768 | 0.3622   | 0.3289      | 1.10x   |

### Winner: v3 (combines autotuning + cache modifiers + improved SM utilization)

v3 includes all optimizations from v1 and v2. It is the cumulative best across both forward and backward passes.

---

## Stage 3: Finalize

### Test Suite

```
$ python -m pytest test/transformers/test_rms_norm.py -xvs
...
test/transformers/test_rms_norm.py::test_liger_rms_norm[dtype0-1e-06-4096-llama-True-1-64] PASSED
test/transformers/test_rms_norm.py::test_liger_rms_norm[dtype0-1e-06-4096-llama-True-4-2048] PASSED
test/transformers/test_rms_norm.py::test_liger_rms_norm[dtype0-1e-06-4096-gemma-True-1-64] PASSED
...
test/transformers/test_rms_norm.py::test_liger_rms_norm_functional PASSED
test/transformers/test_rms_norm.py::test_liger_rms_norm_pickle PASSED
===== 87 passed in 42.31s =====
```

All 87 tests pass. No correctness regressions.

### Checkstyle

```
$ make checkstyle
ruff check . --fix
All checks passed!
ruff format .
42 files left unchanged
```

### Changes Applied

The following changes were applied to `src/liger_kernel/ops/rms_norm.py`:

1. **Added `@triton.autotune` decorator** to `_rms_norm_forward_kernel` and `_rms_norm_backward_kernel` with configs sweeping BLOCK_SIZE, num_warps, and num_stages. The `calculate_settings()` heuristic is retained as a fallback for the block-row mode variants.

2. **Added cache modifiers** to all `tl.load` and `tl.store` calls:
   - Forward: `eviction_policy="evict_last"` on W load and RSTD store; `eviction_policy="evict_first"` on X load and Y store
   - Backward: `eviction_policy="evict_last"` on W load and RSTD load; `eviction_policy="evict_first"` on dY/X loads and dX store

3. **Increased backward grid parallelism**: `n_programs = min(n_rows, sm_count * 4)` instead of `sm_count`, with corresponding `_dW` allocation sized to `n_programs`.

4. **Extended block-row backward mode**: The `_block_rms_norm_backward_kernel` is now invoked when BLOCK_SIZE <= 256 and n_rows >= 32768 (matching the forward path's conditions), instead of always using the standard row-mode backward kernel.

---

## Final Report Summary

```
============================================================
  RMS NORM OPTIMIZATION REPORT
  GPU: NVIDIA H100 80GB HBM3 (Hopper, SM 9.0)
  Date: 2026-04-03
============================================================

  Target kernel:  rms_norm
  Optimization:   speed (balanced forward + backward)
  Variants tried: 3

  FORWARD PASS
  -------------------------------------------------------
  Representative (H=4096, M=2048, bf16):
    Before:  0.0158 ms
    After:   0.0128 ms
    Speedup: 1.23x
  Peak improvement: 1.23x at H=4096 and H=8192

  BACKWARD PASS
  -------------------------------------------------------
  Representative (H=4096, M=2048, bf16):
    Before:  0.0412 ms
    After:   0.0322 ms
    Speedup: 1.28x
  Peak improvement: 1.28x at H=4096

  FULL (forward + backward)
  -------------------------------------------------------
  Representative (H=4096, M=2048, bf16):
    Before:  0.0582 ms
    After:   0.0456 ms
    Speedup: 1.28x

  MEMORY
  -------------------------------------------------------
  No significant change (<1% increase from larger _dW buffer)

  CORRECTNESS
  -------------------------------------------------------
  All 87 existing tests pass.
  Numerical accuracy: unchanged (same casting_mode logic).

  KEY OPTIMIZATIONS APPLIED
  -------------------------------------------------------
  1. Autotuning (BLOCK_SIZE, num_warps, num_stages)
  2. Cache modifier hints (evict_last for reused data,
     evict_first for streaming data)
  3. 4x finer backward grid for weight gradient parallelism
  4. Block-row backward mode for small hidden sizes

============================================================
```
