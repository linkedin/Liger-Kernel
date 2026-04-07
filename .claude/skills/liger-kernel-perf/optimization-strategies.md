# Optimization Strategies Catalog

Reference catalog for the Optimizer Agent. Techniques are ordered by recommended trial order within each category. The agent should **always start with parameter tuning**, then apply diagnosis-driven techniques based on the bottleneck classification from the Profiler stage.

---

## Decision Framework

After reading the optimization profile, follow this priority:

1. **Always first**: Parameter Tuning (Section 1)
2. **Memory-bound kernel**: Section 2 techniques, then Section 5
3. **Compute-bound kernel**: Section 3 techniques, then Section 5
4. **Balanced kernel**: Interleave Section 2 and Section 3, then Section 5
5. **Architecture-specific**: Layer Section 4 techniques on top of the best variant when targeting a specific GPU

---

## 1. Always First: Parameter Tuning

**Why first**: Parameter tuning is the cheapest technique (minimal code changes) and frequently delivers the largest single improvement. The current codebase uses `calculate_settings()` which picks BLOCK_SIZE as the next power of 2 and num_warps via a fixed heuristic. This is a reasonable default but leaves significant performance on the table for specific input shapes and GPU architectures.

### Why NOT @triton.autotune

Liger production kernels **cannot use `@triton.autotune`** for three architectural reasons:

1. **Forward-backward context coupling**: Liger kernels use `torch.autograd.Function` where `forward()` computes `BLOCK_SIZE`/`num_warps` via `calculate_settings()`, stores them in `ctx`, and `backward()` retrieves them to launch its kernel with identical config. With `@triton.autotune`, these become compile-time constants managed by Triton -- they cannot be stored in `ctx`, and the backward kernel has no way to know which config the forward kernel auto-selected for a given input.

2. **Multi-backend incompatibility**: NPU/Ascend does not support `num_warps` or `num_stages` at all (commit `8b965b9` explicitly removed them). `@triton.autotune` relies on these parameters.

3. **Inference warmup latency**: Each new input shape triggers autotuning (benchmarks all configs). Adds 100ms+ first-call penalty per unique shape, unacceptable for LLM inference with variable sequence lengths.

The commented-out `@triton.autotune` decorators in `dyt.py` and `grpo_loss.py` are evidence of prior exploration that was abandoned for these reasons. Do NOT re-introduce `@triton.autotune` on production kernels.

### Parameters to Tune

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `BLOCK_SIZE` | Elements processed per program instance | 256, 512, 1024, 2048, 4096, 8192, 16384 |
| `num_warps` | Warps per program instance (32 threads each) | 2, 4, 8, 16, 32 |
| `num_stages` | Software pipelining depth | 1, 2, 3, 4, 5 |
| `BLOCK_ROW` | Rows per program (block-row mode only) | 1, 2, 4, 8, 16, 32 |

### Method: Manual Sweep with Improved Heuristic

Replace `calculate_settings()` with a better heuristic derived from benchmarking. The variant must still pass `BLOCK_SIZE` and `num_warps` through `ctx` to the backward kernel.

**Step 1**: Identify the current config for the benchmark's input sizes:
```python
# calculate_settings() returns (next_power_of_2(n), num_warps_heuristic)
# For n_cols=4096: BLOCK_SIZE=4096, num_warps=8
# For n_cols=8192: BLOCK_SIZE=8192, num_warps=16
# Note: num_stages is never set (Triton defaults to 1)
```

**Step 2**: Create a variant with an improved selection function:
```python
def optimized_settings(n):
    """Improved heuristic derived from benchmarking on {gpu_arch}."""
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > 65536:
        raise RuntimeError(...)
    # Tuned values (from benchmark sweep):
    if BLOCK_SIZE >= 8192:
        num_warps = 16
        num_stages = 3   # NEW: calculate_settings doesn't set this
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
        num_stages = 2
    else:
        num_warps = 4
        num_stages = 2
    return BLOCK_SIZE, num_warps, num_stages
```

**Step 3**: Update the kernel launcher to pass `num_stages`:
```python
# The kernel launch gains num_stages (previously unset, defaulted to 1):
kernel[(grid,)](..., BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)
```

**Step 4**: Update `ctx` to also store `num_stages` and pass it to the backward kernel.

**Key insight**: `num_stages` is the most commonly overlooked parameter. `calculate_settings()` does not set it at all, so Triton defaults to `num_stages=1`. Simply adding `num_stages=2` or `num_stages=3` to existing kernel launches is often the single biggest win, especially on Hopper.

### Architecture-Specific Guidance

- **Ampere (SM 8.x)**: `num_stages=2` is usually optimal. Higher stages rarely help and can increase register pressure. `num_warps` sweet spot is 4-16.
- **Hopper (SM 9.0)**: Benefits significantly from higher `num_stages` (3-5) due to TMA and async pipeline support. Also tolerates `num_warps=32` better due to deeper warp scheduling.
- **Blackwell (SM 10.0)**: Similar to Hopper but with even more pipeline depth tolerance. Start with `num_stages=4-5`.

### When to Apply

Always. This is the mandatory first optimization attempt for every kernel.

### Expected Impact

5-40% speedup depending on how far the `calculate_settings()` heuristic is from optimal for the target shapes and GPU. Adding `num_stages` alone can yield 10-25% on Hopper.

### Risks/Gotchas

- Configs are shape-dependent. A config optimal for `n_cols=4096` may be suboptimal for `n_cols=128`. Always benchmark across ALL x_values from the existing benchmark suite.
- If BLOCK_SIZE exceeds `n_cols`, the kernel still works (masked loads) but wastes warps. Prune configs where `BLOCK_SIZE >> n_cols`.
- The `calculate_settings()` function enforces `MAX_FUSED_SIZE = 65536`. This limit exists to prevent register spilling. Respect this unless verified otherwise.
- When adding `num_stages` to existing kernels, the backward kernel must also receive the same `num_stages` value via `ctx`.
- On NPU/Ascend, `num_warps` and `num_stages` are ignored. Optimizations targeting these parameters are NVIDIA-specific.

### Kernel Tiers

All tiers benefit. Tier 1 kernels often see the largest relative gains because their simple structure makes the config choice proportionally more impactful.

---

## 2. Memory-Bound Optimizations

Apply these when the profiler classifies the kernel as memory-bound (high memory throughput, low compute utilization, simple operations per load/store).

### 2.1 Memory Access Coalescing

**When to apply**: NCU shows low memory throughput efficiency, or kernel uses non-contiguous access patterns (e.g., strided access across rows, column-major access on row-major data).

**How to implement**:

Ensure that threads in a warp access adjacent memory addresses. In Triton, this means the innermost offset computation should produce contiguous addresses across the `tl.arange` dimension.

```python
# GOOD: contiguous access within a warp
col_offsets = tl.arange(0, BLOCK_SIZE)
X_block = tl.load(X_ptr + row_idx * stride + col_offsets, mask=col_offsets < n_cols)

# BAD: strided access (each thread hits a different cache line)
row_offsets = tl.arange(0, BLOCK_SIZE)
X_block = tl.load(X_ptr + row_offsets * stride + col_idx)
```

In Liger kernels, most access is already contiguous via `col_offsets = tl.arange(0, BLOCK_SIZE)`. Check for cases where stride-based pointer arithmetic breaks coalescing, especially in backward kernels or multi-dimensional grids.

**Expected impact**: 10-50% when fixing uncoalesced access. Minimal gain if access is already coalesced.

**Risks**: Changing access patterns may require transposing data or restructuring the grid, which can be invasive.

**Kernel tiers**: All tiers, but Tier 2/3 backward passes most often have coalescing issues due to accumulation patterns.

### 2.2 Cache Modifiers

**When to apply**: Kernel reads data that is used exactly once (stream through), or reads data that is reused many times (should be cached aggressively). NCU shows poor L2 hit rate.

**How to implement**:

Triton supports cache modifiers via the `eviction_policy` parameter on `tl.load` and `tl.store`:

```python
# Data read once, don't pollute cache (streaming access)
X_block = tl.load(X_ptr + offsets, eviction_policy="evict_first")

# Data reused across iterations, keep in cache
W_row = tl.load(W_ptr + offsets, eviction_policy="evict_last")

# Default (let hardware decide)
X_block = tl.load(X_ptr + offsets)  # no modifier
```

| Modifier | Triton API | When to Use |
|----------|------------|-------------|
| `.ca` (cache all) | `eviction_policy="evict_last"` | Data reused across loop iterations or by other programs |
| `.cs` (cache streaming) | `eviction_policy="evict_first"` | Data used once; large tensors that would thrash cache |
| `.wb` (write-back) | default store behavior | Standard write-back through cache hierarchy |
| `.wt` (write-through) | not directly exposed | Write-through (rarely needed in Triton) |

**Expected impact**: 5-20% on memory-bound kernels with predictable access patterns.

**Risks**: Wrong eviction policy can degrade performance. `.evict_first` on reused data will cause redundant memory fetches. Always benchmark both policies against the default.

**Kernel tiers**: Tier 2 and Tier 3 benefit most (weight reuse in backward passes, multi-pass algorithms).

### 2.3 Reduce Memory Traffic

**When to apply**: Kernel reads or writes data that could be computed on the fly, or writes intermediate results that are only consumed once.

**How to implement**:

Liger already uses several memory-saving patterns. Look for additional opportunities:

```python
# Pattern 1: Recompute instead of saving to memory
# Instead of: save silu(a) to a buffer, load in backward
# Do: recompute silu(a) in backward from saved a
sig_a = tl.sigmoid(a_row)
silu_a = a_row * sig_a  # recomputed in backward

# Pattern 2: In-place gradient storage (gradient-in-forward trick)
# Cross entropy already does this: stores gradient in the input tensor
# Look for other kernels that allocate separate grad tensors unnecessarily

# Pattern 3: Fuse producer-consumer pairs to avoid materializing intermediates
# Instead of: Y = kernel1(X); Z = kernel2(Y)  (Y written to HBM, then read)
# Do: Z = fused_kernel(X)  (Y lives only in registers/SRAM)

# Pattern 4: Write in-place to dY for dX when shapes match
if in_place:
    dX = dY  # reuse memory, zero additional allocation
```

**Expected impact**: 10-40% from eliminating unnecessary memory round-trips. Memory savings also reduce peak memory.

**Risks**: Recomputation trades memory for compute. Only beneficial when the kernel is memory-bound. Can also complicate backward pass correctness -- always verify gradient accuracy.

**Kernel tiers**: All tiers. Tier 3 kernels (fused operations) have the most opportunity for intermediate elimination.

### 2.4 Data Reuse (Pre-Load Outside Loops)

**When to apply**: A value is loaded inside a loop but does not change across iterations. Common with weight tensors and scaling factors.

**How to implement**:

```python
# BEFORE: weight loaded inside the loop (redundant loads if compiler doesn't hoist)
for i in range(0, n_cols, BLOCK_SIZE):
    W_block = tl.load(W_ptr + offsets)  # loaded every iteration if W is large
    X_block = tl.load(X_ptr + offsets)
    Y_block = X_block * W_block

# AFTER: hoist the invariant load
W_row = tl.load(W_ptr + col_offsets, mask=col_offsets < n_cols)
for i in range(0, n_cols, BLOCK_SIZE):
    X_block = tl.load(X_ptr + offsets)
    Y_block = X_block * W_row  # reuse from registers
```

Note: Triton's compiler often hoists loads automatically, but it is not guaranteed for all patterns, especially when the loop body is complex.

**Expected impact**: 5-15% when applicable.

**Risks**: Hoisting large loads outside loops increases register pressure. If the weight vector is very large relative to BLOCK_SIZE, it may cause register spilling that offsets the benefit.

**Kernel tiers**: Tier 2 (norm kernels with weight vectors), Tier 3 (cross-entropy with weight tensors).

### 2.5 Block-Row Mode (Multiple Rows per Program)

**When to apply**: The kernel processes one row per program instance, but each row is small (e.g., `n_cols <= 256`). This under-utilizes each SM because the program does very little work.

**How to implement**:

Liger's `rms_norm.py` already implements this as `_block_rms_norm_forward_kernel`. The pattern:

```python
@triton.jit
def kernel_block_row(
    ...,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,  # number of rows per program
):
    # Each program processes BLOCK_ROW rows
    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols

    # 2D load: [BLOCK_ROW, BLOCK_SIZE]
    X = tl.load(
        X_ptr + row_idx[:, None] * stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    # Process all rows at once...
```

Grid: `(triton.cdiv(n_rows, BLOCK_ROW),)` instead of `(n_rows,)`.

Use conditional dispatch to choose between row mode and block-row mode:

```python
if BLOCK_SIZE > 256 or n_rows < 4096 * 8:
    # Row mode: one row per program
    kernel_row[(n_rows,)](...)
else:
    # Block-row mode: multiple rows per program
    BLOCK_ROW = 16
    kernel_block_row[(triton.cdiv(n_rows, BLOCK_ROW),)](
        ..., BLOCK_ROW=BLOCK_ROW
    )
```

**Expected impact**: 10-30% for kernels with small `n_cols` and large `n_rows`.

**Risks**: Increases register pressure and shared memory usage. The 2D load pattern is less efficient when `BLOCK_ROW` causes register spilling. Also adds code complexity (two kernel variants). Does not help when `n_cols` is already large.

**Kernel tiers**: Tier 1 and Tier 2 primarily. Many Liger Tier 1 kernels (swiglu, geglu, relu_squared) do not yet have block-row mode.

### 2.6 Vectorized Loads

**When to apply**: Kernel loads/stores many small elements individually. Memory bandwidth is limited by the number of transactions rather than total bytes.

**How to implement**:

Triton generally handles vectorization automatically when loads are contiguous and aligned. However, you can help by:

1. Ensuring BLOCK_SIZE is a multiple of 8 (for fp16/bf16) or 4 (for fp32) to enable 128-bit loads.
2. Ensuring pointer alignment (tensors created by PyTorch are typically aligned).
3. Avoiding masked loads at boundaries where possible -- pad the input instead.

```python
# Triton auto-vectorizes contiguous loads. Just ensure alignment:
# BLOCK_SIZE should be power of 2 (already enforced by calculate_settings)
# Data should be contiguous (already enforced by ensure_contiguous decorator)

# For explicit vectorization (rare, advanced):
# Cast pointer to wider type, load fewer elements
# This is rarely needed as Triton handles it, but can help in edge cases
```

**Expected impact**: 5-10% if loads are currently unvectorized. Usually Triton already handles this.

**Risks**: Minimal. Padding inputs wastes a small amount of memory.

**Kernel tiers**: All tiers, but gains are usually small since Triton auto-vectorizes well.

---

## 3. Compute-Bound Optimizations

Apply these when the profiler classifies the kernel as compute-bound (high compute utilization, low memory throughput utilization, complex math per element).

### 3.1 Algorithmic Improvements

**When to apply**: The kernel uses a naive algorithm where a more efficient one exists. Examples: naive softmax vs online softmax, explicit matrix inverse vs iterative method.

**How to implement**:

Liger already uses online softmax in cross_entropy. Look for similar opportunities:

```python
# Online algorithm pattern: single-pass with running statistics
# Instead of: pass 1 = compute max, pass 2 = compute sum, pass 3 = compute output
# Do: single pass maintaining running max and sum

m = float("-inf")
d = 0.0
for i in range(0, n_cols, BLOCK_SIZE):
    X_block = tl.load(...)
    block_max = tl.max(X_block)
    m_new = tl.maximum(m, block_max)
    d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
    m = m_new
# Result: lse = m + tl.log(d)
```

Other algorithmic opportunities:
- Welford's algorithm for variance (numerically stable single-pass)
- Kahan summation for high-precision reduction
- Fast approximations for transcendentals when full precision is not needed

**Expected impact**: 20-60% when applicable (reducing O(n) passes).

**Risks**: Numerical stability can differ from reference implementation. Always validate against PyTorch reference with tight tolerances. Online algorithms may accumulate floating-point errors differently.

**Kernel tiers**: Tier 2 and Tier 3 primarily. Tier 1 kernels rarely have algorithmic improvement opportunities.

### 3.2 Reduce Redundant Computation

**When to apply**: The same sub-expression is computed multiple times within a kernel, or across forward and backward passes.

**How to implement**:

```python
# BEFORE: sigmoid computed twice
silu_a = a_row * tl.sigmoid(a_row)
grad = dc_row * (silu_a * (1 - tl.sigmoid(a_row)) + tl.sigmoid(a_row)) * b_row

# AFTER: compute once, reuse
sig_a = tl.sigmoid(a_row)
silu_a = a_row * sig_a
grad = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
```

Liger already does this well (see swiglu backward). Look for:
- Repeated `tl.exp()`, `tl.sigmoid()`, `tl.log()` calls with the same argument
- Reduction results (sum, max) computed multiple times
- Softmax probabilities recomputed instead of cached in registers

**Expected impact**: 5-20% depending on the cost of the duplicated operation.

**Risks**: Caching sub-expressions increases register pressure. Trade-off with register spilling.

**Kernel tiers**: Tier 2 and Tier 3. Tier 1 kernels are typically simple enough that redundancy is already eliminated.

### 3.3 Tensor Core Utilization

**When to apply**: Kernel performs matrix-multiply-like operations (dot products, outer products) that could leverage tensor cores. This is most relevant for Tier 3 kernels with linear layers or attention-like patterns.

**How to implement**:

Tensor cores require specific alignment:
- Block dimensions must be multiples of 16 (for fp16/bf16) or 8 (for tf32)
- Input/output matrices must be in supported layouts
- Use `tl.dot()` which maps to tensor core instructions when dimensions align

```python
# Ensure BLOCK_M, BLOCK_N, BLOCK_K are multiples of 16
BLOCK_M: tl.constexpr = 64  # multiple of 16
BLOCK_N: tl.constexpr = 64  # multiple of 16
BLOCK_K: tl.constexpr = 32  # multiple of 16

# tl.dot maps to tensor cores automatically
acc = tl.dot(A_block, B_block)  # [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
```

For `tl.dot` to use tensor cores:
- Both operands must be 2D
- Operand dtypes must be fp16, bf16, or tf32
- Inner dimension K must be >= 16

**Expected impact**: 2-10x for matrix-multiply-heavy kernels. Not applicable to element-wise or simple reduction kernels.

**Risks**: Tensor core operations have lower precision than FP32 scalar operations. The `tf32` mode on Ampere+ truncates FP32 mantissa to 10 bits. Always validate numerical accuracy. Not all kernels have operations that map to tensor cores.

**Kernel tiers**: Tier 3 only (fused_linear_cross_entropy, attention-like kernels). Not applicable to Tier 1 or most Tier 2 kernels.

### 3.4 Register Pressure Reduction

**When to apply**: NCU shows high register usage per thread (>128 registers) or occupancy is limited by register count. Kernel body has many live variables.

**How to implement**:

```python
# Strategy 1: Use constexpr branches to eliminate dead code paths
# Triton compiles out branches on constexpr values, reducing register usage
casting_mode: tl.constexpr  # compiler eliminates unused casting branches
HAS_WEIGHT: tl.constexpr    # compiler eliminates weight-related registers when False

# Strategy 2: Split large kernels into smaller passes
# Instead of one kernel doing everything:
# Kernel 1: forward + save intermediates
# Kernel 2: backward using intermediates
# This is a last resort as it adds kernel launch overhead

# Strategy 3: Reduce live range of temporary variables
# Reorder operations so temporary values are consumed immediately
for i in range(0, n_cols, BLOCK_SIZE):
    X_block = tl.load(...)
    # Use X_block immediately, don't hold it across iterations
    result = compute(X_block)
    tl.store(..., result)
    # X_block and result are now dead, registers freed

# Strategy 4: Use smaller data types for intermediates where precision allows
# fp16 uses half the registers of fp32
intermediate = X_block.to(tl.float16)  # only if precision loss is acceptable
```

**Expected impact**: 10-25% when occupancy is register-limited. Minimal impact when occupancy is already high.

**Risks**: Splitting kernels adds launch overhead and memory traffic for intermediates. Reducing precision can cause numerical issues. Constexpr branches are already used extensively in Liger -- check for remaining opportunities.

**Kernel tiers**: Tier 3 (most register pressure due to complex logic). Tier 2 backward passes (accumulation variables).

### 3.5 2D Grid Tiling for Better SM Utilization

**When to apply**: Kernel uses a 1D grid but the problem has two natural dimensions (e.g., batch and feature). A 1D grid may leave SMs idle when the number of programs is not a multiple of SM count.

**How to implement**:

```python
# 1D grid (current Liger pattern for most kernels):
grid = (n_rows,)
# If n_rows = 100 and SM count = 108, 8 SMs are idle in the last wave

# 2D grid for better utilization:
grid = (triton.cdiv(n_rows, BLOCK_ROW), triton.cdiv(n_cols, BLOCK_COL))

@triton.jit
def kernel_2d(
    ...,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    row_start = row_block * BLOCK_ROW
    col_start = col_block * BLOCK_COL
    # Process a tile of [BLOCK_ROW, BLOCK_COL]
```

A 2D grid is most useful when the problem is large in both dimensions and each tile does enough work to amortize launch overhead.

**Expected impact**: 5-15% for kernels with suboptimal SM utilization. More impactful when `n_rows` is small relative to SM count.

**Risks**: 2D tiling for reduction kernels requires careful handling of partial results and synchronization. More complex code. Not beneficial when `n_cols` fits in a single block.

**Kernel tiers**: Tier 2 (reductions with large feature dimensions), Tier 3 (large fused operations).

### 3.6 Loop Unrolling Hints

**When to apply**: Kernel has a hot loop with a known iteration count or small trip count. Triton's compiler unrolls some loops automatically, but explicit hints can help.

**How to implement**:

```python
# Triton unrolls loops with constexpr bounds automatically.
# For dynamic bounds, you can help by:

# Strategy 1: Make the loop bound a constexpr when possible
# If n_cols is known at compile time (e.g., passed as constexpr):
for i in range(0, N_COLS, BLOCK_SIZE):  # N_COLS: tl.constexpr -> unrolled
    ...

# Strategy 2: Use tl.static_range for explicit unrolling
for i in tl.static_range(4):  # always unrolled
    X_block = tl.load(X_ptr + i * BLOCK_SIZE + offsets)
    ...

# Strategy 3: Break a large loop into a known-count outer loop + remainder
CHUNKS: tl.constexpr = n_cols // BLOCK_SIZE  # constexpr division
for i in tl.static_range(CHUNKS):
    ...
# Handle remainder separately
```

**Expected impact**: 5-10% for tight loops with few iterations. Diminishing returns for large trip counts.

**Risks**: Excessive unrolling increases instruction cache pressure and compile time. Can also increase register pressure if the loop body is large.

**Kernel tiers**: Tier 2 (reduction loops), Tier 3 (multi-pass loops).

---

## 4. Architecture-Specific Optimizations

Apply these on top of the best general-purpose variant when targeting a specific GPU architecture.

### 4.1 Ampere (SM 8.x: A100, A10, A30)

**Async Copy (cp.async)**:
- Triton uses `num_stages` to control software pipelining, which maps to `cp.async` on Ampere.
- `num_stages=2` is the sweet spot for most Ampere kernels. Going higher increases shared memory usage without benefit.
- Triton handles `cp.async` automatically when `num_stages > 1`.

**L2 Cache Persistence**:
- A100 has 40MB L2 cache. Tensors smaller than this can benefit from persistence hints.
- Use `eviction_policy="evict_last"` on reused data (weights, biases).
- For cross-entropy, the target tensor and weight tensor are good candidates.

**Optimal Warp Counts**:
- Ampere excels with 4-16 warps per SM. 32 warps often increases scheduling overhead.
- For memory-bound kernels: fewer warps (4-8) with larger BLOCK_SIZE.
- For compute-bound kernels: more warps (8-16) to hide latency.

**Implementation**:
```python
# Ampere-tuned config
triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
```

### 4.2 Hopper (SM 9.0: H100, H200)

**TMA (Tensor Memory Accelerator)**:
- Hopper's TMA offloads address computation and data movement to a dedicated unit.
- Triton >= 3.0 exposes TMA via `tl.make_block_ptr` and experimental APIs.
- TMA is most beneficial for 2D tiled access patterns (matmul-like kernels).

```python
# TMA-friendly block pointer (Triton >= 3.0)
block_ptr = tl.make_block_ptr(
    base=X_ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(row_start, col_start),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0),  # row-major
)
X_block = tl.load(block_ptr, boundary_check=[0, 1])
```

**Higher num_stages**:
- Hopper's async pipeline supports deeper prefetching. `num_stages=3-5` often outperforms 2.
- The deeper pipeline hides memory latency more effectively on H100's higher-bandwidth HBM3.

**WGMMA (Warp Group Matrix Multiply Accumulate)**:
- H100 tensor cores are organized in warp groups (4 warps = 128 threads).
- `num_warps` should be a multiple of 4 for optimal tensor core usage.
- Triton maps `tl.dot` to WGMMA automatically when conditions are met.

**Cluster Launch**:
- Hopper supports thread block clusters (groups of SMs that can share SRAM).
- Currently experimental in Triton. Potentially useful for large reductions.

**Implementation**:
```python
# Hopper-tuned config
triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=5),
triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),
```

### 4.3 Blackwell (SM 10.0: B200, GB200)

**5th-Gen Tensor Cores**:
- Higher throughput for FP8, FP16, BF16, and TF32 operations.
- Block sizes aligned to 16 remain optimal. FP8 support enables even higher throughput.
- Triton support is evolving; check Triton release notes for Blackwell-specific features.

**Larger Shared Memory**:
- Blackwell supports up to 256KB shared memory per SM (vs 228KB on Hopper).
- Larger BLOCK_SIZE values become viable without spilling to registers or HBM.
- Can enable larger `BLOCK_ROW` in block-row mode.

**Higher Memory Bandwidth**:
- HBM3E provides more bandwidth than HBM3. Memory-bound kernels see direct benefit.
- The compute-to-memory ratio shifts, making some previously memory-bound kernels balanced or compute-bound.

**Implementation**:
```python
# Blackwell-tuned config (aggressive block sizes)
triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=5),
triton.Config({"BLOCK_SIZE": 16384}, num_warps=32, num_stages=5),
```

**Note**: Blackwell-specific Triton features are still maturing. Focus on general optimizations first, then add Blackwell-specific configs as Triton support stabilizes.

---

## 5. Balanced Optimizations

These techniques help both memory-bound and compute-bound kernels.

### 5.1 Kernel Fusion Opportunities

**When to apply**: Two or more kernels execute sequentially on the same data, writing intermediate results to HBM between them. The fused version keeps intermediates in registers/SRAM.

**How to implement**:

Liger already excels at fusion (cross_entropy fuses softmax+loss+gradient; fused_linear_cross_entropy fuses matmul+CE). Look for remaining opportunities:

```python
# Pattern: operation followed by elementwise scaling
# BEFORE: two kernel launches
y = rms_norm_kernel(x, w)        # writes y to HBM
z = elementwise_mul_kernel(y, s)  # reads y from HBM, writes z

# AFTER: fused kernel
@triton.jit
def fused_rms_norm_mul(X_ptr, W_ptr, S_ptr, Z_ptr, ...):
    # RMS norm computation
    y = x_row * rstd * (offset + w_row)
    # Elementwise mul (no HBM round-trip)
    z = y * s
    tl.store(Z_ptr + offsets, z)
```

Common fusion candidates in Liger:
- Norm + activation (e.g., rms_norm + swiglu in some architectures)
- Loss + metric computation (already done: cross_entropy + accuracy)
- Dropout + residual add + norm

**Expected impact**: 20-50% from eliminating HBM round-trips between fused operations.

**Risks**: Fused kernels are harder to maintain, test, and debug. Register pressure increases. The fused kernel may have worse autotuning characteristics. Only fuse when the intermediate tensor is large relative to cache.

**Kernel tiers**: Creates new Tier 3 kernels from combinations of Tier 1/2 kernels. Existing Tier 3 kernels may have remaining fusion opportunities.

### 5.2 Better Grid Sizing

**When to apply**: The number of programs (grid size) is much smaller or much larger than the SM count, leading to load imbalance or insufficient parallelism.

**How to implement**:

```python
sm_count = torch.cuda.get_device_properties(device).multi_processor_count

# For backward passes using SM-based parallelism (like rms_norm backward):
grid = (sm_count,)
rows_per_program = math.ceil(n_rows / sm_count)

# For forward passes: ensure grid is a small multiple of sm_count
# If n_rows >> sm_count, this is automatic
# If n_rows < sm_count, consider block-row mode or 2D grid to increase occupancy

# For persistent kernels (advanced):
# Launch exactly sm_count programs, each processes multiple work items
grid = (sm_count,)
for work_item in range(pid, total_work, NUM_SMS):
    process(work_item)
```

The rms_norm backward kernel already uses this SM-based grid pattern. Look for other backward kernels that use `(n_rows,)` grids but could benefit from SM-based parallelism.

**Expected impact**: 5-20% from better load balancing and reduced tail effects.

**Risks**: SM-based grids require careful work distribution and can introduce non-determinism if work items have unequal cost.

**Kernel tiers**: All tiers. Most impactful for Tier 2 backward passes.

### 5.3 Reduce Kernel Launch Overhead

**When to apply**: Kernel is called many times with very short execution time (< 10 microseconds). Launch overhead dominates.

**How to implement**:

```python
# Strategy 1: Batch multiple operations into one kernel
# Instead of launching N small kernels, launch one kernel with N work items

# Strategy 2: Use CUDA graphs (from PyTorch side, not inside Triton)
# Captures a sequence of kernel launches and replays them with minimal overhead
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = liger_kernel(input)
# Replay: graph.replay()

# Strategy 3: Increase work per kernel (larger BLOCK_SIZE, block-row mode)
# Already covered in autotuning and block-row mode sections
```

**Expected impact**: 5-15% for very short kernels. Negligible for kernels that run > 100 microseconds.

**Risks**: CUDA graphs have limitations (fixed input shapes, no dynamic control flow). Batching work items increases kernel complexity.

**Kernel tiers**: Tier 1 primarily (short element-wise kernels). Tier 2/3 kernels usually run long enough that launch overhead is negligible.

### 5.4 Mixed Precision Strategies

**When to apply**: Kernel uses FP32 throughout but only needs FP32 for specific operations (reductions, accumulations). Or kernel uses FP16/BF16 throughout but suffers from numerical issues.

**How to implement**:

Liger already uses mixed precision extensively (e.g., rms_norm with casting_mode). Look for additional opportunities:

```python
# Pattern 1: Compute in fp32 only where needed (Llama-style)
X_row = tl.load(...)              # fp16/bf16
X_fp32 = X_row.to(tl.float32)     # upcast for reduction
mean_sq = tl.sum(X_fp32 * X_fp32) / n_cols  # fp32 reduction
rstd = rsqrt(mean_sq + eps)        # fp32
result = (X_row * rstd.to(X_row.dtype)) * W  # back to original dtype

# Pattern 2: Accumulate in fp32, store in original dtype
acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
for ...:
    acc += (dY_row * X_row).to(tl.float32)  # accumulate in fp32
tl.store(dW_ptr, acc.to(original_dtype))     # store in original dtype

# Pattern 3: Use BF16 instead of FP16 for better dynamic range
# BF16 has same exponent bits as FP32, so less overflow/underflow risk
# But lower mantissa precision than FP16
```

**Expected impact**: 5-20% from reducing unnecessary FP32 computation. Can also improve accuracy if precision is currently insufficient.

**Risks**: Under-casting causes numerical errors. Over-casting wastes compute. The casting_mode pattern in Liger (llama vs gemma vs none) shows that different models need different precision strategies. Always validate against reference implementation.

**Kernel tiers**: All tiers. Tier 2 reductions are most sensitive to precision. Tier 1 kernels with transcendentals (sigmoid, exp) benefit from targeted FP32 upcast.

---

## Quick Reference: Technique Selection by Symptom

| Symptom | Likely Bottleneck | Try These Techniques |
|---------|------------------|---------------------|
| NCU memory throughput > 60% | Memory-bound | 2.1, 2.2, 2.3, 2.5 |
| NCU compute throughput > 60% | Compute-bound | 3.1, 3.2, 3.3, 3.4 |
| Low SM occupancy | Register pressure or grid size | 3.4, 5.2, 1 (num_warps) |
| Small n_cols, large n_rows | Under-utilized programs | 2.5, 5.2, 1 (BLOCK_SIZE) |
| Large n_cols, small n_rows | Poor SM utilization | 3.5, 5.2 |
| Multiple sequential kernel launches | Fusion opportunity | 5.1 |
| Forward fast, backward slow | Backward-specific issue | 2.3, 2.4, 5.2 (SM-based grid) |
| Performance varies wildly by shape | Bad fixed config | 1 (autotune with key) |
| High register count in NCU | Register pressure | 3.4, 1 (smaller BLOCK_SIZE) |
| Poor L2 cache hit rate | Cache misuse | 2.2, 2.4 |

## Quick Reference: Technique Applicability by Kernel Tier

| Technique | Tier 1 (element-wise) | Tier 2 (reduction) | Tier 3 (fused/complex) |
|-----------|----------------------|--------------------|-----------------------|
| 1. Autotuning | High | High | High |
| 2.1 Coalescing | Low (usually ok) | Medium | Medium |
| 2.2 Cache modifiers | Low | Medium | High |
| 2.3 Reduce traffic | Medium | Medium | High |
| 2.4 Data reuse | Low | Medium | Medium |
| 2.5 Block-row mode | High | High | Low |
| 2.6 Vectorized loads | Low | Low | Low |
| 3.1 Algorithmic | Low | High | High |
| 3.2 Reduce redundancy | Low | Medium | High |
| 3.3 Tensor cores | N/A | N/A | High |
| 3.4 Register pressure | Low | Medium | High |
| 3.5 2D grid tiling | Low | Medium | Medium |
| 3.6 Loop unrolling | Low | Medium | Medium |
| 4.x Arch-specific | Medium | Medium | Medium |
| 5.1 Kernel fusion | Medium (fuse into Tier 3) | Medium | High |
| 5.2 Grid sizing | Low | High | Medium |
| 5.3 Launch overhead | High | Low | Low |
| 5.4 Mixed precision | Medium | High | High |
