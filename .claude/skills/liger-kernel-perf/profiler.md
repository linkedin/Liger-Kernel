# Profiler Agent

Profiles an existing Liger kernel to diagnose performance bottlenecks and produce an optimization strategy.

## Inputs

- `target_kernel`: kernel name (e.g., "rms_norm")
- `optimization_goal`: speed / memory / balanced
- `scope`: general / forward / backward / specific input regime
- `target_gpu`: Ampere / Hopper / Blackwell / auto-detect

## Steps

### Step 1: Set Up Workspace

1. Create directory `optimization/{kernel}/` at the repo root
2. Copy `src/liger_kernel/ops/{kernel}.py` to `optimization/{kernel}/original_{kernel}.py`
3. Create `optimization/{kernel}/benchmarks/` subdirectory

### Step 2: Detect GPU Architecture

Run:
```bash
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'GPU: {props.name}, SM: {props.major}.{props.minor}, SMs: {props.multi_processor_count}, Mem: {props.total_mem // (1024**3)}GB')"
```

Classify architecture:
- SM 8.x → Ampere (A100, A10, etc.)
- SM 9.0 → Hopper (H100, H200, etc.)
- SM 10.0 → Blackwell (B200, etc.)

If user specified a target_gpu, note the mismatch (if any) but proceed with the actual hardware.

### Step 3: Run Baseline Benchmarks

Run the existing benchmark script:
```bash
cd benchmark/scripts && python benchmark_{kernel}.py --overwrite
```

Record the baseline results. Copy the relevant rows from `benchmark/data/all_benchmark_data.csv` to `optimization/{kernel}/benchmarks/v0_baseline.csv`.

Parse and summarize the baseline numbers:
- Speed: forward, backward, full (median ms at each x_value)
- Memory: full (median MB at each x_value)
- Identify the slowest operation mode and largest x_values

### Step 4: Check for NCU Availability

```bash
which ncu 2>/dev/null && echo "NCU_AVAILABLE" || echo "NCU_NOT_AVAILABLE"
```

If available, offer to run profiling (interactive mode) or run it (autonomous mode).

To determine the correct import and invocation, read the kernel's `torch.autograd.Function` class from `src/liger_kernel/ops/{kernel}.py` and construct a minimal reproduction script. For example, for `rms_norm`:

```bash
ncu --set full --target-processes all -o optimization/{kernel}/ncu_profile \
    python -c "
import torch
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)
w = torch.ones(4096, device='cuda', dtype=torch.bfloat16)
y = LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, 'llama', False)
y.sum().backward()
"
```

Adapt the import path, function arguments, and tensor shapes for the target kernel.

Extract key metrics from NCU output:
- **SM Occupancy** (%)
- **Compute Throughput** (% of peak)
- **Memory Throughput** (% of peak)
- **L1/L2 Cache Hit Rates**
- **Warp Stall Reasons** (top 3)

### Step 5: Analyze Kernel Code

Read `src/liger_kernel/ops/{kernel}.py` thoroughly. Extract:

1. **Tier classification**:
   - Tier 1 (element-wise): No reductions. One row per program.
   - Tier 2 (reduction): Cross-column reductions. May need SM-based parallelism.
   - Tier 3 (fused/complex): Multi-pass algorithms, gradient-in-forward tricks.

2. **Current configuration**:
   - How BLOCK_SIZE is set (calculate_settings? hardcoded? autotune?)
   - num_warps value
   - num_stages value (if specified)
   - Grid dimensions (1D, 2D, block-row mode?)

3. **Memory access patterns**:
   - Stride-based or contiguous?
   - Cache modifiers used?
   - In-place operations?
   - What is saved_for_backward vs recomputed?

4. **Compute patterns**:
   - Precision-sensitive operations (sigmoid, rsqrt, exp, log)?
   - Casting modes?
   - Online/chunked algorithms?
   - constexpr parameters?

5. **Existing optimization opportunities**:
   - Any TODO/FIXME comments about performance?
   - Missing block-row mode variant?
   - Suboptimal BLOCK_SIZE heuristic?
   - Unnecessary memory traffic?

### Step 6: Classify Bottleneck

Use NCU data (if available) or heuristics:

**Memory-bound indicators**:
- NCU: Memory throughput > 60% of peak, compute throughput < 40%
- Heuristic: Simple element-wise ops, large tensors, few arithmetic operations per load
- Heuristic: Tier 1 kernels are almost always memory-bound

**Compute-bound indicators**:
- NCU: Compute throughput > 60% of peak, memory throughput < 40%
- Heuristic: Complex math per element (exp, log, sigmoid chains), reductions, many operations per load
- Heuristic: Tier 3 kernels with heavy computation

**Balanced (both bottlenecks)**:
- NCU: Both throughputs 30-60%
- Heuristic: Tier 2 kernels often fall here

### Step 7: Produce Optimization Profile

Write `optimization/{kernel}/profile.md` using the [optimization-profile template](templates/optimization-profile.md).

The profile must include:
- GPU architecture and properties
- Baseline benchmark summary (table)
- NCU summary (if available)
- Kernel tier classification
- Bottleneck classification with evidence
- Current configuration analysis
- Recommended strategy order (from [optimization-strategies.md](optimization-strategies.md))
- Specific opportunities identified in the code

### Step 8: Present to User (Interactive Mode)

Show:
1. GPU detected: {name} ({architecture})
2. Baseline performance summary table
3. Bottleneck classification: {memory-bound / compute-bound / balanced}
4. Evidence for classification
5. Proposed optimization strategy order
6. Estimated iteration count

Wait for user confirmation before proceeding to Stage 2.

In autonomous mode, skip this step and proceed directly.
