# Validator Agent

Validates the generated kernel through checkstyle, testing, benchmarking, and optional profiling.

## Validation Flow

### Step 1: Checkstyle (Fast Gate)

```bash
make checkstyle
```

If it fails:
```bash
ruff check . --fix && ruff format .
make checkstyle
```

If it still fails after auto-fix, identify and fix the remaining issues manually, then re-run.

### Step 2: Unit Tests (Hard Gate)

```bash
python -m pytest test/transformers/test_{kernel}.py -xvs
```

If tests fail:
1. Read the error output carefully
2. Identify root cause: kernel logic bug, shape mismatch, dtype issue, or tolerance too tight
3. Fix the issue in the relevant file(s)
4. Re-run `make checkstyle` (in case fix introduced style issues)
5. Re-run the failing test

**Retry up to 3 times total.** If tests still fail after 3 attempts:
- **STOP IMMEDIATELY. Do NOT proceed to benchmarks.**
- Report to the user:
  - Which tests are failing and the exact error messages
  - What fixes were attempted
  - The current state of the code
  - Suggest what might be wrong and ask for guidance

### Step 3: Benchmarks

Run the benchmark script:
```bash
cd benchmark/scripts && python benchmark_{kernel}.py
```

This produces speed and memory measurements for forward, backward, and full modes, comparing Liger vs PyTorch baseline.

### Step 4: Generate Plots

```bash
python benchmark/benchmarks_visualizer.py
```

Plots are saved to `benchmark/visualizations/`. If the visualizer fails or is not available, this step is non-blocking — report the raw numbers instead.

### Step 5: Optional ncu Profiling

Only if the user explicitly requests it. Requires NVIDIA Nsight Compute (`ncu`) to be installed.

Profile both the Liger kernel and PyTorch reference:
```bash
ncu --set full --target-processes all -o liger_profile python -c "
import torch
from liger_kernel.ops.{kernel} import Liger{Kernel}Function
x = torch.randn(..., device='cuda', dtype=torch.float32, requires_grad=True)
out = Liger{Kernel}Function.apply(x, ...)
out.backward(torch.randn_like(out))
"
```

Report key metrics: SM occupancy, memory throughput, compute throughput.

If ncu is not available, report this and skip.

### Step 6: Report Results

Present a clear summary to the user:

```
## Results

**Checkstyle:** PASS
**Unit Tests:** PASS (X tests passed)

**Speed Benchmarks:**
| Mode     | Liger (ms) | PyTorch (ms) | Speedup |
|----------|-----------|-------------|---------|
| Forward  | ...       | ...         | ...x    |
| Backward | ...       | ...         | ...x    |
| Full     | ...       | ...         | ...x    |

**Memory Benchmarks:**
| Mode     | Liger (MB) | PyTorch (MB) | Savings |
|----------|-----------|-------------|---------|
| Forward  | ...       | ...         | ...%    |
| Backward | ...       | ...         | ...%    |
| Full     | ...       | ...         | ...%    |

**Assessment:** [Which dimension improved — speed, memory, or both.
Flag if either metric is catastrophically worse.]

**Plots:** benchmark/visualizations/

**ncu Profiling:** [Results if requested, or "Not requested"]
```
