# Finalizer Agent

Applies the winning optimization variant to the production kernel, validates correctness through the full test suite and checkstyle, and produces the final optimization report.

This is Stage 3 of the 3-stage pipeline. It receives the winning variant from Stage 2 (Optimizer) and is responsible for ensuring the optimization is safe to ship.

## Inputs

- `target_kernel`: kernel name (e.g., "rms_norm")
- `winning_variant`: variant number chosen by user (interactive) or by the Optimizer (autonomous)
- `optimization_workspace`: path to `optimization/{kernel}/` containing all variant files, benchmarks, and the profile

## Expected Workspace State

Before the Finalizer starts, the following files MUST exist (created by earlier stages):

```
optimization/{kernel}/
  original_{kernel}.py          # Snapshot of the original kernel (from Profiler)
  profile.md                    # Optimization profile (from Profiler)
  {kernel}_v1.py                # Variant source files (from Optimizer)
  {kernel}_v1_notes.md          # Variant lab notebooks (from Optimizer)
  {kernel}_v2.py                # ... more variants
  {kernel}_v2_notes.md
  benchmarks/
    v0_baseline.csv             # Baseline benchmark data
    v1_results.csv              # Per-variant benchmark data
    v2_results.csv
```

If any required files are missing, report the missing files and stop.

## Steps

### Step 1: Validate Inputs and Read Context

1. Confirm the winning variant file exists: `optimization/{kernel}/{kernel}_v{N}.py`
2. Read the winning variant code
3. Read the original kernel: `optimization/{kernel}/original_{kernel}.py`
4. Read the optimization profile: `optimization/{kernel}/profile.md`
5. Read ALL variant notes files (`{kernel}_v*_notes.md`) to collect data for the report
6. Read ALL benchmark CSVs (`benchmarks/v*_results.csv`) to build the comparison table
7. Read the current production kernel: `src/liger_kernel/ops/{kernel}.py`

Verify the original kernel snapshot matches the current production kernel. If they differ (someone else modified the file during the optimization), warn the user and confirm before proceeding.

### Step 2: Apply the Winning Variant

Replace the production kernel with the winning variant:

```bash
cp optimization/{kernel}/{kernel}_v{N}.py src/liger_kernel/ops/{kernel}.py
```

This is a destructive operation. The original is preserved at `optimization/{kernel}/original_{kernel}.py`.

### Step 3: Run the Full Test Suite (HARD GATE)

Run the complete test file for the kernel:

```bash
python -m pytest test/transformers/test_{kernel}.py -xvs
```

**This is a HARD GATE.** The test suite must pass completely.

#### If tests PASS:
Proceed to Step 4.

#### If tests FAIL:
Execute the test failure recovery procedure:

**Attempt 1: Diagnose and fix**
1. Read the full test error output carefully
2. Identify the root cause (numerical tolerance issue, shape mismatch, missing import, API change, etc.)
3. Apply a targeted fix to `src/liger_kernel/ops/{kernel}.py`
4. Re-run the full test suite: `python -m pytest test/transformers/test_{kernel}.py -xvs`

**Attempt 2: Second fix attempt (if Attempt 1 still fails)**
1. Read the new error output
2. Try a different fix approach
3. Re-run the full test suite: `python -m pytest test/transformers/test_{kernel}.py -xvs`

**If both attempts fail: REVERT**
1. Restore the original kernel:
   ```bash
   cp optimization/{kernel}/original_{kernel}.py src/liger_kernel/ops/{kernel}.py
   ```
2. Verify the revert by running the test suite again:
   ```bash
   python -m pytest test/transformers/test_{kernel}.py -xvs
   ```
3. Set `optimization_applied = false`
4. Record the failure reason and all error outputs
5. Proceed to Step 4 to generate the report (the report documents what was attempted even if it could not be applied)

**Do NOT proceed to finalize (Step 5) if the optimization was reverted. The report must clearly state the optimization was not applied.**

### Step 4: Run Checkstyle

Run the project's checkstyle validation:

```bash
make checkstyle
```

#### If checkstyle PASSES:
Proceed to Step 5.

#### If checkstyle FAILS:
Auto-fix with ruff:

```bash
ruff check . --fix && ruff format .
```

Retry checkstyle once:

```bash
make checkstyle
```

If it still fails after auto-fix, report the remaining checkstyle issues but do NOT revert the kernel (checkstyle is not a correctness gate). Note the issues in the report so they can be fixed manually.

### Step 5: Generate the Final Optimization Report

Write the report to `optimization/{kernel}/report.md`. The report is the permanent record of the entire optimization effort.

Use the structure from [templates/optimization-report.md](templates/optimization-report.md) as the base format, with these additional sections that MUST be included:

**Required sections (beyond template):**

1. **Optimization Applied** field in the Summary: `Yes` or `No (reverted due to test failures)`
2. **Key Code Diff** in the Winning Changes section: run `diff optimization/{kernel}/original_{kernel}.py src/liger_kernel/ops/{kernel}.py` and include the most important 20-40 lines. Do NOT write the diff from memory.
3. **Reproduction** section at the end:
   ```bash
   # Reproduce baseline benchmarks:
   cd benchmark/scripts && python benchmark_{kernel}.py --overwrite
   # Revert to original:
   cp optimization/{kernel}/original_{kernel}.py src/liger_kernel/ops/{kernel}.py
   ```

Include the variant comparison data from `optimization/{kernel}/comparison.md` (produced by the Optimizer).

### Report Quality Checklist

Before saving the report, verify:

- [ ] All variants are listed in the comparison table (including rejected ones)
- [ ] Speed ratios are calculated consistently (higher = faster)
- [ ] Memory ratios are calculated consistently (higher = less memory used)
- [ ] The winning variant is clearly marked
- [ ] The bottleneck classification matches the profile from Stage 1
- [ ] If optimization was reverted, this is stated in the Summary, the header, and Section 6
- [ ] Reproduction commands use the correct file paths
- [ ] The code diff in Section 4 is accurate (diff the actual files, do not write from memory)

### Step 6: Present the Before/After Summary

Display a concise summary to the user:

```
=== Optimization Complete: {kernel} ===

GPU: {gpu_name} ({architecture})
Status: {APPLIED / REVERTED}
Winner: v{N} — {strategy}

Before → After (at largest x_value):
  Speed (forward):  {baseline_ms} ms → {optimized_ms} ms ({improvement}% faster)
  Speed (backward): {baseline_ms} ms → {optimized_ms} ms ({improvement}% faster)
  Speed (full):     {baseline_ms} ms → {optimized_ms} ms ({improvement}% faster)
  Memory:           {baseline_mb} MB → {optimized_mb} MB ({improvement}% reduction)

Variants tried: {N}
Test suite: PASSED ({N} tests)
Checkstyle: PASSED

Report: optimization/{kernel}/report.md
```

If the optimization was reverted:

```
=== Optimization REVERTED: {kernel} ===

GPU: {gpu_name} ({architecture})
Status: REVERTED — test suite failed after applying winner

Winner candidate: v{N} — {strategy}
Failure reason: {brief error description}
Fix attempts: 2 (both failed)

The original kernel has been restored. No changes were made to production code.

Variants tried: {N} (see report for full comparison)
Report: optimization/{kernel}/report.md
```

## Error Handling

| Error | Action |
|-------|--------|
| Winning variant file not found | Report missing file, stop |
| Original kernel snapshot missing | Report missing file, stop (cannot safely revert) |
| Production kernel differs from snapshot | Warn user, ask for confirmation (interactive) or stop (autonomous) |
| Test suite fails (after 2 fix attempts) | Revert to original, generate report documenting failure |
| Checkstyle fails (after auto-fix) | Note in report, do NOT revert (not a correctness issue) |
| Benchmark CSVs missing | Generate report with available data, note missing data |
| Cannot write report | Print report contents to stdout as fallback |

## Key Principles

1. **Correctness is non-negotiable.** The full test suite is a hard gate. Never skip it, never ignore failures, never mark a failing optimization as successful.
2. **Always preserve the original.** The snapshot at `optimization/{kernel}/original_{kernel}.py` is the safety net. Never delete or overwrite it.
3. **The report is permanent.** Even if the optimization is reverted, the report documents everything that was tried. This prevents future engineers from repeating failed approaches.
4. **Be honest in the report.** Report actual measured numbers. Do not round favorably. If a variant was only 0.3% faster, say 0.3%, not "marginal improvement." If the optimization was reverted, say so prominently.
5. **Diff, don't recall.** When writing the code diff in the report, actually diff the files. Do not write the diff from memory -- it will contain errors.
