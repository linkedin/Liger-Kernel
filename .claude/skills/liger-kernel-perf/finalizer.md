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

### Step 5: Generate Comparison Plots

Generate 3-way comparison plots showing the original Liger kernel, the optimized kernel, and the HuggingFace/PyTorch baseline. This gives the user a visual demonstration of the improvement.

#### Step 5a: Benchmark Both Old and New Kernels for Comparison

The goal is to have 3 providers in `all_benchmark_data.csv` for plotting: the original "liger" kernel (renamed to "liger_original" for the comparison), the optimized kernel (as the new "liger"), and the huggingface/torch baseline.

**Procedure:**

1. **Back up the current CSV** so we can restore if needed:
   ```bash
   cp benchmark/data/all_benchmark_data.csv benchmark/data/all_benchmark_data.csv.bak
   ```

2. **Benchmark the original kernel as "liger_original"**: Temporarily restore the original kernel and run benchmarks. To get a custom provider name, temporarily edit the benchmark script's `kernel_providers` list to use `"liger_original"` instead of `"liger"`, run, then revert the benchmark script:
   ```bash
   # Swap in original kernel
   cp optimization/{kernel}/original_{kernel}.py src/liger_kernel/ops/{kernel}.py
   
   # Edit benchmark script: replace "liger" with "liger_original" in kernel_providers
   sed -i 's/"liger"/"liger_original"/g' benchmark/scripts/benchmark_{kernel}.py
   
   # Run benchmarks (appends "liger_original" rows to CSV)
   cd benchmark/scripts && python benchmark_{kernel}.py
   
   # Revert the benchmark script change
   git checkout benchmark/scripts/benchmark_{kernel}.py
   ```

3. **Benchmark the optimized kernel as "liger"**: Re-apply the optimized kernel and run the standard benchmarks. This overwrites the old "liger" rows with the new optimized results:
   ```bash
   # Swap in optimized kernel
   cp optimization/{kernel}/{kernel}_v{N}.py src/liger_kernel/ops/{kernel}.py
   
   # Run benchmarks (overwrites "liger" rows with optimized kernel results)
   cd benchmark/scripts && python benchmark_{kernel}.py --overwrite
   ```

After this, `all_benchmark_data.csv` contains:
- `"liger"` — the **optimized** kernel (this is the permanent update)
- `"liger_original"` — the old kernel (for comparison plots)
- `"huggingface"` / `"torch"` — the baseline

The "liger" provider in the CSV now reflects the optimized kernel going forward. This is the desired end state — the benchmark data should always represent the current production kernel.

#### Step 5b: Generate Plots

Generate speed and memory comparison plots:

```bash
# Speed plots for all modes (forward, backward, full)
python benchmark/benchmarks_visualizer.py \
  --kernel-name {kernel} \
  --metric-name speed \
  --overwrite

# Memory plot
python benchmark/benchmarks_visualizer.py \
  --kernel-name {kernel} \
  --metric-name memory \
  --overwrite
```

Plots are saved to `benchmark/visualizations/`. Copy them to the optimization workspace:

```bash
cp benchmark/visualizations/{kernel}_speed_*.png optimization/{kernel}/
cp benchmark/visualizations/{kernel}_memory_*.png optimization/{kernel}/
```

#### Step 5c: Generate Additional Custom Plots (Optional)

If the optimization shows interesting size-dependent behavior (e.g., bigger gains at larger sizes), generate targeted plots highlighting this. Use matplotlib directly if the standard visualizer does not cover the comparison you want to show.

Save any additional plots to `optimization/{kernel}/`.

### Step 6: Generate the Final Optimization Report

Write the report to `optimization/{kernel}/report.md`. The report is the permanent record of the entire optimization effort.

The report must include these sections:

1. **Summary**: kernel name, GPU, goal, variants tried, best variant, improvement %, and whether the optimization was applied (`Yes` / `No — reverted due to test failures`)
2. **Baseline Profile**: bottleneck classification with evidence, NCU highlights if available
3. **Variant Comparison**: include the table from `optimization/{kernel}/comparison.md` (produced by the Optimizer)
4. **Winning Changes**: what was changed, why it worked, and a key code diff. Run `diff optimization/{kernel}/original_{kernel}.py src/liger_kernel/ops/{kernel}.py` — do NOT write the diff from memory.
5. **Rejected Approaches**: for each rejected variant, what was tried and why it failed
6. **Comparison Plots**: reference the plots generated in Step 5 with file paths
7. **Correctness Verification**: smoke test results, full test suite results, checkstyle results
8. **Recommendations**: unexplored optimization opportunities, suggested follow-ups
9. **Reproduction** section:
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
- [ ] Comparison plots are generated and referenced in the report

### Step 7: Create Pull Request

Create a PR with the kernel code changes and the updated benchmark data. Do NOT include plots or optimization workspace files in the PR.

#### Step 7a: Stage Only Relevant Files

```bash
git add src/liger_kernel/ops/{kernel}.py
git add benchmark/data/all_benchmark_data.csv
```

Do NOT add:
- `optimization/` directory (workspace files, notes, plots)
- `benchmark/visualizations/` (generated plots -- these are for local review only)

#### Step 7b: Create a Descriptive Commit

```bash
git commit -m "$(cat <<'EOF'
[Perf] Optimize {kernel} kernel: {headline improvement}

{2-3 sentence summary of the optimization technique and its impact.}

Key changes:
- {Change 1}: {what and why}
- {Change 2}: {what and why}

Benchmark results ({gpu_name}):
- Speed (forward): {delta}% {faster/slower}
- Speed (backward): {delta}% {faster/slower}
- Speed (full): {delta}% {faster/slower}
- Memory: {delta}% {reduction/increase}

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

#### Step 7c: Push and Create PR

```bash
git push -u origin HEAD
```

Create the PR using `gh pr create`. The PR body must clearly explain:

1. **What was optimized** -- which kernel, which pass, what bottleneck
2. **What changes were made** -- specific code changes with rationale for each
3. **How they help** -- tie back to the bottleneck diagnosis (e.g., "reduced register pressure from 115 to 95 registers/thread, increasing occupancy from 12.5% to 25%")
4. **Benchmark results** -- before/after table with speed and memory numbers at representative sizes
5. **Correctness** -- confirm all tests pass

Use this PR template:

```bash
gh pr create --title "[Perf] Optimize {kernel}: {headline}" --body "$(cat <<'EOF'
## Summary

{2-3 sentences: what was optimized, the key technique, and the headline result.}

## Bottleneck Diagnosis

- **Kernel**: `{kernel}` ({tier} — {tier_description})
- **GPU**: {gpu_name} ({architecture})
- **Classification**: {memory-bound / compute-bound / balanced}
- **Root cause**: {specific bottleneck explanation}

## Changes

{For each change, explain WHAT was changed and WHY it helps:}

### 1. {Change title}
{Description of the code change and its performance rationale.}

### 2. {Change title}
{Description of the code change and its performance rationale.}

## Benchmark Results

Tested on {gpu_name}. Values are median ms (speed) or MB (memory).

### Speed (ms, lower is better)

| x_value | Mode | Before | After | Improvement |
|---------|------|--------|-------|-------------|
| {x1} | forward | {before} | {after} | {delta}% |
| {x1} | backward | {before} | {after} | {delta}% |
| {x1} | full | {before} | {after} | {delta}% |
| ... | ... | ... | ... | ... |

### Memory (MB, lower is better)

| x_value | Before | After | Improvement |
|---------|--------|-------|-------------|
| {x1} | {before} | {after} | {delta}% |
| ... | ... | ... | ... |

## Correctness

- Full test suite: **PASSED** (`python -m pytest test/transformers/test_{kernel}.py -xvs`)
- Checkstyle: **PASSED** (`make checkstyle`)

## Test Plan

- [x] All existing unit tests pass
- [x] Benchmarks show improvement across all input sizes
- [x] No regression on non-target metrics (speed/memory balance maintained)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Important**: Do NOT include plots as image attachments in the PR. Plots are for local review only and live in the optimization workspace.

### Step 8: Present the Before/After Summary

Include the plot file paths in the summary so the user can view them.

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
