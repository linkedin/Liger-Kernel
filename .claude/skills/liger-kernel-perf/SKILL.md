---
name: liger-kernel-perf
description: "Optimizes the performance of existing Liger Kernel Triton kernels. Profiles kernels, diagnoses bottlenecks (memory-bound vs compute-bound), generates multiple optimization variants with benchmarking, and applies the best variant while maintaining correctness. Supports GPU architecture-specific optimization (Ampere, Hopper, Blackwell). Use when a user asks to optimize, speed up, tune, profile, or reduce memory of an existing Liger kernel."
---

# Liger Kernel Perf

Optimizes existing Liger Kernel Triton kernels through a 3-stage pipeline: Profile, Optimize, Finalize. Supports interactive mode (human checkpoints between stages) and autonomous mode (runs end-to-end). NVIDIA GPUs only.

## Mode Detection

- **Interactive mode** (default): Human checkpoints between each stage
- **Autonomous mode**: User says "just optimize it", "run without asking me", "optimize autonomously" → all stages run end-to-end, user sees only the final report

## Input Parsing

Extract from the user's request:

| Field | Description | Default |
|-------|-------------|---------|
| `target_kernel` | Which kernel to optimize (e.g., "rms_norm", "cross_entropy") | **Required** |
| `optimization_goal` | speed / memory / balanced | balanced |
| `scope` | Specific pass (forward/backward), input regime, or general | general |
| `target_gpu` | Ampere / Hopper / Blackwell / auto-detect | auto-detect |
| `autonomy` | interactive / autonomous | interactive |
| `max_variants` | Max optimization variants to try | 8 |
| `target_metric` | Optional concrete target (e.g., "forward under 0.3ms at hidden_size=4096") | none |

## Pre-Flight Validation

Before starting the pipeline, validate:

1. Kernel file exists: `src/liger_kernel/ops/{kernel}.py`
2. Benchmark script exists: `benchmark/scripts/benchmark_{kernel}.py`
3. Test file exists: `test/transformers/test_{kernel}.py`
4. GPU is available and CUDA works
5. Project is installed in dev mode (`pip install -e ".[dev]"`)

If any validation fails, report clearly and stop.

## Pipeline

### Stage 1: Profile

Spawn a **Profiler** agent (read [profiler.md](profiler.md)).

The agent:
1. Creates the workspace directory `optimization/{kernel}/`
2. Copies the original kernel as a snapshot
3. Runs baseline benchmarks using the existing benchmark script
4. Detects GPU architecture (or uses user-specified target)
5. Optionally runs NCU profiling (if `ncu` is available)
6. Analyzes the kernel code (tier classification, patterns, optimization opportunities)
7. Classifies the bottleneck: memory-bound vs compute-bound
8. Produces an optimization profile with a recommended strategy order
9. Saves profile to `optimization/{kernel}/profile.md`

**Human checkpoint (interactive mode):** Present the optimization profile with bottleneck diagnosis and proposed strategy order. Confirm before proceeding.

### Stage 2: Optimize

Spawn an **Optimizer** agent (read [optimizer.md](optimizer.md)).

The agent runs an autonomous optimization loop:

1. Read the optimization profile and original kernel
2. **Always try parameter tuning first** (BLOCK_SIZE, num_warps, num_stages manual sweep -- NOT @triton.autotune)
3. Then apply diagnosis-driven techniques from [optimization-strategies.md](optimization-strategies.md)
4. For each variant:
   a. Generate the variant code → `optimization/{kernel}/{kernel}_vN.py`
   b. Write the variant lab notebook → `optimization/{kernel}/{kernel}_vN_notes.md`
   c. Run quick smoke test (single shape, float32, forward+backward) → discard on failure
   d. Run the **full existing benchmark script** → `optimization/{kernel}/benchmarks/vN_results.csv`
   e. Check guardrails (no catastrophic regressions)
   f. Update the variant notes with actual results
5. Read all prior variant notes before generating the next variant
6. **Stop when:** budget exhausted, 2 consecutive variants with <1% improvement, or target metric met
7. Produce a comparison table of ALL variants

**Human checkpoint (interactive mode):** Present the comparison table across all variants. User approves the winner (or skill picks best if autonomous).

### Stage 3: Finalize

Spawn a **Finalizer** agent (read [finalizer.md](finalizer.md)).

The agent:
1. Applies the winning variant in-place to `src/liger_kernel/ops/{kernel}.py`
2. Runs the full test suite: `python -m pytest test/transformers/test_{kernel}.py -xvs` (hard gate)
3. Runs checkstyle: `make checkstyle` (auto-fix with `ruff check . --fix && ruff format .`)
4. Generates 3-way comparison plots (original liger vs optimized liger vs huggingface baseline) using `benchmarks_visualizer.py`
5. Generates the final optimization report → `optimization/{kernel}/report.md`
6. Creates a PR with only the kernel code changes (no plots or optimization workspace files)
7. Presents the before/after summary with plots

**Human checkpoint (interactive mode):** Present the final report with before/after numbers, comparison plots, and test results.

## Guardrails

These apply to EVERY variant, regardless of mode:

| Guardrail | Threshold | Action |
|-----------|-----------|--------|
| Non-target metric regression | >5% worse | Reject variant |
| Cross-pass regression | >10% on one pass to marginally improve other | Reject variant |
| Smoke test failure | Any correctness failure | Discard variant immediately |
| Full test suite failure | Any | Do NOT apply winner, report failure, stop |
| Checkstyle failure | Any | Auto-fix with ruff, retry once |

## Reference Files

- [profiler.md](profiler.md) -- Profiler Agent specification
- [optimizer.md](optimizer.md) -- Optimizer Agent specification
- [finalizer.md](finalizer.md) -- Finalizer Agent specification
- [optimization-strategies.md](optimization-strategies.md) -- Catalog of optimization techniques
- Templates in [templates/](templates/) -- Output format templates
- [examples/rms-norm-optimization.md](examples/rms-norm-optimization.md) -- Example: optimizing a Tier 2 kernel
- [examples/canary-kernel.md](examples/canary-kernel.md) -- Sub-optimal test kernel for skill validation
