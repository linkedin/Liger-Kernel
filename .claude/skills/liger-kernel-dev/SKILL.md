---
name: liger-kernel-dev
description: "Develops production-ready Triton kernels for Liger Kernel. Creates new kernels from PyTorch operations (local files, URLs, code snippets, or natural language) with ops, module wrappers, functional APIs, unit tests, benchmarks, and plots. Also modifies existing Liger kernels. Use when adding a new Triton kernel, converting a PyTorch operation to Triton, or updating an existing Liger kernel."
---

# Liger Kernel Dev

Develops Triton kernels for Liger Kernel through a 3-stage pipeline with human review between stages. Supports creating new kernels and modifying existing ones. NVIDIA GPUs only.

## Mode Detection

- **Create mode**: User asks to create/add/generate/write/build a new kernel → full pipeline
- **Modify mode**: User asks to update/fix/change/extend an existing kernel → skip Analyze, modify files, then Validate

## Pipeline (Create Mode)

### Stage 1: Analyze

Spawn an **Analyzer** agent (read [analyzer.md](analyzer.md)).

Accepts any input: local file, URL, code snippet, natural language description, or model component reference. Produces a standalone PyTorch reference implementation and a kernel profile.

**Human checkpoint:** Present PyTorch reference + kernel profile. Confirm before proceeding.

### Stage 2: Generate

Spawn a **Generator** agent (read [generator.md](generator.md)).

Generates/modifies up to 8 files:

1. `src/liger_kernel/ops/{kernel}.py` — NEW Triton kernels + autograd Function
2. `src/liger_kernel/transformers/{kernel}.py` — NEW nn.Module wrapper
3. `src/liger_kernel/transformers/functional.py` — MODIFY add functional API
4. `src/liger_kernel/ops/__init__.py` — MODIFY export Function class
5. `src/liger_kernel/transformers/__init__.py` — MODIFY export Module + `__all__`
6. `test/transformers/test_{kernel}.py` — NEW unit tests
7. `benchmark/scripts/benchmark_{kernel}.py` — NEW benchmark script
8. `benchmark/data/all_benchmark_data.csv` — MODIFY (after benchmarks run)

**Human checkpoint:** Present changes for review.

### Stage 3: Validate

Spawn a **Validator** agent (read [validator.md](validator.md)).

Runs checkstyle, unit tests (hard gate — stops on persistent failure), benchmarks, and generates plots. Optionally runs ncu profiling.

**Human checkpoint:** Report final results with benchmark numbers and plots.

## Pipeline (Modify Mode)

1. Read existing kernel files to understand current implementation
2. Understand the requested modification
3. Make targeted changes (Generator handles this)
4. Run full Validate stage (same as create mode)

## Reference Files

- [kernel-profile-format.md](kernel-profile-format.md) — Kernel profile schema and field descriptions
- [examples/swiglu-profile.md](examples/swiglu-profile.md) — Tier 1 (element-wise) reference
- [examples/rms-norm-profile.md](examples/rms-norm-profile.md) — Tier 2 (reduction) reference
- [examples/cross-entropy-profile.md](examples/cross-entropy-profile.md) — Tier 3 (fused/complex) reference
- Templates in [templates/](templates/) — Code generation patterns for each file type
