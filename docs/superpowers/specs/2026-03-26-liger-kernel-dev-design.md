# Design: liger-kernel-dev Skill

**Date:** 2026-03-26
**Status:** Approved

## Summary

A Claude Code skill that automates the full lifecycle of Triton kernel development for Liger Kernel — from understanding a PyTorch operation to generating a production-ready kernel with tests, benchmarks, and plots. Supports both creating new kernels and modifying existing ones. NVIDIA GPUs only (v1).

## Modes

- **Create mode**: Full 3-stage pipeline (Analyze → Generate → Validate)
- **Modify mode**: Skip Analyze, understand change → modify → Validate

## Input Types

Accepts: local file paths, GitHub/web URLs, code snippets, natural language math descriptions, or model component references (e.g., "the MLP in Phi-4"). All normalized into a standalone PyTorch reference implementation.

## Pipeline

### Stage 1: Analyzer Agent
- Understands the operation from any input form
- Writes standalone PyTorch reference (nn.Module or function)
- Produces kernel profile (operation type, tier, signatures, tiling strategy, etc.)
- **Human checkpoint**: Confirm profile before proceeding

### Stage 2: Generator Agent
- Generates/modifies 8 files: ops kernel, transformer wrapper, functional API, two __init__.py exports, unit tests, benchmark script
- Follows strict Liger patterns via templates
- **Human checkpoint**: Review generated code

### Stage 3: Validator Agent
1. `make checkstyle` (auto-fix with ruff if needed)
2. Unit tests — **hard gate**, 3 retries, STOP on persistent failure
3. Benchmarks (speed + memory, fwd/bwd/full)
4. Generate plots
5. Optional ncu profiling
6. **Human checkpoint**: Report results

## Validation Bar

- **Correctness**: Must pass (hard gate)
- **Performance**: Meaningful improvement on at least one of speed OR memory. The other should not be catastrophically worse.
- **Code quality**: `make checkstyle` must pass
- **Artifacts**: Benchmark plots generated

## Files Generated Per New Kernel

1. `src/liger_kernel/ops/{kernel}.py` — Triton kernels + autograd Function
2. `src/liger_kernel/transformers/{kernel}.py` — nn.Module wrapper
3. `src/liger_kernel/transformers/functional.py` — functional API addition
4. `src/liger_kernel/ops/__init__.py` — export Function class
5. `src/liger_kernel/transformers/__init__.py` — export Module + __all__
6. `test/transformers/test_{kernel}.py` — unit tests
7. `benchmark/scripts/benchmark_{kernel}.py` — benchmarks
8. `benchmark/data/all_benchmark_data.csv` — results (after benchmarks)

## Skill Architecture

Single skill with progressive disclosure: SKILL.md orchestrates, agent instruction files provide stage-specific details, templates provide code patterns, examples provide tier-specific references.

## Out of Scope

- Performance optimization (future `liger-kernel-perf` skill)
- Monkey-patching into HF models (`liger-autopatch` skill)
- Vendor backends (NPU/XPU/HIP)
- Chunked loss kernels
