# AGENTS.md

Guidance for AI coding assistants working in this repository.

## Skills

Repository-specific workflow guides ("skills") live in [`.agents/skills/`](.agents/skills/). Each subdirectory is a self-contained guide for a multi-stage workflow. Read the `SKILL.md` at the top of each subdirectory for an overview; the other files in that directory are referenced from `SKILL.md` and should be read on demand.

| Skill | What it does |
|-------|--------------|
| [`liger-kernel-dev`](.agents/skills/liger-kernel-dev/SKILL.md) | Develops new Triton kernels from a PyTorch reference (or modifies existing kernels). 3-stage pipeline: Analyze → Generate → Validate. NVIDIA GPUs only. |
| [`liger-autopatch`](.agents/skills/liger-autopatch/SKILL.md) | Adds Liger Kernel support for a new HuggingFace Transformers model, or modifies an existing monkey-patch. 3-stage pipeline: Analyze → Generate → Validate. |
| [`liger-kernel-perf`](.agents/skills/liger-kernel-perf/SKILL.md) | Optimizes the performance of an existing Liger Triton kernel. 3-stage pipeline: Profile → Optimize → Finalize. NVIDIA GPUs only. |

The skills are written to be runtime-agnostic — they describe the workflow as a sequence of stages a competent agent (or human) can follow. Where a stage says "Follow the X workflow in `x.md`", that's a directive to read and execute that file's instructions; runtimes that support parallel subagents may delegate the stage, but it is not required.

## Vendor-specific shortcuts

For convenience, some assistants auto-discover skills from vendor-specific paths. These point at the canonical `.agents/skills/` directory:

- `.claude/skills` → symlink → `.agents/skills` (for Claude Code)

If you're adding support for another assistant, add a symlink (or your tool's preferred adapter) pointing to `.agents/skills/`. Do not duplicate the content.

## Repo conventions

- Source layout: `src/liger_kernel/{ops,transformers}/` for Triton ops and `nn.Module` / HF wrappers respectively
- Tests: `test/transformers/` (unit) and `test/convergence/{bf16,fp32}/` (model convergence)
- Benchmarks: `benchmark/scripts/` (scripts) and `benchmark/data/all_benchmark_data.csv` (results)
- Lint/format: `make checkstyle` (uses `ruff`)
- Install dev mode: `pip install -e ".[dev]"`

See `README.md` for the project overview and contribution guide.
