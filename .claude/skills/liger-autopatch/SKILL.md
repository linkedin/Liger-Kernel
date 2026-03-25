---
name: liger-autopatch
description: "Automatically add Liger Kernel support for a new HuggingFace model. Analyzes the model architecture, generates all required files (lce_forward, monkey patch, tests), and validates correctness."
---

# Liger Auto-Patch

Add Liger Kernel optimizations to a new HuggingFace Transformers model with a single command.

## When to Use

Use this skill when the user wants to add Liger Kernel support for a HuggingFace model that is not yet in the `MODEL_TYPE_TO_APPLY_LIGER_FN` dictionary in `monkey_patch.py`.

## Input

The user provides one of:
- A HuggingFace model type string (e.g., `"nemotron"`, `"cohere2"`)
- A HuggingFace model class name (e.g., `"NemotronForCausalLM"`)
- A model name on HuggingFace Hub (e.g., `"nvidia/Nemotron-Mini-4B-Instruct"`)

## Pipeline

This skill runs a 3-stage pipeline with human checkpoints between stages.

### Stage 1: Analyze

Spawn a **Model Analyzer** agent (see `model-analyzer.md`).

The agent reads the HuggingFace `modeling_*.py` source and produces a **model profile** — a structured document answering 12 architectural questions (see `decision-matrix.md`).

**Human checkpoint:** Present the profile to the user. Ask them to confirm or correct any decisions before proceeding.

### Stage 2: Generate

Spawn a **Code Generator** agent (see `code-generator.md`).

The agent takes the confirmed profile and generates/modifies these files:

1. `src/liger_kernel/transformers/model/{model}.py` — NEW
2. `src/liger_kernel/transformers/monkey_patch.py` — MODIFY
3. `src/liger_kernel/transformers/__init__.py` — MODIFY
4. `src/liger_kernel/transformers/model/output_classes.py` — MODIFY (if needed)
5. `test/transformers/test_monkey_patch.py` — MODIFY
6. `test/convergence/bf16/test_mini_models.py` — MODIFY
7. `test/utils.py` — MODIFY
8. `README.md` — MODIFY

**Human checkpoint:** Present the list of changes. Ask the user to review before running tests.

### Stage 3: Validate

Spawn a **Validator** agent (see `validator.md`).

The agent runs:
1. The instance patching test: `pytest test/transformers/test_monkey_patch.py -k {model} -xvs`
2. The convergence test: `pytest test/convergence/bf16/test_mini_models.py -k {model} -xvs`

If tests fail, the Validator reads the traceback, diagnoses the issue, fixes the code, and retries (max 3 attempts).

**Human checkpoint:** Report final test results.

## Setup

Before first use, ensure the project is installed in development mode:

```bash
pip install -e ".[dev]"
```

If the user has a virtual environment, they should activate it first. Ask them to confirm their Python environment has `liger-kernel`, `transformers`, `torch`, and `triton` installed before proceeding.

## Important Rules

- Read `decision-matrix.md` before analyzing any model
- Read `examples/llama-profile.md` and `examples/gemma-profile.md` as reference
- Use templates in `templates/` as the basis for code generation
- Never guess architecture details — always read the HF source code
- Keep all generated files consistent with existing Liger Kernel code style
- Use `ruff` for formatting: line length 120, double quotes, single imports
