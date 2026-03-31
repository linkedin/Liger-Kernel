---
name: liger-autopatch
description: "Adds Liger Kernel support for a new HuggingFace Transformers model, or modifies existing monkey-patching. Generates lce_forward, monkey-patch function, tests, and README entry. Use when adding a new model to Liger Kernel, when a user asks to patch an unsupported model, when extending MODEL_TYPE_TO_APPLY_LIGER_FN, or when modifying/updating/fixing an existing monkey-patch (e.g., adding a new kernel to an already-supported model, fixing instance patching, updating a patch for upstream HF changes)."
---

# Liger Auto-Patch

Adds Liger Kernel optimization support for a new HuggingFace model, or modifies existing monkey-patching, through a staged pipeline with human review between stages. Supports creating new model patches and modifying existing ones.

## Mode Detection

- **Create mode**: User asks to add/patch/support a new model → full pipeline (Analyze → Generate → Validate)
- **Modify mode**: User asks to update/fix/change/extend an existing monkey-patch → lighter pipeline (Change Impact Analysis → Apply Changes → Validate)

Keywords that suggest modify mode: update, fix, change, add [kernel] to [existing model], extend, modify, new activation, new norm, bug in patch, upstream changed

## Pipeline (Create Mode)

### Stage 1: Analyze

Spawn a **Model Analyzer** agent (read [model-analyzer.md](model-analyzer.md)).

The agent reads the HF `modeling_*.py` source and produces a **model profile** answering 12 architectural questions from [decision-matrix.md](decision-matrix.md).

**Human checkpoint:** Present the profile. Confirm before proceeding.

### Stage 2: Generate

Spawn a **Code Generator** agent (read [code-generator.md](code-generator.md)).

Generates/modifies up to 13 files:

1. `src/liger_kernel/transformers/model/{model}.py` — NEW lce_forward
2. `src/liger_kernel/transformers/monkey_patch.py` — MODIFY
3. `src/liger_kernel/transformers/__init__.py` — MODIFY
4. `src/liger_kernel/transformers/model/output_classes.py` — MODIFY if needed
5. `test/transformers/test_monkey_patch.py` — MODIFY
6. `test/convergence/bf16/test_mini_models.py` — MODIFY (FLCE path)
7. `test/convergence/bf16/test_mini_models_with_logits.py` — MODIFY (non-FLCE path)
8. `test/convergence/fp32/test_mini_models.py` — MODIFY (FLCE path)
9. `test/convergence/fp32/test_mini_models_with_logits.py` — MODIFY (non-FLCE path)
10. `test/convergence/bf16/test_mini_models_multimodal.py` — MODIFY if VL model
11. `test/convergence/fp32/test_mini_models_multimodal.py` — MODIFY if VL model
12. `test/utils.py` — MODIFY
13. `README.md` — MODIFY

**Human checkpoint:** Present changes for review.

### Stage 3: Validate

Spawn a **Validator** agent (read [validator.md](validator.md)).

Runs instance patching test, convergence test, and lint check. Retries up to 3 times on failure.

**Human checkpoint:** Report final test results.

## Pipeline (Modify Mode)

### Stage 1: Change Impact Analysis

Read the existing `apply_liger_kernel_to_{model_type}` function in `monkey_patch.py` and the relevant section of the upstream HF `modeling_{model_type}.py`. Produce a short change plan:

- What is being added/changed/fixed
- Which Liger kernel(s) are involved
- Which files need modification (subset of the 13 files from create mode)
- What the expected behavior should be after the change

**Human checkpoint:** Present the change plan. Confirm before proceeding.

### Stage 2: Apply Changes

Spawn the **Code Generator** agent (read [code-generator.md](code-generator.md)) in **modify mode**.

**Human checkpoint:** Present changes for review.

### Stage 3: Validate

Same as create mode — spawn the **Validator** agent (read [validator.md](validator.md)).

**Human checkpoint:** Report final test results.

## Reference Files

- [decision-matrix.md](decision-matrix.md) — 12 architectural decisions to resolve per model
- [examples/llama-profile.md](examples/llama-profile.md) — Reference profile for standard dense model
- [examples/gemma-profile.md](examples/gemma-profile.md) — Reference profile showing GeGLU + offset variant
- Templates in [templates/](templates/) — Code generation patterns for each file type
