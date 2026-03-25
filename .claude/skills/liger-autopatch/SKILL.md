---
name: liger-autopatch
description: "Adds Liger Kernel support for a new HuggingFace Transformers model. Generates lce_forward, monkey-patch function, tests, and README entry. Use when adding a new model to Liger Kernel, when a user asks to patch an unsupported model, or when extending MODEL_TYPE_TO_APPLY_LIGER_FN."
---

# Liger Auto-Patch

Adds Liger Kernel optimization support for a new HuggingFace model through a 3-stage pipeline with human review between stages.

## Pipeline

### Stage 1: Analyze

Spawn a **Model Analyzer** agent (read [model-analyzer.md](model-analyzer.md)).

The agent reads the HF `modeling_*.py` source and produces a **model profile** answering 12 architectural questions from [decision-matrix.md](decision-matrix.md).

**Human checkpoint:** Present the profile. Confirm before proceeding.

### Stage 2: Generate

Spawn a **Code Generator** agent (read [code-generator.md](code-generator.md)).

Generates/modifies up to 12 files:

1. `src/liger_kernel/transformers/model/{model}.py` — NEW lce_forward
2. `src/liger_kernel/transformers/monkey_patch.py` — MODIFY
3. `src/liger_kernel/transformers/__init__.py` — MODIFY
4. `src/liger_kernel/transformers/model/output_classes.py` — MODIFY if needed
5. `test/transformers/test_monkey_patch.py` — MODIFY
6. `test/convergence/bf16/test_mini_models.py` — MODIFY (FLCE path)
7. `test/convergence/bf16/test_mini_models_with_logits.py` — MODIFY (non-FLCE path)
8. `test/convergence/fp32/test_mini_models.py` — MODIFY (FLCE path)
9. `test/convergence/fp32/test_mini_models_with_logits.py` — MODIFY (non-FLCE path)
10. `test/convergence/{bf16,fp32}/test_mini_models_multimodal.py` — MODIFY if VL model
11. `test/utils.py` — MODIFY
12. `README.md` — MODIFY

**Human checkpoint:** Present changes for review.

### Stage 3: Validate

Spawn a **Validator** agent (read [validator.md](validator.md)).

Runs instance patching test, convergence test, and lint check. Retries up to 3 times on failure.

**Human checkpoint:** Report final test results.

## Reference Files

- [decision-matrix.md](decision-matrix.md) — 12 architectural decisions to resolve per model
- [examples/llama-profile.md](examples/llama-profile.md) — Reference profile for standard dense model
- [examples/gemma-profile.md](examples/gemma-profile.md) — Reference profile showing GeGLU + offset variant
- Templates in [templates/](templates/) — Code generation patterns for each file type
