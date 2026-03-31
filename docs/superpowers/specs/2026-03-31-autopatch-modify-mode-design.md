# Design: Extend liger-autopatch Skill to Support Modifications

## Problem

The `liger-autopatch` skill only triggers for adding new model support. When users modify existing monkey-patches (e.g., adding a new kernel to an already-supported model, fixing instance patching, updating for upstream HF changes), Claude does not use the skill at all. This means modification PRs miss the guidelines and patterns the skill enforces.

Example: PR #1172 added `LigerReLUSquared` to nemotron's existing monkey-patch without skill guidance.

## Approach

Follow the same pattern as `liger-kernel-dev`, which handles both create and modify modes via a Mode Detection triage in SKILL.md. Two files change; everything else stays the same.

## Changes

### 1. SKILL.md

**Description field** broadened to cover modifications:

```
"Adds Liger Kernel support for a new HuggingFace Transformers model, or modifies existing
monkey-patching. Generates lce_forward, monkey-patch function, tests, and README entry.
Use when adding a new model to Liger Kernel, when a user asks to patch an unsupported model,
when extending MODEL_TYPE_TO_APPLY_LIGER_FN, or when modifying/updating/fixing an existing
monkey-patch (e.g., adding a new kernel to an already-supported model, fixing instance
patching, updating a patch for upstream HF changes)."
```

**Mode Detection section** added after title, before Pipeline:

- **Create mode**: User asks to add/patch/support a new model -> full pipeline (Analyze -> Generate -> Validate)
- **Modify mode**: User asks to update/fix/change/extend an existing monkey-patch -> lighter pipeline (Change Impact Analysis -> Apply Changes -> Validate)
- Keywords for modify mode: update, fix, change, add [kernel] to [existing model], extend, modify, new activation, new norm, bug in patch, upstream changed

**Modify Pipeline section** added after the existing create pipeline:

- Stage 1: Change Impact Analysis -- read existing `apply_liger_kernel_to_{model}` and relevant HF source, produce a short change plan (what's changing, which kernels, which files, expected behavior). Human checkpoint.
- Stage 2: Apply Changes -- spawn Code Generator in modify mode. Human checkpoint.
- Stage 3: Validate -- same as create mode, reuse Validator agent. Human checkpoint.

### 2. code-generator.md

**Mode section** added at the top distinguishing create vs modify mode.

**Modification Checklist section** added after "Files to Generate", containing:

Six rules (R1-R6):
- R1: Both patching levels -- new kernel must appear in both class-level and instance-level patching
- R2: New parameter with default -- bool parameter on apply function signature
- R3: Update docstring -- add Args entry, remove stale notes
- R4: Update tests -- add import, pre-patch assertion, post-patch assertion in existing test
- R5: Run convergence tests -- verify no regression (don't modify unless needed)
- R6: Update README.md -- update supported operations if visibly new capability

Four common modification patterns with specific guidance:
- Adding an activation kernel (modeled on PR #1172)
- Adding a norm variant
- Fixing missing instance patching
- Updating for upstream HF changes

### 3. Files NOT Changed

- `model-analyzer.md` -- only used in create mode
- `validator.md` -- already works for both modes
- `decision-matrix.md` -- only relevant for new model analysis
- Templates -- modification patterns are short enough to live inline in code-generator.md
