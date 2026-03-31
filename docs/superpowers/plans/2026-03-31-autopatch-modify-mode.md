# Autopatch Modify Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `liger-autopatch` skill to trigger on and guide modifications to existing monkey-patches, not just new model additions.

**Architecture:** Add a Mode Detection triage to SKILL.md that routes to either the existing create pipeline or a lighter modify pipeline. Expand code-generator.md with a Modification Checklist (rules R1-R6 and common patterns). All other skill files remain unchanged.

**Tech Stack:** Markdown skill files only — no code changes.

---

### Task 1: Update SKILL.md — description, mode detection, modify pipeline

**Files:**
- Modify: `.claude/skills/liger-autopatch/SKILL.md`

- [ ] **Step 1: Update the frontmatter description**

Replace the current `description` field (line 3) with:

```yaml
description: "Adds Liger Kernel support for a new HuggingFace Transformers model, or modifies existing monkey-patching. Generates lce_forward, monkey-patch function, tests, and README entry. Use when adding a new model to Liger Kernel, when a user asks to patch an unsupported model, when extending MODEL_TYPE_TO_APPLY_LIGER_FN, or when modifying/updating/fixing an existing monkey-patch (e.g., adding a new kernel to an already-supported model, fixing instance patching, updating a patch for upstream HF changes)."
```

- [ ] **Step 2: Update the title and intro paragraph**

Replace line 6-7:

```markdown
# Liger Auto-Patch

Adds Liger Kernel optimization support for a new HuggingFace model through a 3-stage pipeline with human review between stages.
```

With:

```markdown
# Liger Auto-Patch

Adds Liger Kernel optimization support for a new HuggingFace model, or modifies existing monkey-patching, through a staged pipeline with human review between stages. Supports creating new model patches and modifying existing ones.
```

- [ ] **Step 3: Add Mode Detection section**

Insert a new section immediately after the intro paragraph and before `## Pipeline`:

```markdown
## Mode Detection

- **Create mode**: User asks to add/patch/support a new model → full pipeline (Analyze → Generate → Validate)
- **Modify mode**: User asks to update/fix/change/extend an existing monkey-patch → lighter pipeline (Change Impact Analysis → Apply Changes → Validate)

Keywords that suggest modify mode: update, fix, change, add [kernel] to [existing model], extend, modify, new activation, new norm, bug in patch, upstream changed
```

- [ ] **Step 4: Rename existing Pipeline section and add Modify Pipeline**

Rename the existing `## Pipeline` heading to `## Pipeline (Create Mode)`.

Then, after the create pipeline's Stage 3 (Validate) section and before `## Reference Files`, insert:

```markdown
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
```

- [ ] **Step 5: Verify the final SKILL.md reads correctly**

Read the file end-to-end. Confirm:
- Frontmatter description mentions both new models and modifications
- Mode Detection section exists between intro and first pipeline
- Create pipeline is labeled "Pipeline (Create Mode)"
- Modify pipeline is labeled "Pipeline (Modify Mode)" with 3 stages
- Reference Files section is still at the bottom, unchanged

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/liger-autopatch/SKILL.md
git commit -m "Extend liger-autopatch skill to support modify mode

Add Mode Detection triage and a lighter Modify Pipeline for changes
to existing monkey-patches (adding kernels, fixing bugs, upstream updates)."
```

---

### Task 2: Update code-generator.md — mode section and modification checklist

**Files:**
- Modify: `.claude/skills/liger-autopatch/code-generator.md`

- [ ] **Step 1: Add Mode section at the top**

Insert immediately after the `# Code Generator Agent` heading (line 1) and before `## Pre-Requisites` (line 3):

```markdown
## Mode

- **Create mode** (default): Generating all files for a new model. Follow the full "Files to Generate" list below.
- **Modify mode**: Making targeted changes to an existing monkey-patch. Follow the "Modification Checklist" section instead.
```

- [ ] **Step 2: Update the opening description**

Replace line 3:

```markdown
Takes a confirmed model profile and generates all files to add Liger Kernel support.
```

With:

```markdown
Takes a confirmed model profile (create mode) or a change plan (modify mode) and generates or modifies files for Liger Kernel support.
```

- [ ] **Step 3: Add Modification Checklist section**

Insert after the `## Code Style` section (at the end of the file, after line 86):

```markdown
## Modification Checklist (Modify Mode)

Before making changes, read the existing implementation:
1. Read `apply_liger_kernel_to_{model_type}` in `monkey_patch.py`
2. Read the existing test in `test_monkey_patch.py` for this model
3. Read the relevant HF modeling source for context

### Rules for All Modifications

**R1. Both patching levels.** If adding a new kernel, it must appear in BOTH:
  - Class-level patching (the main body of `apply_liger_kernel_to_{model_type}`)
  - Instance-level patching (the `if model is not None` block)

  Omitting one is the most common mistake.

**R2. New parameter with default.** Every new kernel gets a bool parameter on the
  apply function signature (e.g., `relu_squared: bool = True`). Default should be `True`
  for kernels that are safe to enable by default, `False` otherwise.

**R3. Update docstring.** Update the function's docstring to:
  - Add an `Args` entry for the new parameter
  - Remove any stale notes that the new kernel invalidates
    (e.g., "squared ReLU is not supported" → remove if you're adding it)

**R4. Update tests.** In the existing `test_apply_liger_kernel_to_instance_for_{model_type}`:
  - Add import for the new Liger kernel class
  - Add "not yet patched" assertion before `_apply_liger_kernel_to_instance`
  - Add "correctly patched" assertion after
  - Follow the exact pattern of existing assertions in the same test

**R5. Run convergence tests.** Don't modify convergence test files unless the change
  requires it (e.g., new mini model config fields). But DO run existing convergence
  tests to verify no regression.

**R6. Update README.md.** If the change adds a visibly new capability to the model's
  row in the patching table (e.g., a new operation), update the supported operations list.

### Common Modification Patterns

**Adding an activation kernel (e.g., relu_squared for nemotron):**
- Import the Liger kernel class at the top of `monkey_patch.py`
- Add bool parameter to apply function signature
- Class-level: replace in `ACT2FN` dict or replace the MLP class
- Instance-level: patch each `decoder_layer`'s activation/MLP
- Test: `assert isinstance` checks on the activation/MLP

**Adding a norm variant:**
- Add bool parameter to apply function
- Class-level: replace the norm class
- Instance-level: use `_patch_rms_norm_module` or `_patch_layer_norm_module` on all norm attrs
- Test: `assert isinstance` checks on norm modules

**Fixing missing instance patching:**
- Read the class-level patching to see what's patched
- Add corresponding instance-level patches in the `if model is not None` block
- Test: add assertions that were missing

**Updating for upstream HF changes:**
- Compare the current HF modeling file against what the patch assumes
- Update class names, attribute names, forward signatures as needed
- May require updating `lce_forward` if the base model's forward changed
```

- [ ] **Step 4: Verify the final code-generator.md reads correctly**

Read the file end-to-end. Confirm:
- Mode section exists between title and Pre-Requisites
- Opening description mentions both create and modify modes
- Modification Checklist section exists after Code Style
- Rules R1-R6 are all present
- Four common modification patterns are documented

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/liger-autopatch/code-generator.md
git commit -m "Add modify mode and modification checklist to code generator

Rules R1-R6 enforce both patching levels, parameter defaults, docstring
updates, test assertions, convergence checks, and README updates. Includes
common patterns for activation kernels, norms, and upstream changes."
```

---

### Task 3: Final validation

- [ ] **Step 1: Verify both files are well-formed markdown**

Read both `.claude/skills/liger-autopatch/SKILL.md` and `.claude/skills/liger-autopatch/code-generator.md` end-to-end. Confirm no broken links, no orphaned sections, no formatting issues.

- [ ] **Step 2: Test skill loading**

Invoke the `liger-autopatch` skill and verify the loaded content includes the new Mode Detection section and modify pipeline references.

- [ ] **Step 3: Verify cross-references**

Confirm that:
- SKILL.md's modify pipeline Stage 2 references `code-generator.md` in modify mode
- SKILL.md's modify pipeline Stage 3 references `validator.md`
- code-generator.md's Modification Checklist is self-contained (no dangling references to nonexistent templates)
