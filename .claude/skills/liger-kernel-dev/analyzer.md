# Analyzer Agent

Understands a PyTorch operation from any input form and produces a standalone PyTorch reference implementation + kernel profile.

## Input Handling

The user may provide the operation in any form:

1. **Local file path** → Read the file directly
2. **URL** (GitHub, HuggingFace, etc.) → Fetch via WebFetch tool
3. **Code snippet** → Pasted in the conversation
4. **Natural language** → Mathematical description (e.g., "element-wise SiLU(x) * y")
5. **Model component** → e.g., "the MLP in Phi-4" — locate in transformers source and extract

## Steps

### 1. Understand the Operation

- Read/fetch the source code from whatever input was provided
- Identify the mathematical operation (forward pass)
- Derive the backward pass (gradient computation)
- Identify all inputs, outputs, and their expected shapes/dtypes
- Note any precision-sensitive operations that need float32 upcasting (sigmoid, rsqrt, exp, log, tanh)

### 2. Write PyTorch Reference

Create a standalone implementation that:
- Depends only on `torch` (no external libraries)
- Implements both forward and backward behavior (either as an `nn.Module` or a plain function)
- Will serve as the correctness baseline for testing
- Is clean, readable, and well-named

### 3. Classify Into Tier

Read [kernel-profile-format.md](kernel-profile-format.md) for the full schema.

**Tier 1 — Element-wise**: No reductions across dimensions. One row per program. Examples: SwiGLU, GeGLU, DyT.
- Read reference: `src/liger_kernel/ops/swiglu.py`

**Tier 2 — Reduction**: Cross-column reductions (tl.sum, tl.max). May need to save intermediate state for backward. May need SM-based parallelism for weight gradient reduction. Examples: RMSNorm, LayerNorm, Softmax, Sparsemax.
- Read reference: `src/liger_kernel/ops/rms_norm.py`

**Tier 3 — Fused/Complex**: Multi-pass algorithms, gradient-in-forward tricks, multiple outputs. Examples: CrossEntropy, FusedLinearCrossEntropy.
- Read reference: `src/liger_kernel/ops/cross_entropy.py`

Also read the closest example profile:
- Tier 1 → [examples/swiglu-profile.md](examples/swiglu-profile.md)
- Tier 2 → [examples/rms-norm-profile.md](examples/rms-norm-profile.md)
- Tier 3 → [examples/cross-entropy-profile.md](examples/cross-entropy-profile.md)

### 4. Produce Kernel Profile

Fill in all fields from [kernel-profile-format.md](kernel-profile-format.md).

### 5. Present to User

Show:
1. The PyTorch reference implementation (full code)
2. The kernel profile (all fields)
3. Which existing kernel is closest (for the Generator to use as reference)

Wait for user confirmation before proceeding to Stage 2.
