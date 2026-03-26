# Kernel Profile Format

When analyzing an operation, produce a profile with all of these fields.

## Fields

```yaml
# Identity
operation_name: snake_case name (e.g., "rms_norm", "dyt", "sparsemax")
function_class_name: "Liger{PascalCase}Function" (e.g., "LigerRMSNormFunction")
module_class_name: "Liger{PascalCase}" (e.g., "LigerRMSNorm")
functional_name: "liger_{snake_case}" (e.g., "liger_rms_norm")

# Classification
tier: 1 | 2 | 3
tier_description: "element-wise" | "reduction" | "fused/complex"
closest_existing_kernel: which existing Liger kernel is most similar

# Forward Pass
forward_inputs:
  - name: X
    shape: "(B, T, H)"
    dtype: "input dtype"
  # ... list all inputs with shapes
forward_outputs:
  - name: Y
    shape: "(B, T, H)"
    dtype: "same as input"
forward_computation: brief math description (e.g., "Y = X * rsqrt(mean(X^2) + eps) * W")
precision_sensitive_ops: list ops needing float32 (e.g., [rsqrt, sigmoid, exp])

# Backward Pass
backward_saved_tensors: what to save_for_backward (e.g., [X, W, RSTD])
backward_recompute: what to recompute instead of saving (e.g., "recompute silu in backward")
gradient_formulas:
  dX: "formula or description"
  dW: "formula or description"

# Tiling Strategy
grid_dimensions: 1D | 2D
grid_description: "one program per row" | "one program per (row_block, col_block)" | etc.
block_size_source: "calculate_settings(n_cols)" | "custom" with explanation
needs_sm_parallelism: true | false (for weight gradient reduction in backward)

# Module Parameters
module_init_params:
  - name: hidden_size
    type: int
  - name: eps
    type: float
    default: 1e-6
  # ... list all __init__ parameters
learnable_params:
  - name: weight
    shape: "(hidden_size,)"
    init: "ones"
  # ... list learnable parameters

# Benchmarking
benchmark_variable: "hidden_size" | "seq_len" | "vocab_size" | etc.
benchmark_x_label: human-readable label for x-axis
benchmark_x_values_suggestion: suggested range to sweep
benchmark_providers: ["liger", "torch"] (add "torch_compile" if applicable)
benchmark_fixed_config: dict of fixed params (e.g., {BT: 4096, dtype: torch.bfloat16})
```

## Naming Conventions

| Component | Pattern | Example |
|-----------|---------|---------|
| Ops file | `src/liger_kernel/ops/{operation_name}.py` | `ops/dyt.py` |
| Transformer file | `src/liger_kernel/transformers/{operation_name}.py` | `transformers/dyt.py` |
| Test file | `test/transformers/test_{operation_name}.py` | `test_dyt.py` |
| Benchmark file | `benchmark/scripts/benchmark_{operation_name}.py` | `benchmark_dyt.py` |
| Function class | `Liger{PascalCase}Function` | `LigerDyTFunction` |
| Module class | `Liger{PascalCase}` | `LigerDyT` |
| Functional | `liger_{snake_case}` | `liger_dyt` |
| Forward wrapper | `{snake_case}_forward` or `liger_{snake_case}_fwd` | `liger_dyt_fwd` |
| Backward wrapper | `{snake_case}_backward` or `liger_{snake_case}_bwd` | `liger_dyt_bwd` |
