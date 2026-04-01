# Kernel Profile: SwiGLU (Tier 1 — Element-wise)

## Identity
- operation_name: swiglu
- function_class_name: LigerSiLUMulFunction
- module_class_name: LigerSwiGLUMLP
- functional_name: liger_swiglu

## Classification
- tier: 1
- tier_description: element-wise
- closest_existing_kernel: geglu (same structure, different activation)

## Forward Pass
- forward_inputs:
  - a: shape (B, T, H), gate projection output
  - b: shape (B, T, H), up projection output
- forward_outputs:
  - c: shape (B, T, H), silu(a) * b
- forward_computation: c = silu(a) * b, where silu(x) = x * sigmoid(x)
- precision_sensitive_ops: [sigmoid]

## Backward Pass
- backward_saved_tensors: [a, b] (reshaped to 2D in forward wrapper)
- backward_recompute: recompute silu(a) and sigmoid(a) in backward
- gradient_formulas:
  - da: dc * (silu(a) * (1 - sigmoid(a)) + sigmoid(a)) * b
  - db: dc * silu(a)

## Tiling Strategy
- grid_dimensions: 1D
- grid_description: one program per row, `(n_rows,)`
- block_size_source: calculate_settings(n_cols)
- needs_sm_parallelism: false

## Module Parameters
- module_init_params:
  - config: HuggingFace model config object
- learnable_params:
  - gate_proj: Linear(hidden_size, intermediate_size, bias=False)
  - up_proj: Linear(hidden_size, intermediate_size, bias=False)
  - down_proj: Linear(intermediate_size, hidden_size, bias=False)

## Benchmarking
- benchmark_variable: hidden_size
- benchmark_x_label: "hidden_size"
- benchmark_x_values_suggestion: [1024, 2048, 4096, 8192, 16384]
- benchmark_providers: ["liger", "torch", "torch_compile"]
- benchmark_fixed_config: {BT: 4096, dtype: torch.bfloat16}

## Key Patterns

- **Recomputation over saving**: Forward saves `a, b` but backward recomputes `sigmoid(a)` and `silu(a)` — saves memory, sigmoid is cheap
- **In-place backward**: Writes gradients directly to `a_ptr` and `b_ptr` (the saved tensors) — saves allocation
- **Float32 for sigmoid**: `a_row` cast to `tl.float32` before `tl.sigmoid`, result cast back via `.cast(b_row.dtype)`
- **No intermediate allocations**: Forward kernel writes directly to output `c`; backward kernel overwrites saved `a, b`
