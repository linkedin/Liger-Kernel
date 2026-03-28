# Kernel Profile: RMSNorm (Tier 2 — Reduction)

## Identity
- operation_name: rms_norm
- function_class_name: LigerRMSNormFunction
- module_class_name: LigerRMSNorm
- functional_name: liger_rms_norm

## Classification
- tier: 2
- tier_description: reduction
- closest_existing_kernel: layer_norm (similar pattern with different normalization)

## Forward Pass
- forward_inputs:
  - X: shape (B, T, H), input tensor
  - W: shape (H,), weight tensor
  - eps: float, epsilon for numerical stability
  - offset: float, weight offset (0.0 for Llama, 1.0 for Gemma)
  - casting_mode: str, "llama" | "gemma" | "none"
- forward_outputs:
  - Y: shape (B, T, H), normalized output
- forward_computation: Y = (X / RMS(X)) * (W + offset), RMS = sqrt(mean(X^2) + eps)
- precision_sensitive_ops: [rsqrt]

## Backward Pass
- backward_saved_tensors: [X, W, RSTD] — RSTD cached from forward to avoid recomputation
- backward_recompute: none (RSTD is expensive to recompute)
- gradient_formulas:
  - dX: rstd * (dY*(W+offset) - (1/N) * rstd^2 * dot(dY*(W+offset), X) * X)
  - dW: sum over (B,T) of dY * (X * rstd)

## Tiling Strategy
- grid_dimensions: 1D
- grid_description: forward uses one program per row `(n_rows,)`, backward uses SM-based partitioning `(sm_count,)` with `rows_per_program`
- block_size_source: calculate_settings(n_cols)
- needs_sm_parallelism: true (for dW reduction — each SM accumulates partial dW, then summed)

## Module Parameters
- module_init_params:
  - hidden_size: int
  - eps: float = 1e-6
  - offset: float = 0.0
  - casting_mode: str = "llama"
  - init_fn: str = "ones"
  - in_place: bool = True
  - elementwise_affine: bool = True
- learnable_params:
  - weight: shape (hidden_size,), init ones or zeros

## Benchmarking
- benchmark_variable: hidden_size
- benchmark_x_label: "hidden_size"
- benchmark_x_values_suggestion: [1024, 2048, 4096, 8192, 16384]
- benchmark_providers: ["liger", "huggingface"]
- benchmark_fixed_config: {M: 4096, eps: 1e-6, dtype: torch.float32}

## Key Patterns

- **RSTD caching**: Forward computes and stores `rstd = rsqrt(mean(X^2) + eps)` — 1 value per row, tiny memory cost, saves 4 ops in backward
- **SM-based backward**: Weight gradient needs reduction across all rows. Each SM processes `rows_per_program` rows and accumulates into `_dW[sm_id, :]`. Final `dW = _dW.sum(dim=0)`
- **Casting modes as constexpr**: `casting_mode` is `tl.constexpr` so the compiler eliminates dead branches
- **In-place backward option**: `in_place=True` writes dX into dY tensor to save memory. Set `False` when dY is needed elsewhere (e.g., Gemma2 residual)
- **Two kernel variants**: Row-wise for `BLOCK_SIZE > 256 or n_rows < 4096*8`, block-wise otherwise (processes `BLOCK_ROW=16` rows per program for better GPU utilization)
