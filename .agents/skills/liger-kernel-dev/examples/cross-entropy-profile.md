# Kernel Profile: CrossEntropy (Tier 3 — Fused/Complex)

## Identity
- operation_name: cross_entropy
- function_class_name: LigerCrossEntropyFunction
- module_class_name: LigerCrossEntropyLoss
- functional_name: liger_cross_entropy

## Classification
- tier: 3
- tier_description: fused/complex
- closest_existing_kernel: fused_linear_cross_entropy (extends this with linear layer fusion)

## Forward Pass
- forward_inputs:
  - _input: shape (B*T, V), logits
  - target: shape (B*T,), label indices
  - weight: shape (V,) or None, class weights
  - ignore_index: int, label to ignore
  - label_smoothing: float
  - reduction: str, "mean" | "sum" | "none"
  - softcap: float or None
- forward_outputs:
  - loss: scalar or shape (B*T,) if reduction="none"
  - z_loss: optional auxiliary loss
  - token_accuracy: optional accuracy metric
  - predicted_tokens: optional argmax tokens
- forward_computation: Two-pass online softmax + cross entropy loss with optional smoothing and softcapping
- precision_sensitive_ops: [exp, log]

## Backward Pass
- backward_saved_tensors: [_input] — gradient is computed during forward and stored in-place
- backward_recompute: none (gradient-in-forward trick)
- gradient_formulas:
  - d_input: already computed in forward pass and stored in _input tensor. Backward just scales by grad_output.

## Tiling Strategy
- grid_dimensions: 1D
- grid_description: one program per row (one row = one token's logits over vocab)
- block_size_source: custom — iterates over vocab in chunks of BLOCK_SIZE
- needs_sm_parallelism: false

## Module Parameters
- module_init_params:
  - weight: optional class weights
  - ignore_index: int = -100
  - lse_square_scale: float = 0.0
  - label_smoothing: float = 0.0
  - reduction: str = "mean"
  - softcap: float or None = None
  - return_z_loss: bool = False
- learnable_params: none

## Benchmarking
- benchmark_variable: vocab_size (V)
- benchmark_x_label: "V"
- benchmark_x_values_suggestion: [4096, 8192, 16384, 32768, 65536, 131072]
- benchmark_providers: ["liger", "huggingface"]
- benchmark_fixed_config: {B: 8, T: 512, dtype: torch.bfloat16}

## Key Patterns

- **Online softmax**: Two-pass algorithm. Pass 1: compute running max and logsumexp. Pass 2: compute softmax and gradients. Avoids materializing the full softmax vector.
- **Gradient-in-forward trick**: The forward kernel computes the gradient and stores it directly in `_input` (overwriting logits). The backward pass just retrieves this and multiplies by `grad_output`. This saves having to recompute softmax in backward.
- **Constexpr flags for code elimination**: `HAS_WEIGHT`, `HAS_SOFTCAPPING`, `HAS_GRADIENTS`, `RETURN_Z_LOSS`, `RETURN_TOKEN_ACCURACY` — each is `tl.constexpr`, so the compiler removes unused code paths entirely.
- **Chunked vocab iteration**: The kernel loops over vocabulary in `BLOCK_SIZE` chunks: `for i in range(0, n_cols, BLOCK_SIZE)`. This handles arbitrarily large vocabularies without requiring BLOCK_SIZE >= n_cols.
- **Multiple loss components**: Combines original CE loss, label smoothing loss, and z-loss (for training stability). Each component contributes to both loss and gradient.
