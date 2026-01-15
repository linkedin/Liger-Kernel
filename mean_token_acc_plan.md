# Mean Token Accuracy Without Materializing Logits - Implementation Plan

## Overview

Add mean_token_accuracy calculation to TRL's SFTTrainer when using Liger-Kernel, without materializing logits, following the approach documented in `ACCURACY_WITHOUT_MATERIALIZATION.md`.

## Integration Pattern Analysis

### How Liger Cross Entropy is Currently Used in Transformers

**Integration Flow:**

1. **transformers/trainer.py:492-513** - When `use_liger_kernel=True`, calls `_apply_liger_kernel_to_instance(model, **kernel_config)`

2. **Liger monkey_patch.py** - Patches the model's forward method to use Liger kernels

3. **Model Forward Pass (e.g., llama.py:149-270)** - The patched forward:
   - Computes hidden states
   - When `skip_logits=True` (training with labels), uses fused linear cross entropy
   - Calls `LigerForCausalLMLoss` → `fixed_fused_linear_cross_entropy` → `F.liger_fused_linear_cross_entropy`
   - Returns `CausalLMOutputWithPast(loss=loss, logits=None, ...)`

4. **Key insight**: During training, `logits=None` in the output when using Liger!

### Current API Signature

```python
# functional.py
def liger_fused_linear_cross_entropy(
    input, weight, target, bias=None, ce_weight=None,
    ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0,
    reduction="mean", softcap=None,
    return_z_loss=False,  # ← Returns (loss, z_loss) when True
    accum_dtype=None, use_token_scaling=False
)
```

## Optimal API Design for Accuracy Extension

### Recommended API Design: Follow the `return_z_loss` Pattern

Based on the existing pattern, here's the optimal API:

```python
# 1. Kernel Level (cross_entropy.py & fused_linear_cross_entropy.py)
def liger_cross_entropy_kernel(
    ...,
    accuracy_ptr,  # NEW: pointer to accuracy output
    RETURN_ACCURACY: tl.constexpr,  # NEW: compile-time flag
    ...
)

# 2. Forward Functions
def cross_entropy_forward(..., return_accuracy=False):
    accuracy_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device) if return_accuracy else None
    # ... kernel launch with accuracy_ptr
    return loss, z_loss, accuracy  # NEW: return accuracy (None if not requested)

# 3. Autograd Functions
class LigerCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ..., return_accuracy: bool = False):
        loss, z_loss, accuracy = cross_entropy_forward(..., return_accuracy)
        return loss, z_loss, accuracy  # NEW: 3-tuple return

# 4. Functional API (functional.py)
def liger_fused_linear_cross_entropy(
    input, weight, target, ...,
    return_z_loss: bool = False,
    return_accuracy: bool = False,  # NEW: parallel to return_z_loss
    ...
):
    loss, z_loss, accuracy = LigerFusedLinearCrossEntropyFunction.apply(
        ..., return_z_loss, return_accuracy
    )

    # Return based on what was requested
    if not return_z_loss and not return_accuracy:
        return loss

    result = [loss]
    if return_z_loss:
        result.append(z_loss)
    if return_accuracy:
        result.append(accuracy)
    return tuple(result) if len(result) > 1 else result[0]
```

### Why This Design is Optimal

1. **Consistent with existing patterns**: Mirrors `return_z_loss` behavior exactly

2. **Backward compatible**: Default `return_accuracy=False` maintains current behavior

3. **Flexible return**: Can request loss only, loss+z_loss, loss+accuracy, or all three

4. **Minimal code changes**: Reuses existing kernel infrastructure

5. **Type-safe**: Clear return signature based on flags

### Model Integration

```python
# loss_utils.py - Update LigerForCausalLMLoss
def LigerForCausalLMLoss(
    hidden_states, lm_head_weight, labels, ...,
    return_accuracy: bool = False,  # NEW
    **kwargs
):
    result = fixed_fused_linear_cross_entropy(
        hidden_states, lm_head_weight, shift_labels,
        num_items_in_batch, ignore_index, final_logit_softcapping,
        return_accuracy=return_accuracy,  # NEW: pass through
        **kwargs
    )
    # Return based on what was requested
    return result
```

### Model Forward Pass

```python
# llama.py - Update lce_forward
def lce_forward(self, ..., return_accuracy: bool = False, **kwargs):
    ...
    if skip_logits:
        result = lce_maybe_trainable_lm_head(
            self, hidden_states=kept_hidden_states, ...,
            return_accuracy=return_accuracy,  # NEW
            **kwargs
        )
        # Unpack based on what was returned
        if return_accuracy:
            loss, accuracy = result if isinstance(result, tuple) else (result, None)
        else:
            loss = result
            accuracy = None
    ...

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        accuracy=accuracy,  # NEW field (None if not requested)
        ...
    )
```

### TRL SFTTrainer Integration

```python
# sft_trainer.py - Update compute_loss
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    mode = "train" if self.model.training else "eval"
    labels = inputs["labels"]

    # NEW: Request accuracy when using Liger
    if self.args.use_liger_kernel and self.args.return_accuracy:
        inputs["return_accuracy"] = True

    inputs["use_cache"] = False
    (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True, ...)

    # NEW: Extract accuracy if available
    if hasattr(outputs, 'accuracy') and outputs.accuracy is not None:
        with torch.no_grad():
            accuracy = self.accelerator.gather_for_metrics(outputs.accuracy).mean().item()
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
    elif not self.args.use_liger_kernel:
        # Original argmax-based accuracy calculation
        ...
```

### Configuration

```python
# sft_config.py
@dataclass
class SFTConfig:
    ...
    return_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether to compute token accuracy without materializing logits when using Liger kernel."}
    )
```

## Key Advantages of This API

1. **Zero overhead when disabled**: No performance impact if `return_accuracy=False`

2. **Progressive adoption**: Can be enabled model-by-model

3. **Clear semantics**: `return_accuracy=True` → get accuracy tensor back

4. **Composable**: Works with `return_z_loss`, `use_token_scaling`, etc.

5. **No breaking changes**: Fully backward compatible

6. **Natural extension point**: Easy to add more metrics later (e.g., `return_entropy`, `return_top_k_accuracy`)

## Implementation Phases

### Phase 1: Extend Kernel to Track Argmax and Return Accuracy Tensor

**Files to modify:**
- `src/liger_kernel/ops/cross_entropy.py`
- `src/liger_kernel/ops/fused_linear_cross_entropy.py`
- Add tests: `test/ops/test_cross_entropy_with_accuracy.py`

**Tasks:**
1. Modify `liger_cross_entropy_kernel` to:
   - Add `accuracy_ptr` parameter
   - Add `RETURN_ACCURACY: tl.constexpr` flag
   - Track argmax during max-finding loop (lines 108-139)
   - Store per-token accuracy after computing loss

2. Update `cross_entropy_forward` function:
   - Add `return_accuracy` parameter (default `False`)
   - Allocate `accuracy_1d` tensor when `return_accuracy=True`
   - Pass `accuracy_ptr` to kernel launch
   - Return accuracy tensor alongside loss and z_loss

3. Update `LigerCrossEntropyFunction` class:
   - Add `return_accuracy` parameter to `forward` method
   - Update return signature to include accuracy: `(loss, z_loss, accuracy)`
   - Update `backward` to handle additional return value

4. Apply same changes to `fused_linear_cross_entropy.py`

5. Add comprehensive tests verifying:
   - Accuracy matches naive argmax approach
   - Works with all existing features (label smoothing, z-loss, weight, etc.)
   - Backward compatibility (default behavior unchanged)
   - Memory efficiency (no logits materialization)

### Phase 2: Update Functional API

**Files to modify:**
- `src/liger_kernel/transformers/functional.py`
- `src/liger_kernel/transformers/cross_entropy.py` (if needed)
- Add tests

**Tasks:**
1. Update `liger_fused_linear_cross_entropy` to accept `return_accuracy` parameter
2. Update `liger_cross_entropy` to accept `return_accuracy` parameter
3. Handle flexible return based on flags (loss only, loss+z_loss, loss+accuracy, all three)
4. Add tests for all return combinations

### Phase 3: Thread `return_accuracy` Through Model Forwards

**Files to modify:**
- `src/liger_kernel/transformers/model/loss_utils.py`
- `src/liger_kernel/transformers/model/llama.py`
- Other model files (gemma, mistral, etc.)
- Add tests

**Tasks:**
1. Update `LigerForCausalLMLoss` to accept and pass `return_accuracy`
2. Update `fixed_fused_linear_cross_entropy` to accept and pass `return_accuracy`
3. Update model forward functions to accept `return_accuracy` and handle return values
4. Add accuracy field to `CausalLMOutputWithPast` (or use dict for flexibility)
5. Test with actual model instances

### Phase 4: Update TRL to Request and Log Accuracy

**Files to modify:**
- `/mnt/home/kashif/trl/trl/trainer/sft_trainer.py`
- `/mnt/home/kashif/trl/trl/trainer/sft_config.py`
- Add tests

**Tasks:**
1. Add `return_accuracy` configuration flag to `SFTConfig`
2. Update `compute_loss` to request accuracy when using Liger
3. Extract and log accuracy from model outputs
4. Keep fallback to argmax-based calculation when not using Liger
5. Test with full training loop

### Phase 5: Configuration Flags and Documentation

**Files to modify:**
- Documentation files
- README updates
- Example scripts

**Tasks:**
1. Document the new feature
2. Add usage examples
3. Update benchmarks showing memory savings
4. Add to model cards and training guides

## Technical Details

### Kernel Implementation

The accuracy calculation reuses the existing max-finding logic in the kernel:

```python
# During the max-finding pass for softmax
m = float("-inf")
argmax_idx = 0  # NEW: track argmax

for i in range(0, n_cols, BLOCK_SIZE):
    X_block = tl.load(...)
    block_max = tl.max(X_block)

    # NEW: Track which index has the global max
    if RETURN_ACCURACY and block_max > m:
        is_max_mask = X_block == block_max
        masked_offsets = tl.where(is_max_mask, X_offsets, n_cols)
        argmax_idx = tl.min(masked_offsets)

    m = tl.maximum(m, block_max)

# NEW: Store per-token accuracy
if RETURN_ACCURACY:
    is_correct = 1.0 if argmax_idx == y else 0.0
    tl.store(accuracy_ptr + program_id * accuracy_stride, is_correct)
```

### Memory Savings

For Llama-3-8B with vocab_size=128k, batch_size=4, seq_len=2048:
- Traditional: 4GB for logits tensor
- Liger with accuracy: 0.031 MB for accuracy tensor
- **Savings: ~4GB (99.999% reduction)**

### Backward Compatibility

All changes use optional parameters with safe defaults:
- `return_accuracy=False` → existing behavior unchanged
- No performance impact when disabled
- Gradual adoption without breaking existing code

## Benefits

- ✅ **Massive memory savings** for large vocab models
- ✅ **No accuracy loss** - identical results to naive approach
- ✅ **Minimal overhead** - computed during existing kernel pass
- ✅ **Works with all Liger features**: label smoothing, z-loss, weight, etc.
- ✅ **Backward compatible** - can toggle on/off with `return_accuracy` flag
- ✅ **Consistent API** - follows existing `return_z_loss` pattern

## Future Extensions

This pattern can be extended to other metrics without materialization:

- **Top-k accuracy**: Track top-k indices during max-finding
- **Per-class accuracy**: Use `ce_weight` with accuracy
- **Confidence scores**: Return `max(softmax(logits))`
- **Entropy**: Compute during softmax pass
