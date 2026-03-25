# Template: lce_forward for MoE Models

Reference implementation: `src/liger_kernel/transformers/model/mixtral.py`. Read it first.

## Differences from Dense

1. **Output class**: `LigerMoeCausalLMOutputWithPast` (includes `aux_loss`)
2. **Extra param**: `output_router_logits` in forward signature
3. **Auxiliary loss**: Compute `load_balancing_loss_func` from router logits
4. **Return**: Include `aux_loss` and `router_logits` in output

## Auxiliary Loss Pattern

```python
aux_loss = None
if output_router_logits:
    router_logits = outputs.router_logits if return_dict else outputs[-1]
    aux_loss = load_balancing_loss_func(
        router_logits, self.num_experts, self.num_experts_per_tok, attention_mask,
    )
    if labels is not None and loss is not None:
        loss += self.router_aux_loss_coef * aux_loss
```

## Monkey Patch Differences for MoE

Expert patching is version-aware:

```python
# Transformers v5
modeling_module.{ModelExperts} = LigerExperts
# Instance: _patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)

# Transformers v4
modeling_module.{ModelExpertMLP} = LigerBlockSparseTop2MLP
# Instance: for expert in decoder_layer.block_sparse_moe.experts: _patch_swiglu_module(...)
```

Check exact attribute names in the HF decoder layer class.
