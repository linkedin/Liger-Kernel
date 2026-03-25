# Template: lce_forward for MoE Models

Use this for models with Mixture-of-Experts architecture (e.g., Mixtral, Qwen3-MoE).

## Reference

The canonical MoE implementation is `src/liger_kernel/transformers/model/mixtral.py`. Read it before generating.

## Key Differences from Dense

1. **Output class**: Use `LigerMoeCausalLMOutputWithPast` (includes `aux_loss`)
2. **Router logits**: Add `output_router_logits` parameter to forward signature
3. **Auxiliary loss**: Compute `load_balancing_loss_func` from router logits
4. **Return fields**: Include `aux_loss` and `router_logits` in output

## Additional Imports

```python
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast
```

And the load balancing loss (version-aware):
```python
from transformers.models.{model_type}.modeling_{model_type} import load_balancing_loss_func
```

## Auxiliary Loss Pattern

After computing the main loss, add:

```python
aux_loss = None
if output_router_logits:
    router_logits = outputs.router_logits if return_dict else outputs[-1]
    aux_loss = load_balancing_loss_func(
        router_logits,
        self.num_experts,
        self.num_experts_per_tok,
        attention_mask,
    )
    if labels is not None and loss is not None:
        loss += self.router_aux_loss_coef * aux_loss
```

## Return

```python
return LigerMoeCausalLMOutputWithPast(
    loss=loss,
    aux_loss=aux_loss,
    logits=logits,
    past_key_values=outputs.past_key_values,
    hidden_states=outputs.hidden_states,
    attentions=outputs.attentions,
    router_logits=outputs.router_logits,
    token_accuracy=token_accuracy,
    predicted_tokens=predicted_tokens,
)
```

## Monkey Patch Differences

For MoE models, the SwiGLU patching targets expert modules:

```python
# Transformers v5
if IS_TRANSFORMERS_V5_OR_LATER:
    modeling_module.{ModelExperts} = LigerExperts
else:
    modeling_module.{ModelExpertMLP} = LigerBlockSparseTop2MLP

# Instance patching (v5)
_patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)

# Instance patching (v4)
for expert in decoder_layer.block_sparse_moe.experts:
    _patch_swiglu_module(expert, LigerBlockSparseTop2MLP)
```

Check the exact attribute names by reading the HF decoder layer class.
