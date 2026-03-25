# Template: apply_liger_kernel_to_{model_type}

Reference: `apply_liger_kernel_to_llama` (dense) or `apply_liger_kernel_to_mixtral` (MoE) in `monkey_patch.py`.

## Function Signature

```python
def apply_liger_kernel_to_{model_type}(
    rope: bool = True,              # False if model lacks standard RoPE
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,          # Or layer_norm for LayerNorm models
    swiglu: bool = True,            # Or geglu for Gemma-family
    model: PreTrainedModel = None,
) -> None:
```

## Body Pattern

```python
assert not (cross_entropy and fused_linear_cross_entropy), "..."

from transformers.models.{model_type} import modeling_{model_type}

# Class-level patching
if rope: modeling_{model_type}.apply_rotary_pos_emb = liger_rotary_pos_emb
if rms_norm: modeling_{model_type}.{ModelRMSNorm} = LigerRMSNorm
if swiglu: modeling_{model_type}.{ModelMLP} = LigerSwiGLUMLP
if cross_entropy: modeling_{model_type}.CrossEntropyLoss = LigerCrossEntropyLoss
if fused_linear_cross_entropy:
    modeling_{model_type}.{ModelForCausalLM}.forward = {model_type}_lce_forward

# Instance-level patching
if model is not None:
    base_model = getattr(model, model.base_model_prefix, model)
    if rms_norm: _patch_rms_norm_module(base_model.norm)
    for decoder_layer in base_model.layers:
        if swiglu: _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
        if rms_norm:
            _patch_rms_norm_module(decoder_layer.input_layernorm)
            _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
    if fused_linear_cross_entropy:
        model.forward = MethodType({model_type}_lce_forward, model)
```

## Common Variations

**Gemma-family** (offset + casting):
```python
_patch_rms_norm_module_for_model = partial(_patch_rms_norm_module, casting_mode="gemma", offset=1.0)
```

**MoE experts** (version-aware):
```python
if IS_TRANSFORMERS_V5_OR_LATER:
    _patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)
else:
    for expert in decoder_layer.block_sparse_moe.experts:
        _patch_swiglu_module(expert, LigerBlockSparseTop2MLP)
```

**Multimodal vision patching:**
```python
if layer_norm and model is None:
    modeling_{model_type}.nn.LayerNorm = LigerLayerNorm
```
