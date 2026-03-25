# Template: apply_liger_kernel_to_{model_type}

Use this to generate the patching function in `monkey_patch.py`.

## Reference

Read `apply_liger_kernel_to_llama` (dense) or `apply_liger_kernel_to_mixtral` (MoE) in `monkey_patch.py` as reference.

## Function Signature

```python
def apply_liger_kernel_to_{model_type}(
    rope: bool = {True if model has standard RoPE, False otherwise},
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = {True if model uses RMSNorm},
    {swiglu or geglu}: bool = True,
    {layer_norm: bool = True,}  # only for multimodal models
    model: PreTrainedModel = None,
) -> None:
```

## Function Body Structure

```python
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.{model_type} import modeling_{model_type}

    # Class-level patching (when model=None)
    if rope:
        modeling_{model_type}.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_{model_type}.{ModelRMSNorm} = LigerRMSNorm
    if {swiglu/geglu}:
        modeling_{model_type}.{ModelMLP} = {LigerSwiGLUMLP/LigerGEGLUMLP}
    if cross_entropy:
        modeling_{model_type}.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_{model_type}.{ModelForCausalLM}.forward = {model_type}_lce_forward

    # Instance-level patching (when model is provided)
    if model is not None:
        return _apply_liger_kernel_to_{model_type}_instance(
            model=model,
            rope=rope,
            cross_entropy=cross_entropy,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            rms_norm=rms_norm,
            {swiglu/geglu}={swiglu/geglu},
        )
```

## Instance Patching Pattern

```python
    if model is not None:
        base_model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(
                base_model.norm,
                # For Gemma: offset=1.0, casting_mode="gemma"
                # For Gemma2: offset=1.0, casting_mode="gemma", in_place=False
                # For most: use defaults
            )

        for decoder_layer in base_model.layers:
            if {swiglu}:
                _patch_swiglu_module(decoder_layer.mlp, {LigerMLPClass})
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)

        if fused_linear_cross_entropy:
            from types import MethodType
            model.forward = MethodType({model_type}_lce_forward, model)
```

## Variations

### Gemma-family (offset + casting_mode)
```python
_patch_rms_norm_module_for_{model} = partial(
    _patch_rms_norm_module, casting_mode="gemma", offset=1.0
)
```

### MoE expert patching
```python
if IS_TRANSFORMERS_V5_OR_LATER:
    _patch_swiglu_module(decoder_layer.mlp.experts, LigerExperts)
else:
    for expert in decoder_layer.block_sparse_moe.experts:
        _patch_swiglu_module(expert, LigerBlockSparseTop2MLP)
```

### Additional attention norms (Gemma3, Qwen3-VL)
```python
if rms_norm:
    _patch_rms_norm_module(decoder_layer.self_attn.q_norm)
    _patch_rms_norm_module(decoder_layer.self_attn.k_norm)
```

### Multimodal vision patching
```python
if layer_norm and model is None:
    modeling_{model_type}.nn.LayerNorm = LigerLayerNorm

# Instance patching for vision:
if layer_norm and model is not None:
    for vision_layer in vision_model.encoder.layers:
        _patch_layer_norm_module(vision_layer.layer_norm1)
        _patch_layer_norm_module(vision_layer.layer_norm2)
```
