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

Class-level patching always runs (even when `model` is provided). Instance-level patching runs additionally when `model is not None`. They are NOT mutually exclusive — this matches the existing Llama/Mistral/Gemma patterns.

The only exception is `fused_linear_cross_entropy`: use `MethodType` binding when instance is given, class replacement otherwise.

```python
assert not (cross_entropy and fused_linear_cross_entropy), "..."

from transformers.models.{model_type} import modeling_{model_type}

# Class-level patching (always runs)
if rope: modeling_{model_type}.apply_rotary_pos_emb = liger_rotary_pos_emb
if rms_norm: modeling_{model_type}.{ModelRMSNorm} = LigerRMSNorm
if swiglu: modeling_{model_type}.{ModelMLP} = LigerSwiGLUMLP
if cross_entropy: modeling_{model_type}.CrossEntropyLoss = LigerCrossEntropyLoss
if fused_linear_cross_entropy:
    if model is not None:
        model.forward = MethodType({model_type}_lce_forward, model)
    else:
        modeling_{model_type}.{ModelForCausalLM}.forward = {model_type}_lce_forward

# Instance-level patching (only when model instance provided)
if model is not None:
    base_model = getattr(model, model.base_model_prefix, model)
    if rms_norm: _patch_rms_norm_module(base_model.norm)
    for decoder_layer in base_model.layers:
        if swiglu: _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
        if rms_norm:
            _patch_rms_norm_module(decoder_layer.input_layernorm)
            _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
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

**Multimodal VL instance patching:**

For VL models, extract text_model and vision_model from the composite instance using `isinstance` checks. Each VL model has different accessor paths — read the existing VL patches in `monkey_patch.py` for the exact pattern. Example from Llama4:

```python
if model is not None:
    if isinstance(model, {Model}ForConditionalGeneration):
        language_model = model.language_model          # or model.model.language_model
        vision_model = model.vision_model              # or model.model.visual
        text_model = language_model.model
    elif isinstance(model, {Model}ForCausalLM):
        text_model = model.model
        vision_model = None
    elif isinstance(model, {Model}TextModel):
        text_model = model
        vision_model = None
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Patch text layers
    if text_model:
        if rms_norm: _patch_rms_norm_module(text_model.norm)
        for decoder_layer in text_model.layers:
            ...

    # Patch vision layers
    if vision_model:
        for layer in vision_model.encoder.layers:  # accessor varies by model
            if layer_norm:
                _patch_layer_norm_module(layer.layer_norm1)
                _patch_layer_norm_module(layer.layer_norm2)
```

The exact accessor paths (`model.language_model`, `model.model.language_model`, `model.model.visual`) vary per VL model. Always read the HF model's `__init__` to find the correct paths.
