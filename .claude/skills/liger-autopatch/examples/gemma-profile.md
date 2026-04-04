# Model Profile: Gemma

This profile demonstrates the key differences from Llama: GeGLU activation, RMSNorm offset, and Gemma casting mode.

## Identity
- model_type: gemma
- causal_lm_class: GemmaForCausalLM
- base_model_class: GemmaModel
- base_model_prefix: "model"
- modeling_module: transformers.models.gemma.modeling_gemma
- config_module: transformers.models.gemma.configuration_gemma

## Normalization
- norm_class: GemmaRMSNorm
- norm_type: RMSNorm
- casting_mode: gemma (everything cast to fp32, then computed, then cast back)
- offset: 1.0 (weight uses `1 + self.weight` pattern)
- in_place: true
- final_norm_attr: model.norm
- decoder_norm_attrs:
  - input_layernorm
  - post_attention_layernorm
- attn_norm_attrs: none

## MLP
- mlp_class: GemmaMLP
- activation: gelu (uses GELU activation, not SiLU)
- liger_mlp_class: LigerGEGLUMLP
- gate_proj_attr: gate_proj
- up_proj_attr: up_proj
- down_proj_attr: down_proj

## Structure
- type: dense
- moe_expert_class: n/a
- moe_router_class: n/a
- shared_expert: false

## Positional Embedding
- rope_type: standard
- rope_function: apply_rotary_pos_emb

## Output
- output_class: LigerCausalLMOutputWithPast
- hidden_state_access: outputs[0]
- has_logit_softcapping: false
- softcapping_config_attr: none

## Vision
- has_vision: false

## Forward Signature
Same as Llama — no extra parameters.

## Mini Model Config
```python
GemmaConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    vocab_size=1024,
    rms_norm_eps=1e-6,
    hidden_activation="gelu_pytorch_tanh",
)
```

## Key Differences from Llama

1. **Activation**: Uses `geglu` parameter (not `swiglu`) in the patch function
2. **RMSNorm**: Requires `offset=1.0` and `casting_mode="gemma"`
3. **MLP class**: `LigerGEGLUMLP` instead of `LigerSwiGLUMLP`
4. **Patching uses partial**: `_patch_rms_norm_module_for_gemma = partial(_patch_rms_norm_module, casting_mode="gemma", offset=1.0)`

## Gemma2 Additional Differences
- `in_place: false` (residual between sequential norms)
- Extra norm layers: `pre_feedforward_layernorm`, `post_feedforward_layernorm`
- Has `final_logit_softcapping` in config
- Uses `LigerRMSNormForGemma2` variant
