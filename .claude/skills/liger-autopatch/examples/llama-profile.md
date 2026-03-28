# Model Profile: Llama

This is the reference profile for the simplest and most common model type.

## Identity
- model_type: llama
- causal_lm_class: LlamaForCausalLM
- base_model_class: LlamaModel
- base_model_prefix: "model"
- modeling_module: transformers.models.llama.modeling_llama
- config_module: transformers.models.llama.configuration_llama

## Normalization
- norm_class: LlamaRMSNorm
- norm_type: RMSNorm
- casting_mode: llama
- offset: 0.0
- in_place: true
- final_norm_attr: model.norm
- decoder_norm_attrs:
  - input_layernorm
  - post_attention_layernorm
- attn_norm_attrs: none

## MLP
- mlp_class: LlamaMLP
- activation: silu
- liger_mlp_class: LigerSwiGLUMLP
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
Standard parameters only:
- input_ids, attention_mask, position_ids, past_key_values
- inputs_embeds, labels, use_cache
- output_attentions, output_hidden_states, return_dict
- cache_position, logits_to_keep, skip_logits
- **kwargs

No model-specific extra parameters.

## Mini Model Config
```python
LlamaConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    vocab_size=1024,
    rms_norm_eps=1e-5,
)
```

## Notes
- Llama is the base reference for all dense LLaMA-family models
- Most models (Mistral, Qwen2, OLMo, etc.) follow this exact pattern
- The `lce_forward` for Llama reuses `lce_maybe_trainable_lm_head` for PEFT/FSDP support
- Has `pretraining_tp` check (raises error if > 1)
