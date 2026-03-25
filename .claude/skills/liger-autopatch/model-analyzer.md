# Model Analyzer Agent

Reads a HuggingFace model's source code and produces a structured model profile.

## Input

A model identifier: model_type string (e.g., `"nemotron"`), class name (e.g., `"NemotronForCausalLM"`), or Hub model name.

## Steps

1. Locate `transformers/models/{model_type}/modeling_{model_type}.py` and `configuration_{model_type}.py` on disk
2. Read [decision-matrix.md](decision-matrix.md) for the 12 decisions to resolve
3. Read the modeling file and extract all architectural details
4. Produce the profile in the format below

## What to Extract

From the modeling file, identify:

- `*ForCausalLM`, `*Model`, `*DecoderLayer`, `*RMSNorm`/norm, `*MLP`, `*Expert`/`*MoE` classes
- `ForCausalLM.forward` signature — note all parameters
- Norm class `forward` — determine casting mode and offset
- All norm attribute names on decoder layer and base model
- MLP activation function and gate/up/down projection names
- Config's `model_type` string, `final_logit_softcapping`, `pretraining_tp`

## Output Format

```markdown
# Model Profile: {ModelName}

## Identity
- model_type: {from config}
- causal_lm_class: {e.g., NemotronForCausalLM}
- base_model_class: {e.g., NemotronModel}
- base_model_prefix: {e.g., "model"}
- modeling_module: transformers.models.{model_type}.modeling_{model_type}

## Normalization
- norm_type: RMSNorm | LayerNorm | both
- casting_mode: llama | gemma | none
- offset: 0.0 | 1.0
- in_place: true | false
- final_norm_attr: {e.g., "model.norm"}
- decoder_norm_attrs: {list}
- attn_norm_attrs: {e.g., "self_attn.q_norm" or "none"}

## MLP
- activation: silu | gelu | gelu_new
- liger_mlp_class: LigerSwiGLUMLP | LigerGEGLUMLP | LigerPhi3SwiGLUMLP

## Structure
- type: dense | moe | hybrid
- moe_expert_class: {if MoE}
- shared_expert: true | false

## Positional Embedding
- rope_type: standard | llama4 | qwen2vl_mrope | none

## Output
- output_class: LigerCausalLMOutputWithPast | LigerMoeCausalLMOutputWithPast | custom
- hidden_state_access: outputs[0] | outputs.last_hidden_state
- has_logit_softcapping: true | false

## Vision
- has_vision: true | false
- vision_norm_type: {if applicable}

## Forward Signature
{List model-specific parameters not in standard Llama signature}

## Mini Model Config
{Suggest minimal config values for testing — 2 layers, 32 hidden, 64 intermediate}
```

## Rules

- Never guess — always read the source to determine each value
- Read [examples/llama-profile.md](examples/llama-profile.md) and [examples/gemma-profile.md](examples/gemma-profile.md) as reference
- Compare against existing supported models to find the closest match
