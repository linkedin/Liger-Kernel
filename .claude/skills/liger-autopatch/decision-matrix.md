# Decision Matrix

When analyzing a HuggingFace model for Liger Kernel support, you must resolve these 12 architectural decisions by reading the model's `modeling_*.py` source code.

## 1. Norm Type

**Question:** Does the model use RMSNorm, LayerNorm, or both?

**How to detect:**
- Search for `class *RMSNorm` in the modeling file → RMSNorm
- Search for `nn.LayerNorm` usage → LayerNorm
- Multimodal models often use both (RMSNorm for text, LayerNorm for vision)

**Liger mapping:** `LigerRMSNorm` or `LigerLayerNorm`

## 2. RMSNorm Casting Mode

**Question:** How does the model handle dtype casting during normalization?

**How to detect:** Read the RMSNorm forward method:
- Casts input to fp32, computes variance, casts back → `"gemma"`
- Computes variance in fp32 only (input stays original dtype) → `"llama"`
- No casting at all → `"none"`

**Default:** `"llama"` (most common)

## 3. RMSNorm Offset

**Question:** Does the weight have a +1.0 offset?

**How to detect:** In the RMSNorm forward, look for `(1 + self.weight)` or `self.weight + 1`:
- Present → `offset=1.0` (Gemma family)
- Absent → `offset=0.0` (most models)

## 4. RMSNorm In-Place

**Question:** Can the backward pass modify dY in-place?

**How to detect:** Check if the model has two sequential norm layers with a residual connection between them (like Gemma2's `pre_feedforward_layernorm` + `post_feedforward_layernorm`):
- Sequential norms with residual → `in_place=False`
- Otherwise → `in_place=True`

## 5. MLP Activation Type

**Question:** What activation function does the gated MLP use?

**How to detect:** Read the MLP class forward method:
- `silu` or `F.silu` → SwiGLU → `LigerSwiGLUMLP`
- `gelu` or `gelu_new` or `gelu_fast` → GeGLU → `LigerGEGLUMLP`
- Phi3-style (single gate+up projection split) → `LigerPhi3SwiGLUMLP`

**Also check:** The config's `hidden_act` field.

## 6. Dense vs MoE

**Question:** Is the model dense, MoE, or hybrid (some layers dense, some MoE)?

**How to detect:**
- Search for `Expert`, `MoE`, `SparseMoe`, `TopK` routing classes
- Check if decoder layers have a `block_sparse_moe` or `experts` attribute
- Hybrid: check for `is_moe_layer` or conditional MoE per-layer

**Liger mapping:**
- Dense → standard patching
- MoE (transformers v5) → `LigerExperts`
- MoE (transformers v4) → `LigerBlockSparseTop2MLP`
- Qwen3-style MoE → `LigerQwen3MoeSwiGLUMLP`

## 7. Vision Components

**Question:** Does the model have a vision encoder?

**How to detect:**
- Check for `pixel_values` in the `forward` signature
- Look for a separate vision model class (e.g., `*VisionModel`)
- Check config for `vision_config` or `text_config` sub-configs

**If yes:**
- Vision encoder norms are usually `nn.LayerNorm` → patch with `LigerLayerNorm`
- Text and vision must be patched separately

## 8. RoPE Variant

**Question:** What type of positional embedding does the model use?

**How to detect:** Search for the `apply_rotary_pos_emb` function:
- Standard (q, k, cos, sin) → `liger_rotary_pos_emb` (rope=True)
- Llama4-style → `liger_llama4_text_rotary_pos_emb`
- Qwen2VL MRoPE → `liger_multimodal_rotary_pos_emb`
- No rotary embedding or custom variant → `rope=False`

## 9. Output Class

**Question:** What return type does the model's ForCausalLM.forward use?

**How to detect:** Read the return statement and type annotation:
- Standard → `LigerCausalLMOutputWithPast`
- MoE (has `aux_loss`) → `LigerMoeCausalLMOutputWithPast`
- Custom VL output → create model-specific output class in `output_classes.py`

## 10. Hidden State Access

**Question:** How does the model access hidden states from base model output?

**How to detect:** In the ForCausalLM.forward, after calling `self.model(...)`:
- `outputs[0]` → most models (Llama, Mistral, Gemma, etc.)
- `outputs.last_hidden_state` → Phi3, Qwen3.5 MoE, some newer models

## 11. Logit Softcapping

**Question:** Does the model apply softcapping to logits before loss?

**How to detect:** Check config for `final_logit_softcapping`:
- Present → pass `final_logit_softcapping=self.config.final_logit_softcapping` to `LigerForCausalLMLoss`
- Absent → no softcapping (most models)

**Models with softcapping:** Gemma2, Gemma3

## 12. Decoder Layer Norm Names

**Question:** What are the attribute names of norm layers in each decoder layer?

**How to detect:** Read the decoder layer class `__init__`:
- Standard: `input_layernorm`, `post_attention_layernorm`
- Gemma2 extra: `pre_feedforward_layernorm`, `post_feedforward_layernorm`
- GLM4: `post_self_attn_layernorm`, `post_mlp_layernorm`
- Some models: `q_norm`, `k_norm` on self_attn

Also check the final norm on the base model (usually `model.norm` or `model.final_layernorm`).
