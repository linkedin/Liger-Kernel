# Gemma 4 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `apply_liger_kernel_to_gemma4_text` and `apply_liger_kernel_to_gemma4` (multimodal) to Liger-Kernel so Google Gemma 4 text models can train with Liger's fused kernels — targeting execution on LUMI HPC (AMD MI250X / ROCm).

**Architecture:** Mirror the Gemma 3 port. Add a new Gemma-4-specific RMSNorm subclass (Gemma 4 does **not** use Gemma 3's `(1 + weight)` offset — it inherits from `Gemma3nRMSNorm`, which has `init="ones"` / `offset=0`). Swap `Gemma4TextMLP` → `LigerGEGLUMLP`, `Gemma4RMSNorm` → `LigerRMSNormForGemma4`, `apply_rotary_pos_emb` → `liger_rotary_pos_emb`, and `Gemma4ForCausalLM.forward` / `Gemma4ForConditionalGeneration.forward` → fused-linear-CE forwards. v1 scope: text + multimodal text-path; skip MoE (26B-A4B experts/router), skip custom `Gemma4VisionModel` internals, skip `Gemma4AudioModel`.

**Tech Stack:** Python 3.10+, PyTorch, Triton, HuggingFace Transformers ≥ 5.5.0, pytest, Liger's own kernel ops (`LigerRMSNormFunction`, `liger_rotary_pos_emb`, `LigerGEGLUMLP`, `LigerForCausalLMLoss`).

**Execution environment:** Implementation can be done on any machine. Convergence tests MUST run on LUMI (AMD ROCm) — Triton doesn't work on Apple Silicon / CPU-only systems.

---

## Critical architectural facts established before planning

These were derived from reading HF's `modular_gemma4.py` + `modular_gemma3n.py`:

1. **`Gemma4RMSNorm(Gemma3nRMSNorm)`** — `Gemma3nRMSNorm.__init__(dim, eps=1e-6, with_scale=True)`; forward does `x.float() * mean_sq.pow(-0.5) * self.weight.float()`, cast back to input dtype. **No `(1 + weight)` offset. Weights init to ones, not zeros.** This is DIFFERENT from `Gemma3RMSNorm`.
2. **`Gemma4TextMLP(Gemma3MLP)`** — same `gate/up/down` + `gelu_pytorch_tanh` as Gemma 3; only addition is conditional `intermediate_size *= 2` when `config.use_double_wide_mlp` and the layer is KV-shared. v1 accepts that patching with `LigerGEGLUMLP` loses the doubled width for those specific layers (document + guard behind `geglu` flag).
3. **`Gemma4TextDecoderLayer(Gemma3DecoderLayer)`** — same four norms (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`). Adds `per_layer_input_gate` + `per_layer_projection` (PLE) — module-level Liger patches don't interfere.
4. **`Gemma4TextAttention`** has `q_norm`/`k_norm` **only on non-shared layers** (`layer_idx < num_hidden_layers - num_kv_shared_layers`). Use `getattr(..., None)` when patching per-instance.
5. **`apply_rotary_pos_emb(..., cos, sin, unsqueeze_dim=2)`** is called at module level in `transformers.models.gemma4.modeling_gemma4`. `liger_rotary_pos_emb` is a drop-in replacement (Gemma3 already uses it this way). Proportional vs standard RoPE differs at init (`rope_init_fn`), not at apply.
6. **Multimodal uses `Gemma4VisionModel` (custom)** — not Siglip. Therefore we must not reuse Gemma 3's Siglip-layer-norm patching path.
7. **MoE path** uses `Gemma4TextExperts` + `Gemma4TextRouter` inside `Gemma4TextDecoderLayer` when `config.enable_moe_block=True`. **Out of scope for v1.** The dense decoder path still benefits from RMSNorm/MLP/rope/LCE.
8. **PLE (Per-Layer Embeddings)** adds a second embedding table and per-layer residual. Out of scope — not a kernel target, and won't interfere with our patches.

---

## File Structure

### New files

- `src/liger_kernel/transformers/model/gemma4.py`
  Fused-LCE forwards: `causal_forward` for `Gemma4ForCausalLM`, `multimodal_forward` for `Gemma4ForConditionalGeneration`. Reuses `LigerForCausalLMLoss`, `LigerCausalLMOutputWithPast`, `LigerGemma3CausalLMOutputWithPast` (Gemma 4 multimodal output mirrors Gemma 3's).

- `docs/superpowers/plans/2026-04-16-gemma4-support.md` (this file)

### Modified files

- `src/liger_kernel/transformers/rms_norm.py`
  Add `LigerRMSNormForGemma4(LigerRMSNorm)` with `offset=0.0`, `init_fn="ones"`, `casting_mode="gemma"`, `in_place=False`.

- `src/liger_kernel/transformers/monkey_patch.py`
  Add `apply_liger_kernel_to_gemma4_text(...)` and `apply_liger_kernel_to_gemma4(...)`. Register `"gemma4_text"` and `"gemma4"` in `MODEL_TYPE_TO_APPLY_LIGER_FN`.

- `src/liger_kernel/transformers/__init__.py`
  Add the two new `apply_liger_kernel_to_gemma4*` symbols to the `TYPE_CHECKING` block and the lazy-import machinery (follow existing `apply_liger_kernel_to_gemma3*` pattern).

- `test/utils.py`
  Add `revert_liger_kernel_to_gemma4_text` and `revert_liger_kernel_to_gemma4`.

- `test/convergence/bf16/test_mini_models.py`
  Add `GEMMA4_AVAILABLE` import guard, `mini_gemma4_text` `MINI_MODEL_SETUPS` entry, and `pytest.param("mini_gemma4_text", ...)` case with bf16 tolerances mirroring `mini_gemma3_text`.

- `setup.py`
  Bump dev-extras `transformers>=4.52.0` → `transformers>=5.5.0` (Gemma 4 added in 5.5.0). Note: this affects every model test; see Task 9 for how we handle that.

---

## Out-of-scope (explicit non-goals for v1)

- `Gemma4TextExperts` / `Gemma4TextRouter` MoE kernels (26B-A4B). The v1 patch should leave MoE layers untouched — they use their own non-GEGLU structure.
- `Gemma4VisionModel` / `Gemma4AudioModel` internal kernels. Only the LCE forward on `Gemma4ForConditionalGeneration` is patched.
- `use_double_wide_mlp=True` + KV-shared layer MLP. Documented as a known limitation — users must set `geglu=False` in the monkey-patch call if their model uses double-wide.
- Proportional RoPE correctness verification. `liger_rotary_pos_emb` is assumed to work because it's position-invariant to how cos/sin are initialized. v1 does not separately validate this — convergence test will catch divergence if it breaks.

---

## Task 1: Add `LigerRMSNormForGemma4` subclass

**Files:**
- Modify: `src/liger_kernel/transformers/rms_norm.py` (append after `LigerRMSNormForGemma3` at line 70)

- [ ] **Step 1.1: Read current `rms_norm.py` to confirm insertion point**

Run: Inspect `src/liger_kernel/transformers/rms_norm.py` lines 66–72 — verify `LigerRMSNormForGemma3` ends at line 70 and `LigerRMSNormForOlmo2` starts at line 73.

- [ ] **Step 1.2: Add the new class**

Insert after `LigerRMSNormForGemma3`:

```python
class LigerRMSNormForGemma4(LigerRMSNorm):
    """Gemma4RMSNorm inherits from Gemma3nRMSNorm, NOT from Gemma3RMSNorm.

    Differences from Gemma3 variant:
      - weight initialized to ones (not zeros)
      - no (1 + weight) offset — scales by weight directly
      - still uses fp32 compute (gemma casting mode)
      - with_scale=False is not supported by this Liger kernel path; callers
        must skip patching RMSNorms that were constructed with_scale=False.
    """

    def __init__(self, dim, eps=1e-6, offset=0.0, casting_mode="gemma", init_fn="ones", in_place=False):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)
```

- [ ] **Step 1.3: Commit**

```bash
git add src/liger_kernel/transformers/rms_norm.py
git commit -m "gemma4: add LigerRMSNormForGemma4 (ones init, no +1 offset)

Gemma4RMSNorm inherits Gemma3nRMSNorm, not Gemma3RMSNorm. The Gemma3n variant
initializes weight to torch.ones(dim) and does NOT apply the +1 offset. Using
LigerRMSNormForGemma3 here would give wrong outputs."
```

---

## Task 2: Scaffold `model/gemma4.py` (causal forward)

**Files:**
- Create: `src/liger_kernel/transformers/model/gemma4.py`

Gemma 4's causal forward has the same shape as Gemma 3's: same `final_logit_softcapping`, same `LigerForCausalLMLoss` flow, same `logits_to_keep` handling. We can near-duplicate `model/gemma3.py`'s `causal_forward`.

- [ ] **Step 2.1: Create file with causal_forward**

```python
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from transformers.cache_utils import Cache
from transformers.utils import logging

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast
from liger_kernel.transformers.model.output_classes import LigerGemma3CausalLMOutputWithPast

logger = logging.get_logger(__name__)


def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **loss_kwargs,
) -> Union[Tuple, LigerCausalLMOutputWithPast]:
    """Fused-linear-cross-entropy forward for Gemma4ForCausalLM.

    Mirrors liger's gemma3 causal_forward. Gemma 4 keeps
    final_logit_softcapping (may be None), so the same softcap branch works.
    """
    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma4 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with "
            "`AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **loss_kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]
    shift_labels = loss_kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            final_logit_softcapping=getattr(self.config, "final_logit_softcapping", None),
            **loss_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **loss_kwargs,
            )

    if not return_dict:
        output_tuple = (logits,) + outputs[1:]
        output_tuple = (loss,) + output_tuple if loss is not None else output_tuple
        output_tuple = output_tuple + (token_accuracy,) if token_accuracy is not None else output_tuple
        output_tuple = output_tuple + (predicted_tokens,) if predicted_tokens is not None else output_tuple
        return output_tuple

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
```

- [ ] **Step 2.2: Commit**

```bash
git add src/liger_kernel/transformers/model/gemma4.py
git commit -m "gemma4: scaffold causal_forward for Gemma4ForCausalLM

Mirrors model/gemma3.py's causal_forward. Uses getattr for
final_logit_softcapping (Gemma 4 may set it to None)."
```

---

## Task 3: Add `multimodal_forward` in `model/gemma4.py`

**Files:**
- Modify: `src/liger_kernel/transformers/model/gemma4.py`

Gemma 4 multimodal = text LM head fed from `self.model(...)` that merges vision/audio internally. The output container has `image_hidden_states`; reuse `LigerGemma3CausalLMOutputWithPast`.

- [ ] **Step 3.1: Append `multimodal_forward`**

Add `import torch.nn as nn` to the existing imports, then append:

```python
def multimodal_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **lm_kwargs,
) -> Union[tuple, LigerGemma3CausalLMOutputWithPast]:
    """Fused-linear-cross-entropy forward for Gemma4ForConditionalGeneration.

    Mirrors liger's gemma3 multimodal_forward. We do NOT pass pixel_values_videos
    or input_features here; HF accepts them via **lm_kwargs for the inner
    self.model(...) call to handle vision/audio fusion.
    """
    import torch.nn as nn

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **lm_kwargs,
    )

    shift_labels = lm_kwargs.pop("shift_labels", None)
    hidden_states = outputs[0]

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None
    if skip_logits and labels is None:
        raise ValueError("skip_logits is True, but labels is None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None)

    if skip_logits:
        shift_hidden_states = kept_hidden_states[..., :-1, :]
        shift_labels = labels[..., 1:]

        hidden_device = shift_hidden_states.device
        if attention_mask is not None:
            shift_attention_mask = attention_mask[:, -shift_hidden_states.shape[1] :].to(hidden_device)
            shift_hidden_states = shift_hidden_states[shift_attention_mask.to(hidden_device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
        else:
            shift_hidden_states = shift_hidden_states.contiguous()
            shift_labels = shift_labels.contiguous()

        shift_hidden_states = shift_hidden_states.view(-1, self.config.text_config.hidden_size)
        shift_labels = shift_labels.view(-1).to(hidden_device)

        result = LigerForCausalLMLoss(
            hidden_states=shift_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            shift_labels=shift_labels,
            final_logit_softcapping=getattr(self.config.text_config, "final_logit_softcapping", None),
            **lm_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        elif shift_labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = (loss,) + output if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerGemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=getattr(outputs, "image_hidden_states", None),
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
```

- [ ] **Step 3.2: Commit**

```bash
git add src/liger_kernel/transformers/model/gemma4.py
git commit -m "gemma4: add multimodal_forward for Gemma4ForConditionalGeneration

Uses getattr on image_hidden_states since Gemma4 output class may not always
populate it (video/audio-only prompts). Reuses LigerGemma3CausalLMOutputWithPast."
```

---

## Task 4: Add `apply_liger_kernel_to_gemma4_text` to `monkey_patch.py`

**Files:**
- Modify: `src/liger_kernel/transformers/monkey_patch.py` (insert after `apply_liger_kernel_to_gemma3` ends, currently around line 1239)

Key design choices captured in code comments:
- Use `getattr(decoder_layer.self_attn, "q_norm", None)` — KV-shared layers omit q_norm/k_norm.
- Register `LigerRMSNormForGemma4` at the class level (`modeling_gemma4.Gemma4RMSNorm = ...`) so both text-model RMSNorms and `q_norm`/`k_norm` get the right subclass.
- Swap `Gemma4TextMLP` for `LigerGEGLUMLP`. Document the `use_double_wide_mlp` caveat in the docstring.

- [ ] **Step 4.1: Add the text function**

Append after `apply_liger_kernel_to_gemma3`:

```python
def apply_liger_kernel_to_gemma4_text(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma4
    text models (Gemma4ForCausalLM / Gemma4TextModel).

    Limitations (v1):
      - MoE layers (`Gemma4TextExperts` / `Gemma4TextRouter`, enabled by
        `config.enable_moe_block`) are NOT patched — their dense MLP may still be
        patched but the MoE routing path is not.
      - `use_double_wide_mlp=True` combined with KV-shared layers is not fully
        supported by the GEGLU swap (the doubled intermediate size is lost when
        we replace `Gemma4TextMLP` with `LigerGEGLUMLP`). Pass `geglu=False` to
        keep HF's original MLP if your model uses double-wide.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default False.
        fused_linear_cross_entropy (bool): Fused linear CE for memory efficiency. Default True.
            Mutually exclusive with `cross_entropy`.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default True.
        model (PreTrainedModel): An already-instantiated model to patch in-place.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    from liger_kernel.transformers.model.gemma4 import causal_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma4

    # Gemma4RMSNorm uses ones-init, no +1 offset, fp32 compute.
    _patch_rms_norm_module_for_gemma4 = partial(
        _patch_rms_norm_module, offset=0.0, casting_mode="gemma", in_place=False, init_fn="ones"
    )

    if rope:
        modeling_gemma4.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_gemma4.Gemma4RMSNorm = LigerRMSNormForGemma4

    if geglu:
        modeling_gemma4.Gemma4TextMLP = LigerGEGLUMLP

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(causal_forward, model)
        else:
            modeling_gemma4.Gemma4ForCausalLM.forward = causal_forward

    if model is not None:
        if isinstance(model, Gemma4ForCausalLM) or isinstance(model, Gemma4TextModel):
            base_model = model.model if isinstance(model, Gemma4ForCausalLM) else model

            if rms_norm:
                _patch_rms_norm_module_for_gemma4(base_model.norm)

            for decoder_layer in base_model.layers:
                decoder_layer: Gemma4TextDecoderLayer
                if geglu and not getattr(decoder_layer, "enable_moe_block", False):
                    # Skip MLP rebind on MoE layers in v1.
                    _bind_method_to_module(decoder_layer.mlp, "forward", LigerGEGLUMLP.forward)
                if rms_norm:
                    _patch_rms_norm_module_for_gemma4(decoder_layer.input_layernorm)
                    _patch_rms_norm_module_for_gemma4(decoder_layer.post_attention_layernorm)
                    _patch_rms_norm_module_for_gemma4(decoder_layer.pre_feedforward_layernorm)
                    _patch_rms_norm_module_for_gemma4(decoder_layer.post_feedforward_layernorm)
                    # q_norm / k_norm are absent on KV-shared layers.
                    q_norm = getattr(decoder_layer.self_attn, "q_norm", None)
                    k_norm = getattr(decoder_layer.self_attn, "k_norm", None)
                    if q_norm is not None:
                        _patch_rms_norm_module_for_gemma4(q_norm)
                    if k_norm is not None:
                        _patch_rms_norm_module_for_gemma4(k_norm)

        else:
            raise TypeError("The model must be Gemma4ForCausalLM or Gemma4TextModel.")
```

- [ ] **Step 4.2: Verify `_patch_rms_norm_module` supports `init_fn` kwarg**

Run: Grep `src/liger_kernel/transformers/monkey_patch.py` for `def _patch_rms_norm_module` and inspect its signature.

Expected: if `init_fn` is not an accepted parameter, two options exist:
  (a) Extend `_patch_rms_norm_module` with an optional `init_fn` param forwarded to the new `LigerRMSNormForGemma4` constructor (preferred — unlocks future norms with non-zero init).
  (b) Drop the `init_fn="ones"` kwarg here and rely on `LigerRMSNormForGemma4`'s default (already `"ones"`); the helper must then not override the init.

Pick (b) — fewer cross-cutting changes. Remove `init_fn="ones"` from the `partial(...)` call in Step 4.1's code block if `_patch_rms_norm_module` does not accept it.

- [ ] **Step 4.3: Commit**

```bash
git add src/liger_kernel/transformers/monkey_patch.py
git commit -m "gemma4: add apply_liger_kernel_to_gemma4_text

Patches RMSNorm (all 6 norm locations + q_norm/k_norm where present),
GEGLU MLP, rotary, and fused-linear-CE on Gemma4ForCausalLM. Skips MoE
layers' MLPs when config.enable_moe_block is set (v1 scope)."
```

---

## Task 5: Add `apply_liger_kernel_to_gemma4` multimodal

**Files:**
- Modify: `src/liger_kernel/transformers/monkey_patch.py`

Gemma 4 multimodal does NOT use Siglip; it has its own `Gemma4VisionModel`. We therefore only patch: (a) the fused-linear-CE forward on `Gemma4ForConditionalGeneration`, (b) the text-language-model's RMSNorm / GEGLU / rope via nested `apply_liger_kernel_to_gemma4_text`, and (c) the `multi_modal_projector`'s RMSNorm if present.

- [ ] **Step 5.1: Append the function**

```python
def apply_liger_kernel_to_gemma4(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to Gemma4ForConditionalGeneration (multimodal).

    v1 scope:
      - Patches text-path (delegates to apply_liger_kernel_to_gemma4_text).
      - Patches multi_modal_projector RMSNorm when present.
      - Does NOT patch Gemma4VisionModel internals (custom vision tower; no
        Siglip dependency).
      - Does NOT patch Gemma4AudioModel.

    Args match apply_liger_kernel_to_gemma4_text; `layer_norm` is intentionally
    absent because there's no Siglip-style LayerNorm chain to swap.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    from liger_kernel.transformers.model.gemma4 import multimodal_forward

    _patch_rms_norm_module_for_gemma4 = partial(
        _patch_rms_norm_module, offset=0.0, casting_mode="gemma", in_place=False
    )

    apply_liger_kernel_to_gemma4_text(
        rope=rope,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=rms_norm,
        geglu=geglu,
    )

    if cross_entropy:
        modeling_gemma4.nn.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(multimodal_forward, model)
        else:
            modeling_gemma4.Gemma4ForConditionalGeneration.forward = multimodal_forward

    if model is not None:
        if isinstance(model, Gemma4ForConditionalGeneration):
            if rms_norm:
                mm_projector = getattr(model.model, "multi_modal_projector", None)
                if mm_projector is not None:
                    mm_soft_emb_norm = getattr(mm_projector, "mm_soft_emb_norm", None)
                    if mm_soft_emb_norm is not None:
                        _patch_rms_norm_module_for_gemma4(mm_soft_emb_norm)

            apply_liger_kernel_to_gemma4_text(
                rope=rope,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
                rms_norm=rms_norm,
                geglu=geglu,
                model=model.model.language_model,
            )
        else:
            raise TypeError("The model must be Gemma4ForConditionalGeneration.")
```

- [ ] **Step 5.2: Register both functions in `MODEL_TYPE_TO_APPLY_LIGER_FN`**

Locate `MODEL_TYPE_TO_APPLY_LIGER_FN = {` around line 3180. Insert `gemma4` entries next to `gemma3` entries:

```python
    "gemma3_text": apply_liger_kernel_to_gemma3_text,
    "gemma3": apply_liger_kernel_to_gemma3,
    "gemma4_text": apply_liger_kernel_to_gemma4_text,
    "gemma4": apply_liger_kernel_to_gemma4,
```

Verify the model_type strings. HF model_type is determined by the config `Gemma4TextConfig.model_type` / `Gemma4Config.model_type`. Inspect `transformers/models/gemma4/configuration_gemma4.py` on the HF side to confirm the exact strings — fix here if different.

- [ ] **Step 5.3: Commit**

```bash
git add src/liger_kernel/transformers/monkey_patch.py
git commit -m "gemma4: add apply_liger_kernel_to_gemma4 multimodal + model-type map

Gemma4 multimodal uses a custom Gemma4VisionModel (no Siglip), so we only
patch the text-path + fused-linear-CE + multi_modal_projector RMSNorm.
Registers gemma4 / gemma4_text in MODEL_TYPE_TO_APPLY_LIGER_FN."
```

---

## Task 6: Expose the new API in the top-level package

**Files:**
- Modify: `src/liger_kernel/transformers/__init__.py`

- [ ] **Step 6.1: Add to the `TYPE_CHECKING` block**

Insert after `apply_liger_kernel_to_gemma3`:

```python
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma4  # noqa: F401
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma4_text  # noqa: F401
```

- [ ] **Step 6.2: Check the lazy-import / `__getattr__` machinery**

The file has logic after line 80 that conditionally imports `monkey_patch` symbols based on `transformers` being installed. Inspect that section (read lines 80 to end) and add the two new `apply_liger_kernel_to_gemma4*` names wherever the existing `apply_liger_kernel_to_gemma3*` entries appear. Follow the exact same pattern (likely a list/set of names or similar).

- [ ] **Step 6.3: Commit**

```bash
git add src/liger_kernel/transformers/__init__.py
git commit -m "gemma4: export apply_liger_kernel_to_gemma4(_text) from package root"
```

---

## Task 7: Add revert helpers in `test/utils.py`

**Files:**
- Modify: `test/utils.py` (insert after `revert_liger_kernel_to_gemma3` at line 502)

- [ ] **Step 7.1: Add both revert functions**

Insert after `revert_liger_kernel_to_gemma3`:

```python
def revert_liger_kernel_to_gemma4_text(model_config: MiniModelConfig):
    """Revert all Liger kernel patches applied to Gemma4 text model."""

    from transformers.models.gemma4 import modeling_gemma4

    importlib.reload(modeling_gemma4)

    model_config.model_class = modeling_gemma4.Gemma4ForCausalLM

    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_gemma4(model_config: MiniModelConfig):
    """Revert all Liger kernel patches applied to Gemma4 multimodal model."""

    from transformers.models.gemma4 import modeling_gemma4

    importlib.reload(modeling_gemma4)

    model_config.model_class = modeling_gemma4.Gemma4ForConditionalGeneration
    print("Liger kernel patches have been reverted.")
```

- [ ] **Step 7.2: Commit**

```bash
git add test/utils.py
git commit -m "gemma4: add revert_liger_kernel_to_gemma4(_text) test helpers"
```

---

## Task 8: Add `mini_gemma4_text` convergence test

**Files:**
- Modify: `test/convergence/bf16/test_mini_models.py`

- [ ] **Step 8.1: Add availability guard**

Near the other `*_AVAILABLE` try/except blocks (around line 260–310), add:

```python
try:
    # Gemma4 is only available in transformers>=5.5.0
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    GEMMA4_AVAILABLE = True
except ImportError:
    GEMMA4_AVAILABLE = False
```

- [ ] **Step 8.2: Update the imports at the top of the file**

Add near the existing gemma3 imports:

```python
from liger_kernel.transformers import apply_liger_kernel_to_gemma4_text
```

and:

```python
from test.utils import revert_liger_kernel_to_gemma4_text
```

- [ ] **Step 8.3: Add MINI_MODEL_SETUPS entry**

Insert after the `mini_gemma3_text` block (around line 750):

```python
if GEMMA4_AVAILABLE:
    MINI_MODEL_SETUPS["mini_gemma4_text"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_gemma4_text,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma4_text,
        model_class=Gemma4ForCausalLM,
        mini_model_config=Gemma4TextConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            hidden_activation="gelu_pytorch_tanh",
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            tie_word_embeddings=True,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            # Disable the novel / non-kernel-patched paths for v1:
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            enable_moe_block=False,
            # Per-Layer Embeddings: small dim so test stays cheap; Liger doesn't
            # patch this path but the decoder layer forward expects it to work.
            hidden_size_per_layer_input=128,
            vocab_size_per_layer_input=32000,
        ),
    )
```

If `Gemma4TextConfig` rejects any of the above kwargs, inspect the HF config file and remove/rename them accordingly — verify BEFORE committing.

- [ ] **Step 8.4: Add pytest parametrize entry**

Insert after the `mini_gemma3_text` `pytest.param(...)` block (around line 2232):

```python
        pytest.param(
            "mini_gemma4_text",
            32,
            1e-5,
            torch.bfloat16,
            1e-2,
            1e-2,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            marks=[
                pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
                pytest.mark.skipif(
                    not GEMMA4_AVAILABLE,
                    reason="Gemma4 not available in this version of transformers",
                ),
            ],
        ),
```

- [ ] **Step 8.5: Commit**

```bash
git add test/convergence/bf16/test_mini_models.py
git commit -m "gemma4: add mini_gemma4_text bf16 convergence test

Uses same shapes as mini_gemma3_text. Disables kv-sharing, double-wide MLP,
and MoE (all v1-unsupported). Keeps PLE enabled to prove the decoder-layer
forward still works when our module swaps are applied."
```

---

## Task 9: Bump transformers dev dependency and CI skip logic

**Files:**
- Modify: `setup.py`

Gemma 4 requires `transformers>=5.5.0`. The existing dev extra is `transformers>=4.52.0`. Bumping the floor will force all model tests to run on 5.5.0+, which may break other model tests that were pinned to older API shapes.

Safer v1 approach: keep the floor, but rely on the per-test `GEMMA4_AVAILABLE` skip (already added in Task 8). Only bump if the reviewer / CI requires it.

- [ ] **Step 9.1: Decide — do NOT bump the floor now**

Do not modify `setup.py`. The `GEMMA4_AVAILABLE` guard in the test file is sufficient. Document this decision in the commit log.

- [ ] **Step 9.2: Commit the decision (empty commit for traceability)**

```bash
git commit --allow-empty -m "gemma4: keep transformers>=4.52.0 floor; guard via GEMMA4_AVAILABLE

Rationale: bumping the floor to 5.5.0 would force every existing model
test to run on transformers 5.5.0+, risking regressions across ~35 other
models. The per-test ImportError guard is the conventional pattern here
(see SMOLLM3 / QWEN3_NEXT / FALCONH1 precedents)."
```

---

## Task 10: Lint + smoke-import sanity check

**Files:** none modified; verification step.

- [ ] **Step 10.1: Run ruff**

Run: `ruff check src/liger_kernel/transformers/model/gemma4.py src/liger_kernel/transformers/monkey_patch.py src/liger_kernel/transformers/rms_norm.py src/liger_kernel/transformers/__init__.py test/utils.py`

Expected: no errors. Fix any formatting/import-order issues with `ruff check --fix`.

- [ ] **Step 10.2: Smoke-import test (only if transformers>=5.5.0 is installed on the host)**

Run: `python -c "from liger_kernel.transformers import apply_liger_kernel_to_gemma4_text, apply_liger_kernel_to_gemma4; print('ok')"`

Expected: `ok` printed. If transformers 5.5.0 isn't available locally, skip — LUMI will exercise this.

- [ ] **Step 10.3: Commit any lint fixes**

```bash
git add -A
git commit -m "gemma4: ruff fixes" --allow-empty
```

---

## Task 11: LUMI-only convergence run (manual, not scripted)

**Files:** none.

This is a manual verification step. The plan cannot automate the LUMI run from this session.

- [ ] **Step 11.1: On LUMI, set up env and install**

```bash
module load rocm pytorch
pip install -e ".[dev]"
pip install "transformers>=5.5.0"
```

- [ ] **Step 11.2: Run the new convergence test**

```bash
pytest test/convergence/bf16/test_mini_models.py -k mini_gemma4_text -v
```

Expected: PASS. If FAIL:
- If loss diverges early: inspect `LigerRMSNormForGemma4` — most likely culprit is incorrect `offset` (must be 0.0, not 1.0) or weight init (must be ones).
- If shape errors in MLP: check whether mini model config accidentally enabled `use_double_wide_mlp` or a layer became KV-shared.
- If rope errors: re-verify `apply_rotary_pos_emb` signature matches `liger_rotary_pos_emb` for the installed transformers version.

- [ ] **Step 11.3: On green, push the branch**

```bash
git push -u origin feat/gemma4-support
```

---

## Self-Review (done in-plan)

- **Spec coverage:**
  - Text-only Gemma 4 training support: Tasks 1, 2, 4, 6, 7, 8. ✅
  - Multimodal Gemma 4 (text-path only): Tasks 3, 5. ✅
  - Registration in auto-detection: Task 5 (Step 5.2). ✅
  - Tests: Task 8. ✅
  - Dependency handling: Task 9. ✅
  - Gap: no standalone unit test for `LigerRMSNormForGemma4` correctness vs the HF `Gemma4RMSNorm`. Added implicitly via convergence test, but a direct numerical-parity unit test would be cheaper to debug. **Not adding** in v1 scope — convergence test + mini-model shape matches HF's initialization path, which is sufficient coverage. If convergence diverges on LUMI, Task 11's Step 11.2 debug notes point toward the right next step.
  - Gap: no MoE handling. Explicitly out-of-scope per "Out-of-scope" section. ✅
- **Placeholder scan:** searched for TBD / TODO / "implement later" / "add appropriate" — none present outside the out-of-scope section (which intentionally describes work NOT being done). ✅
- **Type consistency:** `LigerRMSNormForGemma4` in Task 1 matches references in Task 4. `causal_forward` / `multimodal_forward` signatures match imports in Task 4 and Task 5. `MiniModelConfig` field names match `test/utils.py`. `GEMMA4_AVAILABLE` spelling consistent across Task 8 steps. ✅
