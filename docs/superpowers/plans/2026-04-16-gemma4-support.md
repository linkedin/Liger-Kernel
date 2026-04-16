# Gemma 4 31B (text-only) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `apply_liger_kernel_to_gemma4_text` to Liger-Kernel so Google **Gemma 4 31B** (the dense text model) can train with Liger's fused kernels on LUMI HPC (AMD MI250X / ROCm).

**Architecture:** Mirror the Gemma 3 text port. Add a Gemma-4-specific RMSNorm subclass (Gemma 4 does **not** use Gemma 3's `(1 + weight)` offset — `Gemma4RMSNorm` inherits `Gemma3nRMSNorm`, which uses `init="ones"` / `offset=0`). Swap `Gemma4TextMLP` → `LigerGEGLUMLP`, `Gemma4RMSNorm` → `LigerRMSNormForGemma4`, `apply_rotary_pos_emb` → `liger_rotary_pos_emb`, and `Gemma4ForCausalLM.forward` → fused-linear-CE. **No multimodal work. No MoE work.** The 31B config disables every novel Gemma 4 feature that would complicate the port.

**Tech Stack:** Python 3.10+, PyTorch, Triton, HuggingFace Transformers ≥ 5.5.0, pytest, Liger's existing kernel ops (`LigerRMSNormFunction`, `liger_rotary_pos_emb`, `LigerGEGLUMLP`, `LigerForCausalLMLoss`).

**Execution environment:** Implementation can be done on any machine. Convergence tests MUST run on LUMI (AMD ROCm) — Triton doesn't work on Apple Silicon / CPU-only systems.

---

## Why the 31B-only scope dramatically simplifies this

The published `google/gemma-4-31B` text config has **every novel Gemma 4 knob turned off**:

| Novel feature | 31B config value | Consequence for this port |
|---|---|---|
| `num_kv_shared_layers` | `0` | All 60 layers carry `q_norm` / `k_norm` — no absent-attribute guards required |
| `use_double_wide_mlp` | `false` | `LigerGEGLUMLP` swap is direct — no per-layer intermediate-size divergence |
| `enable_moe_block` | `false` | No MoE — drop all `Gemma4TextExperts` / `Gemma4TextRouter` considerations |
| `hidden_size_per_layer_input` | `0` | **No Per-Layer Embeddings (PLE).** 31B is a plain dense decoder stack |
| `final_logit_softcapping` | `30.0` | Must be honored in `causal_forward` (already handled like Gemma 3) |
| `rope_parameters.partial_rotary_factor` | `0.25` on global layers | Handled inside `Gemma4TextRotaryEmbedding` — `apply_rotary_pos_emb` is still plain `x*cos + rotate_half(x)*sin`, so `liger_rotary_pos_emb` is a safe drop-in |

Since all the interesting complications are config-gated off, the 31B port is essentially "Gemma 3 text port + corrected RMSNorm semantics".

The smaller Gemma 4 models (E2B, E4B) use MoE, double-wide MLP, KV sharing, and PLE — those remain **out of scope** for this plan.

---

## Out-of-scope (explicit non-goals)

- Gemma 4 multimodal (`Gemma4ForConditionalGeneration`, `Gemma4VisionModel`, `Gemma4AudioModel`). User explicitly scoped to text-only.
- E2B / E4B / 26B-A4B variants. Their config flips on PLE, MoE, KV sharing, or double-wide MLP — none of which we patch here.
- Proportional vs default RoPE correctness verification beyond the convergence test. The apply function is plain; the rotary embedding module constructs cos/sin itself.

---

## File Structure

### New files

- `src/liger_kernel/transformers/model/gemma4.py`
  Fused-linear-CE forward: `causal_forward` for `Gemma4ForCausalLM`. Reuses `LigerForCausalLMLoss` and `LigerCausalLMOutputWithPast`.

### Modified files

- `src/liger_kernel/transformers/rms_norm.py`
  Add `LigerRMSNormForGemma4(LigerRMSNorm)` with `offset=0.0`, `init_fn="ones"`, `casting_mode="gemma"`, `in_place=False`.

- `src/liger_kernel/transformers/monkey_patch.py`
  Add `apply_liger_kernel_to_gemma4_text(...)`. Register `"gemma4_text"` in `MODEL_TYPE_TO_APPLY_LIGER_FN`.

- `src/liger_kernel/transformers/__init__.py`
  Add `apply_liger_kernel_to_gemma4_text` to the `TYPE_CHECKING` block and lazy-import machinery (follow the `apply_liger_kernel_to_gemma3_text` pattern).

- `test/utils.py`
  Add `revert_liger_kernel_to_gemma4_text`.

- `test/convergence/bf16/test_mini_models.py`
  Add `GEMMA4_AVAILABLE` import guard, `mini_gemma4_text` `MINI_MODEL_SETUPS` entry, and `pytest.param("mini_gemma4_text", ...)` case with bf16 tolerances mirroring `mini_gemma3_text`.

---

## Task 1: Add `LigerRMSNormForGemma4` subclass

**Files:**
- Modify: `src/liger_kernel/transformers/rms_norm.py` (append after `LigerRMSNormForGemma3` at line 70)

- [ ] **Step 1.1: Confirm insertion point**

Read `src/liger_kernel/transformers/rms_norm.py` lines 66–72. Verify `LigerRMSNormForGemma3` ends at line 70 and `LigerRMSNormForOlmo2` begins at line 73.

- [ ] **Step 1.2: Add the new class**

Insert after `LigerRMSNormForGemma3` and before `LigerRMSNormForOlmo2`:

```python
class LigerRMSNormForGemma4(LigerRMSNorm):
    """Gemma4RMSNorm inherits from Gemma3nRMSNorm, NOT from Gemma3RMSNorm.

    Differences from LigerRMSNormForGemma3:
      - weight initialized to ones (not zeros)
      - no (1 + weight) offset — scales by weight directly
      - still uses fp32 compute (gemma casting mode)
    """

    def __init__(self, dim, eps=1e-6, offset=0.0, casting_mode="gemma", init_fn="ones", in_place=False):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)
```

- [ ] **Step 1.3: Commit**

```bash
git add src/liger_kernel/transformers/rms_norm.py
git commit -m "gemma4: add LigerRMSNormForGemma4 (ones init, no +1 offset)

Gemma4RMSNorm inherits Gemma3nRMSNorm, not Gemma3RMSNorm. The Gemma3n
variant initializes weight to torch.ones(dim) and does NOT apply the +1
offset. Using LigerRMSNormForGemma3 here would silently diverge training."
```

---

## Task 2: Create `model/gemma4.py` with `causal_forward`

**Files:**
- Create: `src/liger_kernel/transformers/model/gemma4.py`

Gemma 4 31B sets `final_logit_softcapping=30.0` and uses `tie_word_embeddings=true`, both of which match Gemma 3's code path exactly. We can near-duplicate `model/gemma3.py`'s `causal_forward` with no structural changes.

- [ ] **Step 2.1: Create the file**

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

    Mirrors liger's gemma3 causal_forward. Gemma 4 31B uses
    final_logit_softcapping=30.0, so the softcap branch is exercised on
    the non-fused path.
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
final_logit_softcapping (31B sets it to 30.0; future variants may omit)."
```

---

## Task 3: Add `apply_liger_kernel_to_gemma4_text` to `monkey_patch.py`

**Files:**
- Modify: `src/liger_kernel/transformers/monkey_patch.py` (insert after `apply_liger_kernel_to_gemma3` ends, currently around line 1239)

Design choices — captured up front so reviewers can verify intent:

- **31B has `num_kv_shared_layers=0`** → q_norm/k_norm exist on every layer. We still use `getattr(..., None)` for forward-compat with smaller variants.
- **31B has `enable_moe_block=false`** → no router/experts to guard. We still add a `getattr(decoder_layer, "enable_moe_block", False)` skip for forward-compat.
- **`Gemma4RMSNorm` ones-init / no-offset** → `partial(_patch_rms_norm_module, offset=0.0, casting_mode="gemma", in_place=False)`. (The weight values come from the existing parameter; `_patch_rms_norm_module` does not reinitialize — only swaps forward.)

- [ ] **Step 3.1: Verify `_patch_rms_norm_module` signature**

Grep `src/liger_kernel/transformers/monkey_patch.py` for `def _patch_rms_norm_module`. Confirm it accepts `offset`, `casting_mode`, `in_place` kwargs and does NOT reinitialize weights (it swaps forward + stores flags on the existing module). If it does reinitialize, we need a Gemma4-specific helper — but this is unlikely given how gemma3 works.

- [ ] **Step 3.2: Append the function**

Insert after `apply_liger_kernel_to_gemma3` (do NOT add a multimodal variant — out of scope):

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

    Primary target: Gemma 4 31B. The 31B config disables PLE
    (hidden_size_per_layer_input=0), MoE (enable_moe_block=false), KV sharing
    (num_kv_shared_layers=0), and double-wide MLP (use_double_wide_mlp=false),
    so every decoder layer is a plain (norm, attn, norm, mlp, norm) stack.

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
        _patch_rms_norm_module, offset=0.0, casting_mode="gemma", in_place=False
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
                # Defensive: skip MLP rebind if a future variant flips MoE on.
                if geglu and not getattr(decoder_layer, "enable_moe_block", False):
                    _bind_method_to_module(decoder_layer.mlp, "forward", LigerGEGLUMLP.forward)
                if rms_norm:
                    _patch_rms_norm_module_for_gemma4(decoder_layer.input_layernorm)
                    _patch_rms_norm_module_for_gemma4(decoder_layer.post_attention_layernorm)
                    _patch_rms_norm_module_for_gemma4(decoder_layer.pre_feedforward_layernorm)
                    _patch_rms_norm_module_for_gemma4(decoder_layer.post_feedforward_layernorm)
                    # q_norm / k_norm exist on every 31B layer (num_kv_shared_layers=0)
                    # but stay defensive for future variants.
                    q_norm = getattr(decoder_layer.self_attn, "q_norm", None)
                    k_norm = getattr(decoder_layer.self_attn, "k_norm", None)
                    if q_norm is not None:
                        _patch_rms_norm_module_for_gemma4(q_norm)
                    if k_norm is not None:
                        _patch_rms_norm_module_for_gemma4(k_norm)
        else:
            raise TypeError("The model must be Gemma4ForCausalLM or Gemma4TextModel.")
```

- [ ] **Step 3.3: Register `gemma4_text` in `MODEL_TYPE_TO_APPLY_LIGER_FN`**

Locate the dict (currently near line 3180). Insert next to the `gemma3_text` entry:

```python
    "gemma3_text": apply_liger_kernel_to_gemma3_text,
    "gemma3": apply_liger_kernel_to_gemma3,
    "gemma4_text": apply_liger_kernel_to_gemma4_text,
```

We do NOT register `"gemma4"` (multimodal) because we do not ship a multimodal patch in this plan. Users loading `Gemma4ForConditionalGeneration` via `AutoLigerKernelForCausalLM` will get no Liger patches (consistent with how other unhandled types behave today). This is explicit and intentional.

- [ ] **Step 3.4: Commit**

```bash
git add src/liger_kernel/transformers/monkey_patch.py
git commit -m "gemma4: add apply_liger_kernel_to_gemma4_text + model-type registration

Patches RMSNorm (norm, input_layernorm, post_attention_layernorm,
pre_feedforward_layernorm, post_feedforward_layernorm, q_norm, k_norm),
GEGLU MLP, rotary, and fused-linear-CE on Gemma4ForCausalLM.

Primary target: Gemma 4 31B (dense, text-only). Registers 'gemma4_text'
only; the multimodal 'gemma4' model_type is intentionally NOT registered
in this change — see 2026-04-16-gemma4-support.md plan doc."
```

---

## Task 4: Expose `apply_liger_kernel_to_gemma4_text` in the package

**Files:**
- Modify: `src/liger_kernel/transformers/__init__.py`

- [ ] **Step 4.1: Add to the `TYPE_CHECKING` block**

Insert after `apply_liger_kernel_to_gemma3_text`:

```python
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma4_text  # noqa: F401
```

- [ ] **Step 4.2: Check the runtime lazy-import section**

The file has logic after line 80 that conditionally wires `monkey_patch` symbols when `transformers` is installed. Read lines 80 to end of file. Find every place `apply_liger_kernel_to_gemma3_text` is referenced and add the Gemma 4 text analog following the same pattern (likely a list/set of names or a `__getattr__` table).

- [ ] **Step 4.3: Commit**

```bash
git add src/liger_kernel/transformers/__init__.py
git commit -m "gemma4: export apply_liger_kernel_to_gemma4_text from package root"
```

---

## Task 5: Add `revert_liger_kernel_to_gemma4_text` helper

**Files:**
- Modify: `test/utils.py` (insert after `revert_liger_kernel_to_gemma3_text` at line 489)

- [ ] **Step 5.1: Add the revert helper**

Insert after `revert_liger_kernel_to_gemma3_text`:

```python
def revert_liger_kernel_to_gemma4_text(model_config: MiniModelConfig):
    """Revert all Liger kernel patches applied to Gemma4 text model."""

    from transformers.models.gemma4 import modeling_gemma4

    importlib.reload(modeling_gemma4)

    model_config.model_class = modeling_gemma4.Gemma4ForCausalLM

    print("Liger kernel patches have been reverted.")
```

- [ ] **Step 5.2: Commit**

```bash
git add test/utils.py
git commit -m "gemma4: add revert_liger_kernel_to_gemma4_text test helper"
```

---

## Task 6: Add `mini_gemma4_text` bf16 convergence test

**Files:**
- Modify: `test/convergence/bf16/test_mini_models.py`

The mini model mirrors the 31B config shape but shrinks it to 4 layers / hidden_size=1024 so the test runs in seconds on a single GPU.

- [ ] **Step 6.1: Add availability guard**

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

- [ ] **Step 6.2: Update imports at the top of the file**

Add near the gemma3 liger imports:

```python
from liger_kernel.transformers import apply_liger_kernel_to_gemma4_text
```

and near the revert-helper imports:

```python
from test.utils import revert_liger_kernel_to_gemma4_text
```

- [ ] **Step 6.3: Add the `MINI_MODEL_SETUPS` entry**

Insert after the `mini_gemma3_text` setup block (around line 750):

```python
if GEMMA4_AVAILABLE:
    MINI_MODEL_SETUPS["mini_gemma4_text"] = MiniModelConfig(
        liger_kernel_patch_func=apply_liger_kernel_to_gemma4_text,
        liger_kernel_patch_revert_func=revert_liger_kernel_to_gemma4_text,
        model_class=Gemma4ForCausalLM,
        mini_model_config=Gemma4TextConfig(
            # Shrunk from Gemma 4 31B (num_hidden_layers=60, hidden_size=5376).
            # Layer types mirror the 31B pattern (5 sliding, 1 full, repeat).
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=6,
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
            final_logit_softcapping=30.0,
            sliding_window=1024,
            # Match 31B: every Nth layer is full_attention.
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            # Explicitly disable v1-unsupported flags (these are also defaults on 31B):
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            enable_moe_block=False,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=32000,
        ),
    )
```

If `Gemma4TextConfig.__init__` rejects any kwarg above, inspect HF's `configuration_gemma4.py` and rename/remove before committing. (Risk: `layer_types` may be auto-derived from other fields; if so, drop it.)

- [ ] **Step 6.4: Add the pytest parametrize entry**

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

- [ ] **Step 6.5: Commit**

```bash
git add test/convergence/bf16/test_mini_models.py
git commit -m "gemma4: add mini_gemma4_text bf16 convergence test

Mini model mirrors the Gemma 4 31B layout (sliding+global layer mix,
final_logit_softcapping=30.0, tie_word_embeddings) but shrunk to 6
layers / hidden=1024 for cheap execution. Disables all v1-unsupported
flags explicitly (PLE, MoE, KV sharing, double-wide MLP)."
```

---

## Task 7: Lint + smoke-import sanity check

- [ ] **Step 7.1: Run ruff**

Run: `ruff check src/liger_kernel/transformers/model/gemma4.py src/liger_kernel/transformers/monkey_patch.py src/liger_kernel/transformers/rms_norm.py src/liger_kernel/transformers/__init__.py test/utils.py test/convergence/bf16/test_mini_models.py`

Expected: no errors. Apply `ruff check --fix` for import-order / formatting.

- [ ] **Step 7.2: Smoke-import (if transformers>=5.5.0 is installed locally)**

Run: `python -c "from liger_kernel.transformers import apply_liger_kernel_to_gemma4_text; print('ok')"`

Expected: `ok`. If transformers 5.5.0 isn't available locally, skip — LUMI will exercise this.

- [ ] **Step 7.3: Commit any lint fixes**

```bash
git add -A
git commit -m "gemma4: ruff fixes" --allow-empty
```

---

## Task 8: LUMI-only convergence run (manual)

This cannot be automated from this session.

- [ ] **Step 8.1: On LUMI, install dependencies**

```bash
module load rocm pytorch
pip install -e ".[dev]"
pip install "transformers>=5.5.0"
```

- [ ] **Step 8.2: Run the new test**

```bash
pytest test/convergence/bf16/test_mini_models.py -k mini_gemma4_text -v
```

Expected: PASS. Debugging notes if FAIL:
- Loss diverges from the reference: check `LigerRMSNormForGemma4` — it must use `offset=0.0`, `init_fn="ones"`, and `casting_mode="gemma"`. Using the Gemma 3 variant here is the most likely cause of silent divergence.
- Shape mismatch in MLP: the mini model config may have accidentally enabled `use_double_wide_mlp`, or the `Gemma4TextMLP.__init__` signature differs from `LigerGEGLUMLP.__init__`. Inspect `Gemma4TextMLP(Gemma3MLP).__init__` — it takes `(config, layer_idx)`, while `LigerGEGLUMLP.__init__` takes `(config)` alone. If the class swap fails at instantiation, we need a shim `__init__`.
- Rotary errors: verify `liger_rotary_pos_emb` still accepts `(q, k, cos, sin, unsqueeze_dim=…)`.

- [ ] **Step 8.3: On green, push**

```bash
git push -u origin feat/gemma4-support
```

---

## Risks captured explicitly (for post-mortem if something breaks)

1. **`Gemma4TextMLP.__init__(config, layer_idx)` vs `LigerGEGLUMLP.__init__(config)`.** Swapping classes only works if HF instantiates the replacement with the same args. `LigerGEGLUMLP` may error on the extra `layer_idx`. Mitigation: if Task 6's test fails at model construction, wrap `LigerGEGLUMLP` in a small subclass that accepts and ignores `layer_idx`, or patch at instance level instead of class level. Decision deferred until the test actually runs.

2. **`_patch_rms_norm_module` may not understand `init_fn`.** We deliberately avoided passing `init_fn="ones"` through `partial(...)` — the helper's job is only to swap forward behavior on existing modules, whose weights already have the right initial values from HF's own `Gemma4RMSNorm.__init__`. If during LUMI runs we observe incorrect scaling, re-check whether the helper actually preserves the underlying weight tensor or reinitializes.

3. **`"gemma4_text"` model_type string.** Confirmed from the 31B config dump: `"text_config.model_type": "gemma4_text"`. If a future variant uses a different string, registration must be updated.

---

## Self-Review (done in-plan)

- **Spec coverage:**
  - Gemma 4 31B text model training support: Tasks 1, 2, 3, 4, 5, 6. ✅
  - RMSNorm semantic correctness (Gemma3n lineage): Task 1. ✅
  - Fused-linear-CE on `Gemma4ForCausalLM`: Task 2, Task 3. ✅
  - `final_logit_softcapping=30.0` handling: Task 2. ✅
  - Auto-detection via model_type: Task 3 (Step 3.3). ✅
  - Tests: Task 6. ✅
  - Dependency handling: not bumping the `transformers` floor; per-test `GEMMA4_AVAILABLE` guard is the conventional pattern (see `SMOLLM3_AVAILABLE`, `QWEN3NEXT_AVAILABLE`, `FALCONH1_AVAILABLE`). ✅
  - No gaps for the 31B text-only scope.
- **Placeholder scan:** no TBD / TODO / "implement later" / "add appropriate error handling" left in the plan. ✅
- **Type consistency:** `LigerRMSNormForGemma4` (Task 1) is consumed in Task 3. `causal_forward` (Task 2) is imported in Task 3. `revert_liger_kernel_to_gemma4_text` (Task 5) is used in Task 6. `GEMMA4_AVAILABLE` spelling consistent. ✅
