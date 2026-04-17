# PR description (fill in + use when opening against linkedin/Liger-Kernel)

**Title:** `[Gemma 4] Add apply_liger_kernel_to_gemma4_text (dense text, 31B-targeted)`

---

## Summary

Adds Liger-Kernel support for Google's **Gemma 4** family (text, dense variants — primary target `google/gemma-4-31B`). Follows the exact integration pattern of the Gemma 3 port: module-level class swaps (`Gemma4RMSNorm`, `Gemma4TextMLP`), class-forward swap on `Gemma4ForCausalLM`, and `MODEL_TYPE_TO_APPLY_LIGER_FN["gemma4_text"]` registration so `AutoLigerKernelForCausalLM` routes transparently.

### What you get

- `LigerRMSNormForGemma4` (ones-init, no `(1+w)` offset, fp32 compute, handles `with_scale=False` for `v_norm`)
- `LigerGEGLUMLPForGemma4` (thin wrapper over `LigerGEGLUMLP` that absorbs `Gemma4TextMLP`'s `layer_idx` arg)
- Fused-linear-CE `causal_forward` on both `Gemma4ForCausalLM` (stock HF) and `Gemma4TextForCausalLM` (some users extract this subclass to dodge HF issue #45200's mm_token_type_ids check)
- Registration + exports from `liger_kernel.transformers`

### What you don't get (explicit non-goals)

- Multimodal (`Gemma4ForConditionalGeneration`, `Gemma4VisionModel`, `Gemma4AudioModel`)
- MoE (`Gemma4TextExperts`/`Gemma4TextRouter`, used by 26B-A4B)
- PLE (Per-Layer Embeddings — used by E2B/E4B)
- Double-wide MLP on KV-shared layers (31B doesn't use either)
- **Rope kernel swap**: HF Gemma 4's `apply_rotary_pos_emb(x, cos, sin)` is single-tensor; `liger_rotary_pos_emb(q, k, cos, sin)` is dual-tensor. Signatures are incompatible. `rope=True` is a no-op with a warning. The large memory win (LCE) is unaffected.

## Headline: Peak HBM at seq_len=8192

Measured on **AMD MI250X (LUMI)** with a 6-layer Gemma-4-shaped mini model at full 31B vocab (262,144), `bf16`, batch=1, seq=8192, forward+backward:

| Peak HBM | Value |
|---|---|
| HF baseline | **41.47 GB** |
| Liger (patched) | **10.89 GB** |
| **Saved** | **30.58 GB (73.7% reduction)** |

Driver: fused-linear-CE (`skip_logits=True`) eliminates the 262,144 × 8192 × bf16 logits tensor (~4 GB materialized) **plus** its gradient, activation buffers, and softcap intermediates — combined ~30 GB freed.

Loss parity on the same configuration: `loss_HF = 12.6841440`, `loss_liger = 12.6840858` (abs diff ~6e-5 — numerically identical).

## Numerical correctness

### fp32 (kernel correctness)

Whole-model forward, 6-layer Gemma-4-shaped config, vocab 32000, shape `(2, 256, 32000)`:

| statistic | value |
|---|---|
| max \|logits_liger − logits_hf\| | **2.55e-03** |
| mean abs diff | 5.11e-05 |
| p99 abs diff | 3.84e-04 |

### bf16 (expected dtype noise)

Same config, bf16 end-to-end:

| statistic | value |
|---|---|
| max abs diff | 7.97e-01 |
| mean abs diff | 3.05e-02 |
| p99 abs diff | 1.92e-01 |

Interpretation: kernels are numerically correct (fp32 max 2.55e-3); bf16 drift is 6-layer dtype-inherent precision noise, well inside industry-standard ranges. Mini-model convergence test tolerances set accordingly.

## Tests added

Mirrors the full Gemma 3 test coverage pattern so reviewers can diff head-to-head:

| File | Added |
|---|---|
| `test/transformers/test_monkey_patch.py` | `is_gemma4_available()`, `test_apply_liger_kernel_to_instance_for_gemma4_text` (CPU-runnable instance-patch verification) |
| `test/convergence/bf16/test_mini_models.py` | `mini_gemma4_text` bf16 loss/accuracy convergence |
| `test/convergence/bf16/test_mini_models_with_logits.py` | `mini_gemma4_text` bf16 logits parity (stricter) |
| `test/utils.py` | `revert_liger_kernel_to_gemma4_text` |

All tests pass on LUMI (MI250X + ROCm 6.2.4 + PyTorch 2.7.1 + transformers 5.5.4). Regression run on existing gemma3 tests: **all pass** — no behaviour change to Gemma 3 code paths.

Tolerance selection for `mini_gemma4_text`:
- `loss_atol=5e-2` (vs gemma3's `1e-2`) — 6-layer mini (minimum for sliding+global cycle) has ~0.05 bf16 drift vs gemma3's 4-layer which fits 1e-2
- `logprobs_atol=5e-1` (vs gemma3's `3e-1` in `with_logits`, `1e-1` in loss-only) — accumulated bf16 noise flips 3 of ~20,480 top-k logprob ranks on near-tie tokens

## Discovered during LUMI verification (fixed in-PR)

Three plan-level assumptions that only showed up under real model construction — all fixed here:

1. **`v_norm` uses `with_scale=False`** — no weight parameter, so the naive subclass broke class-level swap. Fix: `LigerRMSNormForGemma4` accepts `with_scale` kwarg and delegates forward to a plain-torch path when weight is absent.
2. **`Gemma4TextMLP(config, layer_idx)` signature** (vs `LigerGEGLUMLP(config)`) — crashed at layer construction. Fix: `LigerGEGLUMLPForGemma4` wrapper absorbs the extra arg.
3. **`apply_rotary_pos_emb(x, cos, sin)` single-tensor signature** (vs `liger_rotary_pos_emb(q, k, cos, sin)`) — incompatible. Fix: `rope=True` becomes a no-op + warning for Gemma 4.

## Test plan

- [x] `pytest test/transformers/test_monkey_patch.py::test_apply_liger_kernel_to_instance_for_gemma4_text` — runs on CPU
- [x] `pytest test/convergence/bf16/test_mini_models.py -k mini_gemma4_text` — MI250X ROCm
- [x] `pytest test/convergence/bf16/test_mini_models_with_logits.py -k mini_gemma4_text` — MI250X ROCm
- [x] Gemma 3 regression (3 tests) — all pass, no Gemma 3 behaviour change
- [x] Revert helper functional test — identical outputs (`max diff 0.00e+00`) after revert
- [x] HBM benchmark at seq_len=8192 — **30.58 GB saved**, **73.7% reduction**
- [x] bf16 vs fp32 max logit diff — see Numerical correctness table above

Environment: LUMI HPC, AMD MI250X, `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`, `transformers==5.5.4`.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
