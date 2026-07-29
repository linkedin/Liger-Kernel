# Megatron-Core integration examples

Two self-contained scripts demonstrating the integration modes shipped by
`liger_kernel.megatron`. Both train a tiny GPT model with mock data on
2 × GPU (TP=1, PP=1, DP=2) for 5 iterations and print the resolved bindings
so you can see which slots picked up Liger.

## Op support matrix

| Op | Mode 1 flag | Patched symbol(s) | Mode 2 class | Slot used in Mode 2 |
|---|---|---|---|---|
| RMSNorm | `rms_norm=True` (on by default) | `LocalSpecProvider.layer_norm`, `transformer_block.LayerNormImpl` | `LigerMegatronRMSNorm` | every norm slot, incl. block-level `final_layernorm` |
| Cross-entropy | `cross_entropy=True` (opt-in) | `fused_cross_entropy.fused_vocab_parallel_cross_entropy`, `tensor_parallel.cross_entropy.vocab_parallel_cross_entropy` | `LigerMegatronCrossEntropy` | none — `GPTModel` subclass overriding `compute_language_model_loss` |
| SwiGLU | `swiglu=True` (opt-in) | `fusions.fused_bias_swiglu.SwiGLUFunction` | `LigerMegatronSwiGLU` | the `mlp` module slot — an `MLP` subclass |

Notes that apply to the table:

- RMSNorm only covers the local (non-TE) backend.
- SwiGLU is used only when `gated_linear_unit=True`, `activation_func=F.silu`,
  and `bias_activation_fusion=True`; otherwise the patch is applied but not
  exercised.
- Liger replaces only `SwiGLUFunction`, so bias and MoE variants stay on
  Megatron.
- Cross-entropy and SwiGLU are wired through subclasses (no dedicated spec
  slot).

## Prerequisites

- A working Megatron-Core install (`pip install megatron-core`).
- `liger-kernel` installed (editable or from PyPI).
- `psutil`.
- At least 2 GPUs.

## Mode 1 — `apply_liger_kernel_to_megatron()` (monkey-patch)

One-line opt-in. Patches the symbols in the matrix above without changing
the spec the user constructs.

```bash
torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    examples/megatron/run_mode1_monkey_patch.py
```

## Mode 2 — hand-assembled `TransformerBlockSubmodules`

Slot-level control — see the "Mode 2 class" column of the matrix above.
Useful when you want to mix Liger with other backends (e.g.
TransformerEngine) on a per-slot basis.

```bash
torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    examples/megatron/run_mode2_hand_spec.py
```

## What you should see

For both scripts:

- 5 lines of `[modeN] iter <i> loss=<float>`. (Five iterations on a
  12-hidden-size model with random mock data is far too short to show a
  trend — the value just hovers.)
- A printed module tree with `LigerMegatronRMSNorm` in **5 of 5** norm
  slots: four per-layer (`input_layernorm`, `pre_mlp_layernorm` × 2 layers)
  and one block-level (`final_layernorm`).
- `Successfully loaded the model` after the distributed checkpoint
  round-trip.

Mode 1 additionally prints `=== Resolved SwiGLU symbols ===` with all three
bindings tagged `[Liger]` — the defining module plus the two consumers that
import the symbol by name. Mode 2 prints `=== Resolved SwiGLU MLPs ===`
listing both `_LigerSwiGLUMLP` layers.

If your environment doesn't have Apex or TransformerEngine installed, you
will see harmless warnings — Megatron falls back to the local backend,
which is exactly where Liger plugs in.
