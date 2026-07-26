# Megatron-Core integration examples

Two self-contained scripts demonstrating the integration modes shipped by
`liger_kernel.megatron`. Both train a tiny GPT model with mock data on
2 × GPU (TP=1, PP=1, DP=2) for 5 iterations and print the resolved norm
classes, CE bindings and SwiGLU bindings so you can see which slots picked
up Liger.

## Prerequisites

- A working Megatron-Core install (`pip install megatron-core`).
- `liger-kernel` installed (editable or from PyPI).
- `psutil` (used by Megatron's async checkpoint worker pool).
- At least 2 GPUs.

## Mode 1 — `apply_liger_kernel_to_megatron()` (monkey-patch)

One-line opt-in. Patches `LocalSpecProvider.layer_norm` and
`transformer_block.LayerNormImpl` so every RMSNorm slot becomes Liger, the
two cross-entropy entry points, and `bias_swiglu_impl` — all without
changing the spec the user constructs.

Note the SwiGLU dispatch flags on `TransformerConfig`
(`gated_linear_unit=True`, `activation_func=F.silu`,
`bias_activation_fusion=True`). `MLP.forward` only routes through
`bias_swiglu_impl` when all three hold, so without them the patch is
applied but never reached — no speedup and no error.

```bash
torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    examples/megatron/run_mode1_monkey_patch.py
```

## Mode 2 — hand-assembled `TransformerBlockSubmodules`

Slot-level control. Explicitly places `LigerMegatronRMSNorm` into each
norm slot, including the block-level `final_layernorm`, and a
`LigerMegatronSwiGLU`-backed `MLP` subclass into the `mlp` slot. Useful
when you want to mix Liger with other backends (e.g. TransformerEngine) on
a per-slot basis.

SwiGLU has no spec slot of its own — `MLPSubmodules.activation_func` is
only read when `config.use_te_activation_func` is set, so it is the
TransformerEngine hook rather than a general one. The MLP *class* is a
slot, so this script subclasses `MLP`, the same way it subclasses
`GPTModel` for cross-entropy.

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
