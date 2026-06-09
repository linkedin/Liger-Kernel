# Megatron-Core integration examples

Two self-contained scripts demonstrating the integration modes shipped by
`liger_kernel.megatron`. Both train a tiny GPT model with mock data on
2 × GPU (TP=2, PP=1) for 5 iterations and print the resolved norm classes
so you can see which slots picked up Liger.

## Prerequisites

- A working Megatron-Core install (`pip install megatron-core`).
- `liger-kernel` installed (editable or from PyPI).
- `psutil` (used by Megatron's async checkpoint worker pool).
- At least 2 GPUs.

## Mode 1 — `apply_liger_kernel_to_megatron()` (monkey-patch)

One-line opt-in. Patches `LocalSpecProvider.layer_norm` and
`transformer_block.LayerNormImpl` so every RMSNorm slot becomes Liger
without changing the spec the user constructs.

```bash
torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    examples/megatron/run_mode1_monkey_patch.py
```

## Mode 2 — hand-assembled `TransformerBlockSubmodules`

Slot-level control. Explicitly places `LigerMegatronRMSNorm` into each
norm slot, including the block-level `final_layernorm`. Useful when you
want to mix Liger with other backends (e.g. TransformerEngine) on a
per-slot basis.

```bash
torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29500 \
    examples/megatron/run_mode2_hand_spec.py
```

## What you should see

For both scripts:

- 5 lines of `[modeN] iter <i> loss=<float>` with the loss decreasing.
- A printed module tree with `LigerMegatronRMSNorm` in **5 of 5** norm
  slots: four per-layer (`input_layernorm`, `pre_mlp_layernorm` × 2 layers)
  and one block-level (`final_layernorm`).
- `Successfully loaded the model` after the distributed checkpoint
  round-trip.

If your environment doesn't have Apex or TransformerEngine installed, you
will see harmless warnings — Megatron falls back to the local backend,
which is exactly where Liger plugs in.
