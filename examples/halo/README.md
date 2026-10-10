# Use Liger Kernel with Halo

[Halo](https://github.com/whitecircle/halo) uses Liger Kernel for faster fused model operations and training losses with lower peak memory. Models stay in their native Hugging Face format. The integration is enabled by default; set `use_liger_kernel: true` to make that choice explicit in a training config.

This example fine-tunes Qwen3-4B with LoRA. It uses fused linear cross entropy, so logits do not need to be fully materialized for the loss.

## Run

Install Halo by following its [installation guide](https://github.com/whitecircle/halo#installation), then run from the Halo repository root:

```bash
halo launch sft /path/to/Liger-Kernel/examples/halo/qwen3-4b-lora.yaml
```

The config runs for 20 steps so the integration can be checked quickly. Remove `max_steps` and set `num_train_epochs` for a complete run.

To compare the same workload without Liger, override one field:

```bash
halo launch sft /path/to/Liger-Kernel/examples/halo/qwen3-4b-lora.yaml \
  --use_liger_kernel=false --output_dir=checkpoints/qwen3-4b-lora-no-liger
```

Keep the model, sequence length, batch size, precision, and hardware unchanged when comparing the two runs. Halo logs tokens per second and peak allocated GPU memory because `enable_efficiency_metrics` is enabled in the config.

Read more about Halo at [whitecircle.com/halo](https://whitecircle.com/halo).
