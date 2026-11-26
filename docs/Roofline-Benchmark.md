# Per-kernel memory bandwidth (roofline) benchmark

Liger's headline numbers are end-to-end. This benchmark measures how close an individual
memory-bound kernel gets to the bandwidth the GPU can actually sustain.

## Method

- `benchmark/benchmark_roofline.py` runs one kernel in a loop (10 warmup + 20 timed launches).
- Nsight Compute measures each launch's time (`gpu__time_duration.sum`). Caches are flushed before every
  launch, and clocks are left unlocked (`--clock-control none`), as in training. Locking them to base clocks
  made these kernels 27-59% slower on a B200.
- Bandwidth is compulsory traffic (each input read once, each output written once) divided by kernel time.
- The ceiling is a **measured baseline**, not the datasheet number: a plain PyTorch op that reads and
  writes the same tensors. That is a copy (`torch.mul(x, 1.0)`) for RMSNorm and `torch.mul(a, b)` for
  SwiGLU. 100% means the Liger kernel is as fast as that op. A single copy is not a fair ceiling for
  every kernel, because the bandwidth a GPU reaches depends on tensor size and on the mix of reads and
  writes.
- `dram__bytes_read.sum` is checked against the model. A ratio above 1.0 means the kernel reads some
  data from DRAM more than once.
- DRAM byte counters are not used for bandwidth. On a GPU with a large L2, a kernel can end with tens
  of MB of its output still in L2. Those writes reach DRAM later and are not counted for the kernel.

## Run it

```bash
NCU=/path/to/ncu bash benchmark/run_roofline.sh roofline_out
```

Requires Nsight Compute with GPU performance counters enabled for your user.
In a Docker container, also pass `--cap-add=SYS_ADMIN`.
Shapes (LLaMA 3-8B, fp16): 4096 tokens, hidden 4096, intermediate 14336.

## Results

NVIDIA B200 (driver 580.173.02) in the NGC PyTorch 26.10 container (torch 2.15, Triton 3.8, CUDA 13.4),
Nsight Compute 2026.3.1. Each time is the mean of 20 launches; three runs agreed within 0.4 points.

| Kernel  | Baseline | Time (us) | GB/s   | Baseline GB/s | % of baseline | DRAM reads / model |
|---------|----------|-----------|--------|---------------|---------------|--------------------|
| RMSNorm | copy     | 13.04     | 5144.5 | 5594.6        | 92.0          | 1.00               |
| SwiGLU  | mul      | 49.21     | 7158.9 | 7154.5        | 100.1         | 1.00               |

SwiGLU runs as fast as `torch.mul(a, b)`, and RMSNorm reaches 92% of a copy of the same size. Neither kernel
reads any data from DRAM twice. Results on other GPUs may differ, which is the point of re-running it on your
hardware.

## Caveats

- Forward pass only, one shape, one dtype.
- The model ignores small extra traffic (RMSNorm's weight read and rstd write).
- Small kernels do not reach full bandwidth (launch ramp-up and tail), so each kernel is compared
  with a baseline of its own shape rather than with one peak number.
- Times are measured with a DRAM counter on, which adds a few percent to each launch. Baselines are
  timed the same way.
