# Per-kernel memory bandwidth (roofline) benchmark

Liger's headline numbers are end-to-end. This benchmark measures how close an individual
memory-bound kernel gets to the bandwidth the GPU can actually sustain.

## Method

- `benchmark/benchmark_roofline.py` runs one kernel in a loop (10 warmup + 20 timed launches).
- Nsight Compute reads `dram__bytes.sum.per_second` (bytes that really reached DRAM, after L2).
- "Peak" is a **measured copy-kernel bandwidth** on the same GPU and tensor size, not the
  datasheet number. Datasheet peaks are rarely reachable and make every kernel look worse.
- `dram__bytes.sum` is compared with a compulsory-traffic model (reads + writes only).
  A ratio near 1.0 means no meaningful cache reuse; well below 1.0 means L2 is absorbing traffic.

## Run it

```bash
NCU=/path/to/ncu bash benchmark/run_roofline.sh roofline_out
```

Requires Nsight Compute with GPU performance counters enabled for your user.
Shapes (LLaMA 3-8B, fp16): 4096 tokens, hidden 4096, intermediate 14336.

## Results

Reference run on NVIDIA GeForce RTX 3050 6GB Laptop GPU (Nsight Compute 2025.3.0). Treat this as an example of the
output format, not as an official Liger number.

| Kernel  | Achieved GB/s | % of copy peak | L2 hit % | DRAM MB measured | DRAM MB model | Measured/Model |
|---------|---------------|----------------|----------|------------------|---------------|----------------|
| copy    | 123.4 | 100.0 | 50.0 | 234.4 | 234.9 | 1.00 |
| RMSNorm | 123.3 | 99.9  | 50.2 | 66.5  | 67.1  | 0.99 |
| SwiGLU  | 124.4 | 100.8 | 33.3 | 351.9 | 352.3 | 1.00 |

On this GPU and shape, both kernels run at the achievable bandwidth ceiling and measured DRAM
traffic matches the simple model. Results on larger GPUs (more L2, higher bandwidth) may differ,
which is the point of re-running it on your hardware.

## Caveats

- Forward pass only, one shape, one dtype.
- The model ignores small extra reads (RMSNorm weight and rstd).
- L2 hit rate counts sector hits inside the kernel and is not by itself evidence of reuse.
