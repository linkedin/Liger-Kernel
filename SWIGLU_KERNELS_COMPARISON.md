# One Activation, Three DSLs: Triton vs cuTile vs CuTe DSL for SwiGLU

*A head-to-head study of Liger-Kernel's SwiGLU backends on an NVIDIA B200.*

This is the companion to the [RoPE comparison](ROPE_KERNELS_COMPARISON.md). Same
question, different op: Liger-Kernel now ships **three** implementations of the SwiGLU
activation, and we want to know whether three different GPU DSLs, targeting the same
memory-bound op, actually behave the same.

| Backend | Language | Source |
|---|---|---|
| **Triton** | OpenAI Triton | `src/liger_kernel/ops/swiglu.py` |
| **cuTile** | NVIDIA cuTile (`cuda.tile`) | `src/liger_kernel/ops/cutile/ops/swiglu.py` |
| **CuTe DSL** | NVIDIA CUTLASS Python DSL (`cutlass.cute`) | `src/liger_kernel/ops/cutedsl/ops/swiglu.py` |

(The cuTile SwiGLU landed recently in PR #1321; before that this was a two-way study.)

All numbers: **NVIDIA B200 (sm_100)**, bf16, via
`benchmark/scripts/benchmark_swiglu_backends.py` (all backends + a HuggingFace-style
unfused reference in one process). Scope is the **Level-1 elementwise kernel**
`c = silu(a)·b` (`a`=gate, `b`=up) — *not* the fused-linear / fused-MLP variants that
fold in the gate/up GEMMs (a separate story).

---

## TL;DR

1. **All three fused backends beat the unfused reference ~2×**, and all use ~1.3× less
   memory. SwiGLU is memory-bound; fusing `silu` + multiply into one kernel removes an
   intermediate and a launch.
2. **Triton ≈ CuTe DSL everywhere** (CuTe DSL is even a hair faster on the full pass).
3. **cuTile matches them — except on FFN widths that aren't multiples of 2048**, where
   it drops to a slower bounds-checked path and runs ~1.1–1.6× slower. This hits
   mainstream models: the **whole Qwen2.5 family, Llama-2, DeepSeek-V2-Lite**.
4. **Unlike RoPE, eager mode already tells the truth here** — because SwiGLU is a big
   enough kernel that host launch overhead is a small fixed cost, not the whole story.

---

## Background: what SwiGLU is, and why fuse it

Every modern transformer MLP is a SwiGLU MLP. Two linear projections of the input —
**gate** (`a`) and **up** (`b`), each `[tokens, intermediate]` — are combined by:

```
out = silu(a) * b        silu(x) = x * sigmoid(x)
```

then projected back down. It's a purely **elementwise, memory-bound** op over large
tensors (`intermediate` is 11k–19k in real models). The unfused way (plain PyTorch)
is two kernels with an intermediate written to HBM:

```python
s   = F.silu(a)   # kernel 1 -> writes s to HBM
out = s * b       # kernel 2 -> reads s, b
```

The fused SwiGLU kernel (`LigerSiLUMulFunction`) does it in one pass — read `a`,`b`,
write `out`, nothing in between. That's the kernel all three backends implement and
what we compare.

---

## Act I — Everybody beats the unfused reference

Eager, tokens = 4096, forward pass (µs), x = intermediate size:

| n_cols | huggingface | triton | cutile | cutedsl |
|-------:|------------:|-------:|-------:|--------:|
| 8192  | 69.6 | 36.9 | 40.9 | 38.9 |
| 11008 | 86.8 | 49.2 | 55.3 | 49.2 |
| 13824 | 106.5 | 59.4 | **94.2** | 59.4 |
| 14336 | 110.6 | 59.4 | 63.5 | 61.4 |
| 18944 | 141.3 | 77.8 | **98.3** | 77.8 |

![SwiGLU forward, eager](swiglu_kernels_assets/swiglu_eager_forward_ncols.png)

The unfused reference (●) is ~2× slower than the fused kernels across the board — it
pays for the extra intermediate and the second launch. Memory tells the same story:
all three fused backends are identical and ~1.3× smaller than unfused.

![SwiGLU memory, eager](swiglu_kernels_assets/swiglu_eager_memory_ncols.png)

But look at the orange triangle (cuTile) at **13824** and **18944** — it jumps well
above Triton and CuTe DSL. That's the story of Act III.

---

## Act II — Here, eager already tells the truth

In the RoPE study, eager numbers were *misleading*: RoPE is a tiny kernel (~5 µs), so
the ~15 µs of Python/launch overhead dominated and hid all differences — we needed
CUDA graphs to see the real device time.

**SwiGLU is different.** It's a much bigger kernel (tensors are `4096 × up-to-18944`),
so the fixed host launch overhead is a *small fraction* of the total. Concretely, for
Triton at 13824: **59.4 µs eager vs 51.5 µs device (CUDA graph)** — only ~8 µs of
overhead on a ~55 µs kernel. So eager and device rank the backends the same way, and
you can just read the eager numbers directly.

Two consequences worth noting:

- **The cuTile cliff is a real device cost**, not a launch artifact — it shows up
  identically in eager and in CUDA-graph timing.
- **CuTe DSL isn't CUDA-graph-capturable here** (a genuine quirk: its SwiGLU launch,
  unlike its RoPE launch, doesn't thread PyTorch's current stream, so a graph capture
  records nothing). Normally that would blind us to its device time — but because the
  kernel is big and it uses the tvm-ffi fast-launch path, its **eager time ≈ device
  time**, so eager is a fair proxy and CuTe DSL is fully measurable anyway.

---

## Act III — cuTile and the multiple-of-2048 rule

cuTile's SwiGLU has a fast path and a slow path, and which one you get depends on the
FFN width in a sharp, discrete way.

### The rule

cuTile's block loads/stores require **power-of-two tile shapes**. Its
`_calculate_block_size` caps the forward tile at 4096 and keeps the fast
(`check_bounds=False`) path only if it can find a power-of-two tile that (a) divides
`n_cols` and (b) is ≥ 2048. That is satisfied exactly when:

```
n_cols % 2048 == 0     ->  fast (coalesced, no bounds checks)
otherwise              ->  slow (bounds-checked "masked" path)
```

Measured forward, tokens = 4096:

| n_cols | ÷2048? | cuTile path | cutile | triton | ratio | model |
|-------:|:------:|:-----------|-------:|-------:|------:|:------|
| 8192  | ✓ | ALIGNED fast | 40.9 | 36.9 | 1.11× | (pow2) |
| 14336 | ✓ | ALIGNED fast | 63.5 | 59.4 | 1.07× | **Llama-3-8B, Mixtral-8x7B** |
| 11008 | ✗ | masked slow  | 55.3 | 49.2 | 1.12× | Llama-2-7B |
| 18944 | ✗ | masked slow  | 98.3 | 77.8 | 1.26× | **Qwen2.5-7B** |
| 13824 | ✗ | masked slow  | 94.2 | 59.4 | **1.59×** | **Qwen2.5-14B** |

The full (fwd+bwd) pass shows the same, dominated by the forward:

![SwiGLU full pass, eager](swiglu_kernels_assets/swiglu_eager_full_ncols.png)

### This is not a rare edge case

Grouping real models by their intermediate size:

- ✅ **Fast** (÷2048): Llama-3-8B & Mixtral-8x7B (14336), DeepSeek-V3 (18432)
- ⚠️ **Slow** (not ÷2048): Llama-2-7B (11008), **Qwen2.5-7B (18944)**,
  **Qwen2.5-14B (13824)**, Qwen2.5-72B (29568), DeepSeek-V2-Lite (10944)

A large swath of mainstream models — the entire Qwen2.5 family, Llama-2, smaller
DeepSeek — land on the slow path.

### The cliff is token-independent

Sweeping tokens at a fixed width shows the ratio is constant — it's purely a function
of `n_cols` and scales proportionally with tokens (forward, µs):

| tokens | 1024 | 2048 | 4096 | 8192 |
|-------:|-----:|-----:|-----:|-----:|
| 13824 triton | 20.5 | 32.8 | 59.4 | 108.5 |
| 13824 **cutile** | **28.7** | **53.2** | **94.2** | **180.1** |
| 14336 cutile | 24.6 | 36.9 | 63.5 | 114.7 |
| 14336 triton | 20.5 | 34.8 | 59.4 | 110.6 |

At the "good" width (14336) cuTile ties Triton at every token count; at the "bad" width
(13824) it's ~1.5–1.65× slower at every token count.

### Why the stock benchmark misses it

Liger's own `benchmark_swiglu.py` benchmarks the whole MLP module and sweeps **tokens**
at a **fixed per-model intermediate size** — and its default model, `llama_3_8b`, has
intermediate 14336, a ÷2048 *fast* width. So the stock benchmark, on its default,
never exercises the slow path. You only see the cliff if you sweep the FFN width (as
here) or happen to benchmark a Qwen2.5-shaped model.

### Why Triton and CuTe DSL don't have it

- **Triton** has a masked block store (`tl.store(..., mask=...)`), so one kernel pads
  and masks *any* width while staying coalesced — plus a Blackwell column-tiling path
  for wide rows.
- **CuTe DSL** uses a vectorized/scalar layout with no power-of-two tile constraint.

Same hardware, same op — the DSL's expressiveness is the variable. (This is the exact
analog of the RoPE non-power-of-2 cliff.)

---

## Can cuTile be fixed?

Yes, and cheaply. The masked path is *correct*, just slower because it bounds-checks a
ragged tail. The fix is to avoid the ragged tail via **power-of-two-sum tile
decomposition**: process `n_cols` as a sum of aligned pow2 chunks, each
`check_bounds=False`, e.g.

```
13824 = 4096 + 4096 + 4096 + 1024 + 512
```

Every chunk is a clean, coalesced, bounds-check-free tile, so the fast path covers any
width. It's the same idea that fixes the RoPE cliff, applied to the column axis —
contained work, kernel body unchanged.

---

## Conclusions

1. **For standard training, pick any of the three** — on ÷2048 widths (Llama-3,
   Mixtral-8x7B) they tie, and all beat the unfused reference ~2× at ~1.3× less memory.
2. **cuTile has a real, discrete footgun**: FFN widths not divisible by 2048 (the whole
   Qwen2.5 family, Llama-2) run ~1.1–1.6× slower. If you select
   `LIGER_KERNEL_IMPL=cutile`, know your model's intermediate size — or prefer Triton /
   CuTe DSL, which have no such cliff.
3. **Methodology note**: eager is a fair measurement for SwiGLU (big kernel, not
   host-bound) — the opposite of RoPE. CUDA graphs were only needed to *confirm* the
   cuTile gap is a device cost (it is).
4. **The differences are about DSL expressiveness, not GPU physics** — the same lesson
   as RoPE. Triton's masked block store and CuTe DSL's unconstrained layout sidestep a
   power-of-two tiling limitation that cuTile's block API currently imposes.

---

### Reproducing

```bash
cd benchmark/scripts

# sweep intermediate size (reveals the cuTile cliff); eager + CUDA-graph
LIGER_BENCH_TARGET=swiglu_ncols  python benchmark_swiglu_backends.py --sweep-dim n_cols  --timing both  --overwrite

# sweep tokens at a good (14336) and bad (13824) width
LIGER_BENCH_TARGET=swiglu_tokens python benchmark_swiglu_backends.py --sweep-dim tokens --timing eager --overwrite
```

- `speed` = eager (host-inclusive) · `speed_graph` = CUDA-graph device-only · `memory` = peak MB (full pass)
- `LIGER_BENCH_TARGET=<name>` routes results to `benchmark/data/all_benchmark_data_<name>.csv`.

*Environment: NVIDIA B200 (sm_100), CUDA 13, torch 2.11 (cu130), triton 3.6,
nvidia-cutlass-dsl 4.6, cuda-tile (PR #1321), bf16.*
