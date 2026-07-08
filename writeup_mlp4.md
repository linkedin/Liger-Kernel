# writeup_mlp4 — MLP4 backward weight gradients on Blackwell (SM100a)

> **Op:** `dB = dUᵀ·X`, `dC = dVᵀ·X` — the two MoE **weight gradients** (backward),
> produced **sequentially in two phases per cell** (phase 0 = `dB` via `dUᵀ`, phase 1 =
> `dC` via `dVᵀ`) with **INDEPENDENT** per-phase accumulators. **Files:**
> `csrc/core/src/moe/mlp4.cuh` (Traits + helpers + UMMA MMA + consumer split),
> `tests/cpp/test_mlp4.cu`. **mlp4 = mlp3 + a two-phase loop:** the mlp3 REDUCE_ADD
> epilogue + store-buf mapping, wrapped in `for(phase=0..1)` with *fresh-cleared*
> accumulators — the **opposite** of mlp5's cross-phase carry.

---

## 1. The nuance of MLP4

During the backward pass, mlp4 produces the gradients w.r.t. the two phase-1 weight
branches by contracting the two upstream grads against the shared block input `X`:

```
dUᵀ, dVᵀ : [I, T]   upstream grads (A operand, M=I, K=T), MN-major (I contiguous)
X         : [T, H]   block input (B operand, N=H, K=T), MN-major (H contiguous) — SHARED
dB = dUᵀ·X : [E, I, H]   phase 0
dC = dVᵀ·X : [E, I, H]   phase 1
```

The nuances that make it distinct:

- **Two GEMMs, TWO phases, INDEPENDENT accumulators.** The consumer wraps the
  mlp3-style mainloop+epilogue in `for(phase=0..1)`. Each phase **clears its
  accumulator fresh** and writes its **own** output — there is **no** cross-phase sum:
  ```
  Phase 0 : clear(acc) ; acc = dUᵀ·X  → REDUCE_ADD into dB
  Phase 1 : clear(acc) ; acc = dVᵀ·X  → REDUCE_ADD into dC
  ```
  This is the **mlp4-unique hazard** and the exact **opposite** of mlp5 (which carries
  one accumulator across both GEMMs). A single reused A-side smem buffer holds `dUᵀ`
  then `dVᵀ`; `X` is shared by both phases and both WGs.
- **BOTH operands MN-major.** `dUᵀ/dVᵀ` is the M-side (A) and is M-major; `X` is the
  N-side (B) and is N-major — so *both* operand descriptors are `UMMA::Major::MN`
  (mlp5, by contrast, has a K-major A).
- **`SM90_TMA_REDUCE_ADD` epilogue** (inherited from mlp3) → **both `dB` and `dC` MUST
  be zero-initialized** by the caller, and re-zeroed between correctness and bench.
- **`EpiChunkN=64`** — the widest epilogue fragment (`TmemLoadOp<64>`), same as mlp5.
- **Persistent chunk-fixed grid**: each CTA fixes one `(chunk, walk-lane)` and runs
  **both** phases internally.

FLOPs (**two** GEMMs): `4·T·H·I` (`2·T·H·I` per phase).

---

## 2. What changed to suit SM100

mlp4 keeps the mlp3 REDUCE_ADD epilogue, the store-buf scatter shape, and the
persistent grid; the SM100 path adds a per-WG UMMA MMA, two **independent** TMEM
accumulators, and a TMEM-pull epilogue — all parameterized by `phase`:

| Piece | Hopper (`sm_90a`) | Blackwell (`sm_100a`) |
|-------|-------------------|------------------------|
| MMA | `TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>` cooperative 2-WG | **per-WG** `SM100_MMA_F16BF16_SS<…,128,128,Major::MN,Major::MN>` (**both** operands MN-major) |
| Accumulator | 2 register fragments (one per WG) | **2 INDEPENDENT TMEM accumulators** — `acc0 @ tmem_base`, `acc1 @ tmem_base+WgTileN`; one MMA warp issues **2 UMMAs / k-step** |
| **Per-phase clear** | `clear(acc)` fresh at the start of each phase | **`UMMA::ScaleOut`**: `Zero` on the *first k of EACH phase*, `One` after — independent per phase, **never carried across** (opposite of mlp5) |
| Epilogue | registers → cast → `store_buf` scatter | `(M,N)` extract → `flat_divide((WgTileM,64))` → **`TmemLoadOp<64>`** → `partition_D`-sized regs → cast → `store_buf` → TMA REDUCE_ADD |
| Store-buf M-row | WGMMA `(m_loc/128)*64 + (m_loc%64)` remap; M-split atom-row `4·out_m + my_wg + 2·a` | **`partition_D` identity coords** (UMMA C-layout is contiguous per-WG, **no remap**); M-split atom-row **`4·out_m + 2·my_wg + a`** |
| Output routing | `phase`-selected `dB`/`dC` descriptor | **unchanged** (already phase-parameterized) |
| TMEM lifetime | n/a | **once-per-CTA** `tcgen05.alloc`/`free`, hoisted **ABOVE both the cell loop AND the phase loop** (B5′) |
| Grid | persistent chunk-fixed | **unchanged** |
| smem | Stages=4 | **Stages=4 ⇒ ~224 KiB < B200's 227 KiB** (reused single A buffer keeps it in budget) |

**The crux — independent per-phase clear (Option B).** UMMA requires `M ∈ {64,128}`, so
the 256-wide M-split tile **cannot** use a single atom. The port uses **Option B**: two
independent `(128,128)` TMEM accumulators (one per consumer WG), and the single MMA warp
issues **two** `tcgen05.mma` per k-step. Each phase sets the accumulate bit
`UMMA::ScaleOut::Zero` on **its own first k** (`k==0 && kb==0`) and `One` thereafter —
so the accumulator is **written fresh** at the start of each phase and the `dUᵀ·X`
result is **not** carried into the `dVᵀ·X` phase. The failure mode to avoid (the mirror
of mlp5's "dropped term") is *accidentally carrying* the acc across phases, or failing to
re-zero the REDUCE_ADD outputs — both surface immediately in the two phase-isolation
diagnostics (§4).

**Why this suits SM100:** two `(128,128)` TMEM accumulators live simultaneously with no
register-resident partials, the `ScaleOut` bit expresses "clear vs accumulate" for free
per instruction, and a single MMA warp feeds both consumer WGs — landing at **93
registers / 0 spills** (M-split) despite the wide `EpiChunkN=64` fragment.

---

## 3. Performance

### B200 (sm_100a / UMMA) — measured, `MLP4_BENCH=1 … TFLOPs_Blackwell`

FLOPs **`4·T·H·I`** (two GEMMs); median CUDA-event timing (pure kernel, memset
excluded); `outer_split` swept over the divisors of the walk axis; `H=I=4096, E=8`;
`grid.x = num_sms = 148`. B200 dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | peak TFLOPS | winning split | grid.x | %peak |
|---|:-----------:|:-------------:|:------:|:-----:|
| 2048  | 471.59  | osplit=4  | 148 | 21.0% |
| 4096  | 724.16  | osplit=16 | 148 | 32.2% |
| 8192  | 921.67  | osplit=8  | 148 | 41.0% |
| 16384 | **1144.18** | osplit=4  | 148 | **50.9%** |

Peak **1144 TFLOPS @ T=16384** (50.9% of peak). %peak climbs monotonically with `T`.

### Cross-tile-size trend (why %peak climbs with T)

mlp4 is a **weight-gradient** kernel: the output `dB/dC : [E·I, H]` is *large* and every
tile is written via an **atomic global REDUCE_ADD**, while the two phases each **reload**
the A operand (`dUᵀ` then `dVᵀ`) — only `X` is shared. It is therefore markedly more
**epilogue/bandwidth-bound** than the forward/input-gradient kernels, which is why its
absolute %peak sits below mlp5's. As `T` (the contraction length) grows, the mainloop
FLOPs amortize the fixed REDUCE_ADD epilogue and the per-phase A reload, so utilization
rises steadily **21% → 32% → 41% → 51%**. The winning `outer_split` (which partitions the
walk axis to fill all 148 SMs) tracks the shape: small `T` prefers a moderate split to
spread the few chunks across SMs, and even at large `T` the epilogue-bound regime keeps a
≥4-way split optimal.

### sm90a → sm100a speedup (measured on H100)

Measured on an **NVIDIA H100 80GB HBM3** (`sm_90a`, 132 SMs), WGMMA `Impl<90>` path in
`build90a` — `Mlp4.TFLOPs_Hopper`, same `outer_split` sweep and shapes (`H=I=4096`,
`E=8`); FLOPs `4·T·H·I`:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a --target test_mlp4 -j
MLP4_BENCH=1 ./liger_cute_kernels/build90a/tests/cpp/test_mlp4 --gtest_filter='*TFLOPs_Hopper*'
```

| T | H100 peak TFLOPS | winning `outer_split` | grid.x | %peak | B200÷H100 |
|---|:----------------:|:---------------------:|:------:|:-----:|:---------:|
| 2048  | 238.91 | 1 | 132 | 24.1% | 1.97× |
| 4096  | 395.96 | 1 | 132 | 40.0% | 1.83× |
| 8192  | 538.14 | 2 | 132 | 54.4% | 1.71× |
| 16384 | **639.12** | 4 | 132 | **64.6%** | 1.79× |

`%peak` is vs the H100 SXM bf16 dense peak ≈ **989.4 TFLOPS** (B200÷H100 computed against
this writeup's B200 table above); correctness passes on the same binary
(`Mlp4Sm90.Correctness`, max_rel ≤ 1.00%). **B200÷H100 = 1.71–1.97×**.

**Measured outcome vs hypothesis.** The prediction — mlp4 leans toward the **~2.4×
bandwidth ratio** as the most epilogue-bound port — is **not** borne out: the measured
speedup is **1.71–1.97×**, *below* both the 2.4× bandwidth and 2.27× compute ratios, and
it actually **declines** from small to mid `T` (1.97× → 1.71×) before ticking back up.
The reason is that mlp4's WGMMA `Impl<90>` path is *relatively efficient* on H100 at mid
`T`: %peak climbs faster there (H100 **24% → 65%**) than on B200 (**21% → 52%**), so the
H100 partially closes the gap exactly where the B200 has not yet saturated — pulling the
ratio to its **1.71× minimum at `T=8192`**. Both arches' `%peak` rise monotonically (the
deeper `K=T` reduction amortizes the atomic REDUCE_ADD write + per-phase `A` reload), so
like mlp3 the speedup stays in a narrow **<2×** band rather than tracking either peak
ratio.

---

## 4. Blockers hit & code changes

The mlp4-unique hazard (independent per-phase accumulator) was pre-empted by design and
**did not** bite at runtime; the substantive change was the **store-buf M-row mapping**,
which *differs* from WGMMA and had to be re-derived from the UMMA thread-value layout.

- **Independent per-phase clear (primary anticipated risk):** handled by
  `ScaleOut::Zero` on the first k of **each** phase, `One` after (k-loop restarts per
  phase, so the bit is reset at every phase start — the *inverse* of mlp5, which must
  **not** reset it). The test adds **two phase-isolation diagnostics** —
  `Mlp4.PhaseDB` (`dV=0`, isolates `dUᵀ·X`; checks `dB` matches and `dC≡0`) and
  `Mlp4.PhaseDC` (`dU=0`, isolates `dVᵀ·X`). Both pass **and** the combined
  `Mlp4.Correctness` passes, confirming no acc-carry-across-phases and correct re-zero.
- **Store-buf mapping pin (the real change) — `partition_D` identity, NOT the WGMMA
  remap.** The UMMA C-layout is **contiguous per-WG**, so the store-buf row comes
  **directly** from the TMEM-load `partition_D` identity coords (0..127) — the WGMMA
  `m_local = (m_loc/128)*64 + (m_loc%64)` interleave remap is **gone**. Consequently the
  **M-split gmem atom-row formula changes** from WGMMA's `4·out_m + my_wg + 2·a` to
  **`4·out_m + 2·my_wg + a`** (= `kRowAtoms·out_m + kAtomsPerWg·my_wg + a`): under Option
  B, WG *w* owns the contiguous tile-rows `[128w, 128w+128)` = stripes `{2w, 2w+1}`, so
  atom *a* covers stripe `2w+a`. The N-split formula (`2·out_m + a`) and both configs'
  `n_tile_idx` are **byte-identical** to WGMMA. Validated **element-by-element** vs the
  fp32 CPU reference on tiny single-tile shapes (PhaseDB/DC) and across both M-split and
  N-split configs (Correctness).
- **TMEM lifecycle vs persistent grid + two phases (B5/B5′):** the alloc/free is
  **consumer-owned** and hoisted **once per CTA, ABOVE both the `cell_idx` loop and the
  `phase` loop**. A persistent grid *and* two phases make a per-phase/per-cell
  `tcgen05.alloc` a **double** trap (`relinquish_alloc_permit` is a permanent
  once-per-CTA relinquish); hoisting above both loops is mandatory. Whole **MMA warp**
  issues the alloc (`if (is_mma_warp) { …allocate…; __syncwarp(); }`), never a single
  `elect_one` lane (**B5**).
- **smem budget:** the single **reused** A buffer (`dUᵀ` then `dVᵀ`, not two buffers) +
  shared `X` + store-buf at `Stages=4` is ~224 KiB, within the B200's ~227 KiB — no
  `Stages` reduction needed.

`Impl<90>` (WGMMA), the producer (already phase-looped), the REDUCE_ADD leader block,
and the persistent grid walk are otherwise unchanged.

---

## 5. Register-spill check (the checker)

Measured on the B200 (`sm_100a`) for the `Compute=100` UMMA consumer, exact `build100a`
flags:

```bash
export CUTLASS_HOME=/usr/local/include/cutlass
nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
     --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
     --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
     -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
     -c liger_cute_kernels/tests/cpp/test_mlp4.cu -o liger_cute_kernels/build100a_mlp4/mlp4_spill.o
```

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp4` M-split `(256,128)` (default) | **93** | **0 B** | **0 B** |
| `mlp4` N-split `(128,256)`           | **91** | **0 B** | **0 B** |

**Zero spills** — the hard requirement. Both configs sit **below the mlp5 yardstick of
116** (same `EpiChunkN=64`): the two independent `(128,128)` TMEM accumulators keep the
epilogue fragment narrow (one WG's `128×64` at a time), and the reused A buffer avoids a
second live operand. No `-maxrregcount` / `__launch_bounds__` cap was required (the
kernel already carries `__launch_bounds__(NumThreads, 1)`).
