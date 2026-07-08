# writeup_mlp1 — MLP1 fused consumer on Blackwell (SM100a)

> **Op:** `Z = SiLU(U) · V` where `U = X·Bᵀ`, `V = X·Cᵀ` — the MoE **phase-1
> up/gate projection**, two GEMMs fused with a SiLU-gated elementwise epilogue.
> **Files:** `csrc/core/src/moe/mlp1_fused.cuh`, `mlp1_fused_act.cuh`,
> `tests/cpp/test_mlp1_fused.cu`. This is the **reference port** every other kernel
> (mlp2, mlp2_t, mlp5) is cloned from.

---

## 1. The nuance of MLP1

In a fused-MoE feed-forward block, phase 1 takes the routed token activations
`X : [T, H]` (per expert) and produces the gated intermediate

```
U = X · Bᵀ            (gate branch,  [T, I])
V = X · Cᵀ            (up   branch,  [T, I])
Z = SiLU(U) · V       (elementwise,  [T, I])   → feeds phase 2 (the down-proj)
```

The nuances that shape the kernel:

- **Two GEMMs that share the operand `X`.** Both `U` and `V` contract the *same*
  `X` tile over `H`. The fused consumer therefore keeps **two accumulators per
  warpgroup** (`acc_B = U`, `acc_C = V`) and loads `X` once — this operand reuse is
  what makes MLP1 more arithmetic-intense (and higher %peak) than a lone GEMM.
- **The activation is fused into the epilogue.** After the two GEMMs, each thread
  computes `SiLU(U)·V` (and, in the `act` variant, also the backward-friendly
  `U'`/`V'` stores) with **no cross-warpgroup sync** — a per-thread epilogue.
- **Cooperative 2-WG consumer**, split axis chosen by `TileM` (M-split at
  `TileM=128`, N-split at `TileM=64`); `EpiChunkN=64`.
- **Two variants:** `mlp1_fused` (GEMM + SiLU·V) and `mlp1_fused_act` (additionally
  stores `U'`,`V'`,`Z` for the backward pass).

FLOPs (two GEMMs, activation ignored as negligible): `4·T·H·I`.

---

## 2. What changed to suit SM100

The Hopper body computes both GEMMs with **WGMMA** (`cute::gemm` over
`GMMA::Layout_K_SW128` operands) into **register** accumulators
(`partition_fragment_C`). The Blackwell body keeps that verbatim as `Impl<90>` and
adds a new `Impl<100>` built on **UMMA / tcgen05**:

| Piece | Hopper (`sm_90a`, WGMMA) | Blackwell (`sm_100a`, UMMA) |
|-------|--------------------------|------------------------------|
| MMA | `TiledMMA<MMA_Atom<GmmaAtom>>`, warpgroup-cooperative | tcgen05 `TiledMMA`, one MMA warp issues the whole tile |
| Accumulator | register fragment (`partition_fragment_C`) | **TMEM** (tensor memory), 2 acc = `2·TileN` cols |
| Mainloop pipe | `PipelineTmaAsync` | **`PipelineTmaUmmaAsync`** (`mlp1_make_pipe_umma`, `num_consumers=1`) |
| Epilogue read | registers → cast/activation | **`TmemLoadOp<64>`** (TMEM→reg) → same cast/activation |
| TMEM lifetime | n/a | whole-MMA-warp `tcgen05.alloc`/`free` + `__syncwarp()` |

The shared recipe introduced here (and reused by mlp2/mlp2_t/mlp5): a compile-time
`TmemLoadOp<EpiChunkN>` selector; extract the `(M,N)` accumulator view first, then
`flat_divide` by the epi tile; size the register fragment from `partition_D`; keep
the existing per-thread epilogue math unchanged.

**Why this suits SM100:** tcgen05 issues a full-tile MMA from a *single* warp (vs a
128-thread warpgroup on Hopper), and the accumulator lives in **TMEM** instead of the
register file — so a large `(M, 2·TileN)` accumulator no longer competes with the
epilogue for registers. That is exactly why the Blackwell consumers hit low register
counts with zero spills (see §5).

---

## 3. Performance (from `blackwell.md`)

### B200 (sm_100a / UMMA) — measured, `MLP1_BENCH=1 … TFLOPs_Blackwell`

FLOPs `4·T·H·I`; median of CUDA-event-timed launches; N-split swept over every
divisor, peak reported. `H=I=4096`, `E=8`. B200 dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | Fused TFLOPS (split) | Fused+Act TFLOPS (split) | Fused %peak |
|---|:--------------------:|:------------------------:|:-----------:|
| 2048  | 983.17  (8) | 779.70 (8)  | 43.7% |
| 4096  | 1081.38 (4) | 884.51 (32) | 48.1% |
| 8192  | **1090.44** (2) | **897.64** (2) | **48.5%** |
| 16384 | 1020.30 (1) | 843.47 (8)  | 45.3% |

Peak **1.09 PFLOPS** (fused) / **0.90 PFLOPS** (act).

### sm90a → sm100a speedup (measured on H100)

Measured on an **NVIDIA H100 80GB HBM3** (SXM5, `sm_90a`, 132 SMs) via the WGMMA
`Impl<90>` path in the `build90a` binary — `Mlp1Fused.TFLOPs_Hopper` +
`Mlp1FusedAct.TFLOPs_Hopper`, same N-split sweep and shapes (`H=I=4096`, `E=8`):

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a --target test_mlp1_fused -j
MLP1_BENCH=1 ./liger_cute_kernels/build90a/tests/cpp/test_mlp1_fused --gtest_filter='*TFLOPs_Hopper*'
```

| T | H100 Fused (split) | H100 Fused+Act (split) | Fused %peak | B200÷H100 (fused) |
|---|:------------------:|:----------------------:|:-----------:|:-----------------:|
| 2048  | 557.68 (16) | 483.67 (8) | 56.4% | 1.76× |
| 4096  | 670.12 (4)  | 550.58 (4) | 67.7% | 1.61× |
| 8192  | **694.71** (2) | **571.65** (2) | **70.2%** | 1.57× |
| 16384 | 595.20 (1)  | 499.87 (1) | 60.2% | 1.71× |

Peak **694.71 TFLOPS** (fused) / **571.65** (act) on H100; `%peak` is vs the H100 SXM
bf16 dense peak ≈ **989.4 TFLOPS**. The fused **B200÷H100 speedup is 1.57–1.76×**
(act 1.57–1.69×) — **below** the ~2.27× compute-peak ratio, because on H100 the WGMMA
path already reaches a *higher* %peak (56–70%) than UMMA does on B200 (44–48%): the
B200's extra FLOPs are partly offset by its lower utilization at these shapes, so the
realized speedup tracks the effective-throughput ratio, not the raw peak ratio.

**Why the gain is what it is (mechanism).** The port trades WGMMA for tcgen05/UMMA on
a much larger machine:

- **Peak compute:** B200 bf16 ≈ 2.25 PFLOPS vs H100 ≈ 0.99 PFLOPS ⇒ **~2.27×**.
- **HBM bandwidth:** B200 HBM3e ≈ 8 TB/s vs H100 HBM3 ≈ 3.35 TB/s ⇒ **~2.4×**.

MLP1 amortizes the shared `X` load across two GEMMs, so it sits partway between
compute- and memory-bound. A naïve peak-ratio estimate would put the speedup in the
**~2.2–2.4×** band, *but that assumed* the WGMMA path reaches a %peak on H100 comparable
to the ~48% UMMA hits on B200. In practice H100 WGMMA runs at a **higher** %peak
(56–70%), so the measured speedup lands lower, at **1.57–1.76×**.

### Cross-tile-size trend (why TFLOPS rises then falls)

The peak walks **983 → 1081 → 1090 → 1020** as `T` grows 2048→16384, and the winning
N-split shrinks **8 → 4 → 2 → 1**. Hypotheses:

- **Small `T` (2048): occupancy-limited.** These consumers are smem-bound (~1
  CTA/SM). At `T=2048` there are only ~16 M-tiles, far below the B200's 148 SMs, so a
  large N-split (×8) is needed to manufacture enough CTAs to fill the machine — and
  the fill is still imperfect (43.7% peak).
- **Mid `T` (4096–8192): best fill.** M-tiles alone approach the SM count; a small
  split tops it off. Arithmetic intensity (shared-`X` reuse) is fully exposed → the
  48% plateau.
- **Large `T` (16384): memory-bound tail.** The working set and total HBM traffic
  grow; the m-dimension alone over-fills the SMs (split→1), and throughput dips
  slightly as the kernel becomes bandwidth-limited rather than issue-limited.

The `act` variant is uniformly lower (~0.90 vs 1.09 PFLOPS) because it additionally
**stores `U'`/`V'`/`Z`** and computes SiLU + its derivative — extra HBM writes and
epilogue work over the same GEMM FLOPs.

---

## 4. Blockers hit & code changes (compile + runtime)

MLP1 was the *first* time this code was ever built for `sm_100a`, so it exposed five
latent bugs (the yardstick set B1–B5 that pre-empted the later ports):

| # | Blocker (symptom) | Fix |
|---|-------------------|-----|
| **B0** | Full build won't configure (`tvm-ffi-config` absent) | Added `LIGER_CUTE_TESTS_ONLY` + `LIGER_CUTE_CUDA_ARCH` CMake knobs → a tests-only `sm_100a` path. |
| **B1** | `static_assert … RegNumDst` — TMEM-load atom hard-coded to `…1x` | Compile-time `TmemLoadOp<EpiChunkN>` selector (`64→…64x`, `32→…32x`). |
| **B2** | `"AtomTVLayout does not exist"` — `zipped_divide` gave a nested rank-1 tile | Switch the epilogue to **`flat_divide`** (flat `(M,N)` tile). |
| **B3** | 5-mode tensor — raw MMA C-fragment fed to `flat_divide` | Extract the `(M,N)` view first: `tCtAcc(make_coord(_,_), _0{}, _0{})`. |
| **B4** | `size(rD) == RegNumDst` (2048 ≠ 64) — regs sized from `partition_S` | Size the register fragment from **`partition_D`**. |
| **B5** | **Runtime deadlock** (100% spin, `synccheck` clean) | `tcgen05.alloc` is **warp-synchronous** but was issued from a single `elect_one` lane → hang. Issue `allocate`/`free`/`release_allocation_lock` from the **whole MMA warp** + `__syncwarp()`. |

B5 was the highest-impact and the defining SM100 footgun: a `.sync.aligned` op from
one divergent lane silently hangs the warp. It was localized with `#ifdef MLP1_DBG`
`printf` breadcrumbs under a hard `timeout` after `synccheck` ruled out a
barrier-count bug; the scaffolding was then removed.

---

## 5. Register-spill check (the checker)

Measured on the B200 (`sm_100a`) for the `Compute=100` UMMA consumer, using the exact
`build100a` flags:

```bash
export CUTLASS_HOME=/usr/local/include/cutlass
nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
     --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
     --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
     -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
     -c liger_cute_kernels/tests/cpp/test_mlp1_fused.cu -o /tmp/mlp1_spill.o
# grep: ptxas info : Used N registers ...  0 bytes spill stores, 0 bytes spill loads
```

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp1_fused`         | 160 | **0 B** | **0 B** |
| `mlp1_fused` (act)   | 162 | **0 B** | **0 B** |

**Zero spills.** These are the highest register counts of the ported family (two
accumulators + SiLU epilogue), and they serve as the **yardstick**: a faithful
single-accumulator port (mlp2/mlp2_t) should land well below, and it does (73/79).
No `-maxrregcount` / `__launch_bounds__` was needed.
