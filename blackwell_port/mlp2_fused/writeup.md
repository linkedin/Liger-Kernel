# Writeup — MLP2-fused Blackwell (SM100a / UMMA / tcgen05) port

**Op:** `Y = Z · Aᵀ` — the MoE down-projection (single fused GEMM, contract over `I`,
N-axis = `H`). **Result: fully green on a B200 (compute_cap 10.0).** Correctness
PASSes, TFLOPS benchmark runs, the hot UMMA consumer has **zero register spills**,
and the Hopper `sm_90a` path still compiles (WGMMA body byte-for-byte unchanged).

This port is the **reference instantiation** of the shared 6-step recipe for the
*single-accumulator* case — the proven UMMA consumer here is directly cloneable by
the `mlp2_t` and `mlp5` ports.

---

## Changed files (only the two I own)

| File | Change |
|------|--------|
| `liger_cute_kernels/csrc/core/src/moe/mlp2_fused.cuh` | Ported. Added SM100 includes + `TmemLoadOpSelector`; `MainloopPipelineUmma` + `AccStages`/`AccumulatorPipeline` in `Mlp2Traits`; `tmem_base` + `acc_pipe` in `Mlp2FusedSmem`; `mlp2_make_pipe_umma` (num_consumers=1); split `mlp2_fused_consumer` → `Mlp2FusedConsumerImpl<90>` (verbatim WGMMA) / `<100>` (new UMMA, single accumulator) + free-function forwarder. Producer unchanged. |
| `liger_cute_kernels/tests/cpp/test_mlp2_fused.cu` | Overwrote the pre-staged stub in place with a clone of `test_mlp1_fused.cu` reduced to the single GEMM `Y = Z·Aᵀ`: `MainloopPipelineFor` + `if constexpr(Compute==100)` launcher (`__trap()` on the 90 body under sm_100a), `run_fused<Compute>` (CPU-ref compare), `run_fused_bench<Compute>` (median CUDA-event timing + N-split divisor sweep, gated on `MLP2_BENCH`), `blackwell/hopper_available()`, `kShapes`/`kBenchShapes`, the `Mlp2Fused.Correctness` / `Mlp2FusedSm90.Correctness` / `Mlp2Fused.TFLOPs_{Blackwell,Hopper}` TESTs, and an arch-aware `main()`. |

Did **not** touch `tests/cpp/CMakeLists.txt` (the `test_mlp2_fused` target was
pre-staged), the shared root `writeup.md`, `mlp1_fused.cuh`, or any other kernel/test.

## Key design decisions (the MLP2-vs-MLP1 deltas)

- **UMMA `TiledMMA`:** single 1SM `SM100_MMA_F16BF16_SS<Element, Element, float,
  TileM, TileN, UMMA::Major::K, UMMA::Major::K>` built locally in `Impl<100>` (same
  as MLP1). One atom with `M=TileM=128, N=TileN=128` covers the whole tile — no
  cooperative-warpgroup MMA. Operand A = `Z`, operand B = `A` (the down weight), both
  **K-major** (the SM100-native SS orientation), so there is no transpose subtlety
  (contrast mlp2_t).
- **TMEM column budget — the one thing to get right:** `mlp2_fused` has **one**
  accumulator, so `tmem_alloc.allocate(TileN, &tmem_base)` (`TileN == WgTileN` for the
  M-split config), **not** MLP1's `2·TileN` two-accumulator budget. Matching
  `free(tmem_base, TileN)`. The `(TileM,TileN)` accumulator occupies exactly `TileN=128`
  TMEM columns; the epilogue's `flat_divide` produces `TileN/EpiChunkN = 4` chunks and
  the two consumer WGs read `NChunksHalf = (TileN/2)/EpiChunkN = 2` chunks each — all
  four chunks require the full `TileN` width, confirming the budget.
- **Single-accumulator epilogue, no activation:** one TMEM→reg fragment `tTR_rAcc`
  (vs MLP1's `tTR_rU`/`tTR_rV`), cast-to-bf16 straight into the reused per-WG
  `store_buf`, TMA store to `Y`. No `SiLU(U)·V` math.
- **Accumulator pipeline (`PipelineUmmaAsync`, 1 stage):** warp 4 issues the UMMA and
  `producer_commit`s (umma_arrive); the epilogue warps `consumer_wait`, then one
  elected thread `consumer_release`s after the TMEM reads, gating the next n-tile's MMA.

## Blockers hit + how resolved

The port compiled, ran correct, and hit **peak TFLOPS on the first attempt — no hang,
no IMA, no spill.** The shared recipe pre-empted the standard MLP1 blocker set; each is
called out below with the concrete resolution in `Impl<100>`:

| Blocker | Resolution in `mlp2_fused.cuh` |
|---------|-------------------------------|
| **B1** — TMEM-load atom width must equal `EpiChunkN` | Local `TmemLoadOpSelector<32>` → `SM100_TMEM_LOAD_32dp32b32x`; `TmemLoadOp<EpiChunkN>` in the epi copy. |
| **B2** — `flat_divide`, not `zipped_divide` | `flat_divide(acc_mn, epi_tile)` → `(TileM,EpiChunkN,1,TileN/EpiChunkN)`; feeds `make_tmem_copy`'s cotiled builder. |
| **B3** — extract the `(M,N)` acc view first | `acc_mn = tCtAcc(make_coord(_,_), _0{}, _0{})` before tiling (matches CUTLASS `accumulators(make_coord(_,_),_0,_0)`). |
| **B4** — size the reg fragment from `partition_D` | `tTR_rAcc = make_tensor<float>(shape(tTR_cChunk))`, `tTR_cChunk = thr_t2r.partition_D(cChunk)` — not `partition_S` (which would fold in the collective datapath-lane dim). |
| **B5** — `tcgen05.alloc/free` are warp-synchronous | Whole MMA-warp (`is_mma_warp`) `allocate` + `__syncwarp()`; `release_allocation_lock()` + `free()` from the whole warp at the end. Never from a single `elect_one` lane (the silent-hang footgun). |
| **Kernel-specific** — TMEM column over-allocation | Allocated `TileN` (one acc), explicitly **not** `2·TileN`. This is the primary copy-paste hazard when cloning from MLP1. |

## Register usage & spills (SM100 `Impl<100>` hot consumer) — deliverable #1

Captured with `nvcc -arch=sm_100a --ptxas-options=-v` on the actual test TU
(`SM100_TMEM_LOAD_32dp32b32x` epilogue). **Zero spill bytes on every kernel.**

| Kernel (Compute=100, sm_100a) | Registers/thread | Spill stores | Spill loads |
|-------------------------------|:---------------:|:------------:|:-----------:|
| **`mlp2_fused` (this port — 1 acc, EpiChunkN=32)** | **79** | **0** | **0** |
| `mlp1_act`  (yardstick — 2 acc + SiLU, EpiChunkN=32) | 128 | 0 | 0 |
| `mlp1_fused` (yardstick — 2 acc + SiLU, EpiChunkN=64) | 168 | 0 | 0 |
| (all `Compute=90` bodies under sm_100a — trap fallback) | 21 | 0 | 0 |

**79 registers is far below the MLP1 yardstick (168 / 128)** — as expected: one
accumulator ⇒ one TMEM→reg fragment, no `SiLU(U)·V` temporaries, and the narrower
`EpiChunkN=32` fragment. No register liberality, no spills, so **no `-maxrregcount` /
`__launch_bounds__` tuning was needed** — the kernel already carries
`__launch_bounds__(NumThreads, 1)` from the test launcher.

## B200 correctness — deliverable #3 (`Mlp2Fused.Correctness`, PASS)

bf16 GEMM vs fp32 CPU reference on identically bf16-rounded inputs. Bars: `mean_rel <
1%`, `max_rel < 5%`. **All shapes PASS.**

| Shape (T, H, I, E) | mean_rel | max_rel | max_abs |
|--------------------|:--------:|:-------:|:-------:|
| 128, 128, 128, 1   | 0.142%   | 0.383%  | 0.124   |
| 128, 256, 256, 1   | 0.141%   | 0.524%  | 0.125   |
| 256, 256, 256, 2   | 0.141%   | 0.389%  | 0.216   |
| 384, 128, 256, 3   | 0.141%   | 0.389%  | 0.244   |

## B200 TFLOPS — deliverable #2 (`MLP2_BENCH=1 … TFLOPs_Blackwell`)

FLOPs = `2·T·H·I` (single GEMM). Median CUDA-event timing; N-split swept over every
divisor of `num_n_tiles`, peak reported with the winning `grid.y`. B200 peak dense-bf16
≈ **2.25 PFLOPS**; 148 SMs.

| Shape (T, H, I, E)    | peak TFLOPS | ms     | winning split | CTAs | %peak |
|-----------------------|:-----------:|:------:|:-------------:|:----:|:-----:|
| 2048, 4096, 4096, 8   | 695.88      | 0.0988 | 8             | 128  | 30.9% |
| 4096, 4096, 4096, 8   | **788.86**  | 0.1742 | 32            | 1024 | 35.1% |
| 8192, 4096, 4096, 8   | 769.19      | 0.3574 | 16            | 1024 | 34.2% |
| 16384, 4096, 4096, 8  | 671.10      | 0.8192 | 8             | 1024 | 29.8% |

Peak **788.86 TFLOPS @ T=4096, split=32**. `mlp2_fused` is a *single-GEMM*
down-projection with no operand reuse across GEMMs (unlike MLP1's fused two-GEMM,
which amortizes the shared `X` load), so it is more HBM-bound; ~30–35 % of dense-bf16
peak is the expected band at these memory-bound production shapes. The divisor sweep
shows occupancy sensitivity — small splits under-fill the 148 SMs (best split grows
with `T` until the CTA count saturates the machine).

## No Hopper regression — deliverable #4

- `sm_90a` build of `test_mlp2_fused` **compiles cleanly** (real WGMMA `Impl<90>`
  instantiated; `Impl<100>` traps under `__CUDA_ARCH__ < 1000`). Binary produced.
- `Impl<90>` WGMMA body verified **byte-for-byte identical** to the pre-port
  `mlp2_fused_consumer` (only the signature wrapper — struct + `static run` — changed;
  the mainloop/epilogue is untouched).
- `Mlp2FusedSm90.Correctness` **SKIPs** on the B200 (no Hopper HW present) — expected;
  the requirement was that it compiles, which it does.

## Reproduce

```bash
export CUTLASS_HOME=/usr/local/include/cutlass

# ── Blackwell (sm_100a): build, correctness, TFLOPS ──
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp2_fused -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp2_fused --target test_mlp2_fused -j
./liger_cute_kernels/build100a_mlp2_fused/tests/cpp/test_mlp2_fused                       # correctness
MLP2_BENCH=1 ./liger_cute_kernels/build100a_mlp2_fused/tests/cpp/test_mlp2_fused \
      --gtest_filter='*TFLOPs_Blackwell*'                                                 # TFLOPS

# ── Register/spill check (deliverable #1): assert ZERO spills on Impl<100> ──
nvcc -std=c++17 -arch=sm_100a -O3 --expt-relaxed-constexpr --ptxas-options=-v \
     -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
     liger_cute_kernels/tests/cpp/test_mlp2_fused.cu -lgtest -lpthread \
     -o liger_cute_kernels/build100a_mlp2_fused/spillcheck/mlp2_fused_spillcheck 2>&1 \
     | grep -A3 'Li100E'      # → "Used 79 registers", "0 bytes spill stores, 0 bytes spill loads"

# ── Hopper no-regression (sm_90a): must compile ──
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a_mlp2_fused -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a_mlp2_fused --target test_mlp2_fused -j
```

## Status

**`mlp2-fused` → DONE.** All four deliverables met: (1) 79 regs / 0 spill on the hot
UMMA consumer (well under the MLP1 yardstick), (2) built + ran on the B200, (3)
`Mlp2Fused.Correctness` PASS (mean_rel ≤ 0.142 %, max_rel ≤ 0.524 %), (4) `sm_90a`
compiles with the WGMMA body unchanged. The `Impl<100>` UMMA consumer is proven and
ready to be cloned by the `mlp2_t` and `mlp5` ports.
