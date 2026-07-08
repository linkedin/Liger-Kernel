# Blackwell (SM100a) MoE consumer kernels — status & how to build/test

This document covers the Blackwell (`sm_100a` / UMMA / tcgen05) port of the fused MoE
**consumer** kernels in `liger_cute_kernels`, and the exact commands to compile them and
run correctness + TFLOPS on a B200.

Each kernel keeps its original Hopper (`sm_90a` / WGMMA) consumer **byte-for-byte** as
`Impl<90>` and adds a new Blackwell UMMA `Impl<100>` selected by an `int Compute`
template parameter. The producer is arch-agnostic and shared. The same test source
compiles for both arches — the opposite-arch body is `__trap()`-guarded via
`__CUDA_ARCH__`, so a `sm_100a` binary runs the UMMA path and a `sm_90a` binary runs the
WGMMA path.

- **Reference / template:** `mlp1_fused` (ported first; the pattern every other kernel follows).
- **This effort added:** `mlp2_fused`, `mlp2_t` ("mlp2"), the two REDUCE_ADD
  weight-gradient kernels `mlp3` and `mlp4`, and `mlp5`.
- **Hardware used:** NVIDIA **B200** (compute capability 10.0, `sm_100a`), CUDA 12.9,
  CUTLASS 4.4.1 (header-only), CMake + Ninja.
- **Pass bar (bf16):** `mean_rel < 1%`, `max_rel < 5%` vs an fp32 CPU reference on
  identically bf16-rounded inputs.
- **B200 dense-bf16 peak reference:** ≈ **2.25 PFLOPS** (2250 TFLOPS), 148 SMs.

---

## The kernels

| Kernel | Op | GEMMs / FLOPs | `EpiChunkN` | Bench env var | Test source |
|--------|-----|---------------|:-----------:|:-------------:|-------------|
| **mlp1_fused** | `Z = SiLU(B·X)·(C·X)` (phase-1, 2 accumulators + SiLU fusion) | 2 GEMMs, `4·T·H·I` | 64 | `MLP1_BENCH` | `test_mlp1_fused.cu` |
| **mlp2_fused** | `Y = Z·Aᵀ` (down-projection) | 1 GEMM, `2·T·H·I` | 32 | `MLP2_BENCH` | `test_mlp2_fused.cu` |
| **mlp2_t** ("mlp2") | `Y = Z·A` (transpose; MN-major weight) | 1 GEMM, `2·T·H·I` | 32 | `MLP2T_BENCH` | `test_mlp2_t_fused.cu` |
| **mlp3** | `dA = dYᵀ·Z` (down-weight grad; REDUCE_ADD) | 1 GEMM, `2·T·H·I` | 64 | `MLP3_BENCH` | `test_mlp3.cu` |
| **mlp4** | `dB = dUᵀ·X`, `dC = dVᵀ·X` (weight grads; 2-phase, REDUCE_ADD) | 2 GEMMs, `4·T·H·I` | 64 | `MLP4_BENCH` | `test_mlp4.cu` |
| **mlp5** | `dX = dU·B + dV·C` (backward input grad) | 2 GEMMs, `4·T·H·I` | 64 | `MLP5_BENCH` | `test_mlp5_fused.cu` |

### Per-kernel status (all GREEN on B200)

| Kernel | `Impl<100>` regs / spills | Correctness (max_rel) | Peak TFLOPS (@ shape, split) | % of B200 peak |
|--------|:------------------------:|:---------------------:|:----------------------------:|:--------------:|
| mlp1_fused | 160 / **0** (yardstick) | ≤ 0.85% (fused) / ≤ 1.39% (act) | **1090.44** @ T=8192, split 2 | 48.5% |
| mlp2_fused | **73 / 0** | ≤ 0.52% | **788.86** @ T=4096, split 32 | 35.1% |
| mlp2_t | **79 / 0** | ≤ 0.40% | **826.27** @ T=4096, split 32 | 36.7% |
| mlp3 | **88 / 0** | ≤ 0.48% | **1214.64** @ T=16384, split 4 | 54.0% |
| mlp4 | **93 / 0** (M-split) · 91 (N-split) | ≤ 1.00% | **1163.87** @ T=16384, split 4 | 51.7% |
| mlp5 | **116 / 0** | ≤ 1.07% | **1156.08** @ T=8192, split 2 | 51.4% |

All ports have **zero register spills** on the hot UMMA consumer (verified with
`--ptxas-options=-v`); no `-maxrregcount` / `__launch_bounds__` cap was needed. The
`sm_90a` (Hopper) build of every kernel still compiles (WGMMA `Impl<90>` unchanged).

### Kernel-specific notes

- **mlp2_fused** — the reference instantiation of the shared recipe: single TMEM
  accumulator (allocate `WgTileN` columns, **not** MLP1's `2·TileN`), both operands
  K-major, cast-only epilogue. Green on the first attempt.
- **mlp2_t** — same epilogue as mlp2_fused, but the weight `A` is consumed **MN-major**.
  Resolved with the `_SS` UMMA atom carrying the major modes explicitly:
  `SM100_MMA_F16BF16_SS<Element,Element,float,TileM,TileN,UMMA::Major::K,UMMA::Major::MN>`
  (A = `Z` K-major, B = `A` MN-major). The existing `Layout_MN_SW128_Atom` + `Step<_2,_1>`
  smem layout feeds the UMMA descriptor unchanged. CPU reference is `Y = Z·A` (**not** `Aᵀ`).
- **mlp3** — the down-weight gradient `dA = dYᵀ·Z`, and the first ported kernel whose
  epilogue is a hardware **atomic-add** (`SM90_TMA_REDUCE_ADD`) into a caller-zeroed
  `[E,H,I]` buffer. `SM90_TMA_REDUCE_ADD` is `__CUDA_ARCH__`-guarded in CUTLASS 4.4.1, so
  the whole reduce-add store block is kept **byte-for-byte** under `sm_100a`; only the step
  that *fills* `store_buf` moves from a register scatter to a TMEM load. It is also the
  first with **both operands MN-major** (`SM100_MMA_F16BF16_SS<…,Major::MN,Major::MN>` —
  A=`dYᵀ`, B=`Z`). The store-buf `(m_local,n_local)` mapping is re-derived from the UMMA
  TMEM-load `partition_D` identity coords (the WGMMA M-atom-interleave remap does **not**
  carry over). One **1-SM tcgen05 atom** spans the whole tile with a single TMEM
  accumulator, so the UMMA path supports the **N-split config (`TileM=128`) only**
  (`static_assert TileM==WgTileM`); the M-split config (`TileM=256`) exceeds the 1-SM M≤128
  cap and stays on the untouched WGMMA `Impl<90>`. TMEM alloc/free hoisted **once per CTA**
  (persistent chunk-fixed grid). 88 regs / 0 spills.
- **mlp4** — the two weight gradients `dB = dUᵀ·X`, `dC = dVᵀ·X`, mirroring mlp3's
  REDUCE_ADD epilogue but produced in **two sequential phases per cell** (phase 0 = dB via
  `dUᵀ`, phase 1 = dC via `dVᵀ`), reusing one A-side smem buffer and the shared `X`. The
  two phases have **independent** accumulators — `ScaleOut::Zero` is issued fresh at the
  first MMA of **each** phase (the opposite of mlp5, which never resets). TMEM alloc/free is
  hoisted **once per CTA — above both the cell loop and the phase loop**. Because each WG
  owns its own `(128×128)` TMEM accumulator (atom M=128, within the 1-SM cap), mlp4's UMMA
  path supports **both** the M-split (`256,128`, default) and N-split configs. Two
  phase-isolation tests (`Mlp4.PhaseDB` with `dV=0`, `Mlp4.PhaseDC` with `dU=0`) confirm no
  acc-carry-across-phases and correct per-phase REDUCE_ADD re-zero. 93 (M-split) / 91
  (N-split) regs, 0 spills.
- **mlp5** — combines the mlp2_fused single-acc epilogue and the mlp2_t MN-major operand,
  plus two unique pieces: (1) **cross-phase accumulate** over one continuous
  `2·num_k_tiles` mainloop using the tcgen05 `ScaleOut` bit — `ScaleOut::Zero` on the
  very first MMA only, `ScaleOut::One` for every MMA after (including the phase-1→phase-2
  boundary, never reset), so `dU·B` survives into phase 2; (2) a persistent **2D grid**
  `(num_sms/NSplit, NSplit)` with cross-CTA `dU/dV` L2 multicast. The `EpiChunkN=64`
  epilogue uses `TmemLoadOp<64>`. TMEM alloc/free is hoisted to **once per CTA** in the
  launcher (the persistent grid reuses each CTA across m-tiles, so per-tile
  `tcgen05.alloc` would trap). At `Stages=4` the smem footprint is ~208 KiB, within the
  B200's 227 KiB.

---

## Build

Prerequisite (once per shell):

```bash
export CUTLASS_HOME=/usr/local/include/cutlass
cd /shared/public/sharing/liger-comms-moe
```

The tests are torch-free: they `#include` the moe device headers directly and need only
CUTLASS (header-only), CUDA, and gtest — not the core library or NVSHMEM. Two canonical
out-of-source build dirs hold **all four** test targets:

- `liger_cute_kernels/build100a` → `sm_100a` (Blackwell / UMMA)
- `liger_cute_kernels/build90a`  → `sm_90a`  (Hopper / WGMMA)

### Blackwell (`build100a`) — all kernels

```bash
# Configure (idempotent; already configured in-repo)
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a

# Build every test target (mlp1, mlp2_fused, mlp2_t, mlp3, mlp4, mlp5)
cmake --build liger_cute_kernels/build100a -j

# …or a single kernel
cmake --build liger_cute_kernels/build100a --target test_mlp5_fused -j
```

### Hopper (`build90a`) — all kernels (regression guard)

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a -j
```

Resulting binaries land in `liger_cute_kernels/build<arch>/tests/cpp/test_<kernel>`.

### Register usage & spills on B200 (`--ptxas-options=-v`)

The default build flags do **not** print ptxas verbose output. To confirm register usage
and zero spills on the hot `Impl<100>` consumer, compile a test TU with the **same flags
`build100a` uses** plus `-v` and read the `Used N registers` / `spill` lines (a full binary
is not needed — `-c` suffices):

```bash
nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
     --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
     --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
     -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
     -c liger_cute_kernels/tests/cpp/test_mlp5_fused.cu -o /tmp/mlp5_spill.o
# ptxas info : Used 116 registers ...  0 bytes spill stores, 0 bytes spill loads
```

Measured on the B200 (`sm_100a`) for each kernel's `Compute=100` UMMA consumer entry
(`*_test_kernel<…, 100, …>`), with the flags above:

| Kernel (Compute=100 consumer) | Registers/thread | Spill stores | Spill loads |
|-------------------------------|:----------------:|:------------:|:-----------:|
| `mlp1_fused` | 160 | **0 B** | **0 B** |
| `mlp1_fused` (act, SiLU·V epilogue) | 162 | **0 B** | **0 B** |
| `mlp2_fused` | 73 | **0 B** | **0 B** |
| `mlp2_t` | 79 | **0 B** | **0 B** |
| `mlp3` (EpiChunkN=64, REDUCE_ADD) | 88 | **0 B** | **0 B** |
| `mlp4` (EpiChunkN=64, 2-phase, M-split) | 93 | **0 B** | **0 B** |
| `mlp4` (N-split) | 91 | **0 B** | **0 B** |
| `mlp5` (EpiChunkN=64, widest fragment) | 116 | **0 B** | **0 B** |

**Zero spill bytes on every hot consumer** — the hard requirement. No `-maxrregcount` /
`__launch_bounds__` cap was applied; register counts sit well under the 255/thread ceiling
at ~1 CTA/SM. mlp5 is the highest (116) because its `EpiChunkN=64` epilogue loads a 2×
wider TMEM→register fragment (`TmemLoadOp<64>`). The `Compute=90` bodies (trapped under
`sm_100a`) are far smaller (~21–30 regs, 0 spills). Counts are mildly flag-sensitive; the
per-kernel `writeup.md` files may list slightly different values from an earlier
measurement basis, but the spills-are-zero conclusion is invariant.

Or, on an already-built binary, count local-memory (spill) instructions in the SASS
(`0` = no spills):

```bash
cuobjdump -sass liger_cute_kernels/build100a/tests/cpp/test_mlp5_fused | grep -Ec '\bSTL\b|\bLDL\b'
```

---

## Test — correctness

Each binary defaults its gtest filter to the **present GPU** (Blackwell → the `Compute=100`
correctness tests; Hopper → the `…Sm90` tests). On a B200, just run the binary:

```bash
# Blackwell correctness (fast)
./liger_cute_kernels/build100a/tests/cpp/test_mlp1_fused        # Mlp1Fused + Mlp1FusedAct
./liger_cute_kernels/build100a/tests/cpp/test_mlp2_fused        # Mlp2Fused.Correctness
./liger_cute_kernels/build100a/tests/cpp/test_mlp2_t_fused      # Mlp2T.Correctness + Mlp2T.SingleTile
./liger_cute_kernels/build100a/tests/cpp/test_mlp5_fused        # Mlp5.Phase1_C0 + Mlp5.Phase2_B0 + Mlp5.Correctness
```

Explicit filters (arch-independent):

```bash
./liger_cute_kernels/build100a/tests/cpp/test_mlp2_fused   --gtest_filter='Mlp2Fused.Correctness'
./liger_cute_kernels/build100a/tests/cpp/test_mlp2_t_fused --gtest_filter='Mlp2T.Correctness:Mlp2T.SingleTile'
./liger_cute_kernels/build100a/tests/cpp/test_mlp5_fused   --gtest_filter='Mlp5.*Correctness:Mlp5.Phase*'
```

Notes:
- **mlp2_t** adds `Mlp2T.SingleTile` — an element-by-element single-tile check that
  localizes MN-major-operand bugs (a transposed operand shows a structured error pattern).
- **mlp5** adds phase-isolation diagnostics: `Mlp5.Phase1_C0` (`C=0`, isolates `dU·B`) and
  `Mlp5.Phase2_B0` (`B=0`, isolates `dV·C`). Both passing **and** the combined
  `Mlp5.Correctness` passing (incl. the `splits=2` 2D-grid rows) confirms the cross-phase
  accumulate bit and the MN-major operands together.
- On Hopper hardware, run the `build90a` binaries; the `…Sm90.Correctness` tests run and the
  `Compute=100` tests `SKIP`. On a B200 it is the reverse.

**mlp1 correctness detail (B200, `Compute=100`)** — both variants PASS (bars `mean_rel<1%`,
`max_rel<5%`); the `act` column is the fused `Z = SiLU(U)·V` output:

| Shape `{T,H,I,E}` | fused mean_rel | fused max_rel | act (Z) mean_rel | act (Z) max_rel |
|-------------------|:--------------:|:-------------:|:----------------:|:---------------:|
| `{128,256,128,1}` | 0.111% | 0.382% | 0.111% | 0.45% |
| `{128,512,256,1}` | 0.101% | 0.738% | 0.100% | 1.39% |
| `{256,256,256,2}` | 0.110% | 0.854% | 0.110% | 0.56% |
| `{384,256,128,3}` | 0.111% | 0.496% | 0.110% | 0.39% |

Run the whole suite with ctest from a build dir:

```bash
cd liger_cute_kernels/build100a && ctest --output-on-failure
```

---

## Test — TFLOPS benchmark

Benchmarks are **opt-in** per kernel (so `ctest` / correctness runs stay fast) via the
kernel's `*_BENCH` env var, and filtered to the `TFLOPs_Blackwell` test. Each reports the
median CUDA-event-timed throughput with an **N-split sweep over every divisor** of the
N-tile count, printing the peak TFLOPS and the winning `grid.y` split per bench shape
(`H=I=4096`, `E=8`, `T ∈ {2048, 4096, 8192, 16384}`).

```bash
MLP1_BENCH=1  ./liger_cute_kernels/build100a/tests/cpp/test_mlp1_fused   --gtest_filter='*TFLOPs_Blackwell*'
MLP2_BENCH=1  ./liger_cute_kernels/build100a/tests/cpp/test_mlp2_fused   --gtest_filter='*TFLOPs_Blackwell*'
MLP2T_BENCH=1 ./liger_cute_kernels/build100a/tests/cpp/test_mlp2_t_fused --gtest_filter='*TFLOPs_Blackwell*'
MLP3_BENCH=1  ./liger_cute_kernels/build100a/tests/cpp/test_mlp3         --gtest_filter='*TFLOPs_Blackwell*'
MLP4_BENCH=1  ./liger_cute_kernels/build100a/tests/cpp/test_mlp4         --gtest_filter='*TFLOPs_Blackwell*'
MLP5_BENCH=1  ./liger_cute_kernels/build100a/tests/cpp/test_mlp5_fused   --gtest_filter='*TFLOPs_Blackwell*'
```

On Hopper, use the `build90a` binaries with `--gtest_filter='*TFLOPs_Hopper*'`.

### Measured peak (B200)

FLOPs: `2·T·H·I` for the single-GEMM kernels (mlp2_fused, mlp2_t, mlp3); `4·T·H·I` for the
two-GEMM kernels (mlp1_fused, mlp4, mlp5).

**mlp1_fused** — base GEMM path, `Mlp1Fused.TFLOPs_Blackwell` (`4·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 983.17 | 8 | 43.7% |
| 4096 | 1081.38 | 4 | 48.1% |
| 8192 | **1090.44** | 2 | 48.5% |
| 16384 | 1020.30 | 1 | 45.3% |

**mlp1_fused (act)** — with the SiLU(U)·V activation epilogue, `Mlp1FusedAct.TFLOPs_Blackwell` (`4·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 779.70 | 8 | 34.7% |
| 4096 | 884.51 | 32 | 39.3% |
| 8192 | **897.64** | 2 | 39.9% |
| 16384 | 843.47 | 8 | 37.5% |

**mlp2_fused** (`2·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 695.88 | 8 | 30.9% |
| 4096 | **788.86** | 32 | 35.1% |
| 8192 | 769.19 | 16 | 34.2% |
| 16384 | 671.10 | 8 | 29.8% |

**mlp2_t** (`2·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 725.50 | 8 | 32.2% |
| 4096 | **826.27** | 32 | 36.7% |
| 8192 | 774.74 | 4 | 34.4% |
| 16384 | 674.09 | 8 | 30.0% |

**mlp3** — down-weight grad `Mlp3.TFLOPs_Blackwell` (`2·T·H·I`, N-split, `grid.x=148`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 453.44 | 8 | 20.2% |
| 4096 | 684.78 | 4 | 30.4% |
| 8192 | 1009.33 | 8 | 44.9% |
| 16384 | **1214.64** | 4 | 54.0% |

**mlp4** — weight grads `Mlp4.TFLOPs_Blackwell` (`4·T·H·I`, M-split default, `grid.x=148`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 471.72 | 4 | 21.0% |
| 4096 | 729.51 | 16 | 32.4% |
| 8192 | 937.31 | 16 | 41.7% |
| 16384 | **1163.87** | 4 | 51.7% |

**mlp5** (`4·T·H·I`, grid = `(num_sms/split, split)`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 1046.40 | 8 | 46.5% |
| 4096 | 1137.89 | 4 | 50.6% |
| 8192 | **1156.08** | 2 | 51.4% |
| 16384 | 1088.45 | 2 | 48.4% |

Both weight-gradient kernels' `%peak` climbs **monotonically with `T`** (peaking at
`T=16384`, unlike the phase-1/backward-input kernels that peak at `T=8192`): the
REDUCE_ADD epilogue writes the full `[E,H,I]` (mlp3) / `[E,I,H]`×2 (mlp4) output at a
fixed per-tile cost, so a deeper `K=T` reduction amortizes that epilogue/atomic traffic —
larger `T` is strictly better until the SMs saturate.

### Measured peak (H100 / Hopper, `build90a` + `*TFLOPs_Hopper*`)

The same WGMMA `Impl<90>` consumers, benchmarked on an **NVIDIA H100 80GB HBM3** (SXM5,
compute capability 9.0, `sm_90a`, **132 SMs**, CUDA 12.9, CUTLASS 4.4.1) as a Hopper
regression. Identical bench shapes and FLOP counts as the B200 tables above (`H=I=4096`,
`E=8`, `T ∈ {2048, 4096, 8192, 16384}`; `2·T·H·I` for the single-GEMM kernels,
`4·T·H·I` for the two-GEMM kernels). `%peak` is vs the **H100 SXM bf16 dense peak
≈ 989.4 TFLOPS** (no sparsity). All six `…Sm90.Correctness` suites pass on the same
binaries (max_rel: mlp1 fused ≤ 0.85% / act ≤ 1.39%, mlp2_fused ≤ 0.52%, mlp2_t ≤ 0.40%,
mlp3 ≤ 0.48%, mlp4 ≤ 1.00%, mlp5 ≤ 1.07%).

| Kernel | Peak TFLOPS (@ shape, split) | % of H100 peak | Correctness (max_rel) |
|--------|:----------------------------:|:--------------:|:---------------------:|
| mlp1_fused | **694.71** @ T=8192, split 2 | 70.2% | ≤ 0.85% (fused) / ≤ 1.39% (act) |
| mlp2_fused | **506.33** @ T=4096, split 4 | 51.2% | ≤ 0.52% |
| mlp2_t | **508.91** @ T=4096, split 4 | 51.4% | ≤ 0.40% |
| mlp3 | **625.43** @ T=16384, split 4 | 63.2% | ≤ 0.48% |
| mlp4 | **639.12** @ T=16384, split 4 | 64.6% | ≤ 1.00% |
| mlp5 | **635.99** @ T=8192, split 4 | 64.3% | ≤ 1.07% |

**mlp1_fused** — base GEMM path, `Mlp1Fused.TFLOPs_Hopper` (`4·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 557.68 | 16 | 56.4% |
| 4096 | 670.12 | 4 | 67.7% |
| 8192 | **694.71** | 2 | 70.2% |
| 16384 | 595.20 | 1 | 60.2% |

**mlp1_fused (act)** — with the SiLU(U)·V activation epilogue, `Mlp1FusedAct.TFLOPs_Hopper` (`4·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 483.67 | 8 | 48.9% |
| 4096 | 550.58 | 4 | 55.6% |
| 8192 | **571.65** | 2 | 57.8% |
| 16384 | 499.87 | 1 | 50.5% |

**mlp2_fused** (`2·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 458.57 | 8 | 46.3% |
| 4096 | **506.33** | 4 | 51.2% |
| 8192 | 475.57 | 2 | 48.1% |
| 16384 | 335.57 | 1 | 33.9% |

**mlp2_t** (`2·T·H·I`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 474.79 | 8 | 48.0% |
| 4096 | **508.91** | 4 | 51.4% |
| 8192 | 451.94 | 4 | 45.7% |
| 16384 | 335.19 | 2 | 33.9% |

**mlp3** — down-weight grad `Mlp3.TFLOPs_Hopper` (`2·T·H·I`, N-split, `grid.x=132`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 238.38 | 16 | 24.1% |
| 4096 | 348.46 | 8 | 35.2% |
| 8192 | 520.08 | 8 | 52.6% |
| 16384 | **625.43** | 4 | 63.2% |

**mlp4** — weight grads `Mlp4.TFLOPs_Hopper` (`4·T·H·I`, M-split default, `grid.x=132`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 238.91 | 1 | 24.1% |
| 4096 | 395.96 | 1 | 40.0% |
| 8192 | 538.14 | 2 | 54.4% |
| 16384 | **639.12** | 4 | 64.6% |

**mlp5** (`4·T·H·I`, grid = `(num_sms/split, split)`):

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048 | 469.47 | 8 | 47.5% |
| 4096 | 439.74 | 2 | 44.4% |
| 8192 | **635.99** | 4 | 64.3% |
| 16384 | 600.20 | 2 | 60.7% |

### B200 vs H100 speedup (peak-to-peak)

Speedup = B200 peak TFLOPS ÷ H100 peak TFLOPS at each shape (each device at **its own**
winning N-split). Because the FLOP count is identical for a given `{kernel, T}`, this ratio
is also the wall-clock speedup. The bf16 dense peak ratio **2250 / 989.4 ≈ 2.27×** is the
theoretical ceiling.

| Kernel | T=2048 | T=4096 | T=8192 | T=16384 |
|--------|:------:|:------:|:------:|:-------:|
| mlp1_fused | 1.76× | 1.61× | 1.57× | 1.71× |
| mlp2_fused | 1.52× | 1.56× | 1.62× | 2.00× |
| mlp2_t | 1.53× | 1.62× | 1.71× | 2.01× |
| mlp3 | 1.90× | 1.97× | 1.94× | 1.94× |
| mlp4 | 1.97× | 1.84× | 1.74× | 1.82× |
| mlp5 | 2.23× | **2.59×** | 1.82× | 1.81× |

Reading the table:
- Most cells land in the **1.5–2.0×** band — below the 2.27× compute ceiling, as expected
  once epilogue/DRAM traffic (not raw MMA throughput) sets the pace.
- The single-GEMM forward kernels (**mlp2_fused**, **mlp2_t**) scale *up* with `T`, hitting
  ~2.0× at `T=16384` where the H100 falls off (335 TFLOPS) but the B200 holds ~670–674.
- The REDUCE_ADD **weight-gradient** kernels (**mlp3**, **mlp4**) are the *steadiest* —
  a tight **1.7–2.0×** across every `T` — because their %peak climbs monotonically on
  **both** devices (a deeper `K=T` reduction amortizes the fixed per-tile epilogue/atomic
  write), so neither device stalls at a particular tile size and the ratio never spikes or
  collapses. mlp4's mild decline (1.97×→1.82×) is the B200 pulling ahead on the epilogue as
  `T` grows; mlp3 holds ~1.9× throughout.
- **mlp5**'s `T=2048–4096` cells (2.23×, 2.59×) exceed the 2.27× ceiling because the
  persistent-2D-grid **H100** path dips there (440–469 TFLOPS), **not** because the B200
  beats its own peak; by `T≥8192` both devices hit stride and the ratio settles to ~1.8×.

---

## Files

Kernel headers (Traits + consumer split) and their tests:

| Kernel | Header(s) | Test |
|--------|-----------|------|
| mlp1_fused | `csrc/core/src/moe/mlp1_fused.cuh`, `mlp1_fused_act.cuh` | `tests/cpp/test_mlp1_fused.cu` |
| mlp2_fused | `csrc/core/src/moe/mlp2_fused.cuh` | `tests/cpp/test_mlp2_fused.cu` |
| mlp2_t | `csrc/core/src/moe/mlp2_t.cuh`, `mlp2_t_fused.cuh` | `tests/cpp/test_mlp2_t_fused.cu` |
| mlp3 | `csrc/core/src/moe/mlp3.cuh` | `tests/cpp/test_mlp3.cu` |
| mlp4 | `csrc/core/src/moe/mlp4.cuh` | `tests/cpp/test_mlp4.cu` |
| mlp5 | `csrc/core/src/moe/mlp5.cuh`, `mlp5_fused.cuh` | `tests/cpp/test_mlp5_fused.cu` |

Test targets are registered in `liger_cute_kernels/tests/cpp/CMakeLists.txt`.
Per-kernel design notes, blocker resolutions, and full result tables live in
`blackwell_port/{mlp2_fused,mlp2,mlp3,mlp4,mlp5}/writeup.md`; the shared port recipe and
isolation rules are in `blackwell_port/README.md`.
