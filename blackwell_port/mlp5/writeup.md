# mlp5 — Blackwell (SM100a / UMMA / tcgen05) port write-up

**Kernel:** `mlp5` — backward input gradient `dX = dU·B + dV·C` (two GEMMs fused
into ONE continuous `2·num_k_tiles` k-loop, accumulated into a single TMEM
accumulator).

**Target:** NVIDIA **B200** (compute_cap 10.0), nvcc 12.9, CUTLASS SM100 headers.

**Status: DONE (GREEN).** Correctness passes on B200 (both phase-isolation
diagnostics + full-shape, incl. the 2D-grid multicast split); TFLOPS bench runs
clean (peak **1156 TFLOPS**, ~51 % of the ~2.25 PFLOPS bf16 B200 peak);
Compute=100 kernel is **116 registers, 0 spill bytes**; the sm_90a (Hopper) build
still compiles with no regression.

This was the hardest of the three ports: it layers three things on top of each
other — the mlp2_fused single-accumulator UMMA epilogue, the mlp2_t **MN-major
operand-B**, and a unique **cross-phase accumulate**. A fourth, port-specific
issue surfaced only at bench scale: the mlp5 **persistent 2D grid** forced the
TMEM lifecycle to be restructured to once-per-CTA.

---

## 1. Changed files

| File | Change |
|------|--------|
| `liger_cute_kernels/csrc/core/src/moe/mlp5.cuh` | **Traits/helpers.** Added SM100 includes (`mma_sm100_umma`, `mma_traits_sm100`, `tmem_allocator_sm100`, `copy_sm100`, `copy_traits_sm100`, `sm100_pipeline`) + `TmemLoadOpSelector<8/16/32/64/128>` → `TmemLoadOp`. In `Mlp5Traits`: `MainloopPipelineUmma` (`PipelineTmaUmmaAsync`, num_consumers=1), `AccStages=1`, `AccumulatorPipeline` (`PipelineUmmaAsync`). In `Mlp5Smem`: `alignas(16) uint32_t tmem_base` + `AccumulatorPipeline::SharedStorage acc_pipe`. Added `mlp5_make_pipe_umma`. **Standalone `mlp5_producer/consumer/fwd` (Hopper) left untouched.** |
| `liger_cute_kernels/csrc/core/src/moe/mlp5_fused.cuh` | **Split consumer** into `Mlp5FusedConsumerImpl<90>` (verbatim WGMMA body) / `Mlp5FusedConsumerImpl<100>` (new UMMA body) + a `Compute`-defaulted forwarder `mlp5_fused_consumer<Traits, Compute=90>`. `mlp5_fused_producer` unchanged. The `<100>` body: MN-major UMMA atom, single continuous `2·num_k_tiles` mainloop with the `ScaleOut` accumulate bit, single TMEM accumulator, mlp2_fused-style `flat_divide` → `TmemLoadOp<64>` → `partition_D`-sized fragment → reused `store_buf` → TMA store. **TMEM alloc/free removed from the consumer** (now launcher-owned — see §2c). |
| `liger_cute_kernels/tests/cpp/test_mlp5_fused.cu` | Overwrote the `int main(){}` stub with the full test: two-GEMM CPU reference `dX=dU·B+dV·C` (B/C MN-major, matched to storage), the **2D-grid launcher** with **once-per-CTA TMEM alloc/free**, correctness (`run5`), phase-isolation diagnostics (`run5_isolate`), and the `4·T·H·I` TFLOPS bench with an N-split (`grid.y`) sweep. |

---

## 2. How the three (four) kernel-unique pieces were resolved

### 2a. Cross-phase accumulate — the primary crux → `UMMA::ScaleOut`
The two GEMMs run as ONE continuous mainloop over `total_k = 2·num_k_tiles` into
ONE TMEM accumulator. The producer already places **B** in the phase-1 stages
(`k<K`) and **C** in the phase-2 stages (`k≥K`) in the *same* pipe slot (W-slot
reused). On WGMMA (Impl<90>) the fragment is `clear()`ed once and `cute::gemm`
accumulates across all `2K` MMAs. On UMMA the accumulate is a **per-instruction
bit** on `tcgen05.mma` (`tiled_mma.accumulate_`):

```cpp
for (int k = 0; k < total_k; ++k) {              // total_k = 2 * num_k_tiles
    ...
    for (int kb = 0; kb < size<2>(tCrZ); ++kb) {
        tiled_mma.accumulate_ = (k == 0 && kb == 0)
            ? UMMA::ScaleOut::Zero    // ONLY the very first MMA clears the acc
            : UMMA::ScaleOut::One;    // everything else accumulates …
        gemm(tiled_mma, tCrZ(_,_,kb), tCrW(_,_,kb), tCtAcc);
    }
}
```

The critical subtlety: the bit is **not reset at the phase-1→phase-2 boundary**
(`k` transitions `K-1 → K` stays `One`). Re-clearing there would drop the entire
`dU·B` term. The two tiny diagnostics prove the bit is right: `Phase1_C0` (C=0,
isolates `dU·B`) and `Phase2_B0` (B=0, isolates `dV·C`) both pass **and** the
combined case passes — which is only possible if phase-1's result survives into
phase-2's accumulation.

### 2b. MN-major operand-B (`B`/`C`) → atom major mode `UMMA::Major::MN`
Operand A (`dU`/`dV`) is K-major; operand B (`B`/`C`) is **MN-major** (the
column-major weight view). The UMMA atom is selected with the B major mode baked
into its type:

```cpp
auto tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_SS<Element, Element, float, TileM, TileN,
                         UMMA::Major::K, UMMA::Major::MN>{});
```

The descriptor for B (`FrgTypeB = UMMA::smem_desc<Major::MN>`) reads the swizzle/
major mode from the smem stride, and mlp5's existing `SmemLayoutW`
(`GMMA::Layout_MN_SW128_Atom` + `Step<_2,_1,_3>`) is byte-identical to the UMMA
MN-major descriptor layout (verified: the SM100 `Layout_MN_SW128_Atom` is a
`using`-alias of the GMMA one). Host-side, B/C are viewed as `(H, E·I)` with
stride `(1, H)` so the H (=N) axis is contiguous. The combined case passing (no
transpose/garble) confirms the major mode is correct.

### 2c. Cross-phase accumulate's structural twin — **TMEM lifecycle vs the persistent 2D grid** (port-specific bug, found at bench scale)
The mlp5 launcher keeps its mandated **2D grid** `grid = (num_sms/NSplit, NSplit)`
with the cross-CTA `dU/dV` L2 multicast, so each CTA runs a *persistent* m-loop
`for (m = blockIdx.x; m < num_m_tiles; m += gridDim.x)`. Correctness shapes never
exposed a problem (there `grid.x ≥ num_m_tiles`, so each CTA does exactly one
m-tile), but the **TFLOPS bench** (`grid.x = num_sms/split < num_m_tiles`)
crashed with:

```
Internal system error during the TMEM allocation.
  $__internal_..._tcgen05_guardrail_trap_phase_invalid_during_alloc
```

Root cause: I had cloned mlp2_fused's **per-consumer** `tcgen05.alloc → … →
release_allocation_lock → free` sequence. `release_allocation_lock`
(`tcgen05.relinquish_alloc_permit`) is a **permanent, once-per-CTA** relinquish —
allocating *after* it (the 2nd m-tile) is illegal → the guardrail trap. mlp2_fused
never hit this because its bench pins `grid.x = num_m_tiles` (one m-tile per CTA).

**Fix (canonical persistent-kernel pattern):** hoist the whole TMEM lifecycle to
**once per CTA** in the launcher — allocate before the m-loop, publish
`smem.tile.tmem_base` via `__syncthreads`, free after the loop:

```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  cute::TMEM::Allocator1Sm tmem_alloc{};
  if constexpr (Compute == 100)
    if (warp_id == 4) { tmem_alloc.allocate(Traits::TileN, &smem.tile.tmem_base); __syncwarp(); }
#endif
  __syncthreads();
  for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) { producer / consumer }
  __syncthreads();
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (Compute == 100)
    if (warp_id == 4) { tmem_alloc.release_allocation_lock(); tmem_alloc.free(smem.tile.tmem_base, Traits::TileN); }
#endif
```

The consumer just reads `smem.tmem_base`. Its accumulator pipeline (`acc_pipe`) is
still (re)constructed per m-tile, which is safe: it is touched only by consumer
warps and is bracketed by `NamedBarrier::sync(ConsumerThreads)` at both ends, so
each m-tile's handshake is fully isolated. The arch macro `#if __CUDA_ARCH__ >=
1000` guard is required because the `Compute==100` kernel is still *instantiated*
on the sm_90a build (where its consumer traps) — the tcgen05 PTX must be compiled
out there.

### 2d. `K_PIPE` (smem budget) — **no reduction needed**
`Mlp5FusedKernelSmem<Traits,100>` = **213 248 bytes = 208.2 KiB** with
`Stages=4` (K_PIPE=4). B200's `sharedMemPerBlockOptin` = **232 448 bytes = 227.0
KiB**, so it fits with **~18.8 KiB headroom**. `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`
succeeded for both correctness and bench, and no `cudaErrorInvalidValue` at
launch. **Stages was kept at 4.** (The Z tile is 4×128×64×2 B = 64 KiB, the W tile
4×256×64×2 B = 128 KiB; those two dominate.)

---

## 3. Register / spill numbers (deliverable #1)

Captured with `nvcc … -Xptxas -v` on `test_mlp5_fused.cu` for `sm_100a`
(`build100a_mlp5/mlp5_ptxas.txt`). Both `mlp5_fused_test_kernel` instantiations
demangled and mapped:

| Kernel (Traits `<bf16,128,256,64,4,64>`) | Registers | Spill stores | Spill loads | Barriers |
|---|---|---|---|---|
| `mlp5_fused_test_kernel<…, **100**>` (UMMA, Blackwell) | **116** | **0 B** | **0 B** | 16 |
| `mlp5_fused_test_kernel<…, 90>` (traps on sm_100a) | 30 | 0 B | 0 B | 1 |

**Zero spill bytes — the hard requirement — is met.** 116 regs is higher than the
mlp2_fused yardstick (79 regs), exactly as expected: mlp5 uses **EpiChunkN=64**
(the widest, most spill-prone epilogue), so the TMEM→reg fragment is loaded with
`TmemLoadOp<64>` (`SM100_TMEM_LOAD_32dp32b64x`) — a 2× wider register fragment
than mlp2_fused's EpiChunkN=32. Live ranges were kept tight per the template
(single accumulator; the WGMMA fragment is never live next to the UMMA one; the
store fragment is sized from `partition_D` and the smem `store_buf` is reused per
chunk), so no `-maxrregcount` / `__launch_bounds__` register cap was needed.
116 regs × 384 threads = 44 544 < 65 536 regs/SM, so the `__launch_bounds__(NumThreads,1)`
1-CTA/SM occupancy target holds.

---

## 4. Correctness on B200 (deliverables #2, #3)

`./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused`
(default filter runs the two diagnostics first, then full-shape). Tolerances:
`mean_rel < 1 %`, `max_rel < 5 %`.

### Phase-isolation diagnostics (tiny single-tile, `128×256×64×1`)
| Test | Isolates | mean_rel | max_rel | max_abs | Result |
|------|----------|----------|---------|---------|--------|
| `Mlp5.Phase1_C0` (C=0) | phase 1, `dU·B` | 0.140 % | 0.386 % | 0.116 | **PASS** |
| `Mlp5.Phase2_B0` (B=0) | phase 2, `dV·C` | 0.141 % | 0.389 % | 0.0854 | **PASS** |

Both isolated phases *and* the combined case below pass ⇒ the cross-phase
`ScaleOut` bit and the MN-major operand are both correct.

### Full-shape `Mlp5.Correctness`
| T | H | I | E | splits | mean_rel | max_rel | Result |
|---|---|---|---|--------|----------|---------|--------|
| 128 | 256 | 64 | 1 | 1 | 0.142 % | 0.388 % | PASS |
| 128 | 512 | 128 | 1 | 1 | 0.141 % | 0.388 % | PASS |
| 128 | 512 | 128 | 1 | 2 (2D grid) | 0.141 % | 0.388 % | PASS |
| 256 | 512 | 256 | 2 | 1 | 0.141 % | 1.073 % | PASS |
| 256 | 512 | 256 | 2 | 2 (2D grid) | 0.141 % | 1.073 % | PASS |
| 384 | 256 | 256 | 3 | 1 | 0.141 % | 0.389 % | PASS |

`[ PASSED ] 3 tests.` The `2D grid` rows use `num_splits=2` (grid.y=2), exercising
the N-split across `blockIdx.y` and the emergent cross-CTA `dU/dV` L2 multicast.

---

## 5. TFLOPS on B200 (deliverable #2)

`MLP5_BENCH=1 ./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused --gtest_filter='*TFLOPs_Blackwell*'`.
FLOPs = **`4·T·H·I`** (two GEMMs). Median CUDA-event timing; N-split
(`grid.y`) swept over the divisors of `num_n_tiles=16`; peak reported. Shapes
`H=I=4096, E=8`. B200 bf16 peak ≈ **2.25 PFLOPS**.

| T | winning split | grid = (num_sms/split, split) | median ms | **peak TFLOPS** | % of 2.25 PFLOPS |
|---|---|---|---|---|---|
| 2048 | 8 | (18, 8) | 0.1313 | 1046.40 | 46.5 % |
| 4096 | 4 | (37, 4) | 0.2416 | 1137.89 | 50.6 % |
| 8192 | 2 | (74, 2) | 0.4755 | **1156.08** | **51.4 %** |
| 16384 | 2 | (74, 2) | 1.0102 | 1088.45 | 48.4 % |

`[ PASSED ] 1 test.` The two GEMMs share operand A across a small per-expert
K (`I/TileK` tiles), so ~51 % of the dense bf16 peak is a reasonable landing spot
for this shape family. The winning split trends from 8 → 2 as T (and thus
`num_m_tiles`) grows and the m-dimension alone fills the SMs.

---

## 6. No Hopper regression (deliverable #4)

```
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a_mlp5 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a_mlp5 --target test_mlp5_fused -j
```

**Compiles clean.** The `Impl<90>` WGMMA body and the 2D-grid launcher + multicast
are byte-for-byte unchanged; only the arch-guarded (`#if __CUDA_ARCH__ >= 1000`)
once-per-CTA TMEM alloc/free was added to the launcher (compiled out on sm_90a).
On B200, `Mlp5Sm90.Correctness` **SKIPs** ("requires an sm_90 (Hopper) GPU") — as
expected.

---

## 7. Reproduce commands

```bash
export CUTLASS_HOME=/usr/local/include/cutlass

# ---- Blackwell (sm_100a) ----
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp5 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp5 --target test_mlp5_fused -j

# correctness (runs the two phase-isolation diagnostics first, then full-shape)
./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused

# TFLOPS
MLP5_BENCH=1 ./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused \
      --gtest_filter='*TFLOPs_Blackwell*'

# register / spill check (Compute=100 → 116 regs, 0 spill bytes)
nvcc -std=c++17 -c -gencode arch=compute_100a,code=sm_100a \
  --use_fast_math --extra-device-vectorization --fmad=true --prec-div=false --prec-sqrt=false \
  --ptxas-options=-O3,--allow-expensive-optimizations=true -Xptxas -v \
  --expt-relaxed-constexpr -DNDEBUG \
  -I liger_cute_kernels/csrc/core/src/moe \
  -isystem /usr/local/cuda/targets/x86_64-linux/include \
  -isystem /usr/local/include/cutlass/include \
  -isystem /usr/local/include/cutlass/tools/util/include \
  liger_cute_kernels/tests/cpp/test_mlp5_fused.cu \
  -o liger_cute_kernels/build100a_mlp5/mlp5_spillcheck.o 2>&1 | grep -iE 'registers|spill'

# ---- Hopper (sm_90a) no-regression ----
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a_mlp5 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a_mlp5 --target test_mlp5_fused -j
```
