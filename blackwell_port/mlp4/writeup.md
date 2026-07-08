# mlp4 — Blackwell (SM100a / UMMA / tcgen05) port write-up

**Kernel:** `mlp4` — backward **weight gradients** `dB = dUᵀ·X`, `dC = dVᵀ·X`
(two GEMMs produced in **two sequential phases** per cell, each into its **own,
freshly-cleared** TMEM accumulator — NOT a cross-phase sum).

**Target:** NVIDIA **B200** (compute_cap 10.0), nvcc 12.9, CUTLASS SM100 headers.

**Status: DONE (GREEN).** Correctness passes on B200 (both phase-isolation
diagnostics + full-shape, across the M-split *and* N-split configs, at
`outer_split ∈ {1,2}`); the TFLOPS bench runs clean (peak **1144 TFLOPS**, ~51 %
of the ~2.25 PFLOPS bf16 B200 peak); the Compute=100 kernel is **93 registers
(M-split) / 91 (N-split), 0 spill bytes**; the sm_90a (Hopper) WGMMA body is
preserved byte-for-byte.

mlp4 is **mlp3 + a two-phase loop**: it reuses mlp3's REDUCE_ADD leader block,
store-buf scatter, M-split remap concept, and persistent grid walk *verbatim*,
and wraps the mainloop+epilogue in `for(phase=0..1)`. The single port-specific
subtlety is the mirror image of mlp5's crux: where mlp5 must **carry** one
accumulator across its two GEMMs, mlp4 must **clear fresh** at the start of each
phase and route each result to its own output (`dB`/`dC`) — *independent*
accumulators. A second, structural subtlety (UMMA's `M ∈ {64,128}` constraint on
the 256-wide tile) forces **Option B**: two independent `(128,128)` TMEM
accumulators driven by one MMA warp.

---

## 1. Changed files

| File | Change |
|------|--------|
| `liger_cute_kernels/csrc/core/src/moe/mlp4.cuh` | **Traits/helpers + consumer split (single file — mlp4 is not split into a `_fused` header).** Added SM100 includes (`mma_sm100_umma`, `mma_traits_sm100`, `tmem_allocator_sm100`, `copy_sm100`, `copy_traits_sm100`, `sm100_pipeline`) + `TmemLoadOpSelector<8/16/32/64/128>` → `TmemLoadOp`. In `Mlp4Traits`: `MainloopPipelineUmma` (`PipelineTmaUmmaAsync`, num_consumers=1), per-WG `TiledMmaUmma` (`SM100_MMA_F16BF16_SS<Element,Element,float,WgTileM=128,WgTileN=128,Major::MN,Major::MN>` — **both** operands MN-major), `AccStages=1`, `AccumulatorPipeline` (`PipelineUmmaAsync`). In `Mlp4Smem`: `alignas(16) uint32_t tmem_base` + `AccumulatorPipeline::SharedStorage acc_pipe` (NOT in `Mlp4FusedSmem` — the Hopper fused path is untouched). Added `mlp4_make_pipe_umma`. Split the consumer into `Mlp4ConsumerImpl<90>` (verbatim WGMMA) / `Mlp4ConsumerImpl<100>` (new UMMA body) + a `Compute`-defaulted forwarder `mlp4_consumer<Traits, Compute=90>`. Templatized `mlp4_fwd<Traits, Compute=90>` (picks `mlp4_make_pipe_umma` vs `mlp4_make_pipe`; traps the WGMMA branch under `sm_100a`). **`mlp4_producer` (already phase-looped) left arch-agnostic.** |
| `liger_cute_kernels/tests/cpp/test_mlp4.cu` | Overwrote the `int main(){}` stub with the full test: two-GEMM CPU reference (`dB=dUᵀ·X`, `dC=dVᵀ·X`, per-expert token ranges), 5 TMA descriptors (X/dU/dV loads + dB/dC **REDUCE_ADD**, both zero-init'd), the persistent chunk-fixed launcher kernel (`mlp4_test_kernel`), correctness (`run4`, both outputs, `outer_split ∈ {1,2}`), the **two phase-isolation diagnostics** (`run4_isolate`), and the `4·T·H·I` TFLOPS bench with an `outer_split` sweep. Arch-aware `main()` (diagnostics registered/run **first**). |

**Not touched (shared / other-pipeline):** `tests/cpp/CMakeLists.txt` (the
`test_mlp4` target is pre-registered), root `blackwell.md`, `mlp3.cuh`,
`test_mlp3.cu`, the shared `build100a`/`build90a` dirs. `mlp_bwd.cuh` still
compiles: it calls `mlp4_{producer,consumer}<Traits4>` with the default
`Compute=90`, so the new template default keeps the Hopper fused path unchanged.

---

## 2. How the port-unique pieces were resolved

### 2a. Independent per-phase accumulate — the primary crux → per-phase `UMMA::ScaleOut::Zero`
The two GEMMs run as **two separate phases** (`for(phase=0..1)`), each a full
k-loop into its **own** accumulator, each REDUCE_ADD'd to its **own** output. On
WGMMA (`Impl<90>`) each phase `clear()`s its fragment. On UMMA the clear is the
**per-instruction `ScaleOut` bit** on `tcgen05.mma`, reset at **every phase
start** (the k-loop restarts per phase, per walk):

```cpp
for (int phase = 0; phase < 2; ++phase) {           // dB then dC
    auto& A_smem  = /* phase==0 ? dUᵀ : dVᵀ (reused A buffer) */;
    auto& out_tma = /* phase==0 ? tma_reduce_db : tma_reduce_dc */;
    ...
    for (int k = 0; k < num_k; ++k) {
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
            tiled_mma.accumulate_ = (k == 0 && kb == 0)
                ? UMMA::ScaleOut::Zero   // FRESH clear at the start of EACH phase
                : UMMA::ScaleOut::One;   // accumulate within the phase only
            gemm(tiled_mma, tCrA(_,_,kb), tCrX0(_,_,kb), tCtAcc0);  // WG0 acc
            gemm(tiled_mma, tCrA(_,_,kb), tCrX1(_,_,kb), tCtAcc1);  // WG1 acc
        }
    }
    /* epilogue → REDUCE_ADD into out_tma */
}
```

This is the **exact inverse** of mlp5, whose bit must **never** reset between its
two GEMMs. The two diagnostics prove it: `PhaseDB` (`dV=0`, isolates `dUᵀ·X` →
`dB` matches, `dC ≡ 0` exactly) and `PhaseDC` (`dU=0`, isolates `dVᵀ·X`) both pass
**and** the combined case passes — impossible if the acc were carried across
phases or the REDUCE_ADD outputs were not re-zeroed.

### 2b. Option B — UMMA `M ∈ {64,128}` on the 256-wide tile → two independent `(128,128)` TMEM accumulators
The cooperative 2-WG tile is `(TileM=256, TileN=128)` (M-split) or `(128, 256)`
(N-split). UMMA cannot issue a single `M=256` atom, so the port uses **Option B**:
a per-WG `(WgTileM=128, WgTileN=128)` UMMA atom and **two** TMEM accumulators —
`acc0 @ tmem_base`, `acc1 @ tmem_base + WgTileN`. The single **MMA warp** (warp 4)
issues **two** `tcgen05.mma` per k-step (one per WG); each consumer WG reads
**only its own** accumulator in the epilogue (`auto& tCtAccEpi = (my_wg==0) ?
tCtAcc0 : tCtAcc1;`). For the M-split, A is sliced per WG (`local_tile(sA_k, …,
make_coord(w,0))`) and X is shared; for the N-split, X is sliced and A is shared.

### 2c. Both operands MN-major → `SM100_MMA_F16BF16_SS<…,Major::MN,Major::MN>`
`dUᵀ/dVᵀ` (A) is M-major and `X` (B) is N-major, so **both** descriptors carry
`UMMA::Major::MN` — mlp4 is the only port with two MN-major operands. The existing
`SmemLayoutA_1`/`SmemLayoutX_1` (`GMMA::Layout_MN_SW128_Atom` + `Step<_2,_1>`) are
byte-identical to the UMMA MN-major descriptor layout; host-side, X is viewed as
`(H, T)` stride `(1, H)` and dU/dV as `(I, T)` stride `(1, I)` so the MN axis is
contiguous. The combined case passing (no transpose/garble) confirms the major
mode.

### 2d. Epilogue store-buf M-row — `partition_D` identity, NOT the WGMMA remap (the correctness pin)
The register→`store_buf` scatter is replaced with a TMEM pull: `(M,N)` extract
(`tCtAcc(make_coord(_,_),_0{},_0{})`) → `flat_divide((WgTileM, EpiChunkN=64))` →
`TmemLoadOp<64>` → a `partition_D`-sized register fragment → cast →
`sStore(m_local, chunk_n)`. The **critical pin**: the UMMA C-layout is
**contiguous per-WG**, so `m_row` is taken **directly from the `partition_D`
identity coords** (0..127) — the WGMMA `m_local = (m_loc/128)*64 + (m_loc%64)`
interleave remap is **gone** (the UMMA thread-value layout differs from `tCcC`).
Consequently the **M-split gmem atom-row formula changes**:

```
WGMMA  : atom_row = 4·out_m + my_wg    + 2·a
UMMA   : atom_row = 4·out_m + 2·my_wg  + a     (= kRowAtoms·out_m + kAtomsPerWg·my_wg + a)
```

Under Option B, WG *w* owns the contiguous tile-rows `[128w, 128w+128)` = stripes
`{2w, 2w+1}`, so store-buf row *r* = tile-row `128w+r` and atom *a* covers stripe
`2w+a` → `atom_row = 4·out_m + 2w + a`. The **N-split** formula (`2·out_m + a`) and
**both** configs' `n_tile_idx` are byte-identical to WGMMA. Validated
element-by-element vs the fp32 CPU reference (tiny PhaseDB/DC single-tile + full
Correctness across both configs).

### 2e. TMEM lifecycle vs the persistent grid + two phases (B5 / B5′)
The alloc is **consumer-owned** (mlp4_consumer runs the persistent `cell_idx`
loop internally, like mlp1_fused — not launcher-owned like mlp5) and hoisted
**once per CTA, ABOVE both the `cell_idx` loop and the `phase` loop**:

```cpp
cute::TMEM::Allocator1Sm tmem_alloc{};
if (is_mma_warp) { tmem_alloc.allocate(2*WgTileN, &smem.tmem_base); __syncwarp(); }
NamedBarrier::sync(ConsumerThreads, 0);
for (cell = cell_start; cell < total_cells; cell += cell_stride)   // persistent
    for (phase = 0; phase < 2; ++phase) { mainloop + epilogue }    // two phases
NamedBarrier::sync(ConsumerThreads, 0);
if (is_mma_warp) { tmem_alloc.release_allocation_lock(); tmem_alloc.free(smem.tmem_base, 2*WgTileN); }
```

A persistent grid **and** two phases make a per-phase/per-cell `tcgen05.alloc` a
**double** trap: `release_allocation_lock` (`relinquish_alloc_permit`) is a
permanent, once-per-CTA relinquish, so any allocate after it (next phase or next
cell) hits the guardrail `phase_invalid_during_alloc`. Hoisting above **both**
loops (blocker **B5′**) is mandatory. The whole **MMA warp** issues the allocate
(never a single `elect_one` lane — **B5**).

---

## 3. Performance (B200 / sm_100a)

`MLP4_BENCH=1 … --gtest_filter='*TFLOPs_Blackwell*'`. FLOPs `4·T·H·I`; median
CUDA-event timing (**pure kernel** — the REDUCE_ADD outputs are zeroed once
before the timing loop, NOT inside it, so the ~0.5 GB double-memset is excluded
and we measure GEMM throughput, matching the mlp5 methodology); `outer_split`
swept over the divisors of the walk axis; `H=I=4096, E=8`; `grid.x = num_sms =
148`. B200 dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | peak TFLOPS | winning split | grid.x | %peak |
|---|:-----------:|:-------------:|:------:|:-----:|
| 2048  | 471.59  | osplit=4  | 148 | 21.0% |
| 4096  | 724.16  | osplit=16 | 148 | 32.2% |
| 8192  | 921.67  | osplit=8  | 148 | 41.0% |
| 16384 | **1144.18** | osplit=4  | 148 | **50.9%** |

**%peak climbs monotonically with `T`.** mlp4 is a weight-gradient kernel: the
`[E·I, H]` outputs are large and written via **atomic global REDUCE_ADD**, and
each of the two phases **reloads** its A operand (`dUᵀ` then `dVᵀ`; only `X` is
shared) — so it is the most epilogue-/bandwidth-bound of the ports, and its
absolute %peak sits below mlp5's. As the contraction length `T` grows, the
mainloop FLOPs amortize the fixed REDUCE_ADD epilogue and the per-phase A reload,
lifting utilization **21% → 32% → 41% → 51%**.

**Correctness (per output, worst-case across shapes):**

| Test | dB `max_rel` | dC `max_rel` | verdict |
|------|:------------:|:------------:|:-------:|
| `Mlp4.PhaseDB` (dV=0) | 0.387 % | 0 % (exact) | PASS |
| `Mlp4.PhaseDC` (dU=0) | 0 % (exact) | 0.387 % | PASS |
| `Mlp4.Correctness` (M-split + N-split, osplit 1&2) | ≤ 1.001 % | ≤ 0.596 % | PASS |

All `mean_rel ≈ 0.14 %` (< 1 %), all `max_rel < 5 %`. `Mlp4Sm90.Correctness` and
`Mlp4.TFLOPs_Hopper` SKIP (no Hopper GPU attached).

---

## 4. Blockers hit & fixes (summary)

| Blocker | Symptom | Fix |
|---------|---------|-----|
| Independent per-phase acc (mlp4-unique) | would leak `dUᵀ·X` into `dC` / drop re-zero | `ScaleOut::Zero` on the first k of **each** phase, `One` after (inverse of mlp5); proven by PhaseDB/PhaseDC |
| **B5′** TMEM alloc under persistent grid + 2 phases | `tcgen05` guardrail trap / hang | hoist `alloc`/`free` **once per CTA, above both cell AND phase loops** |
| **B5** alloc lane | trap | whole **MMA warp** allocates, never a single `elect_one` |
| Store-buf M-row mapping (UMMA ≠ WGMMA) | structured wrong result in both outputs | take `m_row` from `partition_D` **identity** coords (no `(m/128)*64+(m%64)` remap); M-split atom-row `4·out_m + 2·my_wg + a` |
| UMMA `M ∈ {64,128}` on 256-wide tile | can't issue `M=256` atom | **Option B**: two `(128,128)` TMEM accumulators, one MMA warp issues 2 UMMAs/k-step |
| smem budget | `cudaFuncSetAttribute` could fail | single **reused** A buffer (`dUᵀ`→`dVᵀ`) keeps `Stages=4` at ~224 KiB < 227 KiB |

`Impl<90>` (WGMMA), the producer, the `is_my_wg_leader` REDUCE_ADD leader block
(parameterized only by `phase` for the output descriptor), and the persistent
grid walk are otherwise unchanged.

---

## 5. Register-spill check

`sm_100a`, exact `build100a` flags (see command below). Both Compute=100 configs:

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp4` M-split `(256,128)` (default) | **93** | **0 B** | **0 B** |
| `mlp4` N-split `(128,256)`           | **91** | **0 B** | **0 B** |

**Zero spills** — the hard requirement. Both are **below the mlp5 yardstick (116,
same `EpiChunkN=64`)**: the two independent `(128,128)` TMEM accumulators keep the
epilogue fragment narrow (one WG's `128×64` at a time) and the reused A buffer
avoids a second live operand. (The `test_mlp4.cu` TU also emits two 38-register
kernels — the `Compute=90` `__trap()` stubs instantiated for `sm_100a`; they are
never launched on Blackwell.)

---

## 6. Reproduce

```bash
export CUTLASS_HOME=/usr/local/include/cutlass
cd /shared/public/sharing/liger-comms-moe

# Build (private dir).
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp4 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp4 --target test_mlp4 -j

# Diagnostics first, then full correctness (auto-gated to the B200).
./liger_cute_kernels/build100a_mlp4/tests/cpp/test_mlp4 \
    --gtest_filter='Mlp4.PhaseDB:Mlp4.PhaseDC'
./liger_cute_kernels/build100a_mlp4/tests/cpp/test_mlp4 \
    --gtest_filter='Mlp4.Correctness'

# TFLOPS table.
MLP4_BENCH=1 ./liger_cute_kernels/build100a_mlp4/tests/cpp/test_mlp4 \
    --gtest_filter='*TFLOPs_Blackwell*'

# Register/spill audit for the Compute=100 consumer.
nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
     --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
     --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
     -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
     -c liger_cute_kernels/tests/cpp/test_mlp4.cu \
     -o liger_cute_kernels/build100a_mlp4/mlp4_spill.o 2>&1 | grep -A2 -i "registers\|spill"
```
