# writeup_mlp3 — MLP3 down-weight gradient on Blackwell (SM100a)

> **Op:** `dA = dYᵀ · Z` — the MoE **down-weight gradient** (phase-2 backward): one
> GEMM contracting over the token axis `T`, atomically reduced into a per-expert
> `[E, H, I]` output. **Files:** `csrc/core/src/moe/mlp3.cuh` (Traits + helpers + UMMA
> MMA + consumer split), `tests/cpp/test_mlp3.cu`. **The distinctive one** — it is the
> first ported kernel with **both operands MN-major** and the only one whose epilogue is
> a **hardware atomic-add** (`SM90_TMA_REDUCE_ADD`) into a caller-zeroed buffer, so the
> SM100 epilogue *fills* `store_buf` from TMEM but the store path is untouched.

---

## 1. The nuance of MLP3

The MoE backward pass computes the gradient of the down-projection weight `A` by
contracting the upstream gradient `dY` with the phase activations `Z` over **all tokens
routed to each expert**:

```
dY : [T, H]   upstream grad             → A operand dYᵀ : (H, T)   M=H, K=T, MN-major
Z  : [T, I]   phase activations         → B operand Z   : (I, T)   N=I, K=T, MN-major
dA : [E, H, I]  down-weight gradient  = dYᵀ · Z   per expert, contracting T
```

The nuances that make it distinct from the other three ports:

- **One GEMM, contraction over T.** FLOPs `2·T·H·I`. Each expert `e` owns a contiguous
  token-block range `[k_start(e), k_end(e))` (units of `TileK`); the mainloop walks only
  that expert's K-blocks.
- **BOTH operands MN-major.** `dYᵀ` is `(H, T)` with `H` (=M) contiguous; `Z` is `(I, T)`
  with `I` (=N) contiguous. mlp2_t had *one* MN-major operand — mlp3 is the first with
  **two**, so the UMMA atom is `SM100_MMA_F16BF16_SS<…, Major::MN, Major::MN>`.
- **`SM90_TMA_REDUCE_ADD` epilogue (hardware atomic-add).** `dA` is accumulated into
  gmem with a TMA reduce, so **`dA` MUST be zero-initialized by the caller** and
  **re-zeroed between reused launches**. This epilogue is **arch-agnostic** (guarded only
  by `__CUDA_ARCH__` inside CUTLASS 4.4.1), so the entire `is_my_wg_leader` reduce-add
  block is kept **byte-for-byte** across Compute=90 and Compute=100.
- **Persistent chunk-fixed 1-D grid.** `blockIdx.x = cell_start`, `gridDim.x =
  cell_stride`; each CTA walks the `(expert, m-tile, n-lane)` cell space **internally**
  (the cell loop lives in the consumer). `outer_split` subdivides the n-tile walk into
  more, smaller cells for load balance.
- **`EpiChunkN=64`** — the widest ported epilogue fragment (`TmemLoadOp<64>` =
  `SM100_TMEM_LOAD_32dp32b64x`), shared with mlp5.

---

## 2. What changed to suit SM100

The WGMMA `Impl<90>` consumer is kept **verbatim**. The new `Impl<100>` reuses the mlp5
single-TMEM-accumulator epilogue and the mlp2_t MN-major operand resolution, then swaps
in the mlp3-unique pieces:

| Piece | Hopper (`sm_90a`) | Blackwell (`sm_100a`) |
|-------|-------------------|------------------------|
| MMA | `TiledMMA<MMA_Atom<GmmaAtom>>`, 2 WGs partition `(TileM,TileN)`, both operands MN-major | `SM100_MMA_F16BF16_SS<…, Major::MN, Major::MN>` — **one 1SM tcgen05 atom** over the whole `(TileM=128, TileN=256)`, issued by warp 4 |
| Accumulator | per-thread register fragment (`partition_fragment_C`) | **one TMEM accumulator**, `Allocator1Sm.allocate(TileN)` |
| Cross-`(cell,walk)` clear | `cute::gemm` accumulates; a fresh fragment per walk | **`UMMA::ScaleOut` bit**: `Zero` on the first MMA of each `(cell,walk)` K-loop, `One` after |
| Epilogue **store path** | `store_buf` → `tma_store_fence` → `SM90_TMA_REDUCE_ADD` → `tma_store_arrive` | **byte-for-byte identical** (REDUCE_ADD is arch-agnostic) |
| Epilogue **fill** of `store_buf` | scatter register `acc(i)` via SM90 `tCcC` coords + M-split remap | `(M,N)` extract → `flat_divide((WgTileM,EpiChunkN))` → **`TmemLoadOp<64>`** → **`partition_D` regs** → cast → `sStore(m_local, chunk_n)` |
| Store-target coords | `tCcC` (WGMMA thread-value layout) | **TMEM-load `partition_D` identity coords** (the mapping pin — see §4) |
| TMEM lifetime | n/a | **once-per-CTA** `tcgen05.alloc`/`free` in the launcher (before the single consumer call), never per-cell |
| Config support | M-split (`TileM=256`) **and** N-split (`TileM=128`) | **N-split only** — a 1SM UMMA atom's M is ≤128 (see §6) |
| smem | Stages | **Stages=4 ⇒ 224 KiB < B200's 227 KiB** (no `K_PIPE` reduction) |

**The crux — TMEM fill under a byte-identical REDUCE_ADD store.** On WGMMA each thread
owns a register slice of the accumulator and scatters it into `store_buf` using the
`tCcC` identity coords (with the M-split fold `m_local = (m_loc/128)*64 + (m_loc%64)`).
On UMMA the accumulator lives in TMEM, so the epilogue instead **pulls** each
`EpiChunkN`-wide column chunk out of TMEM into a `partition_D`-sized register fragment
and writes the **same** `store_buf` slots — after which the identical reduce-add leader
block stores them. The store path (`tma_store_fence` → `copy(tma_reduce_da,…)` →
`tma_store_arrive`, incl. the `m_atom_row`/`n_tile_idx` index math) is unchanged.

**Why this suits SM100:** one tcgen05 atom covers the full `(128,256)` tile with the
partial-`dA` accumulator resident in TMEM (no register-held partials), the `ScaleOut`
bit expresses "start a fresh sum for this `(cell,walk)`" for free, and the wide
`EpiChunkN=64` TMEM→register load is consumed one chunk at a time — landing at **88
registers / zero spills**, comfortably under the mlp5 yardstick (116).

---

## 3. Performance

### Correctness — B200 (`sm_100a` / UMMA), `Mlp3.SingleTile` + `Mlp3.Correctness`

fp32 CPU reference on the **same bf16-rounded** inputs; gate `mean_rel < 1%`,
`max_rel < 5%`. `dA` zero-initialized (REDUCE_ADD). **PASS.**

| Shape `(T,H,I,E)` | mode | mean_rel | max_rel | max_abs |
|-------------------|------|:--------:|:-------:|:-------:|
| 64, 128, 256, 1   | single-tile (elt-by-elt) | 0.141% | 0.387% | 0.12 |
| 64, 128, 256, 1   | osplit=1 | 0.141% | 0.388% | 0.11 |
| 128, 256, 512, 1  | osplit=1 | 0.141% | 0.477% | 0.125 |
| 128, 256, 512, 1  | osplit=2 (N-split) | 0.141% | 0.477% | 0.125 |
| 256, 256, 512, 2  | osplit=1 | 0.141% | 0.389% | 0.125 |
| 256, 256, 512, 2  | osplit=2 (N-split) | 0.141% | 0.389% | 0.125 |
| 512, 384, 256, 4  | osplit=1 | 0.141% | 0.389% | 0.125 |

`mean_rel` is pinned at ~0.14% — pure bf16 input/output rounding, exactly as expected
for an error-free reduction. The `osplit=2` rows match the `osplit=1` rows bit-for-bit
in error, confirming the multi-lane cell walk reduces into `dA` without double-counting.

### TFLOPS — B200 (`sm_100a` / UMMA), `MLP3_BENCH=1 … TFLOPs_Blackwell`

FLOPs **`2·T·H·I`** (one GEMM); median CUDA-event timing (memset re-zero stream-ordered
**before** the timed window); `outer_split` swept over the divisors of `num_n_tiles=16`;
`H=I=4096, E=8`. B200 dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | peak TFLOPS | winning `outer_split` | grid.x | %peak |
|---|:-----------:|:---------------------:|:------:|:-----:|
| 2048  | 453.44  | 4 | 148 | 20.2% |
| 4096  | 684.78  | 8 | 148 | 30.4% |
| 8192  | 1009.16 | 8 | 148 | 44.9% |
| 16384 | **1159.55** | 8 | 148 | **51.5%** |

Peak **1159.55 TFLOPS @ T=16384, osplit=8** — **51.5% of peak**, on par with the
compute-bound mlp5 sibling (51.4%).

**Cross-tile-size trend (why %peak climbs monotonically).** Unlike the single-GEMM mlp2
kernels (which peak then dip), mlp3's %peak rises **20% → 30% → 45% → 51.5%** with `T`.
The REDUCE_ADD epilogue writes the **entire** `[E,H,I]` output (256 MiB at these dims)
regardless of `T`, so its cost is fixed; growing `T` deepens each expert's K-reduction
(`T/E/TileK` k-blocks: 4 → 8 → 16 → 32), which amortizes that fixed epilogue and fills
the `Stages=4` pipe. At small `T` the mainloop is too short to hide the atomic-add store,
so the kernel is epilogue-bound (20%); at `T=16384` it reaches compute-bound territory
(51.5%). The winning `outer_split` settles at **8**: enough cells (`8·32·8 = 2048`) to
load-balance 148 SMs without over-subdividing the per-cell K-reduction.

### sm90a → sm100a speedup (measured on H100)

Measured on an **NVIDIA H100 80GB HBM3** (`sm_90a`, 132 SMs), WGMMA `Impl<90>` path in
`build90a` — `Mlp3.TFLOPs_Hopper`, same `outer_split` sweep and shapes (`H=I=4096`,
`E=8`); FLOPs `2·T·H·I`:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a --target test_mlp3 -j
MLP3_BENCH=1 ./liger_cute_kernels/build90a/tests/cpp/test_mlp3 --gtest_filter='*TFLOPs_Hopper*'
```

| T | H100 peak TFLOPS | winning `outer_split` | grid.x | %peak | B200÷H100 |
|---|:----------------:|:---------------------:|:------:|:-----:|:---------:|
| 2048  | 238.38 | 16 | 132 | 24.1% | 1.90× |
| 4096  | 348.46 | 8  | 132 | 35.2% | 1.97× |
| 8192  | 520.08 | 8  | 132 | 52.6% | 1.94× |
| 16384 | **625.43** | 4 | 132 | **63.2%** | 1.85× |

`%peak` is vs the H100 SXM bf16 dense peak ≈ **989.4 TFLOPS** (B200÷H100 computed against
this writeup's B200 table above); correctness passes on the same binary
(`Mlp3Sm90.Correctness`, max_rel ≤ 0.48%). **B200÷H100 = 1.85–1.97×**.

**Measured outcome vs hypothesis.** The prediction placed mlp3 between the compute-bound
(mlp5) and bandwidth-bound (mlp2) ends. What the data shows is a **near-flat ~1.9×** at
every `T` — below both the 2.27× compute and 2.4× bandwidth ratios — because mlp3's
`%peak` climbs monotonically on **both** arches in lockstep (H100 **24% → 63%**, B200
**20% → 54%**): as the `K=T` reduction deepens, the fixed REDUCE_ADD epilogue write
amortizes equally on each device, so their ratio barely moves. The kernel is
epilogue/occupancy-limited at small `T` (grid.x=132 but only ~16 M-tiles at `T=2048`) and
approaches compute-bound only at `T=16384`; because that transition happens on both GPUs
together, the speedup neither climbs to the compute ratio nor spikes to the bandwidth
ratio — it holds steady just under 2×.

---

## 4. Blockers hit & code changes

The highest-risk item (the store-buf mapping pin) was pre-empted by design and validated
element-by-element; no numeric hazard bit at runtime.

- **Store-buf mapping pin (primary anticipated risk) — resolved by design.** The UMMA
  thread-value layout differs from WGMMA, so the SM90 `tCcC` coords (and the M-split fold
  `m_local=(m_loc/128)*64+(m_loc%64)`) **do not carry over**. `Impl<100>` re-derives
  `(m_local, chunk_n)` from the **TMEM-load `partition_D` identity coords** of
  `make_identity_tensor((WgTileM, EpiChunkN))` — the DEST coords of the `TmemLoadOp<64>`
  copy — and for the supported N-split config (`TileM=WgTileM=128`) this is the plain
  identity. The `Mlp3.SingleTile` test does an **element-by-element** compare on a
  `(128,256)×64` shape to localize any mapping regression before the larger shapes muddy
  the signal; it passes, and so does full `Mlp3.Correctness` (incl. the N-split rows).
- **TMEM lifetime under the persistent chunk-fixed grid (B5′).** The grid is persistent —
  each CTA's consumer walks *many* cells via an **internal** `cell_idx += cell_stride`
  loop. A per-cell `tcgen05.alloc`/`relinquish` would allocate after the permit was
  relinquished → **`phase_invalid_during_alloc` trap**. **Fix:** the launcher allocs
  `TileN` TMEM columns **once per CTA** — from the **whole MMA warp** (warp 4, never a
  single `elect_one` lane, which silently hangs, **B5**) — *before* the single
  `mlp3_consumer<…,100>` call, publishes `smem.tile.tmem_base` via `__syncthreads`, and
  frees after. Arch-guarded `#if __CUDA_ARCH__ >= 1000` so it compiles out on sm_90a.
- **Both operands MN-major.** Resolved by `SM100_MMA_F16BF16_SS<…, Major::MN, Major::MN>`
  reading the MN-contiguous stride straight from `SmemLayoutDYT`/`SmemLayoutZ` (both
  `Layout_MN_SW128_Atom` + `Step<_2,_1,_3>`); no transpose, no debug loop.
- **Benchmark timing.** `dA` is re-zeroed **every** launch (REDUCE_ADD accumulates), but
  the `cudaMemsetAsync` is stream-ordered **before** `cudaEventRecord(start)`, so only
  the GEMM kernel is timed — the memset is caller-side output prep, not kernel FLOPs.

`Impl<90>` (WGMMA), the arch-agnostic producer, and `mlp3_fwd` are unchanged.

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
     -c liger_cute_kernels/tests/cpp/test_mlp3.cu -o liger_cute_kernels/build100a_mlp3/mlp3_spill.o
```

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp3` UMMA consumer | **88** | **0 B** | **0 B** |

**Zero spills** — the hard requirement. mlp3 shares mlp5's widest-of-the-four epilogue
fragment (`EpiChunkN=64` → `TmemLoadOp<64>`) yet lands at **88 registers** (vs mlp5's
116): the single-GEMM mainloop keeps fewer operands live than mlp5's fused two-phase
loop, and the fragment is loaded/consumed one `EpiChunkN`-chunk at a time. No
`-maxrregcount` / `__launch_bounds__` cap was needed. (The `Compute=90` entry compiles to
37 registers — it is the `__trap` stub under `sm_100a`, since the WGMMA body is only
built for `sm_90a`.)

---

## 6. Deviation from the recipe

**UMMA path supports the N-split config only.** mlp3's Traits allow two cooperative
configs: M-split `(TileM=256, TileN=128)` and N-split `(TileM=128, TileN=256)`. A single
1SM tcgen05 atom's **M extent is ≤128**, so the M-split config cannot map to one TMEM
accumulator on the UMMA path. `Impl<100>` therefore `static_assert`s `TileM==WgTileM`
(N-split), and the test/bench use the N-split default `(128,256)`. The M-split config
remains available on the WGMMA `Impl<90>` path, unchanged. This is the only deviation;
it is inherent to the 1SM UMMA atom geometry, not a shortcut.

---

## 7. Reproduce

```bash
export CUTLASS_HOME=/usr/local/include/cutlass
cd /shared/public/sharing/liger-comms-moe

# configure + build (private dir; nvcc is minutes/TU)
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp3 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp3 --target test_mlp3 -j

# correctness (arch-aware default filter: SingleTile + Correctness on B200)
./liger_cute_kernels/build100a_mlp3/tests/cpp/test_mlp3

# TFLOPS table (peak + winning outer_split per T)
MLP3_BENCH=1 ./liger_cute_kernels/build100a_mlp3/tests/cpp/test_mlp3 \
      --gtest_filter='*TFLOPs_Blackwell*'
```

**Files changed:** `csrc/core/src/moe/mlp3.cuh`, `tests/cpp/test_mlp3.cu`.
