# writeup_mlp2 — MLP2-fused (down-projection) on Blackwell (SM100a)

> **Op:** `Y = Z · Aᵀ` — the MoE **phase-2 down-projection**, a single fused-consumer
> GEMM. **Files:** `csrc/core/src/moe/mlp2_fused.cuh`,
> `tests/cpp/test_mlp2_fused.cu`. This was ported **first** as the *reference
> instantiation* of the shared recipe; mlp2_t and mlp5 are deltas on top of it.

---

## 1. The nuance of MLP2-fused

Phase 2 projects the phase-1 intermediate `Z` back down to the hidden size:

```
Z : [T, I]   (= SiLU(U)·V from phase 1)
A : [H, I]   per-expert down weight, consumed as Aᵀ → [I, H]
Y = Z · Aᵀ : [T, H]   contracts over I (the K axis); N axis = H
```

The nuances that shape the kernel — and make it the *easiest* of the four:

- **One accumulator per warpgroup** (MLP1 had two). One TMEM region, one TMEM→register
  epilogue path.
- **Both operands are K-major** (`Z` and `Aᵀ` are `GMMA::Layout_K_SW128`), which is
  the SM100-native operand orientation — **no transpose subtlety** (contrast mlp2_t).
- **No epilogue activation.** The epilogue is just cast-to-bf16 + TMA store, so the
  per-thread math the UMMA path must preserve is trivial.
- `EpiChunkN=32`; cooperative 2-WG consumer (M-split at `TileM=128`, N-split at
  `TileM=64`).

FLOPs (one GEMM): `2·T·H·I`.

---

## 2. What changed to suit SM100

The original consumer is WGMMA into a register accumulator. The port keeps that as
`Impl<90>` verbatim and adds a single-accumulator UMMA `Impl<100>`:

| Piece | Hopper (`sm_90a`) | Blackwell (`sm_100a`) |
|-------|-------------------|------------------------|
| MMA | `TiledMMA<MMA_Atom<GmmaAtom>>` | `SM100_MMA_F16BF16_SS<…,TileM,TileN,K,K>` (both operands K-major) |
| Accumulator | register fragment | **TMEM**, **`allocate(WgTileN)` — one acc, not `2·TileN`** |
| Mainloop pipe | `PipelineTmaAsync` | **`PipelineTmaUmmaAsync`** (`mlp2_make_pipe_umma`, `num_consumers=1`) |
| Epilogue | registers → cast → store | `(M,N)` extract → `flat_divide` → **`TmemLoadOp<32>`** → `partition_D` regs → cast → reused `store_buf` → TMA store |
| TMEM lifetime | n/a | whole-MMA-warp `tcgen05.alloc`/`free` |

The single most important kernel-specific detail: **allocate exactly `WgTileN` TMEM
columns** (one accumulator), *not* MLP1's `2·TileN` — the primary copy-paste hazard
when cloning the template. Because it carries one accumulator and one `store_buf`,
mlp2_fused is the **least smem-pressured** of the four, so SM100 occupancy never
forced a stage-count reduction.

**Why this suits SM100:** a single tcgen05 tile-MMA with the accumulator in TMEM frees
the register file entirely for the (trivial) cast epilogue — the reason this kernel
posts the lowest register count of the family (73) with zero spills.

---

## 3. Performance (from `blackwell.md`)

### B200 (sm_100a / UMMA) — measured, `MLP2_BENCH=1 … TFLOPs_Blackwell`

FLOPs `2·T·H·I`; median event timing; N-split divisor sweep; `H=I=4096, E=8`. B200
dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048  | 695.88 | 8  | 30.9% |
| 4096  | **788.86** | 32 | 35.1% |
| 8192  | 769.19 | 16 | 34.2% |
| 16384 | 671.10 | 8  | 29.8% |

Peak **788.86 TFLOPS @ T=4096, split 32**.

### sm90a → sm100a speedup (measured on H100)

Measured on an **NVIDIA H100 80GB HBM3** (`sm_90a`, 132 SMs) via the WGMMA `Impl<90>`
path in `build90a` — `Mlp2Fused.TFLOPs_Hopper`, same N-split sweep and shapes:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a --target test_mlp2_fused -j
MLP2_BENCH=1 ./liger_cute_kernels/build90a/tests/cpp/test_mlp2_fused --gtest_filter='*TFLOPs_Hopper*'
```

| T | H100 peak TFLOPS | split | %peak | B200÷H100 |
|---|:----------------:|:-----:|:-----:|:---------:|
| 2048  | 458.57 | 8 | 46.3% | 1.52× |
| 4096  | **506.33** | 4 | 51.2% | 1.56× |
| 8192  | 475.57 | 2 | 48.1% | 1.62× |
| 16384 | 335.57 | 1 | 33.9% | **2.00×** |

`%peak` is vs the H100 SXM bf16 dense peak ≈ **989.4 TFLOPS**. **B200÷H100 = 1.52–2.00×**,
climbing with `T`.

**Why the gain climbs to 2.0× (mechanism).** B200 vs H100: bf16 peak ~2.27×
(2.25 vs 0.99 PFLOPS), HBM bandwidth ~2.4× (8 vs 3.35 TB/s). mlp2_fused is a
**single GEMM with no cross-GEMM operand reuse** — it re-reads `Z` and `A` straight
from HBM — so it is the **most memory-bound** kernel of the four (only ~34–51% of
compute peak on either arch). At small/mid `T` both devices are similarly
bandwidth-limited, so the ratio sits at **~1.5–1.6×** (well under the 2.4× bandwidth
ratio). At `T=16384` the **H100 collapses to 33.9%** (its smaller HBM/L2 saturates
first) while the B200 still holds ~671 TFLOPS, so the speedup jumps to **2.0×** — the
one shape where the B200's extra bandwidth pays off in full.

### Cross-tile-size trend (why it stays in the 30–35% band)

Peak walks **696 → 789 → 769 → 671** (2048→16384), split shrinks **8 → 32 → 16 → 8**.
Hypotheses:

- The low absolute %peak (30–35%) across *all* `T` is the signature of a
  **bandwidth-bound** kernel: with no operand reuse, arithmetic intensity is fixed
  and low, so throughput is capped by HBM well below the tensor-core ceiling — a
  larger machine (B200) helps roughly in proportion to its extra bandwidth, not its
  extra FLOPS.
- **Peak at T=4096** with a large split (×32 → 1024 CTAs): best occupancy fill of the
  148 SMs while the per-CTA working set is still small.
- **T=2048** under-fills even at ×8 (fewer M-tiles) → 30.9%; **T=16384** the largest
  working set + most HBM traffic pulls it back to 29.8% as the kernel saturates
  bandwidth.

---

## 4. Blockers hit & code changes

mlp2_fused **compiled, ran correct, and hit peak TFLOPS on the first attempt — no
hang, no illegal access, no spill.** The shared MLP1 recipe pre-empted the standard
blocker set; each was handled preemptively rather than debugged:

| Blocker | Handled by |
|---------|------------|
| **B1** (TMEM-load atom width) | Local `TmemLoadOp<32>` → `SM100_TMEM_LOAD_32dp32b32x`. |
| **B2** (epi tiling) | `flat_divide(acc_mn, epi_tile)`, not `zipped_divide`. |
| **B3** (acc view) | `(M,N)` extract `tCtAcc(make_coord(_,_),_0,_0)` before tiling. |
| **B4** (fragment sizing) | Register fragment sized from `partition_D`. |
| **B5** (warp-sync alloc) | Whole-MMA-warp `allocate`/`free` + `__syncwarp()`. |
| **Kernel-specific: TMEM over-allocation** | Allocated **`WgTileN`** (one acc), explicitly *not* `2·TileN`. |

Only change to the Hopper side was the mechanical `Impl<90>` struct/forwarder wrapper;
the WGMMA mainloop/epilogue is **byte-for-byte** unchanged.

---

## 5. Register-spill check (the checker)

Measured on the B200 (`sm_100a`) for the `Compute=100` UMMA consumer, exact
`build100a` flags:

```bash
export CUTLASS_HOME=/usr/local/include/cutlass
nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
     --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
     --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
     -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
     -c liger_cute_kernels/tests/cpp/test_mlp2_fused.cu -o /tmp/mlp2_spill.o
```

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp2_fused`         | 73 | **0 B** | **0 B** |

**Zero spills**, and the **lowest** register count of the family — one accumulator, a
trivial cast epilogue, and the narrow `EpiChunkN=32` fragment (`TmemLoadOp<32>`), far
below the MLP1 yardstick (160). No register cap needed.
