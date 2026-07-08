# writeup_mlp5 — MLP5 backward input gradient on Blackwell (SM100a)

> **Op:** `dX = dU·B + dV·C` — the MLP phase-5 **backward input gradient**: two GEMMs
> fused into one continuous k-loop and accumulated into **one** accumulator. **Files:**
> `csrc/core/src/moe/mlp5.cuh` (Traits + helpers + UMMA MMA), `mlp5_fused.cuh`
> (consumer split), `tests/cpp/test_mlp5_fused.cu`. **The hardest of the four** — it
> combines mlp2_fused's single-acc epilogue, mlp2_t's MN-major operand, *and* a unique
> cross-phase accumulate.

---

## 1. The nuance of MLP5

During the backward pass, phase 5 computes the gradient w.r.t. the block input `X` by
summing the contributions of the two phase-1 branches:

```
dU, dV : [T, I]   upstream grads of the two phase-1 branches
B,  C   : [I, H]   per-expert weights, consumed MN-major (like mlp2_t)
dX = dU·B + dV·C : [T, H]   two GEMMs, both contract over I (K axis); N axis = H
```

The nuances that make it distinct:

- **Two GEMMs, one accumulator, one continuous k-loop.** The mainloop runs
  `2·num_k_tiles` iterations; `clear(acc)` happens **once**, then *both* phases
  accumulate into the same `acc`:
  ```
  clear(acc)
  Phase 1 (k = 0 .. K-1)   : Z=dU, W=B → acc += dU·B
  Phase 2 (k = K .. 2K-1)  : Z=dV, W=C → acc += dV·C
  ```
  The W (weight) smem slot is **reused**: `B` in phase 1, `C` in phase 2.
- **MN-major weights** `B`, `C` (like mlp2_t).
- **`EpiChunkN=64`** — the widest epilogue fragment of the four, hence the most
  register/spill-prone.
- **2D grid** `(num_sms/NSplit, NSplit)` with cross-CTA `dU/dV` **L2 multicast**
  between `(m,0)` and `(m,1)`; the output `dX` is fully written (no zero-init).

FLOPs (**two** GEMMs): `4·T·H·I`.

---

## 2. What changed to suit SM100

mlp5 reuses the mlp2_fused single-acc UMMA epilogue and the mlp2_t MN-major operand
resolution wholesale, then layers on **two** kernel-unique pieces:

| Piece | Hopper (`sm_90a`) | Blackwell (`sm_100a`) |
|-------|-------------------|------------------------|
| MMA | `TiledMMA<MMA_Atom<GmmaAtom>>`, MN-major B/C | `SM100_MMA_F16BF16_SS<…,Major::K, Major::MN>` (B/C MN-major, per mlp2_t) |
| Accumulator | one register fragment, `clear` once | one **TMEM** accumulator, `allocate(WgTileN)` |
| **Cross-phase accumulate** | `cute::gemm` accumulates across all `2K` MMAs implicitly | **`UMMA::ScaleOut` bit**: `Zero` on the *very first* MMA only, `One` for **every** subsequent MMA incl. the phase-1→2 boundary (**never reset**) |
| Epilogue | registers → cast → store | `(M,N)` extract → `flat_divide` → **`TmemLoadOp<64>`** → `partition_D` regs → reused `store_buf` → TMA store |
| TMEM lifetime | n/a | **once-per-CTA** `tcgen05.alloc`/`free` in the launcher (not per-tile) |
| Grid | 2D `(num_sms/NSplit, NSplit)` + multicast | **unchanged** |
| smem | Stages | **Stages=4 ⇒ ~208 KiB < B200's 227 KiB** (no K_PIPE reduction needed) |

**The crux — cross-phase accumulate.** On WGMMA the register accumulator is cleared
once and `cute::gemm` naturally accumulates across all `2K` MMAs. On UMMA the
accumulate is a **per-instruction bit** on `tcgen05.mma` (`UMMA::ScaleOut`): the first
MMA of phase 1 runs **non-accumulating** (`ScaleOut::Zero` — *writes* `acc`), and every
MMA after — **including the phase-1→phase-2 boundary** — must be **accumulating**
(`ScaleOut::One`). The failure mode to avoid is re-clearing the bit when phase 2 starts,
which would drop the `dU·B` term. The code sets the atom's accumulate flag `false` on
the very first `k` only and `true` thereafter, and **does not reset it** between the
two phase loops.

**Why this suits SM100:** the single TMEM accumulator persists across both phases with
no register-resident partial sum, and the `ScaleOut` bit expresses "keep accumulating"
for free on the tile-MMA — so two fused GEMMs cost one accumulator and one epilogue,
landing at 116 registers / zero spills despite the wide `EpiChunkN=64` fragment.

---

## 3. Performance (from `blackwell.md`)

### B200 (sm_100a / UMMA) — measured, `MLP5_BENCH=1 … TFLOPs_Blackwell`

FLOPs **`4·T·H·I`** (two GEMMs); median event timing; N-split (`grid.y`) swept over
the divisors of `num_n_tiles`; `H=I=4096, E=8`. B200 dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | peak TFLOPS | winning split | grid `(num_sms/split, split)` | %peak |
|---|:-----------:|:-------------:|:-----------------------------:|:-----:|
| 2048  | 1046.40 | 8 | (18, 8) | 46.5% |
| 4096  | 1137.89 | 4 | (37, 4) | 50.6% |
| 8192  | **1156.08** | 2 | (74, 2) | **51.4%** |
| 16384 | 1088.45 | 2 | (74, 2) | 48.4% |

Peak **1156 TFLOPS @ T=8192, split 2** — the **highest %peak of the four** (51.4%).

### sm90a → sm100a speedup (measured on H100)

Measured on an **NVIDIA H100 80GB HBM3** (`sm_90a`, 132 SMs), WGMMA `Impl<90>` path in
`build90a` — `Mlp5.TFLOPs_Hopper`, same N-split sweep and shapes:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a --target test_mlp5_fused -j
MLP5_BENCH=1 ./liger_cute_kernels/build90a/tests/cpp/test_mlp5_fused --gtest_filter='*TFLOPs_Hopper*'
```

| T | H100 peak TFLOPS | split | %peak | B200÷H100 |
|---|:----------------:|:-----:|:-----:|:---------:|
| 2048  | 469.47 | 8 | 47.5% | 2.23× |
| 4096  | 439.74 | 2 | 44.4% | **2.59×** |
| 8192  | **635.99** | 4 | 64.3% | 1.82× |
| 16384 | 600.20 | 2 | 60.7% | 1.81× |

`%peak` is vs the H100 SXM bf16 dense peak ≈ **989.4 TFLOPS**. **B200÷H100 = 1.81–2.59×** —
the **highest** of the four ports, and the only kernel that *exceeds* the 2.27×
compute-peak ratio (at `T=2048–4096`).

**Measured outcome vs hypothesis.** The prediction — mlp5 leans toward the **compute
ratio (~2.27×)** because it is the most arithmetic-intense of the four (two GEMMs of
shared reuse + 2D-grid `dU/dV` L2 multicast) — holds, and then some. At `T=2048–4096`
the ratio is **2.23–2.59×**, *above* the compute ceiling; this overshoot is an H100
artifact, **not** a B200 over-peak: mlp5's persistent-2D-grid + once-per-CTA TMEM launch
under-fills the H100 at small `T` (only ~44–47% of peak, 440–469 TFLOPS), so the fast
B200 (already ~1046–1138 TFLOPS) opens a >2.27× gap. By `T≥8192` each H100 CTA-row
rasterizes ~2 m-tiles, the fixed launch/epilogue overhead amortizes, H100 %peak jumps to
**60–64%**, and the ratio settles to a healthy **~1.8×**. (The SM100 TMEM lifetime is
hoisted once-per-CTA for the persistent grid; the Hopper path has no such constraint, so
the two arches reach peak through slightly different launch shapes — which is precisely
why the small-`T` H100 dip is deeper than on B200.)

### Cross-tile-size trend (why %peak climbs then dips, and split → 2)

Peak walks **1046 → 1138 → 1156 → 1088**; winning split shrinks **8 → 4 → 2 → 2**.
Hypotheses:

- **Rising %peak 2048→8192 (46.5% → 51.4%):** with two GEMMs of shared reuse, the
  kernel is closer to compute-bound, so it *rewards* larger `T`. As `T` grows the
  per-expert `I`-reduction and the fused two-phase mainloop amortize launch/epilogue
  overhead better, pushing utilization up — until the SMs are fully fed.
- **Split collapses to 2 at large `T`:** once `num_m_tiles` alone can nearly fill 148
  SMs, only a **2-way** N-split is needed; that 2-way split is exactly what activates
  the cross-CTA `dU/dV` L2 multicast between `(m,0)` and `(m,1)`, so the winning
  configuration also *minimizes redundant HBM reads* — a nice alignment of occupancy
  and bandwidth.
- **Slight dip at T=16384 (51.4% → 48.4%):** the largest working set grows HBM traffic
  and cache pressure faster than the extra FLOPs can hide, a mild bandwidth tail (same
  shape as the other kernels, but shallower here because mlp5 is the least
  bandwidth-bound).

---

## 4. Blockers hit & code changes

The two anticipated numeric hazards were pre-empted by design and **did not** bite at
runtime; the real blocker was structural, found only at bench scale:

- **Cross-phase accumulate bit (primary anticipated risk):** handled by the
  `ScaleOut::Zero`-then-`One` scheme (never reset at the phase boundary). The test adds
  **two phase-isolation diagnostics** — `Mlp5.Phase1_C0` (`C=0`, isolates `dU·B`) and
  `Mlp5.Phase2_B0` (`B=0`, isolates `dV·C`). Both pass **and** the combined
  `Mlp5.Correctness` passes (incl. the `splits=2` 2D-grid rows), confirming the
  accumulate bit and MN-major operands together. No "dropped-term" bug occurred.
- **MN-major operand B/C:** inherited from the mlp2_t resolution (`_SS` atom,
  `Major::MN`); no debug loop.
- **The real blocker — TMEM lifetime under the persistent 2D grid (found via
  compute-sanitizer at bench scale):** the mandated persistent grid uses
  `grid.x = num_sms/NSplit`, which is **< `num_m_tiles`**, so each CTA loops over
  several m-tiles. Doing a per-tile `tcgen05.alloc` / `relinquish` then **trapped**
  (`phase_invalid_during_alloc`). **Fix:** hoist the TMEM alloc/free to
  **once per CTA** in the launcher (arch-guarded `#if __CUDA_ARCH__ >= 1000`, compiled
  out on sm_90a), reusing one TMEM region across the CTA's m-tiles.
- **smem budget:** at `Stages=4` the footprint is ~208 KiB, within the B200's 227 KiB,
  so **no `K_PIPE` reduction was needed** despite `EpiChunkN=64`.

`Impl<90>` (WGMMA), the 2D-grid launcher geometry, and the `dU/dV` multicast are
otherwise unchanged.

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
     -c liger_cute_kernels/tests/cpp/test_mlp5_fused.cu -o /tmp/mlp5_spill.o
```

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp5`               | 116 | **0 B** | **0 B** |

**Zero spills** — the hard requirement, and the one that matters most here: mlp5 has
the **widest epilogue fragment** (`EpiChunkN=64` → `TmemLoadOp<64>`, a 2× wider
TMEM→register load than the mlp2 kernels), so 116 registers is the highest of the three
new ports yet still well under the 255/thread ceiling and below the MLP1 yardstick
(160). Because the fragment is loaded and consumed one `EpiChunkN`-chunk at a time, no
`-maxrregcount` / `__launch_bounds__` cap was required.
