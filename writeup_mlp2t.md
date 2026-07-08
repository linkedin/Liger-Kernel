# writeup_mlp2t — MLP2 transpose (`mlp2_t`) on Blackwell (SM100a)

> **Op:** `Y = Z · A` — the MLP phase-2 **transpose** variant, where the weight `A` is
> consumed **MN-major** (as stored, *not* transposed to K-major). **Files:**
> `csrc/core/src/moe/mlp2_t.cuh` (Traits + UMMA MMA), `mlp2_t_fused.cuh` (consumer
> split), `tests/cpp/test_mlp2_t_fused.cu`. Structurally identical to mlp2_fused
> **except the operand layout** — which is the entire reason it is its own port.

---

## 1. The nuance of MLP2-t

```
Z : [T, I]   activation
A : [I, H]   weight, consumed AS STORED (no transpose) → contraction axis I is the
             weight's minor / MN-swizzled axis
Y = Z · A : [T, H]   contracts over I (K axis); N axis = H
```

The nuance — and the whole crux:

- Everything the consumer does (one accumulator, `EpiChunkN=32`, single fused Z+W
  pipe, cooperative 2-WG epilogue) is **line-for-line the same as mlp2_fused**.
- The **only** material difference is the weight's smem layout:
  ```cpp
  using SmemLayoutAtomZ = GMMA::Layout_K_SW128_Atom<Element>;   // Z: K-major
  using SmemLayoutAtomW = GMMA::Layout_MN_SW128_Atom<Element>;  // A: MN-major ← transpose
  ```
  On Hopper WGMMA the MN-major operand-B is expressed through the GMMA descriptor's
  major-mode field and "just works." On SM100 UMMA (`tcgen05.mma`) the operand
  descriptor (`UMMA::SmemDescriptor`) is a **different encoding**, so the risk is
  entirely: *does the MN-major `A` smem layout produce a correct UMMA operand-B
  descriptor?*

FLOPs (one GEMM): `2·T·H·I` (same as mlp2_fused).

---

## 2. What changed to suit SM100

Steps 1–5 are cloned from the green mlp2_fused UMMA consumer (single-acc epilogue,
`PipelineTmaUmmaAsync`, `mlp2_t_make_pipe_umma`, `Impl<90>`/`Impl<100>` split,
`(M,N)` extract → `flat_divide` → `TmemLoadOp<32>` → `partition_D` regs → cast/store,
whole-warp `tcgen05.alloc`/`free`). The **one unique piece** is the operand-B major
mode:

| Piece | Hopper (`sm_90a`) | Blackwell (`sm_100a`) |
|-------|-------------------|------------------------|
| MMA | `TiledMMA<MMA_Atom<GmmaAtom>>`; MN-major B via GMMA descriptor field | **`SM100_MMA_F16BF16_SS<Element,Element,float,TileM,TileN,UMMA::Major::K, UMMA::Major::MN>`** |
| Operand A (`Z`) | K-major | `UMMA::Major::K` |
| Operand B (`A`) | MN-major | **`UMMA::Major::MN`** |
| Weight smem | `Layout_MN_SW128_Atom` + `Step<_2,_1>` | **unchanged** — its stride already feeds the UMMA MN descriptor |

**The resolution:** the `_SS` atom variant (both operands from smem, both major modes
*free*) accepts an explicit `UMMA::Major::MN` for B — whereas the `_TS` variant asserts
`A == K`. With `Major::MN`, CUTLASS's `make_umma_desc<Major::MN>` reads the leading-dim
byte offset straight from the **existing** `Layout_MN_SW128_Atom` + `Step<_2,_1>` smem
tensor, so **no smem-layout change was needed** — only the atom's B-major flag flips
`K → MN`. This was validated against the installed CUTLASS 4.4.1
`mma_traits_sm100.hpp` / `mma_sm100_desc.hpp` rather than guessed.

**Why this suits SM100:** it lets the down-projection consume the weight in its native
storage order (no pre-transpose pass) while still feeding the tcgen05 tile-MMA a
correctly-swizzled operand — at the *same* 79-register / zero-spill cost as the
K-major sibling.

---

## 3. Performance (from `blackwell.md`)

### B200 (sm_100a / UMMA) — measured, `MLP2T_BENCH=1 … TFLOPs_Blackwell`

FLOPs `2·T·H·I`; median event timing; N-split divisor sweep; `H=I=4096, E=8`. B200
dense-bf16 peak ≈ **2.25 PFLOPS**.

| T | peak TFLOPS | winning split | %peak |
|---|:-----------:|:-------------:|:-----:|
| 2048  | 725.50 | 8  | 32.2% |
| 4096  | **826.27** | 32 | 36.7% |
| 8192  | 774.74 | 4  | 34.4% |
| 16384 | 674.09 | 8  | 30.0% |

Peak **826.27 TFLOPS @ T=4096, split 32** — slightly **above** mlp2_fused at every
`T` (e.g. 826 vs 789 at T=4096).

### sm90a → sm100a speedup (measured on H100)

Measured on an **NVIDIA H100 80GB HBM3** (`sm_90a`, 132 SMs), WGMMA `Impl<90>` path in
`build90a` — `Mlp2T.TFLOPs_Hopper`, same N-split sweep and shapes:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a --target test_mlp2_t_fused -j
MLP2T_BENCH=1 ./liger_cute_kernels/build90a/tests/cpp/test_mlp2_t_fused --gtest_filter='*TFLOPs_Hopper*'
```

| T | H100 peak TFLOPS | split | %peak | B200÷H100 |
|---|:----------------:|:-----:|:-----:|:---------:|
| 2048  | 474.79 | 8 | 48.0% | 1.53× |
| 4096  | **508.91** | 4 | 51.4% | 1.62× |
| 8192  | 451.94 | 4 | 45.7% | 1.71× |
| 16384 | 335.19 | 2 | 33.9% | **2.01×** |

`%peak` is vs the H100 SXM bf16 dense peak ≈ **989.4 TFLOPS**. **B200÷H100 = 1.53–2.01×**,
the same climbing-with-`T` shape as mlp2_fused.

**Measured outcome vs hypothesis.** Same single-GEMM, memory-bound profile as
mlp2_fused (34–51% of compute peak on either arch), and the speedup indeed tracks
mlp2_fused's — rising from ~1.5× to **2.0×** at `T=16384` where the H100 collapses to
33.9%. The extra-nuance prediction is **confirmed**: mlp2_t's B200÷H100 ratio sits a
hair **above** mlp2_fused at *every* `T` (1.53/1.62/1.71/2.01 vs 1.52/1.56/1.62/2.00).
The **MN-major operand-B** stresses the smem→tensor-core swizzle path differently on
the two arches — on Hopper a descriptor-field choice on top of WGMMA, on Blackwell a
native UMMA descriptor — and the marginally friendlier B200 handling is exactly the
few-percent edge that shows up in both the raw B200 TFLOPS and this ratio.

### Cross-tile-size trend

Peak walks **725 → 826 → 775 → 674**, the **same inverted-U** as mlp2_fused (peak at
T=4096, roll-off at both ends). The mechanism is identical — occupancy-limited at
`T=2048` (needs ×8 split to fill 148 SMs), best fill at `T=4096` (×32 → 1024 CTAs),
bandwidth-limited tail at `T=16384` (split→8, %peak→30%). That mlp2_t sits a notch
above mlp2_fused throughout is consistent with the MN-major weight read hitting a
marginally friendlier swizzle/coalescing pattern for these shapes, not with any
algorithmic difference (the GEMM FLOP count is identical).

---

## 4. Blockers hit & code changes

The anticipated hard blocker for mlp2_t was a **silent numeric error** from a wrong
operand-B major mode — a wrong `A` orientation compiles and runs but yields a
transposed/garbled `Y` (correctness `max_rel` blows up, no hang, no crash). To catch
it fast, the test adds a **`Mlp2T.SingleTile`** case (`T=I=H=TileM`) that diffs
element-by-element vs the CPU reference — a transposed operand shows a *structured*
error pattern.

In practice:

- **Operand-major (primary risk):** resolved cleanly by selecting the `_SS` atom with
  `UMMA::Major::MN` for B (see §2). **Correctness was green on the first real run** —
  `SingleTile` `max_rel=0.388%`, corner elements matched, **no transpose/swizzle
  pattern** → the MN-major descriptor was correct. No silent-transpose debug loop was
  needed.
- **CPU reference gotcha (avoided by design):** the reference is `Y = Z·A` (**not**
  `Z·Aᵀ`). Getting that backwards would make a *correct* kernel look wrong; the test
  builds the column-major `A` view (`make_shape(I, E·H), stride(1, I)`) to match.
- **Only actual code fix:** a trivial `run_t` return-type change (`ErrStats → void`)
  so the `ASSERT_*` macros compile. No kernel logic changed beyond the atom major mode.
- The B1–B5 recipe blockers did not recur (inherited from the mlp2_fused clone).

`Impl<90>` (WGMMA) and the producer are unchanged.

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
     -c liger_cute_kernels/tests/cpp/test_mlp2_t_fused.cu -o /tmp/mlp2_t_spill.o
```

| Kernel (Compute=100) | Registers/thread | Spill stores | Spill loads |
|----------------------|:----------------:|:------------:|:-----------:|
| `mlp2_t`             | 79 | **0 B** | **0 B** |

**Zero spills**, at **exactly the mlp2_fused single-acc yardstick (79)** — the MN-major
operand adds no register cost, since it changes only the operand *descriptor*, not the
epilogue that consumes register pressure. No register cap needed.
