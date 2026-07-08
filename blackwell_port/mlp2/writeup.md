# Writeup — MLP2-T ("MLP2 transpose") Blackwell (SM100a / UMMA / tcgen05) port

**Op:** `Y = Z · A` — the transpose companion of the MLP2 down-projection. Single
fused GEMM, **contract over `H` (hidden_dim, the K axis), N-axis = `I`
(intermediate_dim)**. The physical `A` weight buffer is the *same* `[E·H, I]`
row-major tensor as `mlp2_fused`, but here it is consumed **MN-major** (through a
column-major TMA view) so the contraction runs over `H` instead of `I`.

Concretely: `Z:[T,H]`, `A:[E,H,I]`, `Y:[T,I]`, and
`Y[t,i] = Σ_h Z[t,h] · A[e,h,i]`, with `e = expert_ids[t / TileM]`.

**Result: fully green on a B200 (compute_cap 10.0).** Correctness PASSes (this is
the key signal that the MN-major operand-B descriptor is right), the TFLOPS
benchmark runs, the hot UMMA consumer has **zero register spills at 79 registers**
(identical to the `mlp2_fused` yardstick), and the Hopper `sm_90a` path still
compiles (WGMMA body byte-for-byte unchanged).

This port is a clone of the proven single-accumulator `mlp2_fused` UMMA consumer;
**the only kernel-unique change is the operand-B major mode** (`UMMA::Major::MN`).

---

## Changed files (only the three I own)

| File | Change |
|------|--------|
| `liger_cute_kernels/csrc/core/src/moe/mlp2_t.cuh` | Traits. Added SM100 includes (`mma_sm100_umma.hpp`, `mma_traits_sm100.hpp`, `tmem_allocator_sm100.hpp`, `copy_sm100.hpp`, `copy_traits_sm100.hpp`, `sm100_pipeline.hpp`) + `TmemLoadOpSelector`/`TmemLoadOp`. In `Mlp2TTraits`: `MainloopPipelineUmma` (`PipelineTmaUmmaAsync`), `AccStages=1`, `AccumulatorPipeline` (`PipelineUmmaAsync<1>`), and the crux **`TiledMmaUmma = make_tiled_mma(SM100_MMA_F16BF16_SS<Element,Element,float,TileM,TileN,UMMA::Major::K,UMMA::Major::MN>{})`**. Kept `SmemLayoutAtomZ = Layout_K_SW128_Atom` (A operand = Z, K-major) and `SmemLayoutAtomW = Layout_MN_SW128_Atom` + `Step<_2,_1>` `tile_to_shape` (B operand = A weight, MN-major) unchanged. |
| `liger_cute_kernels/csrc/core/src/moe/mlp2_t_fused.cuh` | Consumer. Added `tmem_base` + `acc_pipe` to `Mlp2TFusedSmem`; `mlp2_t_make_pipe_umma` (num_consumers=1); split `mlp2_t_fused_consumer` → `Mlp2TFusedConsumerImpl<90>` (verbatim original WGMMA body) / `<100>` (UMMA clone of `mlp2_fused`'s `Impl<100>`, using `Traits::TiledMmaUmma` and the `intermediate_dim` N-extent) + a free-function forwarder preserving the exact signature `mlp_bwd.cuh` calls. Producer unchanged. |
| `liger_cute_kernels/tests/cpp/test_mlp2_t_fused.cu` | Overwrote the pre-staged `int main(){}` stub in place with a clone of `test_mlp2_fused.cu` adapted to `Y = Z·A`: column-major `A` TMA view (`make_shape(I, E·H), stride(1, I)`), CPU reference contracting over `H`, `MainloopPipelineFor` + `if constexpr(Compute==100)` launcher (`__trap()` on the 90 body under sm_100a), `run_t<Compute>` (CPU-ref compare), `run_t_single_tile<Compute>` (element-by-element operand-major localizer), `run_t_bench<Compute>` (median CUDA-event timing + N-split divisor sweep, gated on `MLP2T_BENCH`), `blackwell/hopper_available()`, `kShapes`/`kBenchShapes`, the `Mlp2T.Correctness` / `Mlp2T.SingleTile` / `Mlp2TSm90.Correctness` / `Mlp2T.TFLOPs_{Blackwell,Hopper}` TESTs, and an arch-aware `main()`. |

Did **not** touch `tests/cpp/CMakeLists.txt` (the `test_mlp2_t_fused` target was
pre-registered), the shared root `writeup.md`, `mlp2_fused*`/`mlp5*`/`mlp1*`, or any
other kernel/test.

---

## The operand-major resolution (the whole reason mlp2_t is its own port)

On Hopper WGMMA an MN-major operand-B "just works" via the GMMA descriptor's
major-mode field (`GmmaSelectorKMN`). On SM100 UMMA (`tcgen05.mma`) the operand
descriptor (`UMMA::SmemDescriptor`) is a different encoding, so the major mode must
be selected explicitly on the MMA atom. A wrong major mode is a **silent numeric
error** — it compiles and runs, but `Y` comes out transposed/garbled.

**What worked (validated against CUTLASS 4.4.1 headers, not guessed):**

```cpp
using TiledMmaUmma = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<Element, Element, float,
                         TileM, TileN,
                         UMMA::Major::K,      // operand A = Z   → K-major
                         UMMA::Major::MN>{}));// operand B = A   → MN-major
```

- The **`_SS` (smem×smem) atom variant** is required: it takes *both* `a_major` and
  `b_major` as free template parameters. (The `_TS` variants `static_assert`
  `a_major == UMMA::Major::K` and are unusable for a mixed A=K / B=MN config — though
  here A is K anyway, `_SS` is the variant that lets B be MN.) — `mma_sm100_umma.hpp`.
- `SM100_MMA_F16BF16_SS`'s traits set `FrgTypeB = UMMA::smem_desc<b_major>`; the B
  descriptor is built by **`make_umma_desc<UMMA::Major::MN>`**, which reads the
  *smem tensor's stride* to fill the descriptor's leading-dim / stride-byte-offset
  fields. — `mma_traits_sm100.hpp`, `mma_sm100_desc.hpp`.
- That descriptor builder accepts exactly the canonical MN-major SW128 layout, which
  is what `SmemLayoutW` already is (`Layout_MN_SW128_Atom` + `Step<_2,_1>`
  `tile_to_shape`). **No smem-layout change was needed** — only the atom's `b_major`
  template arg flips from `K` (mlp2_fused) to `MN`.
- The A weight is fed by a **column-major TMA view** of the `[E·H, I]` buffer:
  `make_tensor(pA, make_shape(I, E·H), make_stride(_1{}, I))` — making `I` (the N
  axis) the contiguous MN dimension. This mirrors `moe_bwd.cu`'s production
  `tma_load_a_col`.

**Proof it's right:** `Mlp2T.SingleTile` (a `128×128×128` one-tile case) diffs the
device `Y` against the CPU reference element-by-element. The `got[0:4,0:4]` corner
matches `ref[0:4,0:4]` to bf16 rounding with **no transpose and no block-swizzle
pattern** — a wrong `b_major` would show a structured (transposed / mis-swizzled)
error here. `max_rel = 0.39%`.

---

## Register / spill check (SM100, `--ptxas-options=-v`)

| Kernel (entry) | Registers | Spill stores | Spill loads | Stack frame |
|----------------|-----------|--------------|-------------|-------------|
| `mlp2_t_fused_test_kernel<…, Compute=100>` (UMMA consumer) | **79** | **0 bytes** | **0 bytes** | 0 bytes |
| `mlp2_t_fused_test_kernel<…, Compute=90>` (WGMMA consumer) | 27 | 0 bytes | 0 bytes | 0 bytes |

**Zero spill bytes** on the hot `Impl<100>` path, at **exactly 79 registers — the
`mlp2_fused` yardstick to the register.** No `-maxrregcount` / `__launch_bounds__`
tightening was needed (the epilogue was cloned verbatim: `EpiChunkN=32 →
TmemLoadOp<32>`, single accumulator `allocate(WgTileN)`, `flat_divide` +
`partition_D`-sized fragment + reused `store_buf` + TMA store).

---

## Correctness — B200 (Compute=100 / UMMA), `Mlp2T.Correctness` + `Mlp2T.SingleTile`

Tolerances: `mean_rel < 1%`, `max_rel < 5%` (only error source is bf16 input/output
rounding; fp32 accumulation). **All PASS.**

| Shape `{T,H,I,E}` | mean_rel | max_rel | max_abs |
|-------------------|----------|---------|---------|
| `{128,128,128,1}` | 0.140% | 0.387% | 0.121 |
| `{128, 64,256,1}` | 0.141% | 0.387% | 0.120 |
| `{256,256,256,2}` | 0.141% | 0.404% | 0.201 |
| `{384,128,256,3}` | 0.141% | 0.389% | 0.125 |
| **SingleTile** `{128,128,128}` | 0.140% | 0.388% | 0.125 |

`worst@(t=96,i=25) got=32.2500 ref=32.1252` — a single-ulp bf16 output-rounding
difference, not a structural error.

---

## TFLOPS — B200 (Compute=100), `Mlp2T.TFLOPs_Blackwell` (`MLP2T_BENCH=1`)

FLOPs counted as one GEMM `2·T·H·I`; median CUDA-event timing; grid N-split swept
over the divisors of `num_n_tiles = I/TileN = 32`; peak reported. Device: 148 SMs.
Peak reference: **B200 bf16 ≈ 2.25 PFLOPS = 2250 TFLOPS**.

| Shape `{T,H,I,E}` | peak TFLOPS | median ms | winning split | CTAs | % of B200 peak |
|-------------------|-------------|-----------|---------------|------|----------------|
| `{ 2048,4096,4096,8}` | 725.50 | 0.0947 | 8  | 128  | 32.2% |
| `{ 4096,4096,4096,8}` | **826.27** | 0.1663 | 32 | 1024 | **36.7%** |
| `{ 8192,4096,4096,8}` | 774.74 | 0.3548 | 4  | 256  | 34.4% |
| `{16384,4096,4096,8}` | 674.09 | 0.8156 | 8  | 1024 | 30.0% |

Peak **826 TFLOPS** at `T=4096`. Throughput tracks occupancy: the split that best
fills 148 SMs wins (small `T` under-fills at low split; the largest `T` is slightly
memory/epilogue-bound). These numbers match the `mlp2_fused` envelope — expected,
since the consumer is the same; the MN-major B feed costs nothing at runtime (it's a
descriptor-encoding difference, resolved at compile time).

---

## No Hopper regression (`sm_90a`)

- `test_mlp2_t_fused` **compiles + links clean for `sm_90a`** (build dir
  `build90a_mlp2`). The `Impl<90>` WGMMA body is byte-for-byte the original; the
  `Impl<100>` body is `#if __CUDA_ARCH__ >= 1000`-gated (traps otherwise), so the
  same source builds for both arches.
- `Mlp2TSm90.Correctness` **SKIPs on the B200** (`requires an sm_90 (Hopper) GPU`) —
  as designed; it would run WGMMA on an actual H100.

---

## Reproduce

```bash
export CUTLASS_HOME=/usr/local/include/cutlass

# ── Blackwell build (sm_100a) ──
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp2 -G Ninja \
  -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp2 --target test_mlp2_t_fused -j

# Correctness (arch-aware default filter → Mlp2T.Correctness + Mlp2T.SingleTile)
./liger_cute_kernels/build100a_mlp2/tests/cpp/test_mlp2_t_fused

# TFLOPS
MLP2T_BENCH=1 ./liger_cute_kernels/build100a_mlp2/tests/cpp/test_mlp2_t_fused \
  --gtest_filter='*TFLOPs_Blackwell*'

# ── Register / spill check (per-thread regs + spill bytes for Impl<100>) ──
nvcc -std=c++17 -arch=sm_100a -O3 --expt-relaxed-constexpr --ptxas-options=-v \
  -I "$CUTLASS_HOME/include" -I "$CUTLASS_HOME/tools/util/include" \
  -I liger_cute_kernels/csrc/core/src/moe \
  liger_cute_kernels/tests/cpp/test_mlp2_t_fused.cu -lgtest -lpthread \
  -o liger_cute_kernels/build100a_mlp2/mlp2_t_spillcheck 2>&1 | grep -A2 'Compute=100\|registers'
#   → "0 bytes spill stores, 0 bytes spill loads" ; "Used 79 registers"

# ── Hopper no-regression (sm_90a compiles; Hopper test SKIPs on B200) ──
cmake -S liger_cute_kernels -B liger_cute_kernels/build90a_mlp2 -G Ninja \
  -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build liger_cute_kernels/build90a_mlp2 --target test_mlp2_t_fused -j
./liger_cute_kernels/build90a_mlp2/tests/cpp/test_mlp2_t_fused \
  --gtest_filter='Mlp2TSm90.Correctness'   # → SKIPPED on B200
```

---

## Blockers hit

None fatal. Notes:

- **Operand-B major mode (the port's raison d'être):** resolved by
  `SM100_MMA_F16BF16_SS<…, UMMA::Major::K, UMMA::Major::MN>` — validated against the
  installed CUTLASS 4.4.1 headers before building. Correctness was green on the first
  real run (no silent-transpose debug loop needed).
- **B5 (whole-warp `tcgen05.alloc`):** inherited correctly from the cloned
  `mlp2_fused` consumer — no hang.
- **`run_t` return type:** the initial draft returned `ErrStats` from a function using
  `ASSERT_*`/`CUDA_OK` (which `return;` on failure) — a hard compile error. Fixed by
  making `run_t`/`run_t_single_tile` return `void` (matches the sibling `run_fused`).

**Status: DONE — mlp2_t is fully green on B200.**
