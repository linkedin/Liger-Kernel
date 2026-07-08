# Plan — MLP3 (down-weight gradient) Blackwell (SM100a) port

Execution plan for `dA = dYᵀ · Z`. Self-contained for a subagent. See
[`brief.md`](brief.md) for the design (REDUCE_ADD epilogue + TMEM-fill + store-buf
mapping pin + persistent-grid alloc hoist) and [`../README.md`](../README.md) for the
shared recipe + isolation rules. **Owns only:** `csrc/core/src/moe/mlp3.cuh`,
`tests/cpp/test_mlp3.cu`, build dir `build100a_mlp3`, and
`blackwell_port/mlp3/writeup.md` (+ root `writeup_mlp3.md`).

Env (B200, from the MLP1 bring-up): B200 (SM100), `nvcc` 12.9, CMake+Ninja, CUTLASS
4.4.1 header-only, gtest at `/usr/local`. `export CUTLASS_HOME=/usr/local/include/cutlass`.

> **mlp3 is the reference for mlp4.** Get the REDUCE_ADD TMEM-fill + store-buf mapping
> right here; mlp4 reuses it inside a two-phase (dB/dC) loop.

---

## Step 0 — Isolation setup (orchestrator pre-stages the shared CMake)

The **orchestrator** adds this target to `tests/cpp/CMakeLists.txt` *before* launch
(mirrors the existing `test_mlp5_fused` block):

```cmake
add_executable(test_mlp3 test_mlp3.cu)
target_link_libraries(test_mlp3 PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp3 PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp3 SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp3 PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp3 DISCOVERY_MODE PRE_TEST)
```

If pre-staging was skipped, add exactly this block yourself as your **first and only**
edit to the shared file. **Fallback (no CMake):** `nvcc -std=c++17 -arch=sm_100a -O3
--expt-relaxed-constexpr -I "$CUTLASS_HOME/include" -I csrc/core/src/moe
tests/cpp/test_mlp3.cu -lgtest -lpthread -o /tmp/test_mlp3`.

## Step 1 — Configure for mlp3 (all in `mlp3.cuh`)

- [ ] `#include <cute/arch/tmem_allocator_sm100.hpp>` + local `TmemLoadOpSelector` /
      `TmemLoadOp<EpiChunkN>` (copy from `mlp5.cuh`; `EpiChunkN=64 → …64x`).
- [ ] `Mlp3Traits`: add `MainloopPipelineUmma = PipelineTmaUmmaAsync<Stages>` +
      `mlp3_make_pipe_umma` (`num_consumers=1`); add SM100 UMMA `TiledMmaUmma`
      (`SM100_MMA_F16BF16_SS<…, Major::MN, Major::MN>` — **both operands MN-major**;
      confirm vs `SmemLayoutDYT`/`SmemLayoutZ`, mirror `mlp2_t.cuh`). Accumulator in TMEM.
- [ ] Split `mlp3_consumer` → `Mlp3ConsumerImpl<90>` (verbatim WGMMA) / `<100>` (UMMA)
      + free-function forwarder `mlp3_consumer<Traits,Compute>`. Producer + `mlp3_fwd`
      launcher stay arch-agnostic.
- [ ] `Impl<100>` mainloop: **whole-MMA-warp** `Allocator1Sm.allocate(<cols>,&tmem_base)`
      + `__syncwarp()` **once per CTA, before the `cell_idx` loop** (persistent grid);
      matching `release_allocation_lock()`+`free()` at CTA exit. UMMA `gemm` over the
      k-pipe, `ScaleOut::Zero` on the first k of each `(m_tile,n_tile)` walk, `One` after.
- [ ] `Impl<100>` epilogue: **keep the `is_my_wg_leader` REDUCE_ADD block byte-for-byte.**
      Replace only the register→`store_buf` scatter with: `(M,N)` extract → `flat_divide`
      by `(WgTileM,EpiChunkN)` → `TmemLoadOp<64>` → `partition_D` regs → cast → write
      `sStore(m_local,chunk_n)`. **Re-derive `(m_local,n_local)` from the TMEM-load
      `partition_D` identity coords**, not the SM90 `tCcC` (brief §2.2 pin).
- [ ] `Impl<90>` unchanged.

Configure the isolated Blackwell build:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp3 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
```

## Step 2 — Write the test (`tests/cpp/test_mlp3.cu`)

Clone `test_mlp1_fused.cu`; adapt to the single-GEMM **REDUCE_ADD** shape:

- [ ] `MainloopPipelineFor<Traits,Compute>` + launcher: `if constexpr(Compute==100)` →
      `mlp3_make_pipe_umma` + `mlp3_consumer<…,100>`, else 90-path with `__trap()` under
      `sm_100a`. Build the **chunk-fixed grid** `(cell_stride, …)` the launcher uses.
- [ ] `run3<Compute>(shape)`: random `dY`,`Z`; **zero-init `dA`**; build the
      `SM90_TMA_REDUCE_ADD` desc + input TMA descs; set
      `cudaFuncAttributeMaxDynamicSharedMemorySize`; launch; compare vs fp32 CPU ref
      `dA = dYᵀ·Z`; print `mean_rel/max_rel/max_abs`. **Re-zero `dA` between correctness
      and bench.**
- [ ] `run3_bench<Compute>(shape,cfg)`: FLOPs **`2·T·H·I`**; median CUDA-event timing;
      **N-split sweep over every divisor** of `num_n_tiles` → peak TFLOPS + winning
      `grid.y`. Gate on `std::getenv("MLP3_BENCH")`; re-zero `dA` each launch.
- [ ] `blackwell_available()`(major==10)/`hopper_available()`(major==9); `kBenchShapes`
      (`H=I=4096, E=8, T∈{2048,4096,8192,16384}`, `T` a multiple of `TileM`).
- [ ] TESTs: `Mlp3.Correctness` (Blackwell), `Mlp3Sm90.Correctness` (Hopper),
      `Mlp3.TFLOPs_{Blackwell,Hopper}` (MLP3_BENCH-gated); arch-aware `main()`.
- [ ] Add a **tiny single-tile** correctness case (element-by-element vs CPU) to localize
      a store-buf mapping bug fast.

## Step 3 — Compile for 100a

```bash
cmake --build liger_cute_kernels/build100a_mlp3 --target test_mlp3 -j
```

- [ ] Drive compile errors to zero. Expect **B1–B4** epilogue errors first
      (`RegNumDst`/`AtomTVLayout`/5-mode tensor) — fixes in the brief. MN-major operand
      major mode per `mlp2_t.cuh`. Cross-check CUTLASS 4.4.1 headers
      (`copy_traits_sm100.hpp`, `tmem_allocator_sm100.hpp`, `sm100_epilogue_*`).

## Step 4 — Run correctness + TFLOPs

```bash
./liger_cute_kernels/build100a_mlp3/tests/cpp/test_mlp3                       # correctness
MLP3_BENCH=1 ./liger_cute_kernels/build100a_mlp3/tests/cpp/test_mlp3 \
      --gtest_filter='*TFLOPs_Blackwell*'                                     # TFLOPS
```

Triage order:
- [ ] Hang (100% spin, `compute-sanitizer --tool synccheck` clean) → **B5** whole-warp
      `tcgen05.alloc`, or **B5′** per-cell alloc trap → hoist alloc/free once-per-CTA.
- [ ] Structured wrong result → **store-buf mapping** (run the tiny single-tile case;
      compare element-by-element vs the SM90 result) or MN-major operand.
- [ ] Confirm `Mlp3.Correctness` PASS (`mean_rel<1%`, `max_rel<5%`), then the TFLOPS table.
- [ ] **Register-spill checker** (assert 0 spills on the `Compute=100` consumer):
      ```bash
      nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
           --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
           --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
           -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
           -c liger_cute_kernels/tests/cpp/test_mlp3.cu -o /tmp/mlp3_spill.o
      ```

## Step 5 — Consolidate build + definition of success

- [ ] **Prove it builds under the canonical `build100a`** (remove kernel-specific build):
      `cmake --build liger_cute_kernels/build100a --target test_mlp3 -j`, then **delete
      `build100a_mlp3`**. Rebuild the `sm_90a` regression:
      `cmake --build liger_cute_kernels/build90a --target test_mlp3 -j` (WGMMA path must
      still compile; `Mlp3Sm90.Correctness` SKIPs on B200).
- [ ] `Mlp3.Correctness` PASS on B200 — `mean_rel<1%`, `max_rel<5%`.
- [ ] `Mlp3.TFLOPs_Blackwell` prints a peak-TFLOPS table (`2·T·H·I`, per shape + split).
- [ ] Zero register spills on the `Compute=100` consumer.
- [ ] Write `blackwell_port/mlp3/writeup.md` (+ root `writeup_mlp3.md`): the nuance
      (REDUCE_ADD + TMEM-fill + mapping), a "what changed for SM100" table, correctness +
      TFLOPS tables, blockers hit & fixes, and the reg/spill result. **Do NOT edit root
      `blackwell.md`** — hand the TFLOPS + reg/spill numbers back to the orchestrator.

### Reproduce (paste into writeup)

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a --target test_mlp3 -j
MLP3_BENCH=1 ./liger_cute_kernels/build100a/tests/cpp/test_mlp3
```
