# Plan — MLP4 (weight gradients dB, dC) Blackwell (SM100a) port

Execution plan for `dB = dUᵀ·X`, `dC = dVᵀ·X`. Self-contained for a subagent. See
[`brief.md`](brief.md) for the design (mlp3 + two-phase dB/dC loop) and
[`../README.md`](../README.md) for the shared recipe + isolation rules. **Owns only:**
`csrc/core/src/moe/mlp4.cuh`, `tests/cpp/test_mlp4.cu`, build dir `build100a_mlp4`, and
`blackwell_port/mlp4/writeup.md` (+ root `writeup_mlp4.md`).

Env (B200): `export CUTLASS_HOME=/usr/local/include/cutlass`.

> **Port mlp3 first, then start from its resolved `Impl<100>`.** mlp4 = mlp3's REDUCE_ADD
> TMEM-fill wrapped in `for(phase=0..1)` with independent per-phase accumulators and two
> outputs (`dB`,`dC`). Reuse mlp3's store-buf mapping + REDUCE_ADD block verbatim.

---

## Step 0 — Isolation setup (orchestrator pre-stages the shared CMake)

The **orchestrator** adds this target to `tests/cpp/CMakeLists.txt` *before* launch:

```cmake
add_executable(test_mlp4 test_mlp4.cu)
target_link_libraries(test_mlp4 PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp4 PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp4 SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp4 PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp4 DISCOVERY_MODE PRE_TEST)
```

If pre-staging was skipped, add exactly this block yourself as your **first and only**
edit to the shared file. **Fallback (no CMake):** `nvcc -std=c++17 -arch=sm_100a -O3
--expt-relaxed-constexpr -I "$CUTLASS_HOME/include" -I csrc/core/src/moe
tests/cpp/test_mlp4.cu -lgtest -lpthread -o /tmp/test_mlp4`.

## Step 1 — Configure for mlp4 (all in `mlp4.cuh`)

- [ ] TMEM include + local `TmemLoadOp<EpiChunkN=64>` (copy from mlp3/mlp5).
- [ ] `Mlp4Traits`: add `MainloopPipelineUmma = PipelineTmaUmmaAsync<Stages>` +
      `mlp4_make_pipe_umma` (`num_consumers=1`); SM100 UMMA `TiledMmaUmma`
      (`SM100_MMA_F16BF16_SS<…, Major::MN, Major::MN>` — both operands MN-major; confirm
      vs `SmemLayoutA`/`SmemLayoutX`, mirror `mlp2_t.cuh`). Accumulator in TMEM.
- [ ] Split `mlp4_consumer` → `Mlp4ConsumerImpl<90>` (verbatim WGMMA) / `<100>` (UMMA)
      + forwarder `mlp4_consumer<Traits,Compute>`. Producer (already `phase`-looped) +
      `mlp4_fwd` launcher stay arch-agnostic.
- [ ] `Impl<100>`: **whole-MMA-warp** `Allocator1Sm.allocate(<cols>,&tmem_base)` +
      `__syncwarp()` **once per CTA — above both the `cell_idx` loop AND the `phase`
      loop**; matching `release_allocation_lock()`+`free()` at CTA exit. Inside
      `for(phase=0..1)`: pick A=`dUᵀ`(phase 0)/`dVᵀ`(phase 1) + output `dB`/`dC`; UMMA
      `gemm` with **fresh `ScaleOut::Zero` on the first k of *each* phase** (independent
      accumulators), `One` after.
- [ ] `Impl<100>` epilogue (reuse mlp3): **keep the `is_my_wg_leader` REDUCE_ADD block
      byte-for-byte**, parameterized by `phase` for the output tensor/descriptor. Fill
      `store_buf` from TMEM: `(M,N)` extract → `flat_divide((WgTileM,EpiChunkN))` →
      `TmemLoadOp<64>` → `partition_D` regs → cast → `sStore`, mapping re-derived from the
      TMEM-load `partition_D` identity coords (mlp3 §2.2 pin).
- [ ] `Impl<90>` unchanged.

Configure the isolated Blackwell build:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp4 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
```

## Step 2 — Write the test (`tests/cpp/test_mlp4.cu`)

Clone `test_mlp3.cu`; extend to **two GEMMs / two outputs**:

- [ ] `MainloopPipelineFor<Traits,Compute>` + launcher (`if constexpr(Compute==100)`
      pipe/consumer, `__trap()` 90-fallback under `sm_100a`); chunk-fixed grid.
- [ ] `run4<Compute>(shape)`: random `dU`,`dV`,`X`; **zero-init `dB` and `dC`**; build the
      two `SM90_TMA_REDUCE_ADD` descs + input descs; set
      `cudaFuncAttributeMaxDynamicSharedMemorySize`; launch; compare vs fp32 CPU refs
      `dB=dUᵀ·X`, `dC=dVᵀ·X`; print per-output `mean_rel/max_rel/max_abs`. **Re-zero both
      outputs between correctness and bench.**
- [ ] `run4_bench<Compute>(shape,cfg)`: FLOPs **`4·T·H·I`**; median event timing; N-split
      sweep over divisors → peak TFLOPS + winning `grid.y`. Gate on `MLP4_BENCH`; re-zero
      outputs each launch.
- [ ] avail gates; `kBenchShapes` (`H=I=4096, E=8, T∈{2048,4096,8192,16384}`).
- [ ] TESTs: `Mlp4.Correctness` (Blackwell), `Mlp4Sm90.Correctness` (Hopper),
      `Mlp4.TFLOPs_{Blackwell,Hopper}` (MLP4_BENCH-gated); arch-aware `main()`.
- [ ] **Two phase-isolation diagnostics** (tiny single-tile): `Mlp4.PhaseDB` (`dV=0` /
      dC-side zeroed → isolates `dUᵀ·X`) and `Mlp4.PhaseDC` (`dU=0` → isolates `dVᵀ·X`).
      If one passes but combined fails → **phase output routing / acc-carry / re-zero**
      bug (brief §blockers). Fastest localization.

## Step 3 — Compile for 100a

```bash
cmake --build liger_cute_kernels/build100a_mlp4 --target test_mlp4 -j
```

- [ ] Drive compile errors to zero (**B1–B4** epilogue; MN-major operand per `mlp2_t.cuh`).
      If mlp3 is already green, most of this is copy-adapt.

## Step 4 — Run correctness + TFLOPs

```bash
./liger_cute_kernels/build100a_mlp4/tests/cpp/test_mlp4                       # correctness
MLP4_BENCH=1 ./liger_cute_kernels/build100a_mlp4/tests/cpp/test_mlp4 \
      --gtest_filter='*TFLOPs_Blackwell*'                                     # TFLOPS
```

Triage order:
- [ ] Run `Mlp4.PhaseDB` / `Mlp4.PhaseDC` **first**. One passing + combined failing →
      phase routing / independent-clear / re-zero (fix the phase loop, not the epilogue).
- [ ] Hang (100% spin, `synccheck` clean) → **B5** whole-warp alloc, or **B5′** per-cell/
      per-phase alloc trap → hoist alloc/free once-per-CTA above both loops.
- [ ] Structured wrong result in *both* outputs → store-buf mapping / MN-major operand
      (shared with mlp3).
- [ ] Confirm `Mlp4.Correctness` PASS, then the TFLOPS table.
- [ ] **Register-spill checker** (assert 0 spills on the `Compute=100` consumer):
      ```bash
      nvcc -std=c++17 -arch=sm_100a --use_fast_math --extra-device-vectorization \
           --fmad=true --prec-div=false --prec-sqrt=false --expt-relaxed-constexpr -DNDEBUG \
           --ptxas-options=-O3,--allow-expensive-optimizations=true,-v \
           -I "$CUTLASS_HOME/include" -I liger_cute_kernels/csrc/core/src/moe \
           -c liger_cute_kernels/tests/cpp/test_mlp4.cu -o /tmp/mlp4_spill.o
      ```

## Step 5 — Consolidate build + definition of success

- [ ] **Prove it builds under the canonical `build100a`** (remove kernel-specific build):
      `cmake --build liger_cute_kernels/build100a --target test_mlp4 -j`, then **delete
      `build100a_mlp4`**. Rebuild `sm_90a` regression:
      `cmake --build liger_cute_kernels/build90a --target test_mlp4 -j` (`Mlp4Sm90.*`
      SKIPs on B200).
- [ ] `Mlp4.Correctness` + `Mlp4.PhaseDB` + `Mlp4.PhaseDC` PASS on B200
      (`mean_rel<1%`, `max_rel<5%`).
- [ ] `Mlp4.TFLOPs_Blackwell` prints a peak-TFLOPS table (`4·T·H·I`, per shape + split).
- [ ] Zero register spills on the `Compute=100` consumer.
- [ ] Write `blackwell_port/mlp4/writeup.md` (+ root `writeup_mlp4.md`): the nuance
      (mlp3 + two-phase), a "what changed for SM100" table, correctness (incl. phase
      diagnostics) + TFLOPS tables, blockers hit & fixes, reg/spill result. **Do NOT edit
      root `blackwell.md`** — hand the numbers back to the orchestrator.

### Reproduce (paste into writeup)

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a --target test_mlp4 -j
MLP4_BENCH=1 ./liger_cute_kernels/build100a/tests/cpp/test_mlp4
```
