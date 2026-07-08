# Plan — MLP5 (backward input grad) Blackwell (SM100a) port

Execution plan for `dX = dU·B + dV·C`. Self-contained for a subagent. See
[`brief.md`](brief.md) for the design (cross-phase accumulate + MN-major operand +
smem budget) and [`../README.md`](../README.md) for the shared recipe + isolation
rules. **Owns only:** `mlp5.cuh`, `mlp5_fused.cuh`, `tests/cpp/test_mlp5_fused.cu`,
build dir `build100a_mlp5`, and `blackwell_port/mlp5/writeup.md`.

> **Recommended ordering:** this is the hardest port; it reuses mlp2_fused's
> single-acc UMMA epilogue **and** mlp2_t's MN-major operand resolution. If those
> two are already green, start by copying their solutions; mlp5 then reduces to the
> cross-phase accumulate bit + smem/grid handling. In a fully parallel launch, budget
> the most iteration here.

---

## Step 0 — Isolation setup (orchestrator pre-stages the shared CMake)

Orchestrator adds this to `tests/cpp/CMakeLists.txt` **before** launch:

```cmake
add_executable(test_mlp5_fused test_mlp5_fused.cu)
target_link_libraries(test_mlp5_fused PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp5_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp5_fused SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp5_fused PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp5_fused DISCOVERY_MODE PRE_TEST)
```

If not pre-staged, add exactly this block as your **only** edit to the shared file.
**Fallback (no CMake):** `nvcc -std=c++17 -arch=sm_100a -O3 --expt-relaxed-constexpr
-I "$CUTLASS_HOME/include" -I csrc/core/src/moe tests/cpp/test_mlp5_fused.cu
-lgtest -lpthread -o /tmp/test_mlp5_fused`.

## Step 1 — Configure for mlp5

Traits/helpers in **`mlp5.cuh`**, consumer split in **`mlp5_fused.cuh`**:

- [ ] `mlp5.cuh`: TMEM include + `TmemLoadOpSelector`; add `MainloopPipelineUmma`
      alias + `mlp5_make_pipe_umma` (`num_consumers=1`); add SM100 UMMA `TiledMMA`
      with operand A = K-major (`dU`/`dV`), **operand B = MN-major (`B`/`C`)** (reuse
      mlp2_t's resolution). Accumulator `(AtomTileM, WgTileN)` in TMEM.
- [ ] `mlp5_fused.cuh`: split `mlp5_fused_consumer` → `Impl<90>` (verbatim WGMMA) /
      `Impl<100>` (UMMA) + forwarder.
- [ ] `Impl<100>` mainloop — **the crux**: whole-warp `allocate(WgTileN,…)`; run the
      continuous `2·num_k_tiles` loop with the accumulate bit **false on the first
      MMA only**, **true for all others including the phase-1→phase-2 boundary**
      (`UMMA::ScaleOut::Zero` then `One`; do **not** reset at phase 2). Keep W-slot
      reuse (B in phase 1, C in phase 2).
- [ ] `Impl<100>` epilogue (reuse mlp2_fused): `(M,N)` extract → `flat_divide` →
      `TmemLoadOp<64>` → `partition_D` regs → existing cast+store; whole-warp
      `release`+`free`.
- [ ] Producer + `Impl<90>` unchanged; keep the 2D-grid launcher + dU/dV multicast.

Configure isolated build:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp5 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
```

## Step 2 — Write the tests

Clone `test_mlp1_fused.cu` → `tests/cpp/test_mlp5_fused.cu`, **two-GEMM** shape, CPU
reference `dX = dU·B + dV·C` (B,C MN-major — match the storage exactly):

- [ ] `MainloopPipelineFor<Traits,Compute>` + launcher (`if constexpr(Compute==100)`
      pipe/consumer, `__trap()` 90-fallback). Launcher must build the **2D grid**
      `(num_sms/NSplit, NSplit)`.
- [ ] `run5<Compute>(shape)`: random `dU,dV,B,C`; launch; compare vs CPU
      `dU·B+dV·C`; print `mean_rel/max_rel/max_abs`.
- [ ] `run5_bench<Compute>(shape,cfg)`: FLOPs **`4·T·H·I`**; median event timing;
      N-split sweep over divisors (== grid.y sweep); gate on `MLP5_BENCH`.
- [ ] avail gates; `kBenchShapes` (`H=I=4096, E=8, T∈{2048,4096,8192,16384}`).
- [ ] TESTs: `Mlp5.Correctness` (Blackwell), `Mlp5Sm90.Correctness` (Hopper),
      `Mlp5.TFLOPs_{Blackwell,Hopper}` (MLP5_BENCH-gated); arch-aware `main()`.
- [ ] **Two diagnostic tiny-shape cases** (single tile): one with `C=0` (isolates
      `dU·B` / phase 1) and one with `B=0` (isolates `dV·C` / phase 2). If either
      passes but the combined case fails → the **cross-phase accumulate bit** is
      wrong (brief §blockers). Fastest possible localization.

## Step 3 — Compile for 100a

```bash
cmake --build liger_cute_kernels/build100a_mlp5 --target test_mlp5_fused -j
```

- [ ] Standard B1–B4 epilogue fixups (`…64x` atom). Operand-B major mode per mlp2_t.
- [ ] **If it overflows smem** (`cudaFuncAttributeMaxDynamicSharedMemorySize` set
      fails, or launch returns `cudaErrorInvalidValue`) → **reduce `K_PIPE` stages**
      until it fits 227 KB (brief §smem budget), rebuild.

## Step 4 — Run correctness + TFLOPs

```bash
./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused        # correctness
MLP5_BENCH=1 ./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused \
      --gtest_filter='*TFLOPs_Blackwell*'
```

- [ ] Correctness triage order: run the two tiny diagnostic cases **first**.
      - both isolated cases pass, combined fails → **accumulate bit** at the phase
        boundary (fix the `ScaleOut` handling, don't touch the epilogue).
      - an isolated case itself fails with structured error → **MN-major operand**
        (fix per mlp2_t).
      - hang (100% spin, `synccheck` clean) → **B5** whole-warp `tcgen05.alloc`.
- [ ] Then full-shape correctness, then TFLOPS with the split sweep.

## Step 5 — Definition of success

- [ ] `Mlp5.Correctness` PASS on B200 — `mean_rel<1%`, `max_rel<5%` (confirms
      cross-phase accumulate **and** MN-major operands are both correct).
- [ ] `Mlp5.TFLOPs_Blackwell` prints a peak-TFLOPS table (`4·T·H·I`, per shape +
      winning split).
- [ ] `sm_90a` build still compiles; `Mlp5Sm90.Correctness` PASS where HW exists.
- [ ] `blackwell_port/mlp5/writeup.md`: changed files, how the accumulate bit +
      operand major + any `K_PIPE` reduction were resolved, correctness + TFLOPS
      tables, reproduce cmds.

### Reproduce (paste into writeup)

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp5 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp5 --target test_mlp5_fused -j
MLP5_BENCH=1 ./liger_cute_kernels/build100a_mlp5/tests/cpp/test_mlp5_fused
```
