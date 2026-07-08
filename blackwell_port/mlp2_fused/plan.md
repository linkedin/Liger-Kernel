# Plan — MLP2-fused Blackwell (SM100a) port

Execution plan for `Y = Z·Aᵀ`. Self-contained for a subagent. See
[`brief.md`](brief.md) for the design and [`../README.md`](../README.md) for the
shared recipe + isolation rules. **Owns only:** `mlp2_fused.cuh`,
`tests/cpp/test_mlp2_fused.cu`, build dir `build100a_mlp2_fused`, and
`blackwell_port/mlp2_fused/writeup.md`.

Env assumed (B200, from the MLP1 bring-up): B200 (SM100), `nvcc` 12.9, CMake+Ninja,
CUTLASS 4.4.1 header-only, gtest at `/usr/local`. Set `CUTLASS_HOME` if
`find_package(CUTLASS)` can't locate it.

---

## Step 0 — Isolation setup (orchestrator pre-stages the shared CMake)

To keep three subagents off the one shared file, the **orchestrator** adds this
target to `tests/cpp/CMakeLists.txt` *before* launch (mirrors the existing
`test_mlp1_fused` block):

```cmake
add_executable(test_mlp2_fused test_mlp2_fused.cu)
target_link_libraries(test_mlp2_fused PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp2_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp2_fused SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp2_fused PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp2_fused DISCOVERY_MODE PRE_TEST)
```

If pre-staging was skipped, add exactly this block yourself as your **first and only**
edit to the shared file, then proceed. **Fallback (no CMake):** compile standalone —
`nvcc -std=c++17 -arch=sm_100a -O3 --expt-relaxed-constexpr -I "$CUTLASS_HOME/include"
-I csrc/core/src/moe tests/cpp/test_mlp2_fused.cu -lgtest -lpthread -o /tmp/test_mlp2_fused`.

## Step 1 — Configure for mlp2_fused

Port `mlp2_fused.cuh` per the README recipe (single-accumulator specialization):

- [ ] `#include <cute/arch/tmem_allocator_sm100.hpp>` + local `TmemLoadOpSelector`.
- [ ] `Mlp2Traits`: add `MainloopPipelineUmma` alias + SM100 UMMA `TiledMMA`
      (accumulator `(AtomTileM, WgTileN)` in TMEM; A=`Z`, B=`A`, both K-major).
- [ ] `mlp2_make_pipe_umma` (`num_consumers=1`).
- [ ] Split `mlp2_fused_consumer` → `Mlp2FusedConsumerImpl<90>` (verbatim WGMMA) /
      `<100>` (UMMA) + free-function forwarder `mlp2_fused_consumer<Traits,Compute>`.
- [ ] `Impl<100>`: whole-warp `allocate(WgTileN,…)`+`__syncwarp()`; UMMA gemm;
      `(M,N)` extract → `flat_divide` → `TmemLoadOp<EpiChunkN>` → `partition_D`-sized
      regs → existing cast+store; whole-warp `release_allocation_lock()`+`free()`.
- [ ] Producer unchanged; Hopper `Impl<90>` byte-for-byte unchanged.

Configure the tests-only Blackwell build in an **isolated** dir:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp2_fused -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
```

## Step 2 — Write the tests

Create `tests/cpp/test_mlp2_fused.cu` by cloning `test_mlp1_fused.cu` and reducing
to the single-GEMM shape:

- [ ] `MainloopPipelineFor<Traits,Compute>` + launcher with
      `if constexpr(Compute==100)` pipe/consumer selection and `__trap()` 90-fallback.
- [ ] `run_fused<Compute>(shape)`: random `Z`,`A`; launch; compare vs CPU reference
      `Y = Z·Aᵀ`; print `mean_rel/max_rel/max_abs`.
- [ ] `run_fused_bench<Compute>(shape,cfg)`: FLOPs `2·T·H·I`; median event timing;
      **N-split divisor sweep** → peak TFLOPS + winning `grid.y`. Gate on
      `std::getenv("MLP2_BENCH")`.
- [ ] `blackwell_available()`/`hopper_available()`; `kBenchShapes` (`H=I=4096,E=8`,
      `T∈{2048,4096,8192,16384}`).
- [ ] TESTs: `Mlp2Fused.Correctness` (Blackwell), `Mlp2FusedSm90.Correctness`
      (Hopper), `Mlp2Fused.TFLOPs_{Blackwell,Hopper}` (MLP2_BENCH-gated).
- [ ] Arch-aware `main()` (default filter to the present GPU); link `GTest::gtest`.

## Step 3 — Compile for 100a

```bash
cmake --build liger_cute_kernels/build100a_mlp2_fused --target test_mlp2_fused -j
```

- [ ] Drive compile errors to zero. Expect the B1–B4 epilogue errors first
      (`RegNumDst` / `AtomTVLayout` / 5-mode tensor) — fixes are in the brief.
- [ ] Cross-check any "how does CUTLASS do X" against the installed 4.4.1 headers
      (`copy_traits_sm100.hpp`, `tmem_allocator_sm100.hpp`, the `sm100_epilogue_*`
      collectives).

## Step 4 — Run correctness + TFLOPs

```bash
# Correctness (fast; default filter already targets the present GPU)
./liger_cute_kernels/build100a_mlp2_fused/tests/cpp/test_mlp2_fused
# TFLOPS
MLP2_BENCH=1 ./liger_cute_kernels/build100a_mlp2_fused/tests/cpp/test_mlp2_fused \
      --gtest_filter='*TFLOPs_Blackwell*'
```

- [ ] If it hangs (100% GPU spin, `synccheck` clean) → the **B5** whole-warp
      `tcgen05.alloc` guard; use `#ifdef`-gated `printf` breadcrumbs under a hard
      `timeout` to localize, then remove the scaffolding.

## Step 5 — Definition of success

- [ ] `Mlp2Fused.Correctness` PASS on B200 — `mean_rel<1%`, `max_rel<5%`.
- [ ] `Mlp2Fused.TFLOPs_Blackwell` prints a peak-TFLOPS table (per shape + split).
- [ ] `sm_90a` build still compiles; `Mlp2FusedSm90.Correctness` PASS where HW
      exists (no WGMMA regression).
- [ ] Write `blackwell_port/mlp2_fused/writeup.md`: changed files, blockers hit, the
      B200 correctness + TFLOPS tables, and the exact reproduce commands.

### Reproduce (paste into writeup)

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp2_fused -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp2_fused --target test_mlp2_fused -j
MLP2_BENCH=1 ./liger_cute_kernels/build100a_mlp2_fused/tests/cpp/test_mlp2_fused
```
