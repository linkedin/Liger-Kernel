# Plan — MLP2 (transpose, `mlp2_t`) Blackwell (SM100a) port

Execution plan for `Y = Z·A` (MN-major weight). Self-contained for a subagent. See
[`brief.md`](brief.md) for the design (esp. the MN-major operand crux) and
[`../README.md`](../README.md) for the shared recipe + isolation rules. **Owns
only:** `mlp2_t.cuh`, `mlp2_t_fused.cuh`, `tests/cpp/test_mlp2_t_fused.cu`, build dir
`build100a_mlp2`, and `blackwell_port/mlp2/writeup.md`.

> **Recommended ordering:** if the [`mlp2_fused`](../mlp2_fused/plan.md) port is
> already green, clone its UMMA consumer as the starting point — everything except
> the operand-B major mode is reusable. If running fully parallel, that's fine too;
> just budget for the operand-layout debug in Step 3/4.

---

## Step 0 — Isolation setup (orchestrator pre-stages the shared CMake)

Orchestrator adds this to `tests/cpp/CMakeLists.txt` **before** launch:

```cmake
add_executable(test_mlp2_t_fused test_mlp2_t_fused.cu)
target_link_libraries(test_mlp2_t_fused PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp2_t_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp2_t_fused SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp2_t_fused PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp2_t_fused DISCOVERY_MODE PRE_TEST)
```

If not pre-staged, add exactly this block as your **only** edit to the shared file.
**Fallback (no CMake):** `nvcc -std=c++17 -arch=sm_100a -O3 --expt-relaxed-constexpr
-I "$CUTLASS_HOME/include" -I csrc/core/src/moe tests/cpp/test_mlp2_t_fused.cu
-lgtest -lpthread -o /tmp/test_mlp2_t_fused`.

## Step 1 — Configure for mlp2_t

Traits changes in **`mlp2_t.cuh`**, consumer split in **`mlp2_t_fused.cuh`**:

- [ ] `mlp2_t.cuh`: TMEM include + `TmemLoadOpSelector`; add `MainloopPipelineUmma`
      alias to `Mlp2TTraits`; add SM100 UMMA `TiledMMA` with **operand A = K-major
      (`Z`), operand B = MN-major (`A`)** — this is the crux (see brief §"operand-B
      descriptor"). Keep `SmemLayoutAtomW = Layout_MN_SW128_Atom` + `Step<_2,_1>`.
- [ ] `mlp2_t_fused.cuh`: add `mlp2_t_make_pipe_umma` (`num_consumers=1`); split
      `mlp2_t_fused_consumer` → `Impl<90>` (verbatim WGMMA) / `Impl<100>` (UMMA) +
      free-function forwarder. Single-acc epilogue: whole-warp `allocate(WgTileN,…)`,
      `(M,N)` extract → `flat_divide` → `TmemLoadOp<EpiChunkN>` → `partition_D` regs
      → existing cast+store; whole-warp `release`+`free`.
- [ ] Producer + `Impl<90>` unchanged.

Configure isolated build:

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp2 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
```

## Step 2 — Write the tests

Clone `test_mlp1_fused.cu` → `tests/cpp/test_mlp2_t_fused.cu`, single-GEMM shape,
CPU reference `Y = Z·A` (**note: not `Aᵀ`** — the reference must match the
transpose semantics; getting this backwards will make a *correct* kernel look wrong):

- [ ] `MainloopPipelineFor<Traits,Compute>` + launcher (`if constexpr(Compute==100)`
      pipe/consumer, `__trap()` 90-fallback).
- [ ] `run_t<Compute>(shape)`: random `Z`,`A`; launch; compare vs CPU `Y=Z·A`;
      print `mean_rel/max_rel/max_abs`.
- [ ] `run_t_bench<Compute>(shape,cfg)`: FLOPs `2·T·H·I`; median event timing;
      N-split divisor sweep; gate on `MLP2T_BENCH`.
- [ ] avail gates; `kBenchShapes` (`H=I=4096, E=8, T∈{2048,4096,8192,16384}`).
- [ ] TESTs: `Mlp2T.Correctness` (Blackwell), `Mlp2TSm90.Correctness` (Hopper),
      `Mlp2T.TFLOPs_{Blackwell,Hopper}` (MLP2T_BENCH-gated); arch-aware `main()`.
- [ ] **Add a tiny-shape correctness case** (`T=I=H=TileM`, single tile) — the
      fastest way to localize an operand-major bug (brief §blockers).

## Step 3 — Compile for 100a

```bash
cmake --build liger_cute_kernels/build100a_mlp2 --target test_mlp2_t_fused -j
```

- [ ] Standard B1–B4 epilogue fixups first. Then the operand-B major mode: if the
      tcgen05 MMA atom rejects the MN-major B layout at compile time, consult
      `cute/atom/mma_traits_sm100.hpp` / `cute/arch/mma_sm100_desc.hpp` in the
      installed CUTLASS 4.4.1 for the correct `UMMA::Major::MN` atom variant.

## Step 4 — Run correctness + TFLOPs

```bash
./liger_cute_kernels/build100a_mlp2/tests/cpp/test_mlp2_t_fused        # correctness
MLP2T_BENCH=1 ./liger_cute_kernels/build100a_mlp2/tests/cpp/test_mlp2_t_fused \
      --gtest_filter='*TFLOPs_Blackwell*'
```

- [ ] **If correctness fails (high `max_rel`) but it compiled and didn't hang → the
      MN-major operand is the suspect**, not the epilogue. Run the tiny-shape case;
      a transposed/mis-swizzled B shows a structured error pattern. Fix the operand
      major mode, not the epilogue.
- [ ] If it *hangs* → B5 whole-warp `tcgen05.alloc` guard (same as the other ports).

## Step 5 — Definition of success

- [ ] `Mlp2T.Correctness` PASS on B200 — `mean_rel<1%`, `max_rel<5%` (confirms the
      MN-major UMMA operand is correct).
- [ ] `Mlp2T.TFLOPs_Blackwell` prints a peak-TFLOPS table.
- [ ] `sm_90a` build still compiles; `Mlp2TSm90.Correctness` PASS where HW exists.
- [ ] `blackwell_port/mlp2/writeup.md`: changed files, the operand-major resolution
      (what atom/major mode worked), correctness + TFLOPS tables, reproduce cmds.

### Reproduce (paste into writeup)

```bash
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a_mlp2 -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a_mlp2 --target test_mlp2_t_fused -j
MLP2T_BENCH=1 ./liger_cute_kernels/build100a_mlp2/tests/cpp/test_mlp2_t_fused
```
