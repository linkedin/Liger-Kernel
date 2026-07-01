# MLP1 Blackwell (SM100a) bring-up — process writeup

## Goal

Take the `Compute=100` (Blackwell / UMMA / tcgen05) rewrite of the MLP1 fused
consumers that landed on branch `jkolehma/lck` (commit `b46869a`, *"Adding
Blackwell variant of the MLP1 consumers"*; PR
[vaibhavjindal/liger-comms-moe#11](https://github.com/vaibhavjindal/liger-comms-moe/pull/11))
and:

1. **Compile just the MLP1 kernel** (`mlp1_fused.cuh` / `mlp1_fused_act.cuh`) for
   `sm_100a` — without building the whole NVSHMEM/TVM‑FFI core.
2. **Make `test_mlp1_fused` pass** on the available GPU.

Files in scope:
`liger_cute_kernels/csrc/core/src/moe/mlp1_fused.cuh`,
`liger_cute_kernels/csrc/core/src/moe/mlp1_fused_act.cuh`,
`liger_cute_kernels/tests/cpp/test_mlp1_fused.cu`,
`liger_cute_kernels/tests/cpp/CMakeLists.txt`,
`liger_cute_kernels/CMakeLists.txt`.

## Environment (verified at start)

| Component | Value |
|-----------|-------|
| GPU | **NVIDIA B200**, compute capability **10.0 (SM100)**, driver 580.105.08 |
| Toolchain | `nvcc` 12.9.86, CMake 3.30.3, Ninja, GCC 13.2 |
| CUTLASS | **4.4.1** header‑only at `/usr/local/include/cutlass` |
| gtest | installed at `/usr/local` |
| NVSHMEM | present, but **`tvm-ffi-config` absent** |

Key implication: the code had **never been compiled for SM100** — the `Compute=100`
body is gated behind `#if __CUDA_ARCH__ >= 1000`, and the existing build targeted
`sm_90a`, so the Blackwell path had zero prior compile/run coverage. Every bug below
was latent.

## What was done (in order)

### 1. A tests‑only `sm_100a` CMake path
The top‑level `CMakeLists.txt` hard‑coded `sm_90a`, always ran
`find_package(NVSHMEM)` + `find_program(tvm-ffi-config REQUIRED)`, and always built
the heavy torch‑free core (`moe.cu`, `moe_bwd.cu`, …). None of that is needed to
compile the header‑only MLP1 kernels, and `tvm-ffi-config` isn't even installed.

Added two knobs (defaults preserve the original full build byte‑for‑byte):
- `LIGER_CUTE_CUDA_ARCH` (default `90a`) — drives the `-gencode` string, so
  `-DLIGER_CUTE_CUDA_ARCH=100a` produces `compute_100a,sm_100a`.
- `LIGER_CUTE_TESTS_ONLY` (default `OFF`) — when `ON`, skips NVSHMEM, TVM‑FFI, the
  core library and the bindings; keeps only CUDAToolkit + CUTLASS + gtest +
  `tests/cpp`.

Configure line used throughout:
```
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON \
      -DLIGER_CUTE_CUDA_ARCH=100a
```

### 2. UMMA mainloop‑pipeline helper
Added `mlp1_make_pipe_umma<Traits>()` in `mlp1_fused.cuh` (mirrors the existing
Hopper `mlp1_make_pipe`) that builds `PipelineTmaUmmaAsync` with `num_consumers=1`
(one UMMA‑gated buffer release per stage).

### 3. Test rewired to drive the Blackwell path
`test_mlp1_fused.cu`:
- launcher smem embeds `Traits::MainloopPipelineUmma::SharedStorage`;
- launchers build the pipe via `mlp1_make_pipe_umma` and call
  `mlp1_fused_consumer<Traits, 100>` / `mlp1_fused_act_consumer<Traits, 100>`;
- `hopper_available()` (major==9) → `blackwell_available()` (major==10) gate.

### 4. Kernel bug fixes (see Blockers)

## Blockers and workarounds

The bring‑up split into a **compile phase** (26 → 0 errors) and a **runtime phase**
(one deadlock). Each blocker below is a genuine latent bug in the never‑compiled
`Compute=100` epilogue, plus one build‑infrastructure blocker.

### B0 — Full build can't even configure here
`tvm-ffi-config` is absent and the core glob compiles the heavy `moe*.cu`.
**Workaround:** the `LIGER_CUTE_TESTS_ONLY` path above (which is also exactly
"compile just the MLP1 code").

### B1 — TMEM→register load atom hard‑coded to the `1x` variant
`make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, …)` failed
`static_assert … RegNumDst`. The atom's per‑thread register count must equal the
epilogue chunk width.
**Fix:** a compile‑time `TmemLoadOp<EpiChunkN>` selector
(`EpiChunkN=64 → 64x`, `32 → 32x`, …), matching CUTLASS's
`TMEM::op_repeater<…, EpiChunkN*sizeof_bits(acc)>()`.

### B2 — `make_tmem_copy` fed a nested rank‑1 tile
The epilogue used `zipped_divide(acc, epi_tile)(_, _0{})`, whose tile mode is a
**nested rank‑1** `((TileM,EpiChunkN))` → fails
`"AtomTVLayout does not exist in the DataLayout"`.
**Fix:** switch both epilogues to `flat_divide(...)` and pass
`(_, _, _0{}, _0{})` — a flat rank‑2 `(M,N)` tile — exactly what CUTLASS's SM100
epilogue collective does.

### B3 — Raw MMA C‑fragment fed to `flat_divide`
`flat_divide(tCtAccU, …)` produced a 5‑mode tensor because `tCtAccU` is the raw
UMMA accumulator `(MMA, MMA_M, MMA_N)`, not a flat `(M,N)`.
**Fix:** extract the `(M,N)` view first with
`tCtAcc(make_coord(_,_), _0{}, _0{})` (CUTLASS's `accumulators(make_coord(_,_),…)`
pattern), *then* `flat_divide`.

### B4 — Register fragment sized from the wrong partition
After B1–B3 only one compile error remained: `size(rD) == RegNumDst` (2048 ≠ 64).
The `src` (TMEM) assert **passed**; only the destination register tensor was wrong.
The regs were sized from `partition_S` of the TMEM source — which is the
**warp‑collective** `(64 cols × 32 lanes) = 2048` view — instead of the per‑thread
`partition_D` shape (`64`). The `TMEM_LOAD` atom distributes the 32 datapath lanes
into per‑thread registers internally.
**Fix:** allocate `tTR_rU/rV = make_tensor<float>(shape(tTR_cChunk))` where
`tTR_cChunk` is the `partition_D` result — matching CUTLASS's
`make_tensor<ElementAccumulator>(shape(tTR_sD))`.

➡ At this point the test **compiled, linked, and launched** for `sm_100a`.

### B5 — Runtime deadlock (100% GPU spin) — the real one
The kernel launched but hung; `compute-sanitizer --tool synccheck` reported **no**
barrier‑count error → a *logical* deadlock, not a NamedBarrier mismatch.

**How it was found:** added `#ifdef MLP1_DBG` `printf` breadcrumbs at every
producer/consumer handshake, rebuilt with `-DMLP1_DBG`, and ran under a 40 s
timeout. The trace was decisive:
- the **producer** completed fully (`[prod] ALL DONE`);
- **7 of 8** consumer warps reached the setup `NamedBarrier(256,0)`;
- **warp 4 (the MMA warp) never printed "tmem allocate DONE"** — it was stuck
  *inside* `tmem_alloc.allocate()`, so the other 7 waited on the barrier forever.

**Root cause:** `tcgen05.alloc` is a **warp‑synchronous** instruction
(`.sync.aligned`; CUTLASS doc: *"Must be issued by a single fully active warp"*),
but the kernel guarded it with `if (is_mma_warp && cute::elect_one_sync())` — a
**single thread**. A `.sync.aligned` op issued by one divergent lane hangs the warp.
The matching `free()` / `release_allocation_lock()` (also `tcgen05.…sync.aligned`)
had the same guard.

**Fix:** issue from the whole MMA warp, matching CUTLASS
(`else if (is_participant.mma) { tmem_allocator.allocate(...); __syncwarp(); }`):
```cpp
if (is_mma_warp) {
    tmem_alloc.allocate(2 * TileN, &smem.tmem_base);
    __syncwarp();
}
...
if (is_mma_warp) {                 // end of consumer
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(tmem_base, 2 * TileN);
}
```
Applied to both `mlp1_fused.cuh` and `mlp1_fused_act.cuh`. All `MLP1_DBG` scaffolding
was then removed.

## How debugging was run

- All compiles/tests ran **detached in `tmux`** (session survived several terminal
  drops), writing to a logfile + a `STATUS` file so progress was pollable.
- Ground truth for every "how does CUTLASS do X" question came from reading the
  installed CUTLASS 4.4.1 headers (`copy_traits_sm100.hpp`, `sm100_pipeline.hpp`,
  `tmem_allocator_sm100.hpp`, the `sm100_epilogue_*` collectives, and
  `sm100_gemm_array_tma_warpspecialized.hpp`).
- The runtime deadlock was localized with guarded `printf` breadcrumbs + a hard
  `timeout`, after `synccheck` ruled out a barrier‑count bug.

## Result

Clean (non‑debug) `sm_100a` build, both gtests **PASS** on the B200:

```
[ RUN      ] Mlp1Fused.Correctness
[fused T=128 H=256 I=128 E=1] mean_rel=0.111% max_rel=0.382% max_abs=6.06
[fused T=128 H=512 I=256 E=1] mean_rel=0.101% max_rel=0.738% max_abs=7.99
[fused T=256 H=256 I=256 E=2] mean_rel=0.110% max_rel=0.854% max_abs=7.87
[fused T=384 H=256 I=128 E=3] mean_rel=0.111% max_rel=0.496% max_abs=6.6
[       OK ] Mlp1Fused.Correctness (794 ms)
[ RUN      ] Mlp1FusedAct.Correctness
[act   T=128 H=256 I=128 E=1] U' 0.110% / V' 0.106% / Z 0.111%  (max 0.39/0.39/0.45%)
[act   T=128 H=512 I=256 E=1] U' 0.100% / V' 0.095% / Z 0.100%  (max 1.06/0.39/1.39%)
[act   T=256 H=256 I=256 E=2] U' 0.110% / V' 0.105% / Z 0.110%  (max 0.56/0.39/0.56%)
[act   T=384 H=256 I=128 E=3] U' 0.110% / V' 0.105% / Z 0.110%  (max 0.46/0.39/0.39%)
[       OK ] Mlp1FusedAct.Correctness (45 ms)
[  PASSED  ] 2 tests.
```

All error metrics are well within tolerance (mean_rel < 1 %, max_rel < 5 %).

### Reproduce
```
cmake -S liger_cute_kernels -B liger_cute_kernels/build100a -G Ninja \
      -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=100a
cmake --build liger_cute_kernels/build100a --target test_mlp1_fused -j
./liger_cute_kernels/build100a/tests/cpp/test_mlp1_fused
```

### Changed files
| File | Change |
|------|--------|
| `liger_cute_kernels/CMakeLists.txt` | `LIGER_CUTE_CUDA_ARCH` + `LIGER_CUTE_TESTS_ONLY` |
| `…/moe/mlp1_fused.cuh` | UMMA pipe helper, `TmemLoadOp` selector, `flat_divide`+`(M,N)` extract, `partition_D` regs, whole‑warp `tcgen05.alloc`/`free` |
| `…/moe/mlp1_fused_act.cuh` | same TMEM‑copy + whole‑warp alloc fixes |
| `tests/cpp/test_mlp1_fused.cu` | drive `Compute=100`/UMMA, SM100 gate |

### Takeaways
- The MLP1 Blackwell rewrite was structurally sound but had **four** latent
  compile bugs and **one** latent runtime deadlock — all invisible until it was
  actually built for `sm_100a` and run on Blackwell hardware.
- The single highest‑impact bug was the **warp‑synchronous `tcgen05.alloc` issued
  from one thread**: a classic SM100 footgun that only manifests as a silent hang.
- CUTLASS's own SM100 epilogue collectives are the authoritative reference for the
  TMEM↔register copy plumbing (`flat_divide`, `make_coord(_,_)` accumulator view,
  `op_repeater` atom selection, `partition_D`‑sized register fragments).

## Performance (TFLOPS)

Beyond correctness, the fused consumers were benchmarked for **throughput**. FLOPs
are counted manually as the two GEMMs `U = X·Bᵀ` and `V = X·Cᵀ` (each `2·T·I·H`,
contracting over `H`); the SiLU / elementwise epilogue is ignored (negligible), so

```
TFLOPS = 4·T·H·I / kernel_seconds
```

Kernel time is the **median of 50 CUDA‑event‑timed launches** (10 warm‑up), at
large, GPU‑saturating shapes (`H=I=4096`, `E=8`, `T` a multiple of `TileM` so the
FLOP count is exact — no token padding). The grid is **N‑split** (`grid.y =
num_splits`, derived from the SM count) so small M‑tile counts still fill the
device: even `T=16384` is only 128 M‑tiles vs the B200's 148 SMs, so the extra
N‑parallelism is needed to saturate. The benchmark is **opt‑in** via `MLP1_BENCH=1`
— the default correctness run skips it and stays fast, and the numeric tests are
untouched.

### B200 (sm_100a / UMMA) — NVIDIA B200, 148 SMs

| Shape (T×H×I, E=8) | Fused (ms / TFLOPS) | Fused+Act (ms / TFLOPS) |
|--------------------|---------------------|-------------------------|
| 2048×4096×4096     | 0.146 / **943**     | 0.180 / **762**         |
| 4096×4096×4096     | 0.279 / **986**     | 0.356 / **772**         |
| 8192×4096×4096     | 0.564 / **975**     | 0.717 / **767**         |
| 16384×4096×4096    | 1.089 / **1010**    | 1.316 / **835**         |

Fused sustains **~0.94–1.01 PFLOPS** bf16 (~42–45 % of the B200's ~2.25 PFLOPS
dense‑bf16 peak). The act variant — which additionally stores `U'`/`V'`/`Z` and
computes SiLU plus its derivative — sustains **~0.76–0.84 PFLOPS**.

### H100 (sm_90a / WGMMA) — run pending on Hopper hardware

The **same binary** produces the Hopper numbers; it is arch‑gated
(`hopper_available()`), so on an H100 the Blackwell cases skip and these run
instead. To fill this column:

```
cmake -S liger_cute_kernels -B build90a -G Ninja \
  -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON -DLIGER_CUTE_CUDA_ARCH=90a
cmake --build build90a --target test_mlp1_fused -j
MLP1_BENCH=1 ./build90a/tests/cpp/test_mlp1_fused --gtest_filter='*TFLOPs_Hopper*'
```

| Shape (T×H×I, E=8) | Fused (ms / TFLOPS) | Fused+Act (ms / TFLOPS) |
|--------------------|---------------------|-------------------------|
| 2048×4096×4096     | _tbd_               | _tbd_                   |
| 4096×4096×4096     | _tbd_               | _tbd_                   |
| 8192×4096×4096     | _tbd_               | _tbd_                   |
| 16384×4096×4096    | _tbd_               | _tbd_                   |

> These are **single‑tile MLP1 GEMM‑throughput** numbers (the fused U/V
> projections this PR brought up), not a full‑MoE end‑to‑end figure — measured at
> shapes that saturate the device. The `%peak` is vs the vendor dense‑bf16 rating
> (B200 ≈ 2.25 PFLOPS, H100 ≈ 0.99 PFLOPS).
