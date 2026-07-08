# Blackwell (SM100a) bring-up — MLP2, MLP2-fused, MLP5

This directory holds the **design briefs** and **execution plans** for porting three
more MoE consumer kernels to Blackwell (`sm_100a` / UMMA / tcgen05), following the
template established by the MLP1 port (see the repo-root `writeup.md`, commit
`b46869a` and PR #11).

Each kernel is meant to be executed by an **independent subagent, in parallel**.
This README is the shared contract: the common port recipe, the FLOP/shape
conventions, and — most importantly — the **isolation rules** that keep three
concurrent subagents from clobbering each other.

## Scope & file mapping

There is no plain `mlp2.cuh`; the MLP2 family contains two *different* GEMMs. The
three targets are:

| Brief | Op | Primary kernel file(s) to edit | New test file |
|-------|----|--------------------------------|---------------|
| [`mlp2/`](mlp2/brief.md) | `Y = Z · A` (transpose; MN-major A view) | `csrc/core/src/moe/mlp2_t.cuh` (Traits: add UMMA MMA + pipe) + `mlp2_t_fused.cuh` (fused consumer to split 90/100) | `tests/cpp/test_mlp2_t_fused.cu` |
| [`mlp2_fused/`](mlp2_fused/brief.md) | `Y = Z · Aᵀ` (down-projection) | `csrc/core/src/moe/mlp2_fused.cuh` | `tests/cpp/test_mlp2_fused.cu` |
| [`mlp5/`](mlp5/brief.md) | `dX = dU·B + dV·C` (backward input grad) | `csrc/core/src/moe/mlp5.cuh` (consumer body) + `mlp5_fused.cuh` (single-tile) | `tests/cpp/test_mlp5_fused.cu` |

All three are **`sm_90a`/WGMMA-only today** — `grep` finds *zero* SM100 markers in
any of them (MLP1 has 47). The Compute=90 body stays; we add a Compute=100 sibling.

## The port recipe (identical shape for every kernel)

Replicated verbatim from the MLP1 port. In the kernel header:

1. `#include <cute/arch/tmem_allocator_sm100.hpp>`.
2. **`TmemLoadOpSelector<EpiChunkN>`** → `SM100_TMEM_LOAD_32dp32b{8,16,32,64,128}x`
   atom whose per-thread `RegNumDst` equals the epilogue chunk width (MLP1 blocker
   **B1**).
3. In `Traits`: add a **UMMA `MainloopPipelineUmma`** alias
   (`PipelineTmaUmmaAsync`) next to the existing WGMMA `MainloopPipeline`, and an
   **SM100 UMMA `TiledMMA`** (tcgen05 atom, accumulator in **TMEM**).
4. **`mlpX_make_pipe_umma`** helper mirroring `mlpX_make_pipe` but with
   `num_consumers = 1` (one `umma_arrive` per stage releases the buffer).
5. Split the consumer into `Impl<90>` (existing WGMMA body, **kept verbatim**) and
   `Impl<100>` (new UMMA body), with a free-function forwarder
   `mlpX_..._consumer<Traits, Compute>` — mirrors `Mlp1FusedConsumerImpl`. The
   producer is arch-agnostic and stays shared.
6. In `Impl<100>`:
   - `cute::TMEM::Allocator1Sm`; **whole-MMA-warp** `allocate(<cols>, &tmem_base)`
     + `__syncwarp()`, and matching `release_allocation_lock()` + `free()` at the
     end. Issuing `tcgen05.alloc`/`free` from a single `elect_one` lane is the
     silent-hang footgun (MLP1 blocker **B5**).
   - Extract the `(M,N)` accumulator view first
     (`tCtAcc(make_coord(_,_), _0{}, _0{})`, blocker **B3**), then **`flat_divide`**
     (not `zipped_divide`, blocker **B2**) by the epi tile.
   - Size the TMEM→register fragment from **`partition_D`**, not `partition_S`
     (blocker **B4**): `make_tensor<float>(shape(tTR_cChunk))`.
   - Keep the *existing* per-thread epilogue math (cast / activation) and TMA store
     unchanged.

In a **new** `tests/cpp/test_mlpX_fused.cu` (clone `test_mlp1_fused.cu`):

- `MainloopPipelineFor<Traits,Compute>` alias; a launcher kernel that does
  `if constexpr (Compute==100)` → `_make_pipe_umma` + `consumer<…,100>`, else
  `_make_pipe` + `consumer<…,90>`, with `__trap()` on the 90 body when built for
  `sm_100a`.
- host `run<Compute>()` driver (build TMA descs, set
  `cudaFuncAttributeMaxDynamicSharedMemorySize`, launch, compare to a CPU
  reference; report `mean_rel` / `max_rel` / `max_abs`).
- `run_bench<Compute>()`: median of CUDA-event-timed launches, **N-split sweep over
  every divisor** of `num_n_tiles`, report peak TFLOPS + winning `grid.y`. Opt-in
  via the kernel's `*_BENCH` env var.
- `blackwell_available()` (major==10) / `hopper_available()` (major==9) gates,
  correctness + `TFLOPs_{Blackwell,Hopper}` TESTs, and an arch-aware `main()`.

## Register usage & spills (review on every port)

The `Impl<100>` epilogue pulls the accumulator out of TMEM into a register fragment
(`TmemLoadOp<EpiChunkN>` → `partition_D`-sized regs) — the easiest place to bloat
register pressure. An LLM doing the port tends to be **liberal** with registers
(redundant temporaries, keeping the WGMMA-era fragment live alongside the new UMMA
one, not reusing the `store_buf` staging). These consumers run at ~1 CTA/SM and are
**smem-bound**, so spills go straight to local memory and surface as a latency cliff
that extra occupancy can't hide. Check every port:

- Build the kernel with **`-Xptxas -v`** (`--ptxas-options=-v`) and read the
  per-thread **registers** and **spill stores / spill loads** lines. Any non-zero
  spill bytes on the hot consumer kernel is a red flag to fix, not ship.
- The **MLP1 ported consumer is the yardstick.** A faithful port (fragment sized
  from `partition_D`; the `(M,N)` acc view and the reg fragment not both kept live
  longer than needed; `store_buf` reused) should land at roughly MLP1's register
  count — so following MLP1 closely it should be more or less fine. If a port is
  materially higher, that's LLM register liberality, not an intrinsic need: tighten
  live ranges first, and only then reach for `__launch_bounds__` / `-maxrregcount`.
- `EpiChunkN=64` (mlp5) has the widest epilogue fragment and is the most
  spill-prone; `EpiChunkN=32` (mlp2_fused / mlp2_t) is lighter.

## FLOP & tolerance conventions

- TFLOPS counts the GEMM MACs only (activation epilogue ignored, negligible):
  - mlp2 / mlp2_fused (1 GEMM): `2·T·H·I / s`
  - mlp5 (2 GEMMs): `4·T·H·I / s`
- Bench at large, device-saturating shapes with `T` a multiple of `TileM` (exact
  FLOP count, no token padding), `H=I=4096`, `E=8`.
- Pass bar (matches MLP1): `mean_rel < 1%`, `max_rel < 5%` in bf16.
- `%peak` vs vendor dense-bf16 rating (B200 ≈ 2.25 PFLOPS, H100 ≈ 0.99 PFLOPS).

## Parallel-execution isolation (READ BEFORE LAUNCHING SUBAGENTS)

Three subagents share one worktree. To run conflict-free, each subagent touches
**only** its own files:

| Resource | Rule |
|----------|------|
| Kernel header(s) | Disjoint per kernel (`mlp2_t*.cuh` / `mlp2_fused.cuh` / `mlp5*.cuh`) — no overlap. |
| Test file | **New** per-kernel `test_mlpX_fused.cu` — never edit the shared `test_mlp1_fused.cu`. |
| Build dir | Own out-of-source dir: `build100a_mlp2` / `build100a_mlp2_fused` / `build100a_mlp5`. Ninja state is per-dir, so parallel builds don't race. |
| Results | Each writes its own `blackwell_port/<kernel>/writeup.md`. Do **not** append to the shared root `writeup.md` (serialize that consolidation afterwards). |
| **`tests/cpp/CMakeLists.txt`** | **The one shared file.** See below. |

### The shared `tests/cpp/CMakeLists.txt` hazard

Every kernel needs one `add_executable(test_mlpX_fused …)` + link/include + 
`gtest_discover_tests` stanza there. Three subagents editing it concurrently will
race. **Mitigation (orchestrator, before launch):** pre-stage all three target
stanzas in `tests/cpp/CMakeLists.txt` in a single commit, so each subagent only
*builds* its already-registered target and never edits the file. A ready-to-paste
block is in each kernel's `plan.md` (Step 0). Alternatively a subagent can compile
its test standalone with `nvcc` (fallback command in each plan) and skip CMake
entirely during iteration.

## Definition of "successful port" (all three)

1. Clean (non-debug) `sm_100a` build of the kernel's own test target.
2. `test_mlpX_fused` **Correctness** PASSes on the B200 (`mean_rel<1%`,
   `max_rel<5%`).
3. `*_BENCH=1 test_mlpX_fused --gtest_filter=*TFLOPs_Blackwell*` produces a peak
   TFLOPS table (per bench shape, with winning split).
4. The Hopper path still compiles for `sm_90a` and (where HW is available) its
   Correctness PASSes — i.e. the WGMMA body was not regressed.
5. Results captured in `blackwell_port/<kernel>/writeup.md`.
