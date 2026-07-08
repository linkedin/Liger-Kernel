# Brief — MLP3 (down-weight gradient) Blackwell (SM100a) port

> **Op:** `dA = dYᵀ · Z` — the MoE **down-weight gradient** (backward). Cooperative
> 2-WG consumer, **`SM90_TMA_REDUCE_ADD`** output (hardware atomic-add into gmem).
> **File:** `csrc/core/src/moe/mlp3.cuh` (single file — Traits, pipe helper, producer,
> consumer, and `mlp3_fwd` launcher are all inline). New test: `tests/cpp/test_mlp3.cu`.
> Read [`../README.md`](../README.md) first — the shared 6-step recipe, FLOP
> conventions, register/spill rules, and isolation live there; this brief covers only
> what is **specific** to mlp3. **mlp3 is the reference port for mlp4.**

## What it computes

Per expert, hidden `H`, intermediate `I`, tokens `T`; the contraction is over **T**:

- `dY` : `[T, H]` — output-grad, consumed transposed as `dYᵀ → [H, T]` (A operand,
  `M=H`, `K=T`).
- `Z`  : `[T, I]` — phase-1 activation (B operand, `N=I`, `K=T`).
- `dA = dYᵀ · Z` : `[H, I]` per expert; the global tensor is `dA:[E,H,I]`.

FLOPs (**one** GEMM): `2·T·H·I` → `TFLOPS = 2·T·H·I / s`.

## Current state (`sm_90a` / WGMMA only — 0 SM100 markers)

`Mlp3Traits`: `TiledMma = TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>` with **both** operands
**MN-major** (`SmemLayoutAtom = GMMA::Layout_MN_SW128_Atom`, `Step<_2,_1>`). Accumulator
in **registers** via `partition_fragment_C`. Cooperative 2-WG: only
`(TileM,TileN) ∈ {(256,128) M-split, (128,256) N-split (default)}`; per-WG footprint
`(WgTileM,WgTileN)=(128,128)`, `AtomTileM=64`. Single fused DYT+Z TMA pipe. `EpiChunkN=64`
→ `NumEpiRounds = WgTileN/EpiChunkN = 2` rounds/WG. CTA = 384 threads (12 warps): warp 0
producer, warps 4–11 the two consumer WGs.

**Grid is chunk-fixed / persistent:** `blockIdx.x = cell_start`, `gridDim.x = cell_stride`;
each CTA walks **many** `(chunk, lane, k_slice)` cells in a `for (cell_idx …)` loop, holding
the shared cooperative operand chunk-constant to maximize L2 reuse.

**Epilogue (the mlp3/mlp4-specific bit):** each WG scatters its **register** `acc` into a
`store_buf` slot `(WgTileM × EpiChunkN)` — using identity coords `tCcC(i)` with a
column-chunk guard `chunk_start ≤ n_local < chunk_start+EpiChunkN` and, for M-split, the
atom-interleave remap `m_local = (m_loc/128)*64 + (m_loc%64)` (compensates CUTE's 2-WG
M-atom interleave). The WG leader then issues `SM90_TMA_REDUCE_ADD`
(`copy(tma_reduce_da, partition_S(sStore_a), partition_D(gdA))`), one store per owned atom
row, so hardware **atomically adds** the partial into gmem. **`dA` must be zero-initialized
by the caller.**

## Two things that differ from the batch-1 (mlp1/2/5) STORE ports

1. **The store path is REDUCE_ADD, not STORE — and it is arch-agnostic.**
   `SM90_TMA_REDUCE_ADD` (`cp.reduce.async.bulk.tensor…add.bulk_group`) is guarded only
   by a plain `__CUDA_ARCH__` in CUTLASS 4.4.1, so it compiles and runs under `sm_100a`
   unchanged. **Keep the entire `is_my_wg_leader` reduce-add block (the `tma_store_fence`
   → `copy(tma_reduce_da,…)` → `tma_store_arrive` loop, including the `m_atom_row` /
   `n_tile_idx` index math) byte-for-byte.** Only the step that *fills* `store_buf` changes.

2. **`store_buf` is filled by a hand-rolled scatter from the register `acc`.** On SM100
   the accumulator lives in **TMEM**, so this fill must pull from TMEM. Mirror mlp1/mlp5's
   epilogue: extract the `(M,N)` acc view (`tCtAcc(make_coord(_,_),_0{},_0{})`, **B3**),
   `flat_divide` by the epi tile `(WgTileM, EpiChunkN)` (**B2** — not `zipped_divide`),
   `TmemLoadOp<EpiChunkN=64>` each round's chunk into a `partition_D`-sized register
   fragment (**B1/B4**), cast, and write into `sStore` at the **same `(m_local, chunk_n)`
   positions** the SM90 scatter used. The reduce-add store then reads `sStore` unchanged.
   - **Correctness pin (the single most likely place to get a silently-wrong result):**
     the SM90 M-split remap `m_local=(m_loc/128)*64+(m_loc%64)` compensates for CUTE's
     2-WG M-atom interleave in the **WGMMA** C-partition. The UMMA TMEM-load fragment has a
     **different** thread-value layout, so re-derive the `(m_local, n_local)` mapping from
     the **TMEM-load partition's** identity coords (`partition_D` of
     `make_identity_tensor((WgTileM,EpiChunkN))`), **not** from the old `tCcC`.
     Sanity-check against the SM90 result element-by-element on a small shape.

## Anticipated solution (recipe → mlp3)

Apply the shared 6-step header recipe. Concretely:

1. `#include <cute/arch/tmem_allocator_sm100.hpp>` + a local `TmemLoadOpSelector` /
   `TmemLoadOp<EpiChunkN>` (copy from `mlp5.cuh` ~L84–91; `EpiChunkN=64 →
   SM100_TMEM_LOAD_32dp32b64x`).
2. `Mlp3Traits`: add `MainloopPipelineUmma = cutlass::PipelineTmaUmmaAsync<Stages>` and an
   SM100 UMMA `TiledMmaUmma = make_tiled_mma(SM100_MMA_F16BF16_SS<Element,Element,float,
   TileM,TileN, UMMA::Major::MN, UMMA::Major::MN>{})` — **both MN-major** (confirm against
   `SmemLayoutDYT`/`SmemLayoutZ`; use `_SS`, mirror `mlp2_t.cuh`). Accumulator tile
   `(AtomTileM?, WgTileN)` in TMEM — follow mlp5's cooperative TMEM sizing.
3. `mlp3_make_pipe_umma` mirroring `mlp3_make_pipe` with `num_consumers = 1`.
4. Split `mlp3_consumer` → `Mlp3ConsumerImpl<90>` (existing WGMMA body, **verbatim**) /
   `<100>` (UMMA) + a free-function forwarder `mlp3_consumer<Traits,Compute>`. The producer
   `mlp3_producer` and the `mlp3_fwd` host launcher stay arch-agnostic.
5. `Impl<100>`: **whole-MMA-warp** `Allocator1Sm.allocate(<cols>,&tmem_base)` +
   `__syncwarp()` **once per CTA** (before the `cell_idx` loop — persistent grid, see
   blockers), matching `release_allocation_lock()`+`free()` at CTA exit; UMMA `gemm` over
   the k-pipe with `ScaleOut::Zero` on the first k-step of each `(m_tile,n_tile)` walk and
   `One` after (the `clear(acc)` equivalent); then the TMEM→`store_buf` fill from §2.2.
6. `Impl<90>` byte-for-byte unchanged.

## Anticipated blockers

- **B5′ (persistent-grid TMEM alloc) — HIGH.** mlp3's grid is chunk-fixed; each CTA walks
  many cells. Per the mlp5 finding, issuing `tcgen05.alloc`/`free` per-cell traps
  (`phase_invalid_during_alloc`). **Hoist alloc/free to once-per-CTA** (consumer entry /
  exit), reusing one TMEM region across all cells and epi rounds.
- **B5 (warp-sync alloc):** issue `tcgen05.alloc`/`free` from the whole MMA warp
  (`.sync.aligned`), never a single `elect_one` lane (silent hang).
- **Store-buf mapping mismatch — HIGH (correctness).** See the "Correctness pin" in §2.2.
- **B1–B4 (epilogue):** TMEM-load atom width = `EpiChunkN=64` (`TmemLoadOp<64>` = widest,
  most spill-prone); `flat_divide` not `zipped_divide`; extract `(M,N)` view first; size the
  register fragment from `partition_D`.
- **MN-major operand major modes:** if `_SS<MN,MN>` mis-selects, the UMMA descriptor asserts
  or produces garbage — cross-check the smem layouts, mirror `mlp2_t.cuh`.
- **smem cap:** adding the UMMA pipe + TMEM landing slot should stay < 227 KiB at Stages=4
  (mlp5 didn't need a `K_PIPE` cut); verify with `-Xptxas -v` smem line, reduce `K_PIPE` if
  `cudaFuncAttributeMaxDynamicSharedMemorySize` fails.

## Definition of success

Per the README gate: clean `sm_100a` build of `test_mlp3`; `Mlp3.Correctness` PASS
(`mean_rel<1%`, `max_rel<5%`, output zero-init'd and **re-zeroed between correctness and
bench** — REDUCE_ADD accumulates); a `MLP3_BENCH=1 …TFLOPs_Blackwell` table (`2·T·H·I`,
per shape + winning split); Hopper `sm_90a` path still compiles; a **register-spill
checker** that builds the `Compute=100` consumer with the real `build100a` ptxas flags and
asserts **0** spill stores/loads; and `blackwell_port/mlp3/writeup.md` + root
`writeup_mlp3.md`. Steps: [`plan.md`](plan.md).
