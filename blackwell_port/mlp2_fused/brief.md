# Brief — MLP2-fused Blackwell (SM100a) port

> **Op:** `Y = Z · Aᵀ` — the MoE **down-projection** (single-tile fused consumer).
> **Files:** `csrc/core/src/moe/mlp2_fused.cuh` → new `tests/cpp/test_mlp2_fused.cu`.
> Read [`../README.md`](../README.md) first — the shared recipe, FLOP conventions,
> and parallel-isolation rules live there; this brief only covers what is
> **specific** to mlp2_fused.

## What it computes

Per expert, with hidden `H`, intermediate `I`, tokens `T`:

- `Z` : `[T, I]` — the phase-1 activation (`SiLU(B·X)·(C·X)`).
- `A` : `[H, I]` per-expert down weight; consumed as `Aᵀ` → `[I, H]`.
- `Y = Z · Aᵀ` : `[T, H]`. GEMM contracts over **I** (the K axis); N axis = **H**.

FLOPs (one GEMM): `2·T·H·I` → `TFLOPS = 2·T·H·I / s`.

## Why this is the easiest of the three (do it first / as the template)

`mlp2_fused.cuh` is the closest structural sibling to `mlp1_fused.cuh`, minus the
complexity:

- **One accumulator per WG** (MLP1 had two: `acc_B`/`acc_C`). One TMEM region, one
  TMEM→register epilogue path.
- **Single fused Z+W TMA pipe** (2 copies), same as MLP1's producer shape.
- **No epilogue activation fusion** — the epilogue is just cast-to-bf16 + TMA store
  (MLP1 additionally computed `SiLU(U)·V`). So the per-thread epilogue math the
  UMMA path must preserve is trivial.
- `EpiChunkN` default **32** → `TmemLoadOp<32> = SM100_TMEM_LOAD_32dp32b32x`.
- Cooperative 2-WG consumer, `TileM=128 → M-split` / `TileM=64 → N-split`; identical
  producer/consumer warp layout (WG0 w0 producer, WG1/WG2 consumers).

Treat this port as the **reference instantiation** of the shared recipe — get it
green, then mlp2 and mlp5 are deltas on top.

## Current state

`sm_90a` / WGMMA only. `Traits::TiledMma = TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>`,
accumulator via `partition_fragment_C` (registers), `cute::gemm` over
`SmemLayoutAtom = GMMA::Layout_K_SW128_Atom` operands. Zero SM100 markers.

## Anticipated solution (recipe → mlp2_fused)

Apply the 6-step header recipe from the README. Concretely:

1. Add the TMEM include + `TmemLoadOpSelector` (self-contained file → define locally,
   as MLP1 does).
2. `Mlp2Traits`: add `MainloopPipelineUmma` and an SM100 UMMA `TiledMMA` whose
   accumulator tile is `(AtomTileM=64, WgTileN)` in TMEM. Operand-A = `Z`
   (`Layout_K_SW128`), operand-B = `A` (`Layout_K_SW128`) — **both K-major**, which
   is the SM100-native operand orientation, so no transpose subtlety here (contrast
   mlp2/mlp2_t).
3. `mlp2_make_pipe_umma` (`num_consumers=1`).
4. Split `mlp2_fused_consumer` into `Impl<90>` (verbatim) / `Impl<100>`.
5. `Impl<100>`: whole-warp `tmem_alloc.allocate(WgTileN, &tmem_base)` (**one** acc →
   `WgTileN` cols, *not* `2·TileN`), UMMA `gemm` over the k-pipe, then the
   single-acc epilogue: `(M,N)` extract → `flat_divide(epi_tile)` → `TmemLoadOp<32>`
   copy into a `partition_D`-sized fragment → cast → the existing per-WG
   `store_buf` → TMA store to `Y`.

## Anticipated blockers

Mostly the standard MLP1 set, at lower risk because the epilogue is single-acc and
activation-free:

- **B1 (TMEM-load atom width):** `EpiChunkN=32` → must select `…32x`; the generic
  `TmemLoadOpSelector` handles it.
- **B2/B3 (epi tile shape):** use `flat_divide` on the `(M,N)`-extracted acc view,
  not `zipped_divide` on the raw MMA C-fragment.
- **B4 (fragment sizing):** size regs from `partition_D`.
- **B5 (warp-sync `tcgen05.alloc`):** issue alloc/free from the whole MMA warp.
- **Low kernel-specific risk.** The only thing to watch is the **TMEM column budget**
  (one acc = `WgTileN` cols; ensure `2·WgTileN` isn't over-allocated by a copy-paste
  from MLP1) and the **smem cap** — but with one acc and one store_buf, mlp2_fused is
  the *least* smem-pressured of the three, so occupancy should not force a
  stage-count reduction.

## Definition of success

Per the README's shared "successful port" gate: clean `sm_100a` build,
`test_mlp2_fused` **Correctness** PASS (`mean_rel<1%`, `max_rel<5%`), a
`MLP2_BENCH=1` TFLOPS_Blackwell table, Hopper path uncracked, results in
`blackwell_port/mlp2_fused/writeup.md`. Execution steps: [`plan.md`](plan.md).
