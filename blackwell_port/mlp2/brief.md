# Brief — MLP2 (transpose, `mlp2_t`) Blackwell (SM100a) port

> **Op:** `Y = Z · A` — MLP phase-2 **transpose** variant (weight consumed
> **MN-major**, not transposed to K-major).
> **Files:** Traits + non-fused consumer in `csrc/core/src/moe/mlp2_t.cuh`;
> the **fused consumer to port** is `mlp2_t_fused.cuh` (`#include`s `mlp2_t.cuh`,
> reuses `Mlp2TTraits`). New test: `tests/cpp/test_mlp2_t_fused.cu`.
> Read [`../README.md`](../README.md) first for the shared recipe + isolation rules.

## Naming note (confirm before starting)

There is **no** plain `mlp2.cuh`. "MLP2" maps to the transpose kernel
**`mlp2_t`** (`Y = Z·A`), the sibling of `mlp2_fused` (`Y = Z·Aᵀ`). Both are real,
different GEMMs; this brief covers `mlp2_t`. If the intended target was actually the
down-projection, that is [`../mlp2_fused/`](../mlp2_fused/brief.md).

## What it computes

- `Z` : `[T, I]` activation. `A` : `[I, H]` weight, consumed **as stored** (no
  transpose) so the contraction axis I is the weight's **minor/MN-swizzled** axis.
- `Y = Z · A` : `[T, H]`. Contracts over **I** (K axis); N axis = **H**.
- FLOPs (one GEMM): `2·T·H·I` → `TFLOPS = 2·T·H·I / s` (same as mlp2_fused).

Structurally near-identical to `mlp2_fused` — **one** accumulator per WG,
`EpiChunkN=32`, single fused Z+W TMA pipe, cooperative 2-WG consumer. The **only**
material difference, and the entire reason this is its own brief, is the **operand
layout**.

## The transpose-specific delta (the crux)

From `mlp2_t.cuh`:

```
using SmemLayoutAtomZ = GMMA::Layout_K_SW128_Atom<Element>;   // Z: K-major
using SmemLayoutAtomW = GMMA::Layout_MN_SW128_Atom<Element>;  // A: MN-major  ← transpose
```

vs `mlp2_fused`, where **both** operands are `Layout_K_SW128_Atom`. On Hopper WGMMA
the MN-major operand-B is expressed through the GMMA descriptor's major-mode field
and "just works" via `cute::gemm`. On **SM100 UMMA (`tcgen05.mma`)** the operand
descriptor is a *different* encoding (`UMMA::SmemDescriptor` with its own
leading-dim-byte-offset / major-mode / swizzle fields). The risk is entirely: **does
the MN-major A smem layout produce a correct UMMA operand-B descriptor when fed to
the tcgen05 MMA atom?**

## Current state

`sm_90a` / WGMMA only, zero SM100 markers. `Mlp2TTraits::TiledMma =
TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>`; register accumulator via
`partition_fragment_C`; MN-major W smem via the `Step<_2,_1>` `tile_to_shape` trick
(see the comment at `mlp2_t.cuh:116`).

## Anticipated solution (recipe → mlp2_t)

Same 6-step recipe as mlp2_fused (start from that port if it's already green — the
producer, pipe, epilogue, and single-acc TMEM handling are line-for-line reusable),
**plus** the operand-layout work:

1–5. Identical to mlp2_fused: TMEM include + `TmemLoadOpSelector`;
   `MainloopPipelineUmma` alias + `mlp2_t_make_pipe_umma` (`num_consumers=1`); split
   `mlp2_t_fused_consumer` → `Impl<90>`/`Impl<100>`; single-acc epilogue
   (`(M,N)` extract → `flat_divide` → `TmemLoadOp<32>` → `partition_D` regs → cast →
   store). Add the UMMA `TiledMMA` + pipe alias to `Mlp2TTraits` (in `mlp2_t.cuh`)
   so the `#include`ing fused file sees them.
6. **Operand-B (weight) descriptor:** build the SM100 `TiledMMA` so operand B takes
   the **MN-major** smem layout. Concretely:
   - Select the tcgen05 MMA atom variant whose B operand is MN-major (the CUTLASS
     SM100 MMA atoms carry `UMMA::Major::MN` / `Major::K` template args — pick `MN`
     for B, `K` for A), **or** confirm the atom's descriptor builder reads the
     major mode from the smem layout's stride and needs no change.
   - Keep `SmemLayoutAtomW = Layout_MN_SW128_Atom` and its `Step<_2,_1>`
     `tile_to_shape`; verify the resulting smem tensor's strides still satisfy the
     UMMA descriptor's leading-dim/offset constraints (128-elem swizzle → the same
     SW128 atom UMMA supports).
   - Validate against the installed CUTLASS 4.4.1 SM100 MMA/descriptor headers
     (`cute/atom/mma_traits_sm100.hpp`, `cute/arch/mma_sm100_desc.hpp`) rather than
     guessing the major-mode encoding.

## Anticipated blockers

- **Standard B1–B5** exactly as in mlp2_fused (single-acc, so low).
- **Kernel-unique — MN-major UMMA operand (primary risk):** a wrong operand-B
  major-mode yields a *silent numeric error*, not a compile error — the epilogue
  plumbing is correct but `Y` is wrong/transposed. **Detection:** the correctness
  test's `max_rel` blows up (not a hang, not a crash). **Debug tactic:** first
  bring up a *tiny* single-tile shape (`T=I=H=TileM`) and diff element-by-element vs
  the CPU reference to distinguish "transposed operand" (structured error pattern)
  from "swizzle/descriptor offset" (block-structured error). This is why mlp2_t is
  ranked harder than mlp2_fused despite identical epilogue code.
- **Low smem risk** (one acc, like mlp2_fused) — occupancy shouldn't force stage
  reduction.

## Definition of success

Per the README gate: clean `sm_100a` build, `test_mlp2_t_fused` **Correctness** PASS
(`mean_rel<1%`, `max_rel<5%` — the key signal the MN-major operand is right), a
`MLP2T_BENCH=1` TFLOPS_Blackwell table, Hopper path intact, results in
`blackwell_port/mlp2/writeup.md`. Steps: [`plan.md`](plan.md).
