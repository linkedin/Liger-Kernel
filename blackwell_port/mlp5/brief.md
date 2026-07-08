# Brief — MLP5 (backward input grad) Blackwell (SM100a) port

> **Op:** `dX = dU · B + dV · C` — MLP phase-5 **backward input gradient** (two
> GEMMs fused into one continuous k-loop, accumulated into **one** accumulator).
> **Files:** Traits + non-fused consumer in `csrc/core/src/moe/mlp5.cuh`; the
> **fused consumer to port** is `mlp5_fused.cuh` (`#include`s `mlp5.cuh`, reuses
> `Mlp5Traits`/`Mlp5Smem`/`mlp5_make_pipe`). New test: `tests/cpp/test_mlp5_fused.cu`.
> Read [`../README.md`](../README.md) first for the shared recipe + isolation rules.
> **This is the hardest of the three — do it last, after mlp2_fused (single-acc
> epilogue) and mlp2_t (MN-major operand) are both green, since mlp5 combines both.**

## What it computes

Per expert, hidden `H`, intermediate `I`, tokens `T`:

- `dU`, `dV` : `[T, I]` — upstream grads of the two phase-1 branches.
- `B`, `C`   : `[I, H]` per-expert weights, consumed **MN-major** (like mlp2_t).
- `dX = dU·B + dV·C` : `[T, H]`. Two GEMMs, both contract over **I** (K axis);
  N axis = **H**. Output is fully written (no zero-init).
- FLOPs (**two** GEMMs): `4·T·H·I` → `TFLOPS = 4·T·H·I / s`.

## Structure (from `mlp5.cuh` / `mlp5_fused.cuh`)

A single continuous `2·num_k_tiles` mainloop, **one accumulator** for the whole
thing (`mlp5.cuh:350` `clear(acc)` **once**, then both phases accumulate in):

```
clear(acc);
Phase 1 (k = 0 .. K-1)  : Z=dU, W=B → acc += dU·B     // mlp5.cuh:266, mlp5_fused.cuh:110
Phase 2 (k = K .. 2K-1) : Z=dV, W=C → acc += dV·C     // mlp5.cuh:279, mlp5_fused.cuh:123
```

Other salient traits: `EpiChunkN=64`; `SmemLayoutAtomW = Layout_MN_SW128_Atom`
(MN-major B & C); cooperative 2-WG M-split consumer; **2D grid**
`(num_sms/NSplit, NSplit)` with cross-CTA dU/dV **L2 multicast** between `(m,0)` and
`(m,1)`; single fused Z+W pipe (W slot reused: B in phase 1, C in phase 2).

## Anticipated solution (recipe → mlp5)

Same 6-step header recipe, but with **two** kernel-unique pieces layered on. Reuse
the mlp2_fused UMMA epilogue and the mlp2_t MN-major operand resolution wholesale;
mlp5 adds only the cross-phase accumulate and the smem/grid handling.

1–5. Standard: TMEM include + `TmemLoadOpSelector` (**`EpiChunkN=64` →
   `SM100_TMEM_LOAD_32dp32b64x`**); `MainloopPipelineUmma` alias +
   `mlp5_make_pipe_umma` (`num_consumers=1`) — add to `mlp5.cuh` Traits/helpers so
   the fused file sees them; split `mlp5_fused_consumer` → `Impl<90>` (verbatim) /
   `Impl<100>` (UMMA) + forwarder; single-acc epilogue (`(M,N)` extract →
   `flat_divide` → `TmemLoadOp<64>` → `partition_D` regs → existing cast+store).
   UMMA `TiledMMA` operand B = **MN-major** (reuse mlp2_t's atom/major-mode
   resolution — B and C are the same MN-major layout).
6. **Cross-phase accumulate (the crux):** the WGMMA version clears the register acc
   once and lets `cute::gemm` accumulate across all `2K` MMAs. On UMMA the
   accumulate is a **per-instruction bit** on `tcgen05.mma` (`UMMA::ScaleOut` /
   the MMA atom's `accumulate_` state): the **first** MMA of phase 1 must run
   *non-accumulating* (`ScaleOut::Zero` — writes acc), and **every** subsequent MMA
   — **including the phase-1→phase-2 boundary** — must be *accumulating*
   (`ScaleOut::One`). The failure mode to avoid: re-initializing / re-clearing the
   accumulate bit when phase 2 starts (which would drop the `dU·B` contribution).
   Concretely: set the atom accumulate flag false on the very first `k`, true
   thereafter, and **do not** reset it between the two phase loops.

Plus mechanics that are structural, not numeric:

- **smem budget:** `EpiChunkN=64` + store_buf makes mlp5 the largest consumer. On
  B200 (227 KB smem/SM) an SM100 fused consumer is smem-bound (~1 CTA/SM). If the
  UMMA build overflows `cudaFuncAttributeMaxDynamicSharedMemorySize`, **reduce
  `K_PIPE` stages** (fewer Z/W buffers) until it fits — the mainloop is already
  latency-tolerant at 1 CTA/SM.
- **2D-grid N-split launcher:** the host launcher must set
  `grid = (num_sms/NSplit, NSplit)` and the test's split sweep must iterate over
  `NSplit` (grid.y), which lines up with the existing bench's N-split sweep.

## Anticipated blockers

- **Standard B1–B5**, with `EpiChunkN=64` making the B1 atom-width selection the
  wider `…64x`.
- **Cross-phase UMMA accumulate (primary, mlp5-unique):** wrong accumulate-bit
  handling at the phase boundary → `dX` is missing one term (numerically
  `≈ dV·C` only, or `≈ dU·B` only). **Detection:** correctness `max_rel ≈ O(1)`
  and, tellingly, `dX ≈` exactly one of the two products. Compare partial sums to
  localize.
- **MN-major operand B/C (shared with mlp2_t):** wrong major mode → transposed/
  garbled result; bring up on a tiny single-tile shape to distinguish from the
  accumulate bug.
- **smem overflow / occupancy:** may force a `K_PIPE` reduction for the UMMA path
  (compile/launch-time, not silent).
- **2D-grid indexing:** an off-by-one in the `(num_sms/NSplit, NSplit)` launch or a
  broken multicast pairing → wrong tiles / duplicated work.

## Definition of success

Per the README gate: clean `sm_100a` build, `test_mlp5_fused` **Correctness** PASS
(`mean_rel<1%`, `max_rel<5%` — confirms *both* the cross-phase accumulate and the
MN-major operands are right), a `MLP5_BENCH=1` TFLOPS_Blackwell table (using the
`4·T·H·I` count), Hopper path intact, results in `blackwell_port/mlp5/writeup.md`.
Steps: [`plan.md`](plan.md).
