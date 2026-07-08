# Brief — MLP4 (weight gradients dB, dC) Blackwell (SM100a) port

> **Op:** `dB = dUᵀ · X`, `dC = dVᵀ · X` — the two MoE **weight gradients** (backward),
> produced **sequentially in two phases per cell**. Cooperative 2-WG consumer,
> **`SM90_TMA_REDUCE_ADD`** outputs (hardware atomic-add into gmem).
> **File:** `csrc/core/src/moe/mlp4.cuh` (single file — everything inline, mirrors
> `mlp3.cuh`). New test: `tests/cpp/test_mlp4.cu`.
> **Port mlp3 first.** mlp4 = mlp3's REDUCE_ADD TMEM-fill wrapped in a `for(phase=0..1)`
> loop with **independent** per-phase accumulators and two outputs. Read
> [`../README.md`](../README.md) and [`../mlp3/brief.md`](../mlp3/brief.md) first; this
> brief covers only what is **specific** to mlp4.

## What it computes

Per expert, hidden `H`, intermediate `I`, tokens `T`; the contraction is over **T**:

- `dU`, `dV` : `[T, I]` — upstream grads of the two phase-1 branches; consumed
  transposed as `dUᵀ`/`dVᵀ → [I, T]` (A operand, `M=I`, `K=T`).
- `X` : `[T, H]` — the block input, consumed as B operand (`N=H`, `K=T`); **shared** by
  both phases and both WGs.
- `dB = dUᵀ · X`, `dC = dVᵀ · X` : `[I, H]` per expert (`C[M=I, N=H]`); global tensors
  `dB,dC:[E,I,H]`.

FLOPs (**two** GEMMs): `4·T·H·I` → `TFLOPS = 4·T·H·I / s` (`2·T·H·I` per phase). *Confirm
against the kernel's own bench counter during the port.*

## Current state (`sm_90a` / WGMMA only — 0 SM100 markers)

`Mlp4Traits` mirrors `Mlp3Traits` (both operands **MN-major**, `Layout_MN_SW128_Atom`,
`Step<_2,_1>`; register accumulator via `partition_fragment_C`; cooperative 2-WG;
`EpiChunkN=64`, `NumEpiRounds=2`). Default `(TileM,TileN)=(256,128)` (M-split; `TileM`
tiles `I`, `TileN` tiles `H`). Same persistent chunk-fixed grid as mlp3.

**The mlp4-unique structure — two sequential phases per cell.** For each `(chunk, lane,
k_slice)` cell the consumer runs `for (int phase=0; phase<2; ++phase)`:
- phase 0 → `dB` via `A = dUᵀ`; phase 1 → `dC` via `dVᵀ`. The shared `X` (B operand) is
  re-read per phase (hits L2); a **single** A-side smem buffer is reused for `dUᵀ` then
  `dVᵀ` (this is what lets mlp4's footprint match mlp3 and fit the 228 KiB cap).
- **`clear(acc)` happens per phase** — the two phases have **independent** accumulators
  (this is **not** a cross-phase accumulate like mlp5; do not carry acc across phases).
- The epilogue runs per phase and REDUCE_ADDs into `dB` (phase 0) or `dC` (phase 1). The
  producer likewise loops `phase` and picks `mdUT`/`mdVT` + the matching TMA descriptor.

Everything else (the register→`store_buf` scatter, the M-split atom-interleave remap
`m_local=(m_loc/128)*64+(m_loc%64)`, the column-chunk guard, the `is_my_wg_leader`
REDUCE_ADD leader block) is **identical to mlp3**, just parameterized by `phase` for the
output tensor/descriptor. **Both `dB` and `dC` must be zero-initialized by the caller.**

## What differs from mlp3 (the whole delta)

1. **Two-phase consumer loop.** Wrap the mlp3 `Impl<100>` mainloop+epilogue body in
   `for (phase=0..1)`, selecting the A operand (`dUᵀ`/`dVᵀ`) and output
   descriptor/tensor (`dB`/`dC`) per phase — exactly as the existing WGMMA `Impl<90>`
   already does. **Independent `clear`/`ScaleOut::Zero` at the first k of *each* phase**
   (both phases start fresh; contrast mlp5, which must *not* reset).
2. **Two REDUCE_ADD outputs**, both zero-init and both re-zeroed between correctness and
   bench.
3. **Everything else is mlp3.** The REDUCE_ADD leader block, the TMEM-fill, and the
   store-buf mapping pin are the same — reuse mlp3's resolved code.

## Anticipated solution (recipe → mlp4)

Same 6-step header recipe as mlp3, applied inside the two-phase loop:

1. TMEM include + `TmemLoadOp<EpiChunkN=64>` (copy from mlp3/mlp5).
2. `Mlp4Traits`: `MainloopPipelineUmma` + UMMA `TiledMmaUmma`
   (`SM100_MMA_F16BF16_SS<…, Major::MN, Major::MN>` — both MN-major; confirm vs
   `SmemLayoutA`/`SmemLayoutX`) + accumulator in TMEM.
3. `mlp4_make_pipe_umma` (`num_consumers=1`).
4. Split `mlp4_consumer` → `Mlp4ConsumerImpl<90>` (verbatim) / `<100>` + forwarder;
   producer + `mlp4_fwd` launcher arch-agnostic (producer already loops `phase`).
5. `Impl<100>`: **whole-MMA-warp** TMEM alloc/free **once per CTA** (outside both the
   `cell_idx` loop *and* the `phase` loop — the persistent grid + two phases make a
   per-phase alloc trap doubly certain). Inside `for(phase)`: UMMA `gemm` with fresh
   `ScaleOut::Zero` on the first k of the phase; then mlp3's TMEM→`store_buf` fill →
   byte-for-byte REDUCE_ADD into `dB`/`dC`.
6. `Impl<90>` unchanged.

## Anticipated blockers

- **B5′ (persistent-grid + two-phase TMEM alloc) — HIGH.** Hoist `tcgen05.alloc`/`free`
  to once-per-CTA, above **both** the cell loop and the phase loop. A per-phase or
  per-cell alloc traps (`phase_invalid_during_alloc`).
- **Phase output routing / re-zero — HIGH (correctness).** Writing `dB` into `dC`'s slot,
  carrying acc across phases, or forgetting to re-zero → one output correct, the other
  wrong (or doubled). The test's **phase-isolation diagnostics** (below) localize this.
- **B5, B1–B4, store-buf mapping pin, MN-major operand, smem cap** — identical to mlp3
  (see [`../mlp3/brief.md`](../mlp3/brief.md)). The store-buf mapping is resolved once in
  mlp3 and reused.

## Definition of success

Per the README gate: clean `sm_100a` build of `test_mlp4`; `Mlp4.Correctness` PASS
(`mean_rel<1%`, `max_rel<5%`, both outputs zero-init'd, re-zeroed between correctness and
bench) **plus** the two phase-isolation cases `Mlp4.PhaseDB` (dC-side zeroed, isolates
`dUᵀ·X`) and `Mlp4.PhaseDC` (dB-side zeroed, isolates `dVᵀ·X`); a `MLP4_BENCH=1
…TFLOPs_Blackwell` table (`4·T·H·I`, per shape + winning split); Hopper `sm_90a` path
still compiles; a **register-spill checker** asserting 0 spills on the `Compute=100`
consumer; and `blackwell_port/mlp4/writeup.md` + root `writeup_mlp4.md`. Steps:
[`plan.md`](plan.md).
