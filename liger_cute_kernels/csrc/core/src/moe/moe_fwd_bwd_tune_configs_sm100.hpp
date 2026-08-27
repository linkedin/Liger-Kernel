#pragma once

// ═══════════════════════════════════════════════════════════════════
// MoE fused forward + backward tuner config menu (X-macros)
// ═══════════════════════════════════════════════════════════════════
//
// Single merged menu — replaces the old split moe_tune_configs.hpp +
// moe_bwd_tune_configs.hpp. Consumed by:
//   moe.cu              — LIGER_MOE_TUNE_CONFIGS      → fwd template
//                         instantiations + runtime dispatch table.
//   moe_bwd.cu          — LIGER_MOE_BWD_TUNE_CONFIGS  → bwd template
//                         instantiations + runtime dispatch table.
//   tune_moe_fwd_bwd.cu — both, to build the fwd + bwd registries it
//                         times. The tuner filters each registry by
//                         TileM at runtime, so a single flat union per
//                         direction is all the .cu files need.
//   test_moe_single.cu  — LIGER_MOE_TUNE_CONFIGS for its config menu.
//
// ── Why FOUR groups ─────────────────────────────────────────────────
// The menu is split per direction and per GEMM TileM bucket:
//   LIGER_MOE_FWD_TUNE_CONFIGS_TM64   /  _TM128
//   LIGER_MOE_BWD_TUNE_CONFIGS_TM64   /  _TM128
// SM90-only forward candidates live in
//   LIGER_MOE_FWD_TUNE_CONFIGS_SM90_ONLY
// so they do not instantiate unsupported Blackwell layouts.
//
// tune_moe_fwd_bwd tunes fwd and bwd independently and may select different
// GEMM TileM values. Their communication tile and sort-buffer granularity are
// fixed at 128, so GEMM TileM does not couple the directions. The .cu files
// don't care about the bucket split; the convenience unions below feed them
// the whole menu per direction:
//   LIGER_MOE_TUNE_CONFIGS(X)      = FWD_TM64(X)  FWD_TM128(X)
//   LIGER_MOE_BWD_TUNE_CONFIGS(X)  = BWD_TM64(X)  BWD_TM128(X)
//
// ── FWD row field legend (12 args) ──────────────────────────────────
//   X(NSplit,
//     TileN1, TileK1, Stages1, EpiChunkN1,
//     TileN2, TileK2, Stages2, EpiChunkN2,
//     ZBufferSlots, CommNumStages, TileM)
//
//   - TileM ∈ {64, 128}; shared by MLP1 and MLP2.
//       TileM=128: M-split (Layout<_2,_1,_1>), each WG owns 64 rows × TileN.
//       TileM=64 : N-split (Layout<_1,_2,_1>), each WG owns 64 rows × TileN/2.
//   - MLP1 TileN1 ∈ {64,128,256}.
//   - MLP2 TileN2 ∈ {64,128,256}; SM90 additionally tunes TileN2=192.
//   - Hopper uses ceil-div tile counts and rank-3 expert descriptors; TMA
//     zero-fills load tails and drops store tails without crossing experts.
//   - EpiChunkN must divide WgTileN (= TileN at TM128, TileN/2 at TM64).
//   - TM64 requires TileN even (TileN/2 must be an integer N-half).
//   - NC derived from TileM (NOT a knob): TM128→NC=4, TM64→NC=2.
//     The NSplit % NC divisibility constraint was relaxed (ca1b943,
//     ticket-based comm), so both buckets sweep NSplit ∈ {2,4,6,8,16}.
//
// ── BWD row field legend (14 args) ──────────────────────────────────
//   X(NSplit, NSplit2,
//     TileN1, TileK1, Stages1,
//     TileM3, TileN3, TileK3, Stages3,
//     EpiChunkN1, EpiChunkN25, EpiChunkN34,
//     CommNumStages, TileM)
//
//   EpiChunkN values grouped per smem-shape twin:
//     EpiChunkN1  — mlp1_act ; EpiChunkN25 — mlp2_t + mlp5 ; EpiChunkN34 — mlp3 + mlp4.
//   Derived phases: mlp2_t/mlp5 TileN=256, TileK=TileK1, Stages=Stages1;
//   mlp4 swaps mlp3 (TileM=TileN3, TileN=TileM3).
//   Constraints (mirrored in tuned_config_valid_bwd in moe_bwd.cu):
//     D%TileK1==0, D%TileM3==0, D%256==0, I%256==0, I%TileN3==0,
//     SM100 retains exact divisibility; Hopper permits TMA-padded tails;
//     smem union ≤ 228 KiB.
//
// ── Per-kernel smem (Element = 2 bytes; H100 cap 228 KiB, budget 224) ─
//   FWD: MLP1 = 2·(TileM·TK1·S1 + 2·TN1·TK1·S1 + 2·64·EC1)
//        MLP2 = 2·(TileM·TK2·S2 +   TN2·TK2·S2 + 2·64·EC2); smem = max.
//   BWD: mlp1_act     = 2·(128 + 2·TN1)·TK1·S1 + 6·128·EC1
//        mlp2_t/mlp5  = 2·(128 + 256)·TK1·S1 + 2·128·EC25
//        mlp3         = 2·(TM3 + 2·TN3)·TK3·S3 + 4·TM3·EC34
//        mlp4         = 2·(2·TN3 + TM3)·TK3·S3 + 4·TN3·EC34; union = max.
//
// ── COOPERATIVE mlp3 / mlp4 CONSTRAINT ──────────────────────────────
// mlp3/mlp4 cooperative supports ONLY (TM3,TN3) = (256,128)/(128,256).
// Stages3 is PINNED at 2: at Stages3=4 the mlp3/mlp4 smem union is
// ~320-384 KiB, over the 228 KiB cap, so it fails the kernel's
// static_assert (won't compile) and bwd_shape_valid (would reject). PR
// #104 used Stages3=4 on its pre-K-split kernel; the K-loop split
// (4e39e50) redistributes work across CTAs but does NOT shrink per-CTA
// smem, so Stages3=2 is the deepest Phase-2 pipe that fits here — for
// the classic 1SM (single-CTA) kernel. SM100 always selects the paired-CTA
// 2SM path for both mlp3 and mlp4, using the canonical joined 256x256 shape;
// S3/EN34 drive the shared 2SM Stages/EpiChunkN tuning surface.
//
// ── X-macro structure caveat ────────────────────────────────────────
// Inside a `#define`, the `\` line-continuation joins lines BEFORE
// comments are processed, so a `// X(...)` row in the middle silently
// swallows the rest of the joined line. Keep every group body's rows
// active and uncommented; park inactive rows in the candidate bank at
// the bottom of this file (outside any macro).

// Pruned tuning menu: only rows referenced by the generated static dispatch
// tables remain here. This keeps auto-dispatch coverage identical while avoiding
// template instantiations for configs that no tuned row can select.

// ── FWD · TileM=64 (N-split consumer, NC=2) ─────────────────────────
#define LIGER_MOE_FWD_TUNE_CONFIGS_TM64(X) \
	X(128, 64, 4, 64, 128, 64, 4, 64, 4, 4, 64)

// ── FWD · TileM=128 (M-split consumer, NC=4 — divisibility relaxed) ──
#define LIGER_MOE_FWD_TUNE_CONFIGS_TM128(X) \
	X(128, 64, 4, 64, 128, 64, 4, 64, 4, 2, 128) \
	X(128, 64, 4, 64, 256, 64, 4, 64, 4, 2, 128) \
	X(128, 64, 4, 64, 128, 64, 4, 64, 4, 3, 128) \
	X(128, 64, 4, 64, 256, 64, 4, 64, 4, 3, 128) \
	X(128, 64, 4, 64, 128, 64, 4, 64, 4, 4, 128) \
	X(128, 64, 4, 64, 256, 64, 4, 64, 4, 4, 128) \
	X(128, 64, 4, 64, 128, 64, 4, 64, 4, 8, 128) \
	X(128, 64, 4, 64, 256, 64, 4, 64, 4, 8, 128)

// Mixtral-8x22B T=4096 winner. Keep one CS variant: all four measured within
// 0.2%, and CS2 won at both E_local=1 and E_local=2.
#define LIGER_MOE_FWD_TUNE_CONFIGS_SM90_ONLY(X) \
	X(128, 64, 4, 64, 192, 64, 4, 64, 4, 2, 128)

// ── BWD · TileM=64 (Phase-1 N-split, NC=2) ──────────────────────────
#define LIGER_MOE_BWD_TUNE_CONFIGS_TM64(X) \
	X(6, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 64) \
	X(2, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 64) \
	X(16, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 64) \
	X(8, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 64) \
	X(4, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 64) \
	X(8, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 64) \
	X(2, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 64) \
	X(16, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 64)

// ── BWD · TileM=128 (Phase-1 M-split, NC=4 — divisibility relaxed) ──
#define LIGER_MOE_BWD_TUNE_CONFIGS_TM128(X) \
	X(2, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 128) \
	X(4, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 3, 128) \
	X(8, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 3, 128) \
	X(2, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 3, 128) \
	X(6, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 3, 128) \
	X(16, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 128) \
	X(16, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 3, 128) \
	X(8, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 128) \
	X(4, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 128) \
	X(2, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 3, 128) \
	X(4, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 3, 128) \
	X(2, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 128) \
	X(6, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 128) \
	X(16, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 2, 128) \
	X(16, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 3, 128)


// ── Direction unions (whole menu per direction; fed to the .cu files) ─
#define LIGER_MOE_TUNE_CONFIGS(X) \
	LIGER_MOE_FWD_TUNE_CONFIGS_TM64(X) \
	LIGER_MOE_FWD_TUNE_CONFIGS_TM128(X)

#define LIGER_MOE_BWD_TUNE_CONFIGS(X) \
	LIGER_MOE_BWD_TUNE_CONFIGS_TM64(X) \
	LIGER_MOE_BWD_TUNE_CONFIGS_TM128(X)

// ═══════════════════════════════════════════════════════════════════
// Candidate bank — rows not winning any shape in the last sweep.
// ═══════════════════════════════════════════════════════════════════
// To re-enable a row: copy it into the matching group macro above (as a
// new line ending in `\`, and add a `\` to the row that was previously
// last). The full pruned menus (~130 fwd / ~30 bwd variants) live in the
// git history of the deleted moe_tune_configs.hpp / moe_bwd_tune_configs.hpp;
// the highest-signal alternates are reproduced here.
//
// ── FWD alternates ──────────────────────────────────────────────────
// X(/*NS=*/ 4, /*TN1=*/128, /*TK1=*/64, /*S1=*/4, /*EC1=*/32, /*TN2=*/128, /*TK2=*/64, /*S2=*/4, /*EC2=*/32, /*ZB=*/ 4, /*CS=*/2, /*TM=*/128)
// X(/*NS=*/ 4, /*TN1=*/128, /*TK1=*/64, /*S1=*/4, /*EC1=*/64, /*TN2=*/128, /*TK2=*/64, /*S2=*/4, /*EC2=*/64, /*ZB=*/ 8, /*CS=*/2, /*TM=*/128)
// X(/*NS=*/ 4, /*TN1=*/128, /*TK1=*/64, /*S1=*/4, /*EC1=*/64, /*TN2=*/256, /*TK2=*/64, /*S2=*/3, /*EC2=*/64, /*ZB=*/ 4, /*CS=*/2, /*TM=*/128)
// X(/*NS=*/ 6, /*TN1=*/128, /*TK1=*/64, /*S1=*/4, /*EC1=*/64, /*TN2=*/128, /*TK2=*/64, /*S2=*/4, /*EC2=*/64, /*ZB=*/ 4, /*CS=*/2, /*TM=*/128)
// X(/*NS=*/ 8, /*TN1=*/128, /*TK1=*/64, /*S1=*/3, /*EC1=*/64, /*TN2=*/128, /*TK2=*/64, /*S2=*/3, /*EC2=*/64, /*ZB=*/ 4, /*CS=*/4, /*TM=*/128)
// X(/*NS=*/ 8, /*TN1=*/128, /*TK1=*/64, /*S1=*/4, /*EC1=*/64, /*TN2=*/256, /*TK2=*/64, /*S2=*/4, /*EC2=*/64, /*ZB=*/ 4, /*CS=*/4, /*TM=*/ 64)
// X(/*NS=*/ 2, /*TN1=*/128, /*TK1=*/64, /*S1=*/4, /*EC1=*/64, /*TN2=*/128, /*TK2=*/64, /*S2=*/4, /*EC2=*/64, /*ZB=*/ 4, /*CS=*/1, /*TM=*/ 64)
//
// ── BWD alternates ──────────────────────────────────────────────────
// X(4, 8, 128, 64, 4, 128, 128, 64, 4, 32, 128, 64, 2, 128)  // full-shape Phase-2 (TM3=TN3=128, S3=4)
// X(8, 4, 128, 64, 4, 128, 128, 64, 4, 32, 128, 64, 2, 128)
// X(6, 4, 128, 64, 4, 256, 128, 64, 2, 32,  64, 64, 2, 128)  // NS=6 TM128 (Qwen3 I=768/TN1=128=6)
// X(2, 4, 128, 64, 4, 256, 128, 64, 2, 32, 128, 64, 2,  64)  // TM64 EN25=128
