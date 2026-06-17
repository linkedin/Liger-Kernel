#pragma once

// Umbrella header for the auto-tuned MoE fwd+bwd config tables.
//
// The tuner emits TWO world-size CLASSES into two separate generated files so a
// single-GPU sweep and a multi-GPU sweep never clobber each other:
//   * moe_fwd_bwd_tuning_configs_single.cuh — kTunedConfigsSingle / kNumTunedConfigsSingle
//       tuned at n_pes == 1 (no cross-PE comm; all experts local).
//   * moe_fwd_bwd_tuning_configs_multi.cuh  — kTunedConfigsMulti  / kNumTunedConfigsMulti
//       tuned at n_pes  > 1 (8 GPUs); the active comm path.
// Both define the shared TunedConfigFwdBwd struct under a one-time include guard.
//
// The auto-dispatch lookup (moe.cu / moe_bwd.cu) selects the matching class at
// runtime from nvshmem_team_n_pes: n_pes <= 1 → Single, else → Multi.
//
// Regenerate via:  srun -n 1 ./benchmarks/tune_moe_fwd_bwd   (single class)
//                  srun -n 8 ./benchmarks/tune_moe_fwd_bwd   (multi class)
#include "moe_fwd_bwd_tuning_configs_single.cuh"
#include "moe_fwd_bwd_tuning_configs_multi.cuh"
