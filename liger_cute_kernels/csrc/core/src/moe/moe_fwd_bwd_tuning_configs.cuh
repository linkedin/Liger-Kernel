#pragma once

// Umbrella header for the auto-tuned MoE fwd+bwd config tables.
//
// The tuner emits TWO world-size CLASSES. Each class wrapper then exposes
// compute-specific subtables:
//   * moe_fwd_bwd_tuning_configs_single.cuh — kTunedConfigTablesSingle
//       tuned at n_pes == 1 (no cross-PE comm; all experts local).
//   * moe_fwd_bwd_tuning_configs_multi.cuh  — kTunedConfigTablesMulti
//       tuned at n_pes  > 1 (8 GPUs); the active comm path.
// The auto-dispatch lookup first selects by world-size class, then by Compute
// (Hopper=90, Blackwell=100), then nearest-neighbor searches that subtable.
//
// The auto-dispatch lookup (moe.cu / moe_bwd.cu) selects the matching class at
// runtime from nvshmem_team_n_pes: n_pes <= 1 → Single, else → Multi.
//
// Regenerate via:  srun -n 1 ./benchmarks/tune_moe_fwd_bwd   (single class)
//                  srun -n 8 ./benchmarks/tune_moe_fwd_bwd   (multi class)
#include "moe_fwd_bwd_tuning_configs_single.cuh"
#include "moe_fwd_bwd_tuning_configs_multi.cuh"
