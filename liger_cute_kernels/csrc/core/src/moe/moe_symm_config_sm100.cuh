#pragma once

#include "moe_comm_config_sm100.cuh"
#include "moe_context.h"

// ============================================================================
// Process-wide capacity and per-EP topology, shared by forward and backward.
// Requires <nvshmem.h>. Configure collectively before launching either path.
// ============================================================================

namespace liger {

struct MoeSymmConfig {
	int max_total_slots;   // upper bound across all configs
	int max_num_experts;   // max experts across all configs
	int hidden_dim;        // fixed across configs
	int num_pes;
	int num_hosts;
	int gpus_per_host;
	int experts_per_pe;    // max_num_experts / num_pes
	int max_top_k;         // max top_k across configs (exposed via the flat ABI)
	nvshmem_team_t team;   // NVSHMEM team
	// Worst-case comm-staging shape across every config the symmetric
	// session may run. The symmetric staging pool (moe_src/dst_staging) is
	// a single shared key sized ONCE; because get_symmetric aborts on grow,
	// it must be reserved at the largest CommNumStages × TileM any config
	// uses (the tuner sweeps NS=16/CS=8).
	// Per-config sizing would
	// let a small first config lock in a buffer a later large config can't
	// grow into. Same reasoning as sizing by max hidden_dim, extended to
	// the (CS, TileM) axes. The bwd path doesn't read these, but they MUST
	// stay in the layout so its get_symm_config() view matches the fwd's.
	int max_comm_stages = kMaxMoeCommStages;
	int max_tile_m      = kMaxMoeCommTileM;
	bool initialized = false;
	std::int64_t slot = -1;
	int max_inflight = 0;
};

// Resolves the current host-call context; kernels receive its values by value.
MoeSymmConfig& get_symm_config();

} // namespace liger
