#pragma once

// ============================================================================
// MoeSymmConfig — fixed parameters for symmetric-memory sizing.
//
// Single source of truth for the MoE symmetric-session config, shared by
// moe.cu (forward) and moe_bwd.cu (backward). BOTH translation units alias
// the one object returned by get_symm_config(), so the layout MUST be defined
// exactly once — here. Previously each .cu defined its own copy; when moe.cu
// gained max_comm_stages / max_tile_m (2026-06-03, ca1b943) and moe_bwd.cu's
// copy wasn't updated, the two TUs read members at different offsets and the
// bwd path saw a stale `initialized` flag (spurious "Call
// moe_configure_symmetric before moe_bwd_fwd_bf16"). A single definition makes
// that drift impossible.
//
// Requires nvshmem_team_t — include <nvshmem.h> before this header.
// Must be set once (via moe_configure_symmetric) before the first kernel call;
// all PEs must agree.
// ============================================================================

namespace liger {

struct MoeSymmConfig {
	int max_total_slots;   // upper bound across all configs
	int max_num_experts;   // max experts across all configs
	int hidden_dim;        // fixed across configs
	int num_pes;
	int experts_per_pe;    // max_num_experts / num_pes
	int max_top_k;         // max top_k across configs (exposed via the flat ABI)
	nvshmem_team_t team;   // NVSHMEM team
	// Worst-case comm-staging shape across every config the symmetric
	// session may run. The symmetric staging pool (moe_src/dst_staging) is
	// a single shared key sized ONCE; because get_symmetric aborts on grow,
	// it must be reserved at the largest CommNumStages × TileM any config
	// uses (the tuner sweeps NS=16/CS=8/TileM=128). Per-config sizing would
	// let a small first config lock in a buffer a later large config can't
	// grow into. Same reasoning as sizing by max hidden_dim, extended to
	// the (CS, TileM) axes. The bwd path doesn't read these, but they MUST
	// stay in the layout so its get_symm_config() view matches the fwd's.
	int max_comm_stages = 8;   // max CommNumStages over LIGER_MOE_TUNE_CONFIGS
	int max_tile_m      = 128; // max TileM (WGMMA-bounded)
	bool initialized = false;
};

// Single instance lives in moe.cu (a function-local static). moe_bwd.cu reuses
// it through this same declaration.
MoeSymmConfig& get_symm_config();

} // namespace liger
