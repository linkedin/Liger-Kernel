#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// dX tensor-parallel reduction contract — CTA-owned staging, tile grouping
// and one-team NVLS synchronization.
//
// The fused SM90 backward kernel computes a *partial* dX on every rank: the
// vocabulary is sharded, so `dX_local = dZ_local @ W_local` is one term of a
// SUM over the tensor-parallel team. Instead of materialising the whole
// partial dX and running one big collective afterwards, each completed
// M128xN256 FP32 accumulator tile is written straight into a compact symmetric
// FP32 staging ring owned by that same CTA. Its own warps 1..2 reduce and
// scatter the group while the WGMMA consumers keep working on later groups.
//
// ── Pipeline ──────────────────────────────────────────────────────────────
//   consumer epilogue   FP32 accumulator -> generic coalesced global stores
//                       into partial[slot], CTA named-barrier release,
//                       then one CTA-scope release-store of a unique epoch
//   owning comm warp    local_chunk_index % 2 selects warp 1 or 2;
//                       it acquire-loads that epoch, publishes a block-prefixed
//                       system-scope NVLS ready epoch, performs the two-shot
//                       multicast SUM, then publishes a completion epoch
//   owning comm warp    vector-load the complete FP32 result,
//                       convert once to X.dtype (BF16 in this specialization),
//                       scatter into row-major dX,
//                       CTA-local consumed mbarrier arrival
//
// ── Grouping ──────────────────────────────────────────────────────────────
// `TilesPerReduce` contiguous N tiles of one M tile form one reduction
// message. The last group of an M tile is a ragged tail of
// `num_n_tiles % TilesPerReduce` tiles; its message is exactly that many tiles
// long, which is identical on every PE because every PE sees the same shapes.
//
// ── CTA ownership and independent NVLS regions ────────────────────────────
// Wave-local group `unit` belongs to CTA `unit % gridDim.x`; warp 0 and warps
// 4..11 produce only that CTA's groups, and warps 1..2 consume exactly the
// same sequence. Ring slot `j % NumStages` is therefore private to one CTA,
// and ready/reuse handoff is entirely CTA-local.
//
// Every ring slot has disjoint symmetric data and signal prefixes. All CTAs
// share one tensor-parallel team's immutable NVLS multicast mapping; no
// collective counter or team scratch is used inside the fused kernel.
//
// Warp 3 is reserved and idle on every path.
//
// This header stays NVSHMEM-free and CUTLASS-free so torch-free consumers can
// include the umbrella header.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "config.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

// Hopper has at most 132 resident SMs/CTAs for this 1-CTA/SM kernel.
inline constexpr int kMaxDxResidentCtas = 132;
inline constexpr int kDxCommWarpsPerChannel = 2;
inline constexpr int kDxSyncPhases = 2;
inline constexpr int kDxReadyPhase = 0;
inline constexpr int kDxCompletePhase = 1;

// Ring depth. Compile time: the kernel template, symmetric capacity and
// CTA-local mbarrier arrays all have to agree, so there is exactly one place
// to change it. 2, 3 and 4 are supported and exercised; 4 is the default
// because the comm warps run one pipeline step behind the collective, so a
// producer can run ahead by kDxRingStages - 1 groups.
inline constexpr int kDxRingStages = 4;
static_assert(kDxRingStages >= 2,
	"the dX staging ring needs at least a double buffer");

// Compile-time shape of the staging ring.
//
//   NumStages       ring depth per CTA; >= 2 or the producer of a group
//                   would wait on the very slot it is about to fill.
//   TilesPerReduce  contiguous N tiles coalesced into one reduction message.
template <
	typename GemmConfig,
	int NumStages = 4,
	int TilesPerReduce = 2,
	int Compute = GemmConfig::kCompute>
struct DxCommConfig {
	static_assert(Compute == 90, "DxCommConfig requires Compute=90");
	static_assert(NumStages >= 2,
		"the dX staging ring needs at least a double buffer; with one stage "
		"the producer of group j would wait on group j itself");
	static_assert(TilesPerReduce >= 1,
		"TilesPerReduce must be positive");

	static constexpr int kCompute = Compute;
	static constexpr int kNumStages = NumStages;
	static constexpr int kTilesPerReduce = TilesPerReduce;
	// config.cuh's make_dx_tile_group()/dx_groups_per_m_tile() spell this
	// knob kCoalesceTiles; keep the alias so both names name one value.
	static constexpr int kCoalesceTiles = TilesPerReduce;

	static constexpr int kTileM = GemmConfig::kTileM;
	static constexpr int kTileN = GemmConfig::kDxTileN;
	static constexpr int kTileElements = kTileM * kTileN;
	static constexpr int kGroupElements = kTileElements * kTilesPerReduce;

	static constexpr int kProducerWarp = 0;
	static constexpr int kFirstCommWarp = 1;
	static constexpr int kNumCommWarps = kDxCommWarpsPerChannel;
	static constexpr int kLastCommWarp = kFirstCommWarp + kNumCommWarps - 1;
	static constexpr int kReservedWarp = 3;

	static_assert(kLastCommWarp + 1 == kReservedWarp,
		"warps 1..2 cooperatively communicate and warp 3 stays reserved");
	static_assert(backward_warp_role(kProducerWarp) ==
		BackwardWarpRole::kProducer);
	static_assert(backward_warp_role(kFirstCommWarp) ==
		BackwardWarpRole::kDxCommunication);
	static_assert(backward_warp_role(kLastCommWarp) ==
		BackwardWarpRole::kDxCommunication);
	static_assert(backward_warp_role(kReservedWarp) ==
		BackwardWarpRole::kReserved);
};

// FSLCE-specific symmetric ring storage. Backend mappings are supplied through
// the minimal views in liger_cute/detail/tp_reduce.cuh.
//
// partial / reduced are symmetric (NVSHMEM) and laid out identically on every
// PE as
// [max_resident_ctas][2 comm warps][kNumStages][kTilesPerReduce][kTileM][kTileN],
// FP32 and
// fully packed, so one group is one contiguous message. FP32 is intentional:
// TP=1 and TP>1 use the same reduction path and accumulation semantics, and
// the final row-major grad_input store is the only conversion to X.dtype
// (BF16 in the SM90 specialization).
//
template <typename Element>
struct DxReduceWorkspace {
	Element* partial;                 // symmetric, SUM source
	Element* reduced;                 // symmetric, SUM destination
	std::uint64_t* sync;              // local symmetric signal replicas
	const std::uint64_t* launch_epoch;  // device-resident graph-safe generation
};

#if defined(__CUDACC__)
template <typename Element>
__device__ __forceinline__ std::uint64_t dx_epoch_base(
		const DxReduceWorkspace<Element>& workspace) {
	return *workspace.launch_epoch;
}
#endif

template <typename Config>
__host__ __device__ inline std::size_t dx_slot_index(
		int cta, int comm_warp, int stage) {
	return (static_cast<std::size_t>(cta) * kDxCommWarpsPerChannel +
			comm_warp) *
			Config::kNumStages +
		stage;
}

// Element offset of a slot's first tile inside the packed staging arena. The
// compile-time communication config is the single source of truth for the
// staging stride used by all producers and consumers.
template <typename Config>
__host__ __device__ inline std::size_t dx_slot_offset(
		int cta, int comm_warp, int stage) {
	std::size_t slot = dx_slot_index<Config>(cta, comm_warp, stage);
	return slot * static_cast<std::size_t>(Config::kTilesPerReduce) *
		static_cast<std::size_t>(Config::kTileElements);
}

template <typename Config>
__host__ __device__ inline std::size_t dx_sync_offset(
		int cta,
		int comm_warp,
		int stage,
		int phase,
		int team_size) {
	std::size_t slot = dx_slot_index<Config>(cta, comm_warp, stage);
	return (slot * kDxSyncPhases + static_cast<std::size_t>(phase)) *
		static_cast<std::size_t>(team_size);
}

// ───────────────────────────────────────────────────────────────────────────
// Deterministic group schedule
// ───────────────────────────────────────────────────────────────────────────

// Index in one CTA's deterministic group sequence -> ring coordinates.
// Both communication warps cooperatively reduce disjoint halves of the same
// slot, so the payload always lives in data slot zero.
struct DxCtaGroupSlot {
	int comm_warp;
	int index_in_warp;
	int stage;
	int pass;
};

template <typename Config>
__host__ __device__ inline DxCtaGroupSlot dx_cta_group_slot(
		int index_in_cta) {
	DxCtaGroupSlot slot;
	slot.comm_warp = 0;
	slot.index_in_warp = index_in_cta;
	slot.stage = slot.index_in_warp % Config::kNumStages;
	slot.pass = slot.index_in_warp / Config::kNumStages;
	return slot;
}

// Wave-local unit index -> the (m_tile, first_n_tile, num_tiles) it covers.
// `unit` counts M-fastest-last: unit = m_tile * groups_per_m + group_in_m.
template <typename Config>
__host__ __device__ inline DxTileGroup dx_unit_to_group(
		int unit, int num_n_tiles, int group_id) {
	int groups_per_m = dx_groups_per_m_tile<Config>(num_n_tiles);
	int m_tile = unit / groups_per_m;
	int group_in_m = unit - m_tile * groups_per_m;
	return make_dx_tile_group<Config>(
		m_tile, group_in_m, num_n_tiles, group_id);
}

__host__ __device__ inline int dx_groups_for_cta_per_wave(
		int groups_per_wave, int num_ctas, int cta) {
	return groups_per_wave > cta
		? (groups_per_wave - cta + num_ctas - 1) / num_ctas
		: 0;
}

// Global group -> index in its owning CTA's sequence across all waves.
__host__ __device__ inline int dx_cta_group_index(
		int group_id, int groups_per_wave, int num_ctas) {
	int wave = group_id / groups_per_wave;
	int unit = group_id - wave * groups_per_wave;
	int cta = unit % num_ctas;
	int index_in_wave = unit / num_ctas;
	return wave * dx_groups_for_cta_per_wave(
		groups_per_wave, num_ctas, cta) + index_in_wave;
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
