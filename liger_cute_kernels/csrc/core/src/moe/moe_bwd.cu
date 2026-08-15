// Fused MoE Backward: sort → fwd-dispatch (for X) → reverse-combine (for dY)
//   → MLP_bwd + comm overlap → reverse-dispatch (for dX).
// Compiled with separable compilation. Do NOT link liger_comm_kernels library.
//
// Torch-free core (liger_cute): ported from LigerCommKernels' moe_bwd.cu. The
// device kernel + templated launcher are upstream-verbatim (namespace liger).
// Tensor-carrying backward calls are exported through TVM FFI and lower directly
// into MoeBwdArgs.
#define LIGER_CUTE_BUILDING 1

#include <nvshmem.h>
#include <nvshmemx.h>  // pulls in nvshmemi_device_state_d, required by the
                       // inline device helpers in <nvshmem_defines.h>.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
// SM100 2SM (paired-CTA cluster) TMA descriptor factories (make_tma_copy_A_sm100
// / make_tma_copy_B_sm100) and the SM100_TMA_2SM_LOAD copy atom used below to
// build pair-aware Phase-2 mlp3/mlp4 descriptors. Already pulled in
// transitively via moe_bwd.cuh -> mlp_bwd.cuh -> mlp3.cuh/mlp4.cuh; included
// directly here too since this TU calls these factories itself.
#include <cute/atom/copy_traits_sm100_tma.hpp>

#include "liger_cute/moe.h"
#include "liger_cute/check.h"
#include "liger_cute/detail/status.h"

#include "moe.cuh"
#include "moe_bwd.cuh"
#include "moe_symm_config.cuh"
#include "moe_launch.h"         // MoeFwdArgs / MoeBwdArgs (shared with the tuner)
#include "moe_utils.cuh"
#include "moe_fwd_bwd_tune_configs.hpp"
#include "moe_dispatch_configs_sm90.cuh"
#include "moe_dispatch_configs_sm100.cuh"
#include "mlp_comms_bwd.cuh"
#include "tile_iterator_bwd.cuh"
#include "dispatch.cuh"
#include "dispatch_bwd.cuh"
#include "combine_bwd.cuh"
#include "tma_copy_atom.cuh"
#include "radix_sort.cuh"
#include "buffer_pool.cuh"  // shim: liger:: aliases of liger_cute::detail infra

using namespace cute;

namespace liger {

// 0 = build every dispatch family (developer fallback). CMake sets this to 90
// for Hopper builds and 100 for Blackwell builds so normal native builds do not
// instantiate kernels for hardware that cannot run in that binary.
#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

// External symbol: moe_configure_symmetric (defined in moe.cu) is reused.
// MoeSymmConfig + get_symm_config() now come from the shared header (included
// below), so this TU can't drift from moe.cu's layout.

// Global iteration counter — same role as the fwd counter in moe.cu.
// Lives in device memory (allocated from BufferPool under the key
// "moe_bwd_iter_counter"), atomicAdded at the end of each kernel by
// CTA 0 thread 0. See PeSync::advance_iteration_counter() and the
// detailed comment in mlp_comms.cuh for why the counter has to be
// device-resident: a host-side atomic would be baked into a captured
// CUDA graph as a constant, breaking the cross-PE wait-until-greater
// protocol on every replay.

// ============================================================================
// MoeBwdConfig — bundles bwd-side traits + comm/grid constants
// ============================================================================
//
// All five MLP_bwd phase traits + grid params. Mirrors MoeFusedConfig.
// Trait choices match the values that mlp_bwd.cu landed on after the
// 2026-04-26 perf tuning round (see project_fused_bwd_tile_bump.md).

// Trait defaults match mlp_bwd.cu's validated set (lines 28–33). MoE
// backward shares mlp_fused_bwd; any drift from those values has hit
// illegal-instruction / smem-overrun bugs in the past.
template <
    int NSplit2_        = 4,    // Phase 2 (mlp3/mlp4) virtual n-split (autotuning surface)
	// Trait defaults match the standalone benches (bench_mlp{1_act,2_t,3,4,5})
	// at Stages=4 cooperative-M-split TileM=128.
	int TileM1_         = 128,
	int TileN1_         = 128,
	int TileK1_         = 64,
	int Stages1_        = 4,
	int EpiChunkN1_     = 32,         // mlp1_act
	int TileN2_         = 256,
	int TileK2_         = 64,
	int Stages2_        = 4,
	int TileM2T_        = 128,        // Phase 1b' (mlp2_t): TileM ∈ {64, 128}
	// mlp3/mlp4 cooperative layout (#102) supports only (256,128) or (128,256)
	// and needs Stages=2 to fit the 228 KiB smem cap — Stages=4 at TileN=256
	// alone is ~384 KiB. (256,128) M-split is the smallest-smem legal choice
	// (~192 KiB; the binding floor is mlp1_act ≈ 216 KiB).
	int TileM3_         = 256,        // mlp3 cooperative: (256,128) M-split
	int TileN3_         = 128,
	int TileK3_         = 64,
	int Stages3_        = 2,
	int TileM4_         = 128,        // mlp4 cooperative: swap of mlp3 → (128,256)
	int TileN4_         = 256,
	int TileK4_         = 64,
	int Stages4_        = 2,
	int TileM5_         = 128,
	int TileN5_         = 256,
	int TileK5_         = 64,
	int Stages5_        = 4,
	int EpiChunkN25_    = 64,         // shared by mlp2_t and mlp5 (same smem shape)
	int EpiChunkN34_    = 64,         // shared by mlp3 and mlp4 (same smem shape)
    typename Element_   = bfloat16_t,
    int CommNumStages_  = 2,
	int NC_             = 4,
	// Phase-1 sub-batch grouping count: run SubBatch Phase-1 sub-batches
	// (mlp1/2/5) before one Phase-2 (mlp3/4) over their combined K-window.
	// 1 = original per-batch loop. The only structural requirement is a ring
	// deeper than the group span (CommNumStages·mc_remote > SubBatch·grid_x);
	// the grouped window wraps the ring exactly as the un-grouped loop does.
	int SubBatch_       = 2,
	// CommTileM_ decouples the COMM staging tile (ring slots, semaphores,
	// tile_expert_ids granularity) from the Phase-1 GEMM tile (Traits1::TileM =
	// mlp1a/mlp2t/mlp5 = the GemmTileM hyperparameter ∈ {64,128}). CommTileM_ is
	// fixed at 128; when Traits1::TileM < CommTileM_, Phase-1 steps through
	// kSubTiles = CommTileM_/Traits1::TileM sub-tiles of GemmTileM rows per
	// 128-wide comm slot (mirrors the forward). Default == TileM1_ → kSubTiles=1
	// → bit-identical to the old coupled path.
	int CommTileM_      = 128,
	int Compute_        = 90>
struct MoeBwdConfig {
	using Element  = Element_;
	using Traits1  = Mlp1Traits <
		Element, TileM1_, TileN1_, TileK1_, Stages1_, EpiChunkN1_>;                            // Phase 1a
	using Traits2T = Mlp2TTraits<Element, TileM2T_, TileN2_, TileK2_, Stages2_, EpiChunkN25_>;  // Phase 1b'
	// Blackwell always uses the paired-CTA MLP3/MLP4 path. Hopper keeps the
	// existing single-CTA traits and never instantiates SM100-only machinery.
	// The existing TileK3_/Stages3_/EpiChunkN34_ fields remain the shared
	// tuning surface for both weight-gradient kernels.
	static constexpr bool kUsesTwoSm = Compute_ == 100;
	static constexpr int  kClusterM  = kUsesTwoSm ? 2 : 1;
	using Traits3  = cute::conditional_t<
		kUsesTwoSm,
		Mlp3Traits2Sm<Element, 256, 256, TileK3_, Stages3_, EpiChunkN34_>,
		Mlp3Traits   <Element, TileM3_, TileN3_, TileK3_, Stages3_, EpiChunkN34_>>;  // Phase 2 mlp3
	using Traits4  = cute::conditional_t<
		kUsesTwoSm,
		Mlp4Traits2Sm<Element, 256, 256, TileK4_, Stages4_, EpiChunkN34_>,
		Mlp4Traits   <Element, TileM4_, TileN4_, TileK4_, Stages4_, EpiChunkN34_>>;  // Phase 2 mlp4
	using Traits5  = Mlp5Traits <Element, TileM5_, TileN5_, TileK5_, Stages5_, EpiChunkN25_>;  // Phase 1d

	// No divisibility constraints between NSplit/NC/CommNumStages.
	// MLP side is ticket-based (RemoteMlpTileIteratorBwd); comm side
	// uses K = CommNumStages. Only runtime requirement is that NC
	// evenly divides gridDim.x · gridDim.y so MC is integer.

	static constexpr int kNSplit2       = NSplit2_;
	static constexpr int kNC            = NC_;
	static constexpr int kCommNumStages = CommNumStages_;
	static constexpr int kSubBatch      = SubBatch_;
	// kTileM is the COMM staging tile (ring slots / semaphores / tile_expert_ids
	// granularity), fixed at CommTileM_=128 — decoupled from the Phase-1 GEMM
	// tile. kGemmTileM = Traits1::TileM (mlp1/2/5 GEMM tile, the hyperparameter);
	// the mlp3/4 ratios (efkb_stride / bk↔Kblock / ring_kb) are computed from it.
	static constexpr int kTileM         = CommTileM_;       // comm tile (128)
	static constexpr int kGemmTileM     = Traits1::TileM;   // Phase-1 gemm tile (64/128)
	static constexpr int kSubTiles      = CommTileM_ / Traits1::TileM;  // 1 or 2
	static constexpr int kCompute       = Compute_;
	static constexpr int kNumThreads    = 384;

	static_assert(CommTileM_ % Traits1::TileM == 0,
		"CommTileM must be an integer multiple of the Phase-1 GEMM TileM (Traits1::TileM)");
	static_assert(kSubTiles == 1 || kSubTiles == 2,
		"kSubTiles (CommTileM/GemmTileM) must be 1 or 2");
	// mlp1a/mlp2t/mlp5 must share the same Phase-1 GEMM tile so their sub-tiling
	// and the Z/dU/dV/dZ buffer rows stay aligned.
	static_assert(Traits1::TileM == Traits2T::TileM && Traits1::TileM == Traits5::TileM,
		"Phase-1 GEMM TileM must match across mlp1a / mlp2t / mlp5");

	// Compile-time smem budget. H100 dynamic-smem cap is 228 KiB. The bwd
	// union sits very close to that ceiling (Mlp3/Mlp4 with TileK=128 are
	// 224 KiB on their own, and pipe storage pushes the total a bit
	// further), so we use the full hardware cap here. The runtime-sized
	// remote_offsets region appended by the launcher is ≤ a few hundred
	// bytes for autotuner-scale E and num_pes; relying on the kernel's
	// cudaFuncSetAttribute call to surface any overflow.
	static constexpr size_t kSmemBytes  =
		sizeof(MoeBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute_>);
	static constexpr size_t kSmemBudget = 228 * 1024;
	static_assert(kSmemBytes <= kSmemBudget,
		"MoE fused bwd kernel smem footprint exceeds H100 228 KiB cap — "
		"reduce TileN/TileK/Stages of one of the bwd phase traits");
	
	using LocalIter  = LocalMlpTileIteratorBwd<Element, kTileM>;
	using RemoteIter = RemoteMlpTileIteratorBwd<Element, kCommNumStages, kNC, kTileM>;
	using FusedIter  = FusedMlpTileIteratorBwd<Element, kCommNumStages, kNC, kTileM>;
};

// ============================================================================
// MoeBwdParams — input/output pointers + sort buffers + comm + dims
// ============================================================================

template <typename Config>
struct MoeBwdParams {
	using Element = typename Config::Element;

	// Router outputs (caller-provided).
	const int*     expert_indices;      // [T, K]
	const Element* expert_weights;      // [T, K]

	// Sort buffers.
	SortBuffers sort;

	// Inputs.
	// Note: there is no `tokens` (X) field — BWD reads token data only
	// via `x_sorted` (FWD's expert-sorted view from its symmetric stack
	// entry). Removed 2026-05-18 after confirming p.tokens was set in the
	// host wrapper and never read by the kernel or any helper.
	const Element* dY;                  // [T, D] (regular device)
	// Forward MoE output Y_fwd. Caller is responsible for providing this
	// — either retained from the most recent forward pass or recomputed
	// under gradient checkpointing. Read by combine_tokens_bwd to compute
	// dW[t,k] = Σ_d dO[t,d] * Y_fwd[slot(t,k), d]. Layout: [total_slots, D]
	// (one row per routed (token, expert-slot) pair, in the same
	// expert-sorted order produced by the forward dispatch).
	const Element* output_buffer;

	// Outputs (caller-allocated).
	Element* dX_out;                    // [T, D] (regular device)
	Element* dB_out;                    // [epp, I, D]
	Element* dC_out;                    // [epp, I, D]
	Element* dA_out;                    // [epp, D, I]
	Element* dW_out;                    // [T, K]

	// Symmetric staging buffers (allocated by allocate_moe_bwd_buffers).
	Element* x_sorted;                  // [total_slots, D]   — fwd dispatch dst
	Element* dy_sorted;                 // [total_slots, D]   — combine_bwd dst
	Element* dx_sorted;                 // [total_slots, D]   — MLP_bwd dX dst

	// Comm (forward, for sort/dispatch/pe_sync) and bwd (for X+dY gets, dX puts).
	CommBuffers     comm;
	CommBuffersBwd  comm_bwd;

	// Dimensions.
	int num_tokens;
	int hidden_dim;
	int intermediate_dim;
	int top_k;
	int total_slots;
	int num_m_tiles;                    // sort input tiles = T / TileM

	// Output stride for dB/dC/dA per-expert.
	int experts_per_pe;
};

// ============================================================================
// MoeBwdBuffers — owned device + symmetric allocations
// ============================================================================

template <typename Config>
struct MoeBwdBuffers {
	using Element = typename Config::Element;

	// Symmetric (NVSHMEM).
	Element* x_sorted;
	Element* dy_sorted;
	Element* dx_sorted;

	// Symmetric offsets.
	int*      expert_offsets_sym;        // for sort

	// Device-only.
	int*     cta_counter;
	int*     phase_counter;              // mlp_bwd uses this
	int*     barrier_counter;            // mlp_global_barrier
	Element* z_buf;                      // [scratch_rows, I]
	Element* du_buf;                     // [scratch_rows, I] -> dU after silu_bwd
	Element* dv_buf;                     // [scratch_rows, I]
	Element* dz_buf;                     // [scratch_rows, I]
	int      scratch_rows;               // max(local padded rows, remote ring rows)
	int*     expert_for_k_block;         // num_k_blocks; populated post-sort

	// Phase 2 per-expert K-range scratch. Filled in bwd preamble (once
	// per pass) by mlp_bwd_phase_2_build_ranges; shared between mlp3 and
	// mlp4 (they share TileK). Sized experts_per_pe.
	int*     expert_k_starts;
	int*     expert_k_ends;

	// Comm staging (X + dY gets, dX puts).
	Element* x_staging;
	Element* dy_staging;
	Element* dx_staging;

	int* x_src_ready;
	int* x_src_consumed;
	int* dy_src_ready;
	int* dy_src_consumed;
	int* dst_ready;
	int* dst_consumed;
	// Two per-staging-slot expert-id arrays, one per get pipe. Comm warps
	// write the X-side array during the X getmem and the dY-side array
	// during the dY getmem. Phase 2 mlp4 reads tile_expert_ids_x; Phase 2
	// mlp3 reads tile_expert_ids_dy. See CommBuffersBwd docstring for why.
	int* tile_expert_ids_x;
	int* tile_expert_ids_dy;

	// Sort-internal scratch.
	int* tile_expert_counts;
	int* sorted_token_ids;
	int* token_expert_slots;
	int* cta_done;
	int* cta_sums;
};

template <typename Config>
static MoeBwdBuffers<Config> allocate_moe_bwd_buffers(
		BufferPool& pool, cudaStream_t stream,
		int grid_x, int num_blocks,
		int hidden_dim, int intermediate_dim, int num_tokens, int top_k,
		int experts_per_pe, int total_slots, int num_m_tiles) {

	using Element = typename Config::Element;
	constexpr int kTileM = Config::kTileM;
	// kCommNumStages / kNSplit / kNC are not needed here: the comm staging is
	// sized at the config-independent UB (kCommStagesUB · kTileMUB) below.

	MoeBwdBuffers<Config> b;

	auto& scfg = get_symm_config();
	LIGER_CHECK(scfg.initialized,
		"Call moe_configure_symmetric before moe_bwd_fwd_bf16");

	size_t symm_slot_bytes =
		static_cast<size_t>(scfg.max_total_slots) * scfg.hidden_dim * sizeof(Element);

	// ── Symmetric ────────────────────────────────────────
	// x_sorted comes from the FWD pass — the caller passes its symm
	// pointer in the `x_sorted` torch tensor (FWD's pool entry under
	// key "moe_x_sorted"). BWD reads it directly and never allocates
	// or writes to its own copy.
	b.dy_sorted  = static_cast<Element*>(pool.get_symmetric("moe_bwd_dy_sorted", symm_slot_bytes));
	b.dx_sorted  = static_cast<Element*>(pool.get_symmetric("moe_bwd_dx_sorted", symm_slot_bytes));

	// dy_sorted MUST be zeroed: combine_tokens_bwd only writes routed slots
	// (top_k per token). Padding slots stay 0, contributing nothing to MLP_bwd.
	cudaMemsetAsync(b.dy_sorted, 0, symm_slot_bytes, stream);

	// all_expert_offsets is no longer allocated here — the caller passes
	// the broadcasted offsets in via the BWD's `expert_offsets` tensor
	// (populated by FWD's broadcast_expert_offsets and returned as part
	// of its intermediates).
	b.expert_offsets_sym = static_cast<int*>(
		pool.get_symmetric("moe_sort_offsets",
			(scfg.max_num_experts + 1) * sizeof(int)));

	// ── Device-only ──────────────────────────────────────
	b.cta_counter     = static_cast<int*>(pool.get_device("moe_bwd_cta_counter",   sizeof(int)));
	b.phase_counter   = static_cast<int*>(pool.get_device("moe_bwd_phase_counter", grid_x * sizeof(int)));
	b.barrier_counter = static_cast<int*>(pool.get_device("moe_bwd_barrier",       sizeof(int)));
	cudaMemsetAsync(b.cta_counter,     0, sizeof(int),                stream);
	cudaMemsetAsync(b.phase_counter,   0, grid_x * sizeof(int),       stream);
	cudaMemsetAsync(b.barrier_counter, 0, sizeof(int),                stream);

	// Local Phase 1 indexes scratch by padded token row, while the remote pass
	// indexes the same buffers by staging-ring slot. A deep comm ring can exceed
	// the local padded extent (for example, 33 MC columns * CS8 * TileM128 =
	// 33792 rows versus 17408 local rows for Mixtral-8x7B T=8K). Size for both
	// coordinate spaces or a valid remote slot writes past z/du/dv/dz.
	const int remote_m_tiles =
		(num_blocks / Config::kNC) * Config::kCommNumStages;
	const size_t remote_rows =
		static_cast<size_t>(remote_m_tiles) * kTileM;
	const size_t scratch_rows =
		std::max(static_cast<size_t>(total_slots), remote_rows);
	b.scratch_rows = static_cast<int>(scratch_rows);
	size_t intermediate_bytes =
		scratch_rows * intermediate_dim * sizeof(Element);
	b.z_buf  = static_cast<Element*>(pool.get_device("moe_bwd_z_buf",  intermediate_bytes));
	b.du_buf = static_cast<Element*>(pool.get_device("moe_bwd_du_buf", intermediate_bytes));
	b.dv_buf = static_cast<Element*>(pool.get_device("moe_bwd_dv_buf", intermediate_bytes));
	b.dz_buf = static_cast<Element*>(pool.get_device("moe_bwd_dz_buf", intermediate_bytes));

	// expert_for_k_block: one int per K-block. The K-block / m-tile size
	// ratio depends on Traits3::TileK vs Traits1::TileM; see
	// build_expert_for_k_block which uses both dims directly.
	// Size by scfg.max_total_slots, not per-call total_slots — the buffer
	// pool caches by name and won't resize on later calls. Otherwise the
	// first (smallest) config locks in a tiny allocation and a later
	// larger config (e.g. E=64, K=8) reads OOB past the cached size.
	int max_num_k_blocks = scfg.max_total_slots / Config::Traits3::TileK;
	b.expert_for_k_block = static_cast<int*>(
		pool.get_device("moe_bwd_expert_for_k_block", max_num_k_blocks * sizeof(int)));

	// Phase 2 per-expert K-range scratch — sized by scfg.max_num_experts
	// (the pool caches by name, so we have to size to the worst case
	// across configs; same rationale as expert_for_k_block above).
	// Shared between mlp3 and mlp4 (they share TileK).
	size_t expert_range_bytes = scfg.max_num_experts * sizeof(int);
	b.expert_k_starts = static_cast<int*>(
		pool.get_device("moe_bwd_expert_k_starts", expert_range_bytes));
	b.expert_k_ends   = static_cast<int*>(
		pool.get_device("moe_bwd_expert_k_ends",   expert_range_bytes));

	// ── Comm staging ─────────────────────────────────────
	// Flat staging list — L = MC · kCommNumStages slots, where
	// MC = (grid_x · NSplit) / NC. Comm strides MC through NumStages
	// stages; MLP is ticket-based and keys on the absolute slot index
	// modulo L. With NC = NSplit (legacy config), MC = grid_x and this
	// collapses to grid_x · kCommNumStages — unchanged behavior. The live
	// length MC·kCommNumStages (MC = (grid_x·NSplit)/NC) is bounded by
	// comm_l_ub below; the buffers are sized at that UB, not the live value.
	// Staging must be SYMMETRIC: the comm path issues nvshmemx_getmem/putmem
	// whose LOCAL side (x/dy/dx staging) is the NIC-registered buffer.
	// Device-only memory faults with IBV_WC_LOC_PROT_ERR over IB (works on
	// NVLink either way, which masked it single-host). Size at an
	// NC-independent upper bound (mc <= sm_count) so the symmetric
	// allocation never has to grow across NC variants — it aborts on grow;
	// larger-than-needed is fine. Mirrors the fwd moe.cu comm_l_ub sizing.
	int sm_count = 0;
	{
		int dev = 0;
		cudaGetDevice(&dev);
		cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
	}
	// Worst-case sizing across the whole tuner sweep: the symmetric/device
	// pools cache by key and ABORT on grow, so the FIRST config to allocate a
	// key locks its size. A small-TileM (64) or small-CommStages config must
	// not lock in a buffer that a later TileM=128 / deeper-pipe config can't
	// grow into. Size every comm buffer at the configured maxima (TileM=128,
	// CommStages=8, hidden_dim=scfg.hidden_dim); the kernel indexes by the
	// per-call (mc·kCommNumStages, kTileM) ≤ these UBs, so over-allocation is
	// safe. Mirrors moe.cu's MoeSymmConfig.max_tile_m / max_comm_stages fix.
	constexpr int kCommStagesUB = 8;    // max CommNumStages over the sweep
	constexpr int kTileMUB      = 128;  // max TileM over the sweep (WGMMA-bounded)
	int comm_l_ub = sm_count * kCommStagesUB;
	size_t stage_slots_bytes =
		(size_t)comm_l_ub * kTileMUB * scfg.hidden_dim * sizeof(Element);
	b.x_staging  = static_cast<Element*>(pool.get_symmetric("moe_bwd_x_staging",  stage_slots_bytes));
	b.dy_staging = static_cast<Element*>(pool.get_symmetric("moe_bwd_dy_staging", stage_slots_bytes));
	b.dx_staging = static_cast<Element*>(pool.get_symmetric("moe_bwd_dx_staging", stage_slots_bytes));

	// X / dY pipe signals: per-slot, [L = MC * NumStages] ints. Comm
	// producer indexes by `+ xc` (stride MC); MLP consumer is
	// ticket-based and walks slot = (m_base + idx·gridDim.x) mod L.
	// Allocate at the UB (comm_l_ub ≥ MC·kCommNumStages); the kernel only
	// touches [0, MC·kCommNumStages). Per-call resets (below) zero the live prefix.
	size_t tile_sig_bytes = (size_t)comm_l_ub * sizeof(int);
	b.x_src_ready    = static_cast<int*>(pool.get_device("moe_bwd_x_src_ready",    tile_sig_bytes));
	b.x_src_consumed = static_cast<int*>(pool.get_device("moe_bwd_x_src_consumed", tile_sig_bytes));
	b.dy_src_ready   = static_cast<int*>(pool.get_device("moe_bwd_dy_src_ready",   tile_sig_bytes));
	b.dy_src_consumed= static_cast<int*>(pool.get_device("moe_bwd_dy_src_consumed",tile_sig_bytes));
	cudaMemsetAsync(b.x_src_ready,    0, tile_sig_bytes, stream);
	cudaMemsetAsync(b.x_src_consumed, 0, tile_sig_bytes, stream);
	cudaMemsetAsync(b.dy_src_ready,   0, tile_sig_bytes, stream);
	cudaMemsetAsync(b.dy_src_consumed,0, tile_sig_bytes, stream);

	// dX pipe signals: per-slot, [L = MC * NumStages] ints. UB-sized (see above).
	size_t tile_pipe_bytes = (size_t)comm_l_ub * sizeof(int);
	b.dst_ready     = static_cast<int*>(pool.get_device("moe_bwd_dst_ready",    tile_pipe_bytes));
	b.dst_consumed  = static_cast<int*>(pool.get_device("moe_bwd_dst_consumed", tile_pipe_bytes));
	cudaMemsetAsync(b.dst_ready,    0, tile_pipe_bytes, stream);
	cudaMemsetAsync(b.dst_consumed, 0, tile_pipe_bytes, stream);

	size_t eid_bytes = (size_t)comm_l_ub * sizeof(int);  // UB-sized (see above)
	// One array per get pipe (see MoeBwdBuffers / CommBuffersBwd docstrings).
	// Same shape as the staging signal arrays.
	b.tile_expert_ids_x = static_cast<int*>(
		pool.get_device("moe_bwd_tile_expert_ids_x", eid_bytes));
	b.tile_expert_ids_dy = static_cast<int*>(
		pool.get_device("moe_bwd_tile_expert_ids_dy", eid_bytes));

	// ── Sort scratch (sized to scfg max for stability across calls) ──
	int max_sorted_slots = scfg.max_total_slots;
	int max_m_tiles      = scfg.max_total_slots / kTileM;
	int max_tile_counts  = max_m_tiles * scfg.max_num_experts;

	b.tile_expert_counts = static_cast<int*>(
		pool.get_device("moe_bwd_sort_tile_counts", max_tile_counts * sizeof(int)));
	b.sorted_token_ids = static_cast<int*>(
		pool.get_device("moe_bwd_sort_sorted_ids", max_sorted_slots * sizeof(int)));
	b.token_expert_slots = static_cast<int*>(
		pool.get_device("moe_bwd_sort_token_slots", max_sorted_slots * sizeof(int)));
	b.cta_done = static_cast<int*>(
		pool.get_device("moe_bwd_sort_cta_done", num_blocks * sizeof(int)));
	b.cta_sums = static_cast<int*>(
		pool.get_device("moe_bwd_sort_cta_sums",
			num_blocks * scfg.max_num_experts * sizeof(int)));

	cudaMemsetAsync(b.cta_done, 0, num_blocks * sizeof(int), stream);
	cudaMemsetAsync(b.sorted_token_ids, 0xff, max_sorted_slots * sizeof(int), stream);

	return b;
}

// ============================================================================
// build_expert_for_k_block — small helper kernel section
// ============================================================================
//
// Post-sort: derive expert_for_k_block[k] = tile_expert_ids[(k * TileK) / TileM].
// Works for both directions:
//   TileK > TileM (K-block spans multiple m-tiles): uses the first m-tile's expert.
//   TileK < TileM (m-tile spans multiple K-blocks): all K-blocks in a m-tile
//     resolve to the same tile_expert_ids slot via integer division.
// Assumes per-expert ranges are TileK-aligned, OR that any K-block straddling
// an expert boundary has only padding tokens in its second m-tile (ensured
// by zeroed dy_sorted + zero-filled X-padding from forward dispatch).

__device__ __forceinline__ void build_expert_for_k_block(
		int* expert_for_k_block,
		const int* tile_expert_ids,
		int num_k_blocks,
		int num_actual_k_blocks,
		int tile_k,
		int tile_m,
		int num_threads) {
	// For k < num_actual_k_blocks, copy the expert from the sort output.
	// For padding entries (k beyond expert_offsets[num_experts] / TileK),
	// write a sentinel that the Phase 3/4 bounds check will reject. Without
	// this, stale memory in tile_expert_ids[k] could pass the bounds check
	// and cause Phase 4 to load garbage X/dU into the dB/dC accumulators.
	for (int k = threadIdx.x; k < num_k_blocks; k += num_threads) {
		expert_for_k_block[k] = (k < num_actual_k_blocks)
			? tile_expert_ids[(k * tile_k) / tile_m]
			: -1;
	}
}

// ============================================================================
// Fused kernel: sort → fwd-dispatch → combine_bwd → MLP_bwd+comm → dispatch_bwd
// ============================================================================

template <typename Config,
          // Local-phase MLP_bwd TMA descriptors
          typename TmaLoadX,    typename TmaLoadW1,
          typename TmaStoreZ,   typename TmaStoreU,  typename TmaStoreV,
          typename TmaLoadDY,   typename TmaLoadW2T, typename TmaStoreDZ,
          typename TmaLoadDU,   typename TmaLoadDV,
          typename TmaLoadB,    typename TmaLoadC,   typename TmaStoreDX,
          // Phase 2
          typename TmaLoadDYT3, typename TmaLoadZT3, typename TmaReduceDA,
          typename TmaLoadXT4,  typename TmaLoaddUT4, typename TmaLoaddVT4,
          typename TmaReduceDB, typename TmaReduceDC,
          // Remote (staging)
          typename TmaLoadXR,   typename TmaLoadDYR, typename TmaStoreDXR,
          // Remote Phase 2 TMA (staging-backed dY/X for dA/dB/dC)
          typename TmaLoadDYT3R, typename TmaLoadXT4R>
__global__ void __launch_bounds__(Config::kNumThreads)
moe_bwd_kernel(
		// Local TMA
		__grid_constant__ TmaLoadX    const tma_load_x,
		__grid_constant__ TmaLoadW1   const tma_load_b_fwd,
		__grid_constant__ TmaLoadW1   const tma_load_c_fwd,
		__grid_constant__ TmaStoreZ   const tma_store_z,
		__grid_constant__ TmaStoreU   const tma_store_u,
		__grid_constant__ TmaStoreV   const tma_store_v,
		__grid_constant__ TmaLoadDY   const tma_load_dy,
		__grid_constant__ TmaLoadW2T  const tma_load_a_col,
		__grid_constant__ TmaStoreDZ  const tma_store_dz,
		__grid_constant__ TmaLoadDU   const tma_load_du,
		__grid_constant__ TmaLoadDV   const tma_load_dv,
		__grid_constant__ TmaLoadB    const tma_load_b_col,
		__grid_constant__ TmaLoadC    const tma_load_c_col,
		__grid_constant__ TmaStoreDX  const tma_store_dx,
		__grid_constant__ TmaLoadDYT3 const tma_load_dyt3,
		__grid_constant__ TmaLoadZT3  const tma_load_zt3,
		__grid_constant__ TmaReduceDA const tma_reduce_da,
		__grid_constant__ TmaLoadXT4  const tma_load_xt4,
		__grid_constant__ TmaLoaddUT4 const tma_load_dut4,
		__grid_constant__ TmaLoaddVT4 const tma_load_dvt4,
		__grid_constant__ TmaReduceDB const tma_reduce_db,
		__grid_constant__ TmaReduceDC const tma_reduce_dc,
		// Remote TMA
		__grid_constant__ TmaLoadXR   const tma_load_x_remote,
		__grid_constant__ TmaLoadDYR  const tma_load_dy_remote,
		__grid_constant__ TmaStoreDXR const tma_store_dx_remote,
		// Remote Phase 2 TMA (staging-backed dY/X for dA/dB/dC)
		__grid_constant__ TmaLoadDYT3R const tma_load_dyt3_remote,
		__grid_constant__ TmaLoadXT4R  const tma_load_xt4_remote,
		// Dims + bufs
		__grid_constant__ MlpBwdDims  const mlp_dims,
		__grid_constant__ MlpBwdBufs<typename Config::Element> const mlp_bufs,
		__grid_constant__ MoeBwdParams<Config>                  const p,
		__grid_constant__ int                                  const static_nsplit,
		__grid_constant__ GetTmaDescsBwd                        const get_descs_bwd) {

	using Element  = typename Config::Element;
	using Traits1  = typename Config::Traits1;
	using Traits2T = typename Config::Traits2T;
	using Traits3  = typename Config::Traits3;
	using Traits4  = typename Config::Traits4;
	using Traits5  = typename Config::Traits5;
	using FusedIter  = typename Config::FusedIter;

	constexpr int kTileM         = Config::kTileM;
	constexpr int kNumThreads    = Config::kNumThreads;
	constexpr int kCommNumStages = Config::kCommNumStages;
	constexpr int kNC            = Config::kNC;

	// Target-based cross-CTA barrier — single instance per kernel call.
	SyncThreadsCtaCounterBarrier pe_barrier(p.comm.pe_sync.cta_counter,
		(int)gridDim.x * (int)gridDim.y);

	// ── Phase 0: Sort — SKIPPED.
	// Caller provides token_expert_slots, tile_expert_ids, and
	// expert_offsets from the most recent forward call (these come out of
	// moe_fused_fwd_bf16, populated by the forward
	// kernel's sort_tokens). We trust those instead of re-sorting.

	// ── Phase 1: Broadcast expert offsets — SKIPPED.
	// Forward returned the broadcasted offsets in the `expert_offsets`
	// tensor we forward verbatim to p.comm.all_expert_offsets, and
	// forward ended with nvshmem_team_sync(team), so every PE's writes
	// are visible here. No bwd-side re-broadcast needed.

	// ── Phase 2a: Build expert_for_k_block (post-sort, before MLP) ───
	// Operates on caller-provided tile_expert_ids (forward's sort output,
	// threaded through p.sort.tile_expert_ids). BWD_TileM is constrained
	// to match FWD_TileM (enforced in moe_bwd_fwd_bf16_auto), so the
	// FWD-populated array indexes correctly at BWD's tile granularity.
	{
		int num_k_blocks = p.total_slots / Traits3::TileK;
		// Real data ends at expert_offsets[num_experts]; everything beyond
		// is buffer-pool padding from the symmetric x_sorted allocation.
		int total_real_rows = p.sort.expert_offsets[p.comm.num_experts];
		int num_actual_k_blocks = total_real_rows / Traits3::TileK;
		// Both _mlp4 and _mlp3 pointers alias buf.expert_for_k_block in the
		// local pass (set above), so writing through either populates the
		// shared post-sort array.
		build_expert_for_k_block(
			const_cast<int*>(mlp_bufs.expert_for_k_block_mlp4),
			p.sort.tile_expert_ids,
			num_k_blocks, num_actual_k_blocks,
			Traits3::TileK, kTileM, kNumThreads);
	}
	__syncthreads();

	// ── Phase 2b: Forward dispatch — SKIPPED.
	// p.x_sorted points to the forward's x_sorted (already dispatched).
	// MLP TMAs read from there directly; nothing to do here.

	// ── Phase 3: Reverse-combine — dY → dy_sorted (+ dW side-output) ─
	// Caller-provided output_buffer is Y_fwd in expert-sorted layout
	// [total_slots, D] (must match the expert-sort produced by the
	// forward dispatch). combine_tokens_bwd computes:
	//   dW[t,k]          = Σ_d dY[t,d] * Y_fwd[slot(t,k), d]   → p.dW_out
	//   dy_sorted[slot,d]= dY[t,d] * w[t,k]                    → fed to mlp_fused_bwd
	combine_tokens_bwd<Element, kNumThreads>(
		p.dY,
		/*expert_out=*/p.output_buffer,
		p.expert_weights,
		p.sort.token_expert_slots,
		p.dW_out,
		p.dy_sorted,
		p.hidden_dim,
		p.num_tokens,
		p.top_k);
	__threadfence_system();
	__syncthreads();

	p.comm.pe_sync.barrier_all(pe_barrier);

	// ── Phase 4: MLP_bwd with comm overlap ───────────────────────────
	{
		extern __shared__ char raw_smem[];
		auto& smem = *reinterpret_cast<MoeBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Config::kCompute>*>(raw_smem);

		size_t offsets_region = sizeof(MoeBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Config::kCompute>);
		offsets_region = (offsets_region + sizeof(int) - 1) & ~(sizeof(int) - 1);
		smem.comm.remote_offsets = reinterpret_cast<int*>(raw_smem + offsets_region);

		// Bwd TMA-GET bounce: 128-aligned region right after remote_offsets.
		// Mirrors the host smem sizing (align128(smem_mlp) + bounce). Only
		// dereferenced when get_descs_bwd.enabled; harmless otherwise.
		size_t bounce_off = offsets_region
			+ (size_t)p.comm_bwd.num_pes * (p.comm_bwd.experts_per_pe + 1) * sizeof(int);
		bounce_off = (bounce_off + 127) & ~((size_t)127);
		char* get_bounce = raw_smem + bounce_off;

		// No sentinel write here: nvshmem_comm_prologue_bwd's warp-2 leader
		// is guaranteed to write smem.comm.total_tiles (either to 0 in the
		// early-exit case or to comm_count in the normal case) before the
		// __syncthreads below. A `smem.comm.total_tiles = -1` from
		// threadIdx.x == 0 racing with the prologue's warp-2 write is what
		// caused the BWD-TM=64 + 2 PE deadlock: when thread 0's sentinel
		// write landed AFTER warp 2's write, total_tiles ended up as -1,
		// comm skipped nvshmem_comm_main_bwd, the comm pipes (x_src_ready
		// etc.) never got a producer_release, and the MLP remote pass'
		// iter.acquire_src spun forever.
		nvshmem_comm_prologue_bwd<kTileM, kNC>(smem.comm, p.comm_bwd, static_nsplit);
		__syncthreads();

		int flat_id = (int)blockIdx.x;
		int runtime_nsplit = static_nsplit;
		int n_gemm = p.comm_bwd.n_gemm;
		int grid_x = n_gemm / runtime_nsplit;
		int col = flat_id / runtime_nsplit;
		if (threadIdx.x == 0) {
			int total_m_tiles = smem.comm.global_total;
			runtime_nsplit = select_runtime_nsplit_bwd_from_tiles(
				total_m_tiles,
				mlp_dims.num_n_tiles_1,
				mlp_dims.num_n_tiles_2t,
				mlp_dims.num_n_tiles_5,
				mlp_dims.num_k_tiles_1,
				mlp_dims.num_k_tiles_2t,
				mlp_dims.num_k_tiles_5,
				p.comm_bwd.experts_per_pe,
				(int)gridDim.x,
				static_nsplit);
			n_gemm = ((int)gridDim.x / runtime_nsplit) * runtime_nsplit;
			grid_x = n_gemm / runtime_nsplit;
			col = flat_id / runtime_nsplit;
			int mlp_count = 0;
			if (flat_id < n_gemm && col < total_m_tiles)
				mlp_count = (total_m_tiles - 1 - col) / grid_x + 1;
			smem.comm.per_cta_tiles = mlp_count;
			smem.comm.runtime_nsplit = runtime_nsplit;
			smem.comm.runtime_n_gemm = n_gemm;
			smem.comm.runtime_grid_x = grid_x;
		}
		__syncthreads();
		runtime_nsplit = smem.comm.runtime_nsplit;
		n_gemm = smem.comm.runtime_n_gemm;
		grid_x = smem.comm.runtime_grid_x;
		col = flat_id / runtime_nsplit;
		int split = flat_id % runtime_nsplit;
		bool gemm_active = flat_id < n_gemm;

		int warp_id = threadIdx.x / 32;
		bool is_comm_warp = (warp_id == kCommBwdGetWarp0 || warp_id == kCommBwdPutWarp);

		if (is_comm_warp) {
			if (smem.comm.total_tiles > 0) {
				nvshmem_comm_main_bwd<Element, kTileM, kCommNumStages, kNC>(
					smem.comm, p.comm_bwd, &get_descs_bwd, get_bounce);
				nvshmem_quiet();
			}
		} else {
			// MLP warps. The fused iter is initialized for the staging ring
			// inside moe_fused_bwd (init_remote); every tile — local or remote —
			// flows through that staging path, so there is no local sub-iter to
			// seed here. Gap CTAs skip Phase 1; moe_fused_bwd gates it on
			// gemm_active.
			FusedIter fused_iter;
			MlpBwdDims runtime_mlp_dims = mlp_dims;
			runtime_mlp_dims.grid_x = grid_x;
			runtime_mlp_dims.n_gemm = n_gemm;

			moe_fused_bwd<Traits1, Traits2T, Traits3, Traits4, Traits5,
				kCommNumStages, Config::kNSplit2, Config::kSubBatch,
				Config::kSubTiles, Config::kCompute, FusedIter>(
				smem, fused_iter,
				tma_load_x, tma_load_b_fwd, tma_load_c_fwd,
				tma_store_z, tma_store_u, tma_store_v,
				tma_load_dy, tma_load_a_col, tma_store_dz,
				tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col, tma_store_dx,
				tma_load_dyt3, tma_load_zt3, tma_reduce_da,
				tma_load_xt4, tma_load_dut4, tma_load_dvt4, tma_reduce_db, tma_reduce_dc,
				tma_load_x_remote, tma_load_dy_remote, tma_store_dx_remote,
				tma_load_dyt3_remote, tma_load_xt4_remote,
				p.comm_bwd.x_src_ready, p.comm_bwd.x_src_consumed,
				p.comm_bwd.dy_src_ready, p.comm_bwd.dy_src_consumed,
				p.comm_bwd.dst_ready,    p.comm_bwd.dst_consumed,
				p.comm_bwd.tile_expert_ids_x,
				p.comm_bwd.tile_expert_ids_dy,
				runtime_mlp_dims, mlp_bufs,
				col, grid_x, split, runtime_nsplit, gemm_active);
		}

		__threadfence();
		__syncthreads();
		p.comm.pe_sync.barrier_all(pe_barrier);
	}

	// ── Phase 5: Reverse-dispatch — dx_sorted → dX_out (gather-sum) ──
	dispatch_tokens_bwd<Element, kNumThreads>(
		p.dx_sorted,
		p.sort.token_expert_slots,
		p.dX_out,
		p.hidden_dim,
		p.num_tokens,
		p.top_k);

}

// ============================================================================
// GET-via-TMA descriptor builder for the bwd comm path (raw CUtensorMap).
// ============================================================================
//
// Box = [kGetBoxRowsBwd, kGetKChunk] over a row-major [n_rows, hidden_dim]
// bf16 tensor (SWIZZLE_NONE keeps global bytes row-major so the MLP's
// swizzled reader sees correct data). Mirrors moe.cu's make_get_tma_map but
// with the bwd's small box height. Returns false on encode failure → caller
// disables the bwd TMA-get path (falls back to getmem/int4).
static inline bool make_get_tma_map_bwd(CUtensorMap& m, void* base,
		int hidden_dim, int n_rows) {
	uint64_t gdim[2]    = { (uint64_t)hidden_dim, (uint64_t)n_rows };
	uint64_t gstride[1] = { (uint64_t)hidden_dim * sizeof(bfloat16_t) };
	uint32_t bdim[2]    = { (uint32_t)liger::kGetKChunk, (uint32_t)liger::kGetBoxRowsBwd };
	uint32_t estride[2] = { 1, 1 };
	CUresult r = cuTensorMapEncodeTiled(
		&m, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, base, gdim, gstride, bdim, estride,
		CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
		CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
	return r == CUDA_SUCCESS;
}

// ============================================================================
// Host wrapper
// ============================================================================

// MoeBwdArgs (the POD launch bundle) is defined in moe_launch.h, shared with the
// tuner (mirrors MoeFwdArgs). dB/dC/dA/dW are gradient accumulation targets —
// the launcher zeroes them before the kernel.

template <typename Config = MoeBwdConfig<>>
void
moe_bwd_fwd_bf16(const MoeBwdArgs& a, int static_nsplit) {

	const int num_experts = a.num_experts;
	const int top_k       = a.top_k;
	const nvshmem_team_t team = a.team;
	cudaStream_t stream   = a.stream;

	using Element  = typename Config::Element;
	using Traits1  = typename Config::Traits1;
	using Traits2T = typename Config::Traits2T;
	using Traits3  = typename Config::Traits3;
	using Traits4  = typename Config::Traits4;
	using Traits5  = typename Config::Traits5;
	using TmaLoadAtom = TmaLoadAtomForCompute<Config::kCompute>;
	using TmaStoreAtom = TmaStoreAtomForCompute<Config::kCompute>;
	using TmaReduceAddAtom = TmaReduceAddAtomForCompute<Config::kCompute>;
	constexpr int kTileM      = Config::kTileM;
	constexpr int kNumThreads = Config::kNumThreads;

	// Iteration counter lives on the device; CTA 0 thread 0 atomicAdds it at
	// the end of the kernel via PeSync::advance_iteration_counter().
	int num_tokens       = a.num_tokens;
	int hidden_dim       = a.hidden_dim;
	int experts_per_pe   = a.experts_per_pe;
	int intermediate_dim = a.intermediate_dim;

	// max_total_slots uses kTileM because forward dispatch always pads each
	// expert range to its own TileM. moe_bwd_fwd_bf16_auto enforces
	// BWD_TileM == FWD_TileM, so kTileM is the right alignment here.
	int max_total_slots = num_tokens * top_k + num_experts * kTileM;
	int num_m_tiles_in  = (num_tokens + kTileM - 1) / kTileM;

	int num_pes = nvshmem_team_n_pes(team);
	int my_pe   = nvshmem_team_my_pe(team);
	int local_gpus = 0;
	cudaGetDeviceCount(&local_gpus);
	int gpus_per_node =
		(local_gpus > 0 && num_pes % local_gpus == 0) ? local_gpus : num_pes;

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, a.device);
	// Flat 1-D launch: use ALL SMs. Phase 1 + comm partition over the
	// GEMM-active subset n_gemm = floor_NS (largest multiple of static_nsplit ≤
	// num_sms); Phase 2 (mlp3/mlp4) grid-strides over all num_blocks CTAs
	// (no divisibility needed — cell stride = num_blocks); the gap CTAs
	// [n_gemm, num_blocks) fall through Phase 1/comm but still run Phase 2
	// and every barrier. This decouples the Phase-1 runtime NS and Phase-2
	// (NSplit2) CTA counts: e.g. NS4/NS2-8 on 132 SMs no longer rounds to
	// 128 — Phase 1 gets 132 and Phase 2 gets 132.
	int n_gemm     = num_sms - num_sms % static_nsplit;  // static seed floor_NS
	int grid_x     = n_gemm / static_nsplit;             // initial Phase 1 column count
	int num_blocks = num_sms;
	// Even-grid handling for the SM100 2SM paired-CTA cluster path: a
	// cudaLaunchKernelEx cluster launch requires gridDim.x to be an exact
	// multiple of ClusterM so every CTA pairs off cleanly (no partial
	// cluster at the tail). Rounds DOWN so num_blocks never exceeds the
	// device's SM count. No-op (kClusterM == 1) for every 1SM config.
	if constexpr (Config::kUsesTwoSm)
		num_blocks -= num_blocks % Config::kClusterM;
	int max_runtime_grid_x = (num_blocks + 1) / 2;       // min runtime NS candidate = 2
	dim3 grid(num_blocks);
	LIGER_CHECK(top_k <= kCombineMaxTopK,
		"top_k (", top_k, ") exceeds kCombineMaxTopK (", kCombineMaxTopK,
		"). Bump the constant in combine.cuh.");

	// stream comes from a.stream (the binding passes the current torch stream).
	auto& pool = global_buffer_pool();

	auto buf = allocate_moe_bwd_buffers<Config>(pool, stream, max_runtime_grid_x, num_blocks,
		hidden_dim, intermediate_dim, num_tokens, top_k,
		experts_per_pe, max_total_slots, num_m_tiles_in);

	// Caller-owned gradient outputs (binding-allocated). dB/dC/dA/dW are
	// accumulation targets — zero them on the stream (upstream used torch::zeros);
	// dX is fully overwritten so it needs no pre-zero.
	constexpr std::size_t kElemBytes = 2;  // sizeof(bf16)
	cudaMemsetAsync(a.dB, 0,
		static_cast<std::size_t>(experts_per_pe) * intermediate_dim * hidden_dim * kElemBytes, stream);
	cudaMemsetAsync(a.dC, 0,
		static_cast<std::size_t>(experts_per_pe) * intermediate_dim * hidden_dim * kElemBytes, stream);
	cudaMemsetAsync(a.dA, 0,
		static_cast<std::size_t>(experts_per_pe) * hidden_dim * intermediate_dim * kElemBytes, stream);
	cudaMemsetAsync(a.dW, 0,
		static_cast<std::size_t>(num_tokens) * top_k * kElemBytes, stream);

	// ── Build MoeBwdParams ──────────────────────────────────────────
	MoeBwdParams<Config> p;
	p.expert_indices  = a.expert_indices;
	p.expert_weights  = reinterpret_cast<const Element*>(a.expert_weights);
	// The BWD kernel never reads the original X token layout — token data
	// is read only from x_sorted (FWD's expert-sorted view of X). Shape
	// and device for dX_out come from dY.
	p.dY              = reinterpret_cast<const Element*>(a.dY);
	p.output_buffer   = reinterpret_cast<const Element*>(a.Y_fwd);

	p.dX_out = reinterpret_cast<Element*>(a.dX);
	p.dB_out = reinterpret_cast<Element*>(a.dB);
	p.dC_out = reinterpret_cast<Element*>(a.dC);
	p.dA_out = reinterpret_cast<Element*>(a.dA);
	p.dW_out = reinterpret_cast<Element*>(a.dW);

	// Caller-provided forward intermediates (x_sorted, sort outputs).
	// These come from moe_fused_fwd_bf16 and replace
	// what the bwd allocator + internal sort/dispatch used to populate.
	p.x_sorted  = reinterpret_cast<Element*>(a.x_sorted);
	p.dy_sorted = buf.dy_sorted;
	p.dx_sorted = buf.dx_sorted;

	// Sort outputs come from forward; we don't run sort_tokens in bwd.
	// The fields we leave from buf.* (tile_expert_counts, sorted_token_ids,
	// cta_done, cta_sums) are only consumed by sort_tokens — now skipped.
	// expert_offsets is now the broadcasted [num_pes, num_experts + 1]
	// view from FWD. The local-PE single-row view is offset by
	// my_pe * (num_experts + 1) — needed by sort lookup
	// (total_real_rows = local_offsets[num_experts]) and by comm code
	// that walks this PE's expert range.
	int* local_expert_offsets =
		a.expert_offsets + my_pe * (num_experts + 1);

	p.sort.tile_expert_counts = buf.tile_expert_counts;
	p.sort.sorted_token_ids   = buf.sorted_token_ids;
	p.sort.token_expert_slots = a.token_expert_slots;
	p.sort.expert_offsets     = local_expert_offsets;
	p.sort.tile_expert_ids    = a.tile_expert_ids;
	p.sort.cta_done           = buf.cta_done;
	p.sort.cta_sums           = buf.cta_sums;

	// Forward CommBuffers (used for pe_sync only — bwd no longer
	// re-broadcasts expert offsets; it reads what fwd wrote).
	p.comm.src_staging  = nullptr;
	p.comm.dst_staging  = nullptr;
	p.comm.src_ready    = nullptr;
	p.comm.src_consumed = nullptr;
	p.comm.dst_ready    = nullptr;
	p.comm.dst_consumed = nullptr;
	p.comm.expert_offsets    = local_expert_offsets;
	p.comm.sorted_token_ids  = buf.sorted_token_ids;
	p.comm.local_tokens      = reinterpret_cast<Element*>(a.x_sorted);
	p.comm.local_output      = buf.dx_sorted;
	// Forward CommBuffers field — preserved from before the bwd-side split.
	// The X-pipe array is the canonical "tile_expert_ids" (the dY-pipe array
	// is a bwd-only mirror gated by release_dy).
	p.comm.tile_expert_ids   = buf.tile_expert_ids_x;
	p.comm.all_expert_offsets = a.expert_offsets;
	p.comm.pe_sync.cta_counter   = buf.cta_counter;
	p.comm.pe_sync.num_pes       = num_pes;
	p.comm.pe_sync.my_pe         = my_pe;
	p.comm.pe_sync.team          = team;
	p.comm.hidden_dim       = hidden_dim;
	p.comm.num_experts      = num_experts;
	p.comm.experts_per_pe   = experts_per_pe;
	p.comm.all_expert_offsets_stride = num_experts + 1;

	// Backward CommBuffers (used for X+dY gets and dX puts).
	p.comm_bwd.x_staging       = buf.x_staging;
	p.comm_bwd.dy_staging      = buf.dy_staging;
	p.comm_bwd.dx_staging      = buf.dx_staging;
	p.comm_bwd.x_src_ready     = buf.x_src_ready;
	p.comm_bwd.x_src_consumed  = buf.x_src_consumed;
	p.comm_bwd.dy_src_ready    = buf.dy_src_ready;
	p.comm_bwd.dy_src_consumed = buf.dy_src_consumed;
	p.comm_bwd.dst_ready       = buf.dst_ready;
	p.comm_bwd.dst_consumed    = buf.dst_consumed;
	p.comm_bwd.tile_expert_ids_x  = buf.tile_expert_ids_x;
	p.comm_bwd.tile_expert_ids_dy = buf.tile_expert_ids_dy;
	// Cross-PE pull sources. Both x_sorted and dy_sorted are in the
	// same global-expert-sorted layout on every PE:
	//   • x_sorted: forward's sort_tokens → dispatch wrote it.
	//   • dy_sorted: combine_tokens_bwd writes [slot, d] using the same
	//     token_expert_slots → matches forward's per-PE expert sort.
	// all_expert_offsets is the broadcasted offset tensor produced by
	// fwd (broadcast_expert_offsets) and passed through here verbatim —
	// it tells remote PEs the [start, end) slice in our buffers that
	// holds data for their experts.
	p.comm_bwd.remote_x  = p.x_sorted;
	p.comm_bwd.remote_dy = buf.dy_sorted;
	p.comm_bwd.remote_dx = buf.dx_sorted;
	p.comm_bwd.all_expert_offsets        = a.expert_offsets;
	p.comm_bwd.all_expert_offsets_stride = num_experts + 1;
	p.comm_bwd.my_pe         = my_pe;
	p.comm_bwd.num_pes       = num_pes;
	p.comm_bwd.team          = team;
	p.comm_bwd.hidden_dim    = hidden_dim;
	p.comm_bwd.num_experts   = num_experts;
	p.comm_bwd.experts_per_pe = experts_per_pe;
	// GEMM/comm-active count: comm derives MC = n_gemm/NC and the MLP column
	// count grid_x = n_gemm/NSplit from this (the flat launch over-covers it).
	p.comm_bwd.n_gemm        = n_gemm;

	p.num_tokens       = num_tokens;
	p.hidden_dim       = hidden_dim;
	p.intermediate_dim = intermediate_dim;
	p.top_k            = top_k;
	p.total_slots      = max_total_slots;
	p.num_m_tiles      = num_m_tiles_in;
	p.experts_per_pe   = experts_per_pe;

	// ── MlpBwdDims (passed to inner mlp_fused_bwd) ──────────────────
	int total_n_rows_1 = experts_per_pe * intermediate_dim;
	int total_k_cols_2 = experts_per_pe * hidden_dim;

	MlpBwdDims mlp_dims;
	mlp_dims.hidden_dim       = hidden_dim;
	mlp_dims.intermediate_dim = intermediate_dim;
	mlp_dims.total_n_rows_1   = total_n_rows_1;
	mlp_dims.num_n_tiles_1    = (Config::kCompute == 90)
		? (intermediate_dim + Traits1::TileN - 1) / Traits1::TileN
		: intermediate_dim / Traits1::TileN;
	mlp_dims.num_k_tiles_1    = (Config::kCompute == 90)
		? (hidden_dim + Traits1::TileK - 1) / Traits1::TileK
		: hidden_dim / Traits1::TileK;
	mlp_dims.total_k_cols_2t  = total_k_cols_2;
	mlp_dims.num_n_tiles_2t   = (Config::kCompute == 90)
		? (intermediate_dim + Traits2T::TileN - 1) / Traits2T::TileN
		: intermediate_dim / Traits2T::TileN;
	mlp_dims.num_k_tiles_2t   = (Config::kCompute == 90)
		? (hidden_dim + Traits2T::TileK - 1) / Traits2T::TileK
		: hidden_dim / Traits2T::TileK;
	mlp_dims.total_k_cols_5   = total_n_rows_1;
	mlp_dims.num_n_tiles_5    = (Config::kCompute == 90)
		? (hidden_dim + Traits5::TileN - 1) / Traits5::TileN
		: hidden_dim / Traits5::TileN;
	mlp_dims.num_k_tiles_5    = (Config::kCompute == 90)
		? (intermediate_dim + Traits5::TileK - 1) / Traits5::TileK
		: intermediate_dim / Traits5::TileK;
	mlp_dims.num_m_tiles_3    = (Config::kCompute == 90)
		? (hidden_dim + Traits3::TileM - 1) / Traits3::TileM
		: hidden_dim / Traits3::TileM;
	mlp_dims.num_n_tiles_3    = (Config::kCompute == 90)
		? (intermediate_dim + Traits3::TileN - 1) / Traits3::TileN
		: intermediate_dim / Traits3::TileN;
	mlp_dims.total_n_rows_3   = total_k_cols_2;
	mlp_dims.num_m_tiles_4    = (Config::kCompute == 90)
		? (intermediate_dim + Traits4::TileM - 1) / Traits4::TileM
		: intermediate_dim / Traits4::TileM;
	mlp_dims.num_n_tiles_4    = (Config::kCompute == 90)
		? (hidden_dim + Traits4::TileN - 1) / Traits4::TileN
		: hidden_dim / Traits4::TileN;
	mlp_dims.total_m_rows_4   = total_n_rows_1;
	mlp_dims.phase_counter    = buf.phase_counter;
	// Flat-grid geometry: Phase 1 columns = grid_x, GEMM-active count = n_gemm.
	mlp_dims.grid_x                 = grid_x;
	mlp_dims.n_gemm                 = n_gemm;
	mlp_dims.num_tokens             = max_total_slots;
	mlp_dims.num_m_tiles            = max_total_slots / kTileM;
	// Local pass: ring/buffer size and the K-block iteration bound coincide,
	// so total_m_tiles_in_pass equals num_m_tiles. The remote pass overrides
	// this in moe_bwd.cuh to comm.global_total.
	mlp_dims.total_m_tiles_in_pass  = mlp_dims.num_m_tiles;
	// Phase 1 processes grid_x m-tiles per batch (one per column).
	mlp_dims.num_batches            = (mlp_dims.num_m_tiles + grid_x - 1) / grid_x;
	// Sort emits GLOBAL expert IDs; mlp_fused_bwd computes
	// `local_expert = tile.expert - local_expert_start` to index this PE's
	// local all_B/all_C/all_A. Forward kernel does the same (moe.cu:469).
	mlp_dims.local_expert_start = my_pe * experts_per_pe;
	mlp_dims.experts_per_pe     = experts_per_pe;
	// Local pass: expert_for_k_block is built per-K-block by
	// build_expert_for_k_block (one entry per Traits{3,4}::TileK), so the
	// K-block index can be used directly as the array index. The remote pass
	// overrides these via EfkbStride{3,4} template params in moe_fused_bwd
	// because its array is per-comm-tile.

	// Host-computable constants (grid-constant — saves device-side arithmetic).
	// phase2_num_blocks is no longer used for the Phase 2 cell stride (that now
	// reads gridDim.x·gridDim.y = num_blocks directly), kept for compatibility.
	mlp_dims.phase2_num_blocks   = num_blocks;
	// Remote pass: TMA descriptors cover the full flat staging list of
	// L = MC · kCommNumStages slots, each TileM rows tall.
	// MC = launched NC-complete comm columns (matches nvshmem_comm_main_bwd).
	int mc_remote = num_blocks / Config::kNC;
	mlp_dims.remote_num_m_tiles  = mc_remote * Config::kCommNumStages;
	mlp_dims.remote_num_tokens   = mlp_dims.remote_num_m_tiles * kTileM;

	// Phase-1 sub-batch grouping (mlp_bwd.cuh, SubBatch): the remote pass groups
	// up to SubBatch Phase-1 sub-batches before one Phase-2 over their combined
	// K-window. The kernel ADAPTS the group size to the ring depth (a group must
	// fit in the ring so its grid_x·gcount live slots map to distinct ring slots),
	// degrading to fewer sub-batches — down to 1 = the original per-batch loop —
	// on shallow rings. So no host-side ring constraint is needed here.

	MlpBwdBufs<Element> mlp_bufs;
	mlp_bufs.du_buf                  = buf.du_buf;
	mlp_bufs.dv_buf                  = buf.dv_buf;
	mlp_bufs.dz_buf                  = buf.dz_buf;
	// Local pass: no comm staging, no race — both Phase 2 sub-passes alias
	// the same post-sort expert_for_k_block. The remote sub-call in
	// moe_fused_bwd overrides these to the per-pipe staging arrays.
	mlp_bufs.expert_for_k_block_mlp4 = buf.expert_for_k_block;
	mlp_bufs.expert_for_k_block_mlp3 = buf.expert_for_k_block;
	mlp_bufs.expert_k_starts         = buf.expert_k_starts;
	mlp_bufs.expert_k_ends           = buf.expert_k_ends;
	mlp_bufs.gmem_da                 = reinterpret_cast<Element const*>(a.dA);
	mlp_bufs.gmem_db                 = reinterpret_cast<Element const*>(a.dB);
	mlp_bufs.gmem_dc                 = reinterpret_cast<Element const*>(a.dC);
	mlp_bufs.barrier_counter         = buf.barrier_counter;

	// ── TMA descriptors ─────────────────────────────────────────────
	// X comes from caller's x_sorted (forward's pool, post-dispatch).
	// dy_sorted is bwd-allocated (combine_tokens_bwd will populate it).
	auto* pX  = reinterpret_cast<Element const*>(a.x_sorted);  // sorted X
	auto* pDY = reinterpret_cast<Element const*>(buf.dy_sorted); // expanded dY
	auto* pB  = reinterpret_cast<Element const*>(a.all_B);
	auto* pC  = reinterpret_cast<Element const*>(a.all_C);
	auto* pA  = reinterpret_cast<Element const*>(a.all_A);
	auto* pDX = reinterpret_cast<Element*>(buf.dx_sorted);

	// All Hopper expert weights and weight-gradient targets use rank-3 tensor
	// maps. TMA handles partial edge boxes as zero-filled loads / dropped
	// stores, while the explicit expert coordinate prevents cross-expert tails.
	// (See mlp_bwd.cu for the full pattern; here we mirror it but the
	// "X" tensor is the symmetric x_sorted and "dY" is the symmetric
	// dy_sorted, both of shape [max_total_slots, hidden_dim].)
	auto tma_load_x = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(pX), make_shape(max_total_slots, hidden_dim),
			make_stride(hidden_dim, Int<1>{})),
		typename Traits1::SmemLayoutX_1{});

	auto tma_load_b_fwd = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pB),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(hidden_dim),
						Int<1>{},
						static_cast<int64_t>(intermediate_dim) * hidden_dim)),
				typename Traits1::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pB),
					make_shape(total_n_rows_1, hidden_dim),
					make_stride(hidden_dim, Int<1>{})),
				typename Traits1::SmemLayoutW_1{});
		}
	}();
	auto tma_load_c_fwd = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pC),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(hidden_dim),
						Int<1>{},
						static_cast<int64_t>(intermediate_dim) * hidden_dim)),
				typename Traits1::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pC),
					make_shape(total_n_rows_1, hidden_dim),
					make_stride(hidden_dim, Int<1>{})),
				typename Traits1::SmemLayoutW_1{});
		}
	}();

	auto tma_store_z = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.z_buf), make_shape(buf.scratch_rows, intermediate_dim),
			make_stride(intermediate_dim, Int<1>{})),
		typename Traits1::SmemLayoutStoreSlot{});
	auto tma_store_u = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.du_buf), make_shape(buf.scratch_rows, intermediate_dim),
			make_stride(intermediate_dim, Int<1>{})),
		typename Traits1::SmemLayoutStoreSlot{});
	auto tma_store_v = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.dv_buf), make_shape(buf.scratch_rows, intermediate_dim),
			make_stride(intermediate_dim, Int<1>{})),
		typename Traits1::SmemLayoutStoreSlot{});

	auto tma_load_dy = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(pDY), make_shape(max_total_slots, hidden_dim),
			make_stride(hidden_dim, Int<1>{})),
		typename Traits2T::SmemLayoutZ_1{});
	auto tma_load_a_col = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pA),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						Int<1>{},
						static_cast<int64_t>(intermediate_dim),
						static_cast<int64_t>(hidden_dim) * intermediate_dim)),
				typename Traits2T::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pA),
					make_shape(intermediate_dim, total_k_cols_2),
					make_stride(Int<1>{}, intermediate_dim)),
				typename Traits2T::SmemLayoutW_1{});
		}
	}();
	// One TMA box per sub-tile; consumer issues NSub stores per outer N-tile.
	auto tma_store_dz = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.dz_buf),
			make_shape(buf.scratch_rows, intermediate_dim),
			make_stride(intermediate_dim, Int<1>{})),
		typename Traits2T::SmemLayoutStore_1{});

	auto tma_load_du = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.du_buf)),
			make_shape(buf.scratch_rows, intermediate_dim),
			make_stride(intermediate_dim, Int<1>{})),
		typename Traits5::SmemLayoutZ_1{});
	auto tma_load_dv = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.dv_buf)),
			make_shape(buf.scratch_rows, intermediate_dim),
			make_stride(intermediate_dim, Int<1>{})),
		typename Traits5::SmemLayoutZ_1{});
	auto tma_load_b_col = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pB),
					make_shape(static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						Int<1>{},
						static_cast<int64_t>(hidden_dim),
						static_cast<int64_t>(intermediate_dim) * hidden_dim)),
				typename Traits5::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pB),
					make_shape(hidden_dim, total_n_rows_1),
					make_stride(Int<1>{}, hidden_dim)),
				typename Traits5::SmemLayoutW_1{});
		}
	}();
	auto tma_load_c_col = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pC),
					make_shape(static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						Int<1>{},
						static_cast<int64_t>(hidden_dim),
						static_cast<int64_t>(intermediate_dim) * hidden_dim)),
				typename Traits5::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(pC),
					make_shape(hidden_dim, total_n_rows_1),
					make_stride(Int<1>{}, hidden_dim)),
				typename Traits5::SmemLayoutW_1{});
		}
	}();
	// dX TMA: split-N store (TileN_half boxes per Mlp5::SmemLayoutStore_1).
	auto tma_store_dx = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(pDX),
			make_shape(max_total_slots, hidden_dim),
			make_stride(hidden_dim, Int<1>{})),
		typename Traits5::SmemLayoutStore_1{});

	// Phase 2 TMAs. mlp3's dY^T / Z operands and mlp4's X / dU^T / dV^T
	// operands need pair-aware SM100_TMA_2SM_LOAD descriptors (built via
	// make_tma_copy_A_sm100 / make_tma_copy_B_sm100, keyed off the joined
	// TileShape + TiledMma2Sm) when Config::kUsesTwoSm (all SM100 configs);
	// the dA/dB/dC TMA_REDUCE_ADD outputs are unchanged either way (each
	// peer CTA still reduce-adds its own CtaTileM rows independently).
	auto tma_load_dyt3 = [&] {
		auto tensor = make_tensor(make_gmem_ptr(pDY),
			make_shape(hidden_dim, max_total_slots),
			make_stride(Int<1>{}, hidden_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_A_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits3::SmemLayoutDYT_1{},
				typename Traits3::TileShape{},
				typename Traits3::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits3::SmemLayoutDYT_1{});
		}
	}();
	auto tma_load_zt3 = [&] {
		auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.z_buf)),
			make_shape(intermediate_dim, buf.scratch_rows),
			make_stride(Int<1>{}, intermediate_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_B_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits3::SmemLayoutZ_1{},
				typename Traits3::TileShape{},
				typename Traits3::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits3::SmemLayoutZ_1{});
		}
	}();
	auto tma_reduce_da = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaReduceAddAtom{},
				make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(a.dA)),
					make_shape(static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(intermediate_dim),
						Int<1>{},
						static_cast<int64_t>(hidden_dim) * intermediate_dim)),
				typename Traits3::SmemLayoutStore{});
		} else {
			return make_tma_copy(TmaReduceAddAtom{},
				make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(a.dA)),
					make_shape(total_k_cols_2, intermediate_dim),
					make_stride(intermediate_dim, Int<1>{})),
				typename Traits3::SmemLayoutStore{});
		}
	}();

	auto tma_load_xt4 = [&] {
		auto tensor = make_tensor(make_gmem_ptr(pX),
			make_shape(hidden_dim, max_total_slots),
			make_stride(Int<1>{}, hidden_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_B_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits4::SmemLayoutX_1{},
				typename Traits4::TileShape{},
				typename Traits4::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits4::SmemLayoutX_1{});
		}
	}();
	auto tma_load_dut4 = [&] {
		auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.du_buf)),
			make_shape(intermediate_dim, buf.scratch_rows),
			make_stride(Int<1>{}, intermediate_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_A_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits4::SmemLayoutA_1{},
				typename Traits4::TileShape{},
				typename Traits4::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits4::SmemLayoutA_1{});
		}
	}();
	auto tma_load_dvt4 = [&] {
		auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.dv_buf)),
			make_shape(intermediate_dim, buf.scratch_rows),
			make_stride(Int<1>{}, intermediate_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_A_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits4::SmemLayoutA_1{},
				typename Traits4::TileShape{},
				typename Traits4::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits4::SmemLayoutA_1{});
		}
	}();
	auto tma_reduce_db = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaReduceAddAtom{},
				make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(a.dB)),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(hidden_dim),
						Int<1>{},
						static_cast<int64_t>(intermediate_dim) * hidden_dim)),
				typename Traits4::SmemLayoutStore{});
		} else {
			return make_tma_copy(TmaReduceAddAtom{},
				make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(a.dB)),
					make_shape(total_n_rows_1, hidden_dim),
					make_stride(hidden_dim, Int<1>{})),
				typename Traits4::SmemLayoutStore{});
		}
	}();
	auto tma_reduce_dc = [&]() {
		if constexpr (Config::kCompute == 90) {
			return make_tma_copy(TmaReduceAddAtom{},
				make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(a.dC)),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(hidden_dim),
						Int<1>{},
						static_cast<int64_t>(intermediate_dim) * hidden_dim)),
				typename Traits4::SmemLayoutStore{});
		} else {
			return make_tma_copy(TmaReduceAddAtom{},
				make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(a.dC)),
					make_shape(total_n_rows_1, hidden_dim),
					make_stride(hidden_dim, Int<1>{})),
				typename Traits4::SmemLayoutStore{});
		}
	}();

	// Remote (staging) TMAs — buffers cover L = MC · CommNumStages slots,
	// each TileM rows tall. MC = (grid.x · NSplit) / NC.
	int remote_total_rows = mc_remote * Config::kCommNumStages * kTileM;
	auto tma_load_x_remote = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.x_staging)),
			make_shape(remote_total_rows, hidden_dim),
			make_stride(hidden_dim, Int<1>{})),
		typename Traits1::SmemLayoutX_1{});
	auto tma_load_dy_remote = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.dy_staging)),
			make_shape(remote_total_rows, hidden_dim),
			make_stride(hidden_dim, Int<1>{})),
		typename Traits2T::SmemLayoutZ_1{});
	// Remote dX TMA: split-N store (matches local tma_store_dx).
	auto tma_store_dx_remote = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.dx_staging),
			make_shape(remote_total_rows, hidden_dim),
			make_stride(hidden_dim, Int<1>{})),
		typename Traits5::SmemLayoutStore_1{});

	// Remote Phase 3/4 TMAs: transposed staging sources. The remote phase
	// of moe_fused_bwd previously reused tma_load_dyt3/tma_load_xt4 (which
	// point at LOCAL dy_sorted/x_sorted), so dA/dB/dC for cross-PE-routed
	// tokens used the wrong rows. These descriptors pull from the same
	// staging buffers that Phase 1/2 of the remote pass already use.
	auto tma_load_dyt3_remote = [&] {
		auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.dy_staging)),
			make_shape(hidden_dim, remote_total_rows),
			make_stride(Int<1>{}, hidden_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_A_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits3::SmemLayoutDYT_1{},
				typename Traits3::TileShape{},
				typename Traits3::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits3::SmemLayoutDYT_1{});
		}
	}();
	auto tma_load_xt4_remote = [&] {
		auto tensor = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.x_staging)),
			make_shape(hidden_dim, remote_total_rows),
			make_stride(Int<1>{}, hidden_dim));
		if constexpr (Config::kUsesTwoSm) {
			return make_tma_copy_B_sm100(SM100_TMA_2SM_LOAD{}, tensor,
				typename Traits4::SmemLayoutX_1{},
				typename Traits4::TileShape{},
				typename Traits4::TiledMma2Sm{});
		} else {
			return make_tma_copy(TmaLoadAtom{}, tensor,
				typename Traits4::SmemLayoutX_1{});
		}
	}();

	// ── GET-via-TMA descriptors for the bwd comm path (#111 port) ────
	// Per-peer source maps over nvshmem_ptr(x_sorted/dy_sorted, peer) + one
	// local dst map each over x_staging / dy_staging. Enabled only on NVLink
	// P2P + divisible hidden_dim + bounce smem fit (checked after smem sizing).
	GetTmaDescsBwd get_descs_bwd{};
	get_descs_bwd.enabled = 0;
	get_descs_bwd.gpus_per_node = gpus_per_node;
	get_descs_bwd.my_pe = my_pe;
	size_t get_bounce_bytes_bwd = (size_t)liger::GetBounceBwd<Element>::kTotalBytes;
	{
		// The single get warp's row band is TileM/NC; the small TMA box height
		// must divide it (else the last band would over-read). Holds for NC≤8
		// at TileM=128; larger NC (band < 16) disables the path.
		constexpr int kRowBandBwd = Config::kTileM / Config::kNC;
		bool ok = (hidden_dim % liger::kGetKChunk == 0)
			&& (gpus_per_node <= liger::kGetMaxPes)
			&& (kRowBandBwd % liger::kGetBoxRowsBwd == 0);
		int host_start = (my_pe / gpus_per_node) * gpus_per_node;
		for (int local_peer = 0; ok && local_peer < gpus_per_node; ++local_peer) {
			int tp = host_start + local_peer;
			if (tp >= num_pes) { ok = false; break; }
			int gpe = nvshmem_team_translate_pe(team, tp, NVSHMEM_TEAM_WORLD);
			void* px_peer  = nvshmem_ptr(p.comm_bwd.remote_x,  gpe);  // peer x_sorted
			void* pdy_peer = nvshmem_ptr(p.comm_bwd.remote_dy, gpe);  // peer dy_sorted
			if (px_peer == nullptr || pdy_peer == nullptr) { ok = false; break; }
			ok = make_get_tma_map_bwd(get_descs_bwd.src_x_desc[local_peer], px_peer,
				hidden_dim, max_total_slots);
			if (ok) ok = make_get_tma_map_bwd(get_descs_bwd.src_dy_desc[local_peer], pdy_peer,
				hidden_dim, max_total_slots);
		}
		if (ok) ok = make_get_tma_map_bwd(get_descs_bwd.dst_x_staging_desc,
			buf.x_staging, hidden_dim, remote_total_rows);
		if (ok) ok = make_get_tma_map_bwd(get_descs_bwd.dst_dy_staging_desc,
			buf.dy_staging, hidden_dim, remote_total_rows);
		get_descs_bwd.enabled = ok ? 1 : 0;
	}

	// ── Smem sizing ─────────────────────────────────────────────────
	int sort_num_warps = kNumThreads / 32;
	size_t smem_sort   = sort_num_warps * num_experts * sizeof(int);

	size_t smem_mlp = sizeof(MoeBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Config::kCompute>);
	smem_mlp = (smem_mlp + sizeof(int) - 1) & ~(sizeof(int) - 1);
	smem_mlp += num_pes * (experts_per_pe + 1) * sizeof(int);  // remote_offsets

	// Query the device's actual opt-in dynamic-smem cap rather than trust
	// Config::kSmemBudget's compile-time 228 KiB constant: the 2SM paired-CTA
	// path (Mlp3Traits2Sm / Mlp4Traits2Sm) pushes some candidates close to
	// that ceiling, and the real per-SKU limit is what
	// cudaFuncAttributeMaxDynamicSharedMemorySize is actually bounded by.
	int device = 0;
	int dynamic_smem_limit = 0;
	if (cudaError_t e = cudaGetDevice(&device); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: cudaGetDevice failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaDeviceGetAttribute(&dynamic_smem_limit,
			cudaDevAttrMaxSharedMemoryPerBlockOptin, device); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: querying the opt-in shared-memory limit failed: ",
			cudaGetErrorString(e));

	// GET-via-TMA bounce: append a 128-aligned region after remote_offsets
	// (the kernel computes the same offset). The bwd union sits near the
	// device's dynamic-smem cap, so disable the TMA-get path if the bounce
	// won't fit against the QUERIED limit (not the static 228 KiB constant)
	// — that config just keeps the original 2-get/1-put getmem layout (no
	// regression).
	if (get_descs_bwd.enabled) {
		size_t mlp_with_bounce =
			((smem_mlp + 127) & ~((size_t)127)) + get_bounce_bytes_bwd;
		if (mlp_with_bounce <= static_cast<size_t>(dynamic_smem_limit)) smem_mlp = mlp_with_bounce;
		else get_descs_bwd.enabled = 0;
	}
	// Unified BWD comm layout always uses one get warp and one put warp.
	// get_descs_bwd.enabled only controls whether same-host tiles use TMA;
	// remote/unsupported tiles fall back inside do_get_bwd_tma.
	mlp_dims.tma_get_enabled = 1;

	size_t smem_size = std::max(smem_sort, smem_mlp);

	// ── Launch ──────────────────────────────────────────────────────
	auto kernel_fn = moe_bwd_kernel<Config,
		decltype(tma_load_x), decltype(tma_load_b_fwd),
		decltype(tma_store_z), decltype(tma_store_u), decltype(tma_store_v),
		decltype(tma_load_dy), decltype(tma_load_a_col), decltype(tma_store_dz),
		decltype(tma_load_du), decltype(tma_load_dv),
		decltype(tma_load_b_col), decltype(tma_load_c_col), decltype(tma_store_dx),
		decltype(tma_load_dyt3), decltype(tma_load_zt3), decltype(tma_reduce_da),
		decltype(tma_load_xt4), decltype(tma_load_dut4), decltype(tma_load_dvt4),
		decltype(tma_reduce_db), decltype(tma_reduce_dc),
		decltype(tma_load_x_remote), decltype(tma_load_dy_remote), decltype(tma_store_dx_remote),
		decltype(tma_load_dyt3_remote), decltype(tma_load_xt4_remote)>;

	if (cudaError_t e = cudaFuncSetAttribute(kernel_fn,
			cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: cudaFuncSetAttribute(MaxDynamicSharedMemorySize=",
			smem_size, ", device opt-in limit=", dynamic_smem_limit, ") failed: ",
			cudaGetErrorString(e));
	// The SM100 2SM paired-CTA cluster path launches with an explicit
	// clusterDim; opt in to non-portable cluster sizes (the CUDA default
	// portable max is 8, so ClusterM=2 is already portable in practice,
	// but every 2SM launch enables this defensively, mirroring CUTLASS's
	// own cluster-launch helpers).
	if constexpr (Config::kUsesTwoSm) {
		if (cudaError_t e = cudaFuncSetAttribute(kernel_fn,
				cudaFuncAttributeNonPortableClusterSizeAllowed, 1); e != cudaSuccess)
			LIGER_FAIL_CUDA("moe_bwd: enabling non-portable cluster size (ClusterM=",
				Config::kClusterM, ") failed: ", cudaGetErrorString(e));
	}

	// Fresh per-call counter resets.
	cudaMemsetAsync(buf.cta_counter,     0, sizeof(int),               stream);
	cudaMemsetAsync(buf.barrier_counter, 0, sizeof(int),               stream);
	cudaMemsetAsync(buf.phase_counter,   0, max_runtime_grid_x * sizeof(int), stream);
	// L = MC · CommNumStages, MC = launched NC-complete comm columns.
	int mc_local = ((int)grid.x / Config::kNC);
	int comm_l_local = mc_local * Config::kCommNumStages;
	size_t per_call_tile_sig_bytes =
		static_cast<size_t>(comm_l_local) * sizeof(int);
	cudaMemsetAsync(buf.x_src_ready,    0, per_call_tile_sig_bytes, stream);
	cudaMemsetAsync(buf.x_src_consumed, 0, per_call_tile_sig_bytes, stream);
	cudaMemsetAsync(buf.dy_src_ready,   0, per_call_tile_sig_bytes, stream);
	cudaMemsetAsync(buf.dy_src_consumed,0, per_call_tile_sig_bytes, stream);
	cudaMemsetAsync(buf.dst_ready,    0, per_call_tile_sig_bytes, stream);
	cudaMemsetAsync(buf.dst_consumed, 0, per_call_tile_sig_bytes, stream);
	cudaMemsetAsync(buf.cta_done,     0, num_blocks * sizeof(int),     stream);
	cudaMemsetAsync(buf.sorted_token_ids, 0xff, max_total_slots * sizeof(int), stream);
	// dy_sorted MUST be zeroed: combine_tokens_bwd writes routed slots
	// only (no padding fill — unlike forward's dispatch_tokens). Partial-
	// expert tile rows of dy_sorted are read by Phase 1b's TMA load; if
	// those rows aren't zero, the resulting dZ/dU/dV poison Phase 2 dA/dB/dC
	// for K-blocks that straddle an expert boundary (see the
	// build_expert_for_k_block comment for the alignment assumption).
	cudaMemsetAsync(buf.dy_sorted,    0, static_cast<size_t>(max_total_slots) * hidden_dim * sizeof(Element), stream);
	// dx_sorted: write-only buffer for routed slots — Phase 1d writes valid
	// tiles, comm-warp dX put writes remote-routed slots, dispatch_tokens_bwd
	// reads routed slots only via token_expert_slots. Padding rows are never
	// read so they don't need pre-zero'ing.
	//
	// x_staging / dy_staging: comm warps fill the slots THIS CTA fetches;
	// unfetched slots are gated out by tile_expert_ids_x/dy = -1 (the 0xff
	// memset below), and the RemoteIter only iterates per_cta_tiles so Phase
	// 1 never reads unfetched slots either. Stale prev-call contents are
	// inert under both bounds checks.
	//
	// du_buf / dv_buf / dz_buf / z_buf: Phase 1 writes the full TileM rows
	// of every tile the local-expert pass / staging-ring iterates over;
	// Phase 2 mlp{3,4} read at K-block granularity and bounds-check via
	// expert_for_k_block (build_expert_for_k_block stamps -1 past
	// num_actual_k_blocks, and the remote pass aliases tile_expert_ids).
	// Fake K-blocks are skipped → any stale slot from a previous call is
	// inert.
	//
	// Total: ~2.5 GB of per-call memset bandwidth (4× intermediate_bytes +
	// 2× stage_bytes + 1× symm_slot_bytes for dx_sorted) eliminated. The
	// 0xff sentinel fill of tile_expert_ids_{x,dy} below is what makes
	// these drops safe — keep that.

	// Sentinel-fill BOTH comm-side tile_expert_ids arrays so unfetched
	// slots produce expert_id = -1, which the Phase 3/4 bounds check
	// rejects. Without this, stale memory could land in [0, experts_per_pe)
	// and trigger spurious dB/dC/dA writes for empty staging slots. This
	// 0xff fill is also what lets us drop the x_staging / dy_staging /
	// du_buf / dv_buf / dz_buf / z_buf memsets above.
	size_t comm_eid_bytes =
		static_cast<size_t>(comm_l_local) * sizeof(int);
	if (cudaError_t e = cudaMemsetAsync(buf.tile_expert_ids_x, 0xff, comm_eid_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: cudaMemsetAsync(tile_expert_ids_x) failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaMemsetAsync(buf.tile_expert_ids_dy, 0xff, comm_eid_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: cudaMemsetAsync(tile_expert_ids_dy) failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: CUDA error before kernel launch: ", cudaGetErrorString(e));

	// Compute=90 keeps the ordinary <<<>>> launch. Compute=100 always uses
	// the explicit paired-CTA cluster launch below.
	if constexpr (Config::kUsesTwoSm) {
		cudaLaunchConfig_t launch_config = {};
		launch_config.gridDim = grid;
		launch_config.blockDim = kNumThreads;
		launch_config.dynamicSmemBytes = smem_size;
		launch_config.stream = stream;
		cudaLaunchAttribute cluster_attr = {};
		cluster_attr.id = cudaLaunchAttributeClusterDimension;
		cluster_attr.val.clusterDim.x = Config::kClusterM;
		cluster_attr.val.clusterDim.y = 1;
		cluster_attr.val.clusterDim.z = 1;
		launch_config.attrs = &cluster_attr;
		launch_config.numAttrs = 1;
		cudaLaunchKernelEx(&launch_config, kernel_fn,
			tma_load_x, tma_load_b_fwd, tma_load_c_fwd,
			tma_store_z, tma_store_u, tma_store_v,
			tma_load_dy, tma_load_a_col, tma_store_dz,
			tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col, tma_store_dx,
			tma_load_dyt3, tma_load_zt3, tma_reduce_da,
			tma_load_xt4, tma_load_dut4, tma_load_dvt4, tma_reduce_db, tma_reduce_dc,
			tma_load_x_remote, tma_load_dy_remote, tma_store_dx_remote,
			tma_load_dyt3_remote, tma_load_xt4_remote,
			mlp_dims, mlp_bufs, p, static_nsplit, get_descs_bwd);
	} else {
		kernel_fn<<<grid, kNumThreads, smem_size, stream>>>(
			tma_load_x, tma_load_b_fwd, tma_load_c_fwd,
			tma_store_z, tma_store_u, tma_store_v,
			tma_load_dy, tma_load_a_col, tma_store_dz,
			tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col, tma_store_dx,
			tma_load_dyt3, tma_load_zt3, tma_reduce_da,
			tma_load_xt4, tma_load_dut4, tma_load_dvt4, tma_reduce_db, tma_reduce_dc,
			tma_load_x_remote, tma_load_dy_remote, tma_store_dx_remote,
			tma_load_dyt3_remote, tma_load_xt4_remote,
			mlp_dims, mlp_bufs, p, static_nsplit, get_descs_bwd);
	}
	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_bwd: kernel launch failed for grid=", grid.x,
			" threads=", kNumThreads, " smem=", smem_size, ": ",
			cudaGetErrorString(e));

	// Post-kernel stream-ordered barrier: mirrors the fwd wrapper. The
	// kernel's PeSync wait_mlp guarantees this PE has received peer
	// signals, but does not guarantee this PE's outbound signals have
	// landed on peers. Under back-to-back captured replays (fwd → bwd →
	// fwd), bwd's tail-end signals to peers can race against the next
	// fwd's wait_dispatch: a peer's wait_until(>= target) may briefly
	// see a stale value if bwd's older signal arrives after a newer
	// one. Forcing host-side completion here serialises remote
	// signal delivery against the next captured kernel.
	if (num_pes > 1) {
		nvshmemx_barrier_on_stream(team, stream);
	}
}

// ============================================================================
// Tuned-config wrapper: 13 free knobs → full MoeBwdConfig
// ============================================================================
//
// Function template that takes the X-macro's free knobs and derives the
// dependent traits per the symmetric-traits constraints documented in
// moe_fwd_bwd_tune_configs.hpp:
//   Phase 1a:  TileN1, TileK1, Stages1, EpiChunkN1
//   Phase 1b': TileN = 256 (fixed), TileK = TileK1, Stages = Stages1, EpiChunkN25
//   Phase 1d (mlp5): TileM=128, TileN=256, TileK = TileK1, Stages = Stages1, EpiChunkN25
//   Phase 2 mlp3:    TileM3, TileN3, TileK3, Stages3, EpiChunkN34
//   Phase 2 mlp4:    TileM = TileN3, TileN = TileM3, TileK3, Stages3, EpiChunkN34
//                    (mlp3/mlp4 swap)
//   NC = NSplit
//
// The tuner registry and dispatch table are built by emitting one explicit
// instantiation of this template per X-macro entry.

template <
	int NSplit2,
	int TileN1, int TileK1, int Stages1,
	int TileM3, int TileN3, int TileK3, int Stages3,
	int EpiChunkN1, int EpiChunkN25, int EpiChunkN34,
	int CommNumStages,
	int TileM,            // COMM tile (staging/semaphores/NC); must match FWD's TileM. 64 or 128.
	int GemmTileM = TileM, // Phase-1 GEMM tile (mlp1a/mlp2t/mlp5) ∈ {64,128}; default == comm.
	int Compute = 90>
void
moe_bwd_fwd_bf16_tuned(const MoeBwdArgs& a, int static_nsplit) {
	// NC paired with TileM (mirrors moe.cu's moe_fused_fwd_bf16):
	//   TileM=128 → NC=4 (~2× the comm bandwidth of NC=2; deeper comm pipeline).
	//   TileM=64  → NC=2 (tiles are half-size, NC=2 matches NC=4 throughput).
	// Ticket-based iterator + K=NumStages comm means NSplit and NC have no
	// divisibility constraint — only requirement is that NC evenly divides
	// gridDim.x · gridDim.y so MC is integer (e.g. NS=6, NC=4: MC = 22·6/4
	// = 33).
	// COMM tile FIXED at 128 (NC=4) for every config — uniform across PEs so the
	// all-to-all dispatch padding (tile_expert_ids / x_sorted / expert_offsets)
	// aligns. `TileM` is purely the Phase-1 GEMM-tile knob (flows to GemmTileM,
	// default == TileM): a "TM64" bwd config = comm128/gemm64.
	using Config = MoeBwdConfig<
		NSplit2,
		GemmTileM, TileN1, TileK1, Stages1, EpiChunkN1,              // Phase 1a (mlp1_act) — GEMM tile
		/*TileN2=*/256, TileK1, Stages1,                              // Phase 1b' (Mlp2TTraits caps TileN ≤ 256)
		/*TileM2T=*/GemmTileM,                                        // Phase 1b' GEMM tile
		TileM3, TileN3, TileK3, Stages3,                              // Phase 2 mlp3
		TileN3, TileM3, TileK3, Stages3,                              // Phase 2 mlp4 (mlp3/mlp4 swap; cooperative)
		GemmTileM, 256, TileK1, Stages1,                             // Phase 1d mlp5 — GEMM tile
		EpiChunkN25,                                                  // shared mlp2_t + mlp5
		EpiChunkN34,                                                  // shared mlp3 + mlp4
		bfloat16_t,
		CommNumStages,
		/*NC_=*/4,
		/*SubBatch_=*/2,
		/*CommTileM_=*/128,                                           // comm tile fixed at 128
		/*Compute_=*/Compute>;
	moe_bwd_fwd_bf16<Config>(a, static_nsplit);
}

// ── Runtime dispatch table for tuned-config lookup ────────────────────

using MoeBwdFwdFn = void (*)(const MoeBwdArgs&, int static_nsplit);

struct DispatchEntryBwd {
	int Compute;
	int NSplit2;
	int TileN1, TileK1, Stages1;
	int TileM3, TileN3, TileK3, Stages3;
	int EpiChunkN1, EpiChunkN25, EpiChunkN34;
	int CommNumStages;
	int TileM;            // COMM tile (staging/semaphores/NC).
	int GemmTileM;        // Phase-1 GEMM tile (mlp1a/mlp2t/mlp5); == TileM for coupled configs.
	MoeBwdFwdFn fn;
};

#define LIGER_MOE_BWD_DISPATCH_ENTRY_G_C(Compute, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM) \
	{ Compute, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM, \
	  &moe_bwd_fwd_bf16_tuned<NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM, Compute> },
#define LIGER_MOE_BWD_DISPATCH_ENTRY_C(Compute, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	LIGER_MOE_BWD_DISPATCH_ENTRY_G_C(Compute, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, TM)
#define LIGER_MOE_BWD_DISPATCH_ENTRY_G_SM90(NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM) \
	LIGER_MOE_BWD_DISPATCH_ENTRY_G_C(90, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM)
#define LIGER_MOE_BWD_DISPATCH_ENTRY_SM90(NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	LIGER_MOE_BWD_DISPATCH_ENTRY_C(90, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM)
#define LIGER_MOE_BWD_DISPATCH_ENTRY_G_SM100(NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM) \
	LIGER_MOE_BWD_DISPATCH_ENTRY_G_C(100, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM)
#define LIGER_MOE_BWD_DISPATCH_ENTRY_SM100(NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	LIGER_MOE_BWD_DISPATCH_ENTRY_C(100, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM)

static const DispatchEntryBwd kDispatchTableBwd[] = {
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 90
	LIGER_MOE_BWD_DISPATCH_CONFIGS_SM90(LIGER_MOE_BWD_DISPATCH_ENTRY_SM90, LIGER_MOE_BWD_DISPATCH_ENTRY_G_SM90)
#endif
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 100
	LIGER_MOE_BWD_DISPATCH_CONFIGS_SM100(LIGER_MOE_BWD_DISPATCH_ENTRY_SM100, LIGER_MOE_BWD_DISPATCH_ENTRY_G_SM100)
#endif
};
#undef LIGER_MOE_BWD_DISPATCH_ENTRY_SM100
#undef LIGER_MOE_BWD_DISPATCH_ENTRY_G_SM100
#undef LIGER_MOE_BWD_DISPATCH_ENTRY_SM90
#undef LIGER_MOE_BWD_DISPATCH_ENTRY_G_SM90
#undef LIGER_MOE_BWD_DISPATCH_ENTRY_C
#undef LIGER_MOE_BWD_DISPATCH_ENTRY_G_C
static constexpr int kNumDispatchEntriesBwd =
	sizeof(kDispatchTableBwd) / sizeof(kDispatchTableBwd[0]);

} // namespace liger

// Joint fwd+bwd tuning table (auto-generated by benchmarks/tune_moe_fwd_bwd).
// We only consume the Bwd_* fields here; moe.cu consumes the same table's
// Fwd_* fields. Picking fwd and bwd from a single table guarantees the two
// directions are tuned as a pair — in particular their TileM agrees.
#include "moe_fwd_bwd_tuning_configs.cuh"

namespace liger {

static constexpr int kBwdRuntimeNsplitSeed = 8;

// TunedConfigBwd is the bwd-only projection of TunedConfigFwdBwd. The rest
// of the dispatch code (validity check, dispatch-entry match) was written
// against this struct and is reused unchanged.
struct TunedConfigBwd {
	int Compute;
	int TK, TKE, D, I;
	int NSplit2;
	int TileN1, TileK1, Stages1;
	int TileM3, TileN3, TileK3, Stages3;
	int EpiChunkN1, EpiChunkN25, EpiChunkN34;
	int CommNumStages;
	int TileM;      // comm tile
	int GemmTileM;  // gemm tile (auto path: == TileM; FORCE_CONFIG may set 64)
};

// Project a TunedConfigFwdBwd row down to its bwd template params. The tuned
// table has no GemmTileM column, so the auto path keeps gemm == comm (TileM).
static TunedConfigBwd project_bwd(const TunedConfigFwdBwd& c, int compute) {
	return TunedConfigBwd{
		compute,
		c.TK, c.TKE, c.D, c.I,
		c.Bwd_NSplit2,
		c.Bwd_TileN1, c.Bwd_TileK1, c.Bwd_Stages1,
		c.Bwd_TileM3, c.Bwd_TileN3, c.Bwd_TileK3, c.Bwd_Stages3,
		c.Bwd_EpiChunkN1, c.Bwd_EpiChunkN25, c.Bwd_EpiChunkN34,
		c.Bwd_CommNumStages,
		c.Bwd_TileM, c.Bwd_TileM,
	};
}

// Find the dispatch table entry matching a TunedConfigBwd's knobs.
static const DispatchEntryBwd* find_dispatch_entry_bwd(const TunedConfigBwd& c) {
	for (int i = 0; i < kNumDispatchEntriesBwd; ++i) {
		const auto& e = kDispatchTableBwd[i];
		if (e.Compute == c.Compute &&
		    e.NSplit2 == c.NSplit2 &&
		    e.TileN1 == c.TileN1 && e.TileK1 == c.TileK1 && e.Stages1 == c.Stages1 &&
		    e.TileM3 == c.TileM3 && e.TileN3 == c.TileN3 && e.TileK3 == c.TileK3 &&
		    e.Stages3 == c.Stages3 &&
		    e.EpiChunkN1 == c.EpiChunkN1 &&
		    e.EpiChunkN25 == c.EpiChunkN25 && e.EpiChunkN34 == c.EpiChunkN34 &&
		    e.CommNumStages == c.CommNumStages &&
		    e.TileM == c.TileM && e.GemmTileM == c.GemmTileM) {
			return &e;
		}
	}
	return nullptr;
}

// Structural divisibility checks on (D, I) for a given tuned config.
// Mirrors the constraint list in moe_fwd_bwd_tune_configs.hpp.
static bool tuned_config_valid_bwd(int D, int I, const TunedConfigBwd& c) {
	// The current Blackwell UMMA MLP1/MLP1-act consumers only support the
	// 128-row comm tile. Keep mirrored SM90 TileM=64 tuning rows out of SM100
	// auto-dispatch until that TMEM epilogue layout is implemented.
	if (c.Compute == 100 && c.TileM != 128) return false;
	if (c.Compute == 100) {
		if (D % c.TileK1 != 0)        return false;  // mlp1/mlp2_t/mlp5 K
		if (D % (2 * c.TileN1) != 0)  return false;  // mlp5 N (= TileN2)
		if (I % (2 * c.TileN1) != 0)  return false;  // mlp2_t N
		if (I % c.TileK1 != 0)        return false;  // mlp5 K
		if (I % c.TileK3 != 0)        return false;  // mlp3/mlp4 K
	}
	if (D % 8 != 0 || I % 8 != 0) return false;  // vectorization
	if (c.Compute == 100) {
		if (D % 256 != 0 || I % 256 != 0) return false;
	} else {
		if (!((c.TileM3 == 256 && c.TileN3 == 128) ||
		      (c.TileM3 == 128 && c.TileN3 == 256))) return false;
		if (D % c.TileM3 != 0 || I % c.TileN3 != 0) return false;
	}
	return true;
}

static bool find_tuned_table_bwd(int compute, int n_pes,
                                 const TunedConfigFwdBwd** table,
                                 int* table_n) {
	const TunedConfigFwdBwdTable* tables =
		(n_pes <= 1) ? kTunedConfigTablesSingle : kTunedConfigTablesMulti;
	const int tables_n =
		(n_pes <= 1) ? kNumTunedConfigTablesSingle : kNumTunedConfigTablesMulti;
	for (int i = 0; i < tables_n; ++i) {
		if (tables[i].Compute == compute) {
			*table = tables[i].configs;
			*table_n = tables[i].count;
			return *table_n > 0;
		}
	}
	return false;
}

// Nearest-neighbor lookup in log-space (TK, TKE, D, I), where TK = T*top_k
// is the total routed-token count (the bwd's M-axis post-dispatch) and
// TKE = TK / E_local (E_local = E / n_pes) is the per-LOCAL-expert avg
// K-range. mlp3/mlp4 walk each PE's local experts, so adding TKE to the
// distance lets the dispatcher distinguish configs whose walk-scheduler
// load differs even when (TK, D, I) is the same.
// Configs that violate structural constraints for the runtime (D, I) are
// skipped — the nearest tuned point is not always valid for the actual shape.
// If fwd_tile_m > 0, configs with TileM != fwd_tile_m are filtered out:
// BWD's sort-output reads (tile_expert_ids granularity, expert_offsets
// alignment, x_sorted slot layout) must match the forward's TileM.
// Out-param API instead of pointer-into-static-array because each candidate
// TunedConfigBwd is projected on the fly from kTunedConfigsFwdBwd (no stable
// address). Returns true when a valid config was found.
static bool find_nearest_tuned_bwd(int compute, int TK, int TKE, int D, int I,
                                   int fwd_tile_m, int n_pes,
                                   TunedConfigBwd* out) {
	// World-size class: single-GPU (n_pes <= 1) and multi-GPU tables are tuned
	// and stored separately (mirrors find_nearest_tuned on the fwd side).
	const TunedConfigFwdBwd* table = nullptr;
	int table_n = 0;
	if (!find_tuned_table_bwd(compute, n_pes, &table, &table_n)) return false;
	bool found = false;
	double best_dist = 1e30;
	double lTK  = std::log((double)TK);
	double lTKE = std::log((double)TKE);
	double lD   = std::log((double)D);
	double lI   = std::log((double)I);
	for (int i = 0; i < table_n; ++i) {
		TunedConfigBwd c = project_bwd(table[i], compute);
		// No fwd/bwd TileM match required: the comm tile is fixed at 128 for
		// both directions, so tile_expert_ids / x_sorted / expert_offsets align
		// regardless of each direction's GEMM tile. (fwd_tile_m retained for ABI
		// but no longer constrains the bwd GEMM-tile choice.)
		(void)fwd_tile_m;
		if (!tuned_config_valid_bwd(D, I, c)) continue;
		// Guard against TKE=0 in legacy rows (would yield -inf in log).
		double dTK  = std::log((double)c.TK)  - lTK;
		double dTKE = c.TKE > 0
			? std::log((double)c.TKE) - lTKE
			: 0.0;  // legacy / placeholder rows contribute 0 to the TKE axis
		double dD   = std::log((double)c.D)   - lD;
		double dI   = std::log((double)c.I)   - lI;
		double d    = dTK*dTK + dTKE*dTKE + dD*dD + dI*dI;
		if (d < best_dist) { best_dist = d; *out = c; found = true; }
	}
	return found;
}

// ============================================================================
// Auto-dispatch wrapper — pure tuned-table lookup
// ============================================================================
//
// Picks the nearest tuned config (log-space distance over (TK, D, I)) from
// moe_fwd_bwd_tuning_configs.cuh and dispatches to the matching template
// instantiation in kDispatchTableBwd. Both tables are populated by the
// tune_moe_bwd sweep — shapes outside the swept range error out instead of
// falling back to a heuristic.
//
// fwd_tile_m: BWD's TileM must match the forward's TileM. Forward sort
//   pads expert ranges to FWD's TileM, populates tile_expert_ids at that
//   granularity, and writes x_sorted in FWD-TileM-aligned slots. BWD
//   reads all three at the same granularity, so a mismatch corrupts both
//   the LocalIter expert lookup and the symmetric-memory bounds.
//   Callers must pass the FWD's TileM (= 128 for the default fwd path).
//
// Override path: LIGER_MOE_BWD_FORCE_CONFIG=NS2,TN1,TK1,S1,TM3,TN3,TK3,S3,EN1,EN25,EN34,CS,TM
// Lets a test suite sweep every compiled config without rebuilding.

// Internal auto-dispatch: nearest tuned bwd config for the runtime shape (or
// LIGER_MOE_BWD_FORCE_CONFIG), then invoke the matching template instantiation.
// All shape info comes from the MoeBwdArgs bundle; the launcher writes the grads.
void moe_bwd_dispatch(const MoeBwdArgs& a, int fwd_tile_m) {
	const int T  = a.num_tokens;
	const int D  = a.hidden_dim;
	const int I  = a.intermediate_dim;
	const int top_k = a.top_k;
	const int num_experts = a.num_experts;
	const int TK = T * top_k;
	const int n_pes = nvshmem_team_n_pes(a.team);
	const int E_local = (n_pes > 0) ? std::max(1, num_experts / n_pes) : num_experts;
	// Keep backward lookup consistent with forward for tiny token batches.
	const int TKE = std::max(1, TK / E_local);
	const int compute = moe_detect_compute_dispatch_key(a.device, "moe_fused_bwd_bf16_auto");

	if (const char* s = std::getenv("LIGER_MOE_BWD_FORCE_CONFIG")) {
		TunedConfigBwd tc{};
		tc.Compute = compute;
		tc.GemmTileM = -1;  // sentinel → default to comm TileM after parse
		int vals[14] = {};
		int n = std::sscanf(s, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
			&vals[0], &vals[1], &vals[2], &vals[3], &vals[4], &vals[5], &vals[6],
			&vals[7], &vals[8], &vals[9], &vals[10], &vals[11], &vals[12], &vals[13]);
		LIGER_CHECK(n == 13 || n == 14,
			"LIGER_MOE_BWD_FORCE_CONFIG must have 13 or 14 comma-separated ints "
			"(NS2,TN1,TK1,S1,TM3,TN3,TK3,S3,EN1,EN25,EN34,CS,TM[,GemmTM]); got '", s, "'");
		tc.NSplit2 = vals[0];
		tc.TileN1 = vals[1];
		tc.TileK1 = vals[2];
		tc.Stages1 = vals[3];
		tc.TileM3 = vals[4];
		tc.TileN3 = vals[5];
		tc.TileK3 = vals[6];
		tc.Stages3 = vals[7];
		tc.EpiChunkN1 = vals[8];
		tc.EpiChunkN25 = vals[9];
		tc.EpiChunkN34 = vals[10];
		tc.CommNumStages = vals[11];
		tc.TileM = vals[12];
		tc.GemmTileM = (n == 14) ? vals[13] : tc.TileM;
		if (tc.GemmTileM <= 0) tc.GemmTileM = tc.TileM;  // omitted → gemm == comm
		// No fwd/bwd TileM match check: comm tile is fixed at 128 for both
		// directions, so the sort/dispatch alignment holds for any GEMM tile.
		(void)fwd_tile_m;
		const DispatchEntryBwd* de = find_dispatch_entry_bwd(tc);
		LIGER_CHECK(de != nullptr,
			"LIGER_MOE_BWD_FORCE_CONFIG=", s, " is not in the compiled dispatch "
			"table. Add it to LIGER_MOE_BWD_TUNE_CONFIGS in moe_fwd_bwd_tune_configs.hpp.");
		de->fn(a, kBwdRuntimeNsplitSeed);
		return;
	}

	TunedConfigBwd tc{};
	LIGER_CHECK(find_nearest_tuned_bwd(compute, TK, TKE, D, I, fwd_tile_m, n_pes, &tc),
		"moe_bwd_fwd_bf16_auto: no valid tuned config for compute=", compute,
		" shape (T=", T,
		" K=", top_k, " TK=", TK, " TKE=", TKE, " D=", D, " I=", I,
		" fwd_tile_m=", fwd_tile_m,
		"). Either moe_fwd_bwd_tuning_configs.cuh has no entry with TileM=",
		fwd_tile_m, ", or every matching entry violates structural "
		"constraints for this (D, I). Add a compatible config to "
		"LIGER_MOE_BWD_TUNE_CONFIGS and re-tune.");

	const DispatchEntryBwd* de = find_dispatch_entry_bwd(tc);
	LIGER_CHECK(de != nullptr,
		"moe_bwd_fwd_bf16_auto: tuned config (Compute=", tc.Compute,
		" NS2=", tc.NSplit2,
		" TN1=", tc.TileN1, "/", tc.TileK1, "/", tc.Stages1,
		" TM3=", tc.TileM3, " TN3=", tc.TileN3, "/", tc.TileK3,
		"/", tc.Stages3,
		" EN1=", tc.EpiChunkN1,
		" EN25=", tc.EpiChunkN25, " EN34=", tc.EpiChunkN34,
		" CS=", tc.CommNumStages, " TM=", tc.TileM,
		") for shape (T=", T, " K=", top_k, " TK=", TK, " D=", D, " I=", I,
		") is not in the compiled dispatch table. "
		"Add it to LIGER_MOE_BWD_TUNE_CONFIGS in moe_fwd_bwd_tune_configs.hpp.");

	de->fn(a, kBwdRuntimeNsplitSeed);
}

// Returns the compiled bwd config table as a list of 13-tuples
// (Compute, NSplit2, TileN1, TileK1, Stages1, TileM3, TileN3, TileK3,
//  Stages3, EpiChunkN1, EpiChunkN25, EpiChunkN34, CommNumStages).
std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int>>
moe_bwd_list_compiled_configs() {
	std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int>> out;
	out.reserve(kNumDispatchEntriesBwd);
	for (int i = 0; i < kNumDispatchEntriesBwd; ++i) {
		const auto& e = kDispatchTableBwd[i];
		out.emplace_back(e.Compute, e.NSplit2,
			e.TileN1, e.TileK1, e.Stages1,
			e.TileM3, e.TileN3, e.TileK3, e.Stages3,
			e.EpiChunkN1, e.EpiChunkN25, e.EpiChunkN34,
			e.CommNumStages);
	}
	return out;
}

} // namespace liger
