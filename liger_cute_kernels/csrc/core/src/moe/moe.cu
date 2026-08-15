// Fused MoE: sort+dispatch → MLP+combine — torch-free core (liger_cute).
//
// Ported from LigerCommKernels' moe.cu. The device kernels and the templated
// host launcher are upstream-verbatim (namespace liger, internal to the .so);
// only the boundary changed: the remaining public entry points here are flat
// extern "C" control/configuration APIs. Tensor-carrying MoE calls are exposed
// through TVM FFI and lower directly into the internal MoeFwdArgs/MoeBwdArgs
// launch bundles. torch::Tensor never appears here.
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

#include "liger_cute/moe.h"
#include "liger_cute/check.h"
#include "liger_cute/detail/status.h"
#include "liger_cute/detail/comm_schedule.cuh"

#include "moe.cuh"
#include "moe_symm_config.cuh"
#include "moe_launch.h"         // MoeFwdArgs / MoeBwdArgs (shared with the tuner)
#include "moe_utils.cuh"
#include "moe_fwd_bwd_tune_configs.hpp"
#include "moe_dispatch_configs_sm90.cuh"
#include "moe_dispatch_configs_sm100.cuh"
#include "tma_copy_atom.cuh"
#include "radix_sort.cuh"
#include "dispatch.cuh"
#include "combine.cuh"
#include "buffer_pool.cuh"      // shim: liger:: aliases of liger_cute::detail infra
#include "fused_memset.cuh"

using namespace cute;

namespace liger {

// 0 = build every dispatch family (developer fallback). CMake sets this to 90
// for Hopper builds and 100 for Blackwell builds so normal native builds do not
// instantiate kernels for hardware that cannot run in that binary.
#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

// init_comm_schedule lives in liger_cute::detail (comm_schedule.cuh); the host
// configure path below calls it through this alias.
using liger_cute::detail::init_comm_schedule;

// ============================================================================
// Global iteration counter for PeSync
// ============================================================================
//
// PeSync's dispatch_done / mlp_done symmetric arrays must increase
// monotonically across calls (the wait-side uses NVSHMEM_CMP_GE). The
// iteration value is what each call SETs the signal to via
// nvshmemx_signal_op(SET).
//
// The counter lives in device memory (allocated by BufferPool as
// "moe_fwd_iter_counter") and is atomicAdded at the end of every
// kernel by CTA 0 thread 0 — see PeSync::advance_iteration_counter().
// This is the form that survives CUDA graph capture: a host-side
// atomic would be read once on the host before the kernel launch and
// baked into the captured graph, so all replays would re-SET the same
// already-met value and the cross-PE wait would race; a GPU-resident
// counter incremented inside the captured kernel re-evaluates on
// every replay, giving each launch a fresh monotonic target.
//
// Buffer pool keys "moe_fwd_iter_counter" (fwd) and
// "moe_bwd_iter_counter" (bwd) name the two separate counters.

// ============================================================================
// MoeFusedConfig — bundles all tunable kernel parameters
// ============================================================================
//
// TileM is fixed at 64 by the WGMMA hardware.
// NumThreads is fixed at 384 (10 MLP warps + 2 comm warps).
// All other parameters are exposed for autotuning.

template <
    typename Element_   = bfloat16_t,
    int TileM_          = 128,
    int TileN1_         = 128,
    int TileK1_         = 64,
    int Stages1_        = 4,
    int EpiChunkN1_     = 64,
    int TileN2_         = 256,
    int TileK2_         = 64,
    int Stages2_        = 4,
    int EpiChunkN2_     = 64,
    int ZBufferSlots_   = 4,
    int CommNumStages_  = 2,
    int NC_             = 4,
    // GemmTileM_ decouples the GEMM compute tile (mlp1/mlp2 wgmma + TMA) from
    // the COMM staging tile (TileM_). TileM_ is the comm tile — staging ring
    // slots, per-tile semaphores, NC, sort/dispatch — and is fixed at 128.
    // GemmTileM_ ∈ {64,128}: when < TileM_, the GEMM waits on the 128-wide comm
    // semaphore once, then steps through kSubTiles = TileM_/GemmTileM_ sub-tiles
    // of GemmTileM_ rows each (finer TMA + wgmma) within that one comm slot.
    // Default GemmTileM_ = TileM_ → kSubTiles = 1 → bit-identical to the old
    // single-TileM path.
    int GemmTileM_      = TileM_,
    int Compute_        = 90>
struct MoeFusedConfig {
    using Element = Element_;
    // GEMM tile (mlp1/mlp2): GemmTileM_ ∈ {64,128}, independent of the comm tile.
    using Traits1 = Mlp1Traits<
	    Element, GemmTileM_, TileN1_, TileK1_, Stages1_, EpiChunkN1_>;
    using Traits2 = Mlp2Traits<Element, GemmTileM_, TileN2_, TileK2_, Stages2_, EpiChunkN2_>;

    // No compile-time constraint between NC, NSplit and CommNumStages
    // — the GEMM-side iterator is ticket-based (slot = T mod L,
    // ticket = T / L) and synchronizes per-slot via raw atomics
    // computed from the global tile index, so the only requirement
    // (runtime) is that NC evenly divides gridDim.x · gridDim.y so MC
    // is integer. CommNumStages is now a free buffering knob.

    // Compile-time smem budget. H100 dynamic-smem cap is 228 KiB; we leave
    // ~4 KiB for the runtime-sized remote_offsets region appended by the
    // launcher (see moe.cu: smem_mlp += num_pes * (experts_per_pe+1) * 4).
    // Catches over-budget configs at link time instead of cudaErrorInvalidValue
    // at launch.
    static constexpr size_t kSmemBytes  = sizeof(MoeSmem<Traits1, Traits2, Compute_>);
    static constexpr size_t kSmemBudget = 224 * 1024;
    static_assert(kSmemBytes <= kSmemBudget,
        "MoE fused fwd kernel smem footprint exceeds 224 KiB budget — "
        "reduce TileN/TileK/Stages for this config");

    static constexpr int kNC            = NC_;
    static constexpr int kZBufferSlots  = ZBufferSlots_;
    static constexpr int kCommNumStages = CommNumStages_;
    // kTileM is the COMM tile (staging ring slots, per-tile semaphores, NC,
    // sort/dispatch granularity). Fixed at 128 for the decoupled path.
    static constexpr int kTileM         = TileM_;          // comm tile (128)
    static constexpr int kGemmTileM     = GemmTileM_;      // gemm tile ∈ {64,128}
    static constexpr int kSubTiles      = TileM_ / GemmTileM_;  // 1 or 2
    static constexpr int kCompute       = Compute_;
    static constexpr int kNumThreads    = 384;             // fixed

    static_assert(TileM_ % GemmTileM_ == 0,
        "CommTileM must be an integer multiple of GemmTileM");
    static_assert(kSubTiles == 1 || kSubTiles == 2,
        "kSubTiles (CommTileM/GemmTileM) must be 1 or 2");
    static_assert(GemmTileM_ == 64 || GemmTileM_ == 128,
        "GemmTileM must be 64 or 128");

    // Iterators operate on the COMM tile (kTileM): staging slots / semaphores
    // are 128-wide regardless of the GEMM sub-tile size.
    using LocalIter  = LocalMlpTileIterator<Element, kTileM>;
    using RemoteIter = RemoteMlpTileIterator<Element, kCommNumStages, kNC, kTileM>;
};

// ============================================================================
// MoeParams -- sort + dispatch + MLP + combine (single kernel)
// ============================================================================

template <typename Config>
struct MoeParams {
	using Element = typename Config::Element;

	// Router outputs (read-only, from caller's tensors).
	const int* expert_indices;      // [T, K]
	const Element* expert_weights;  // [T, K]

	// Sort buffers.
	SortBuffers sort;

	// Dispatch.
	const Element* tokens;         // [T, D] input (symmetric)
	Element* dispatch_dst;         // [total_slots, D] sorted output (symmetric)

	// Combine output.
	Element* combine_output;       // [T, D] final output

	// MLP + comm.
	CommBuffers comm;

	// Dimensions.
	int num_tokens;
	int intermediate_dim;
	int top_k;
	int total_slots;
	int num_m_tiles;               // sort input tiles
};

// ============================================================================
// MoeBuffers -- all device/symmetric allocations
// ============================================================================

template <typename Config>
struct MoeBuffers {
	using Element = typename Config::Element;

	// MLP intermediate Z.
	Element* z_buf;
	int z_rows;

	// Sorted X (dispatch destination) -- symmetric, remote PEs get from here.
	Element* x_sorted;

	// Output Y -- symmetric, local MLP writes here, remote PEs put here.
	Element* y_buf;

	// Phase counters.
	int* phase_counter;

	// Comm staging.
	Element* src_staging;
	Element* dst_staging;
	int* src_ready;
	int* src_consumed;
	int* dst_ready;
	int* dst_consumed;
	int* tile_expert_ids;

	// All PEs' expert offsets -- symmetric.
	int* all_expert_offsets;
	int* cta_counter;             // [1] device memory — shared monotonic

	// Sort buffers.
	int* tile_expert_counts;
	int* sorted_token_ids;
	int* token_expert_slots;
	int* expert_offsets;       // symmetric
	int* cta_done;
	int* cta_sums;

	// Per-tile expert IDs for MLP local iterator.
	int* mlp_tile_expert_ids;
};

// MoeSymmConfig + get_symm_config() declaration live in the shared header so
// moe.cu and moe_bwd.cu can't disagree on the layout (see header). This TU
// owns the single instance.
MoeSymmConfig& get_symm_config() {
	static MoeSymmConfig cfg;
	return cfg;
}

void moe_configure_symmetric(
		int max_tokens,
		int hidden_dim,
		int max_num_experts,
		int max_top_k,
		int num_pes,
		int num_hosts,
		int gpus_per_host) {
	static constexpr int kTileM = 128;  // fixed by WGMMA

	// num_hosts × gpus_per_host must equal the team size so the comm
	// schedule (g_dest_table / g_rank_table) addresses exactly the PEs
	// in the team. Mismatch would silently map ranks to non-existent
	// destinations (or skip some).
	LIGER_CHECK(num_hosts * gpus_per_host == num_pes,
		"moe_configure_symmetric: num_hosts (", num_hosts, ") * gpus_per_host (",
		gpus_per_host, ") = ", num_hosts * gpus_per_host,
		" must equal num_pes (", num_pes, ")");

	// Populate g_dest_table / g_rank_table in device constant memory.
	// Idempotent — safe to re-call with the same (N, M).
	init_comm_schedule(num_hosts, gpus_per_host);

	auto& cfg = get_symm_config();
	cfg.max_total_slots = max_tokens * max_top_k + max_num_experts * kTileM;
	cfg.max_num_experts = max_num_experts;
	cfg.hidden_dim = hidden_dim;
	cfg.num_pes = num_pes;
	cfg.experts_per_pe = max_num_experts / num_pes;
	cfg.max_top_k = max_top_k;
	cfg.initialized = true;

	// Preset sizes for the symmetric stack used to hold X / Y / offsets
	// across fwd → bwd. Element type is bf16 (= 2 bytes); kept in sync
	// with allocate_moe_buffers below.
	constexpr std::size_t kElemBytes = 2;  // sizeof(__nv_bfloat16)
	std::size_t symm_slot_bytes =
		static_cast<std::size_t>(cfg.max_total_slots) * cfg.hidden_dim * kElemBytes;
	std::size_t all_off_bytes =
		static_cast<std::size_t>(cfg.num_pes) * (cfg.max_num_experts + 1) * sizeof(int);

	auto& stack = global_symmetric_stack();
	stack.set_size(kSymmKeyXSorted,          symm_slot_bytes);
	stack.set_size(kSymmKeyYBuf,             symm_slot_bytes);
	stack.set_size(kSymmKeyAllExpertOffsets, all_off_bytes);
}

// Releases the x_sorted / y_buf / all_expert_offsets entries pushed onto the
// symmetric stack by the most recent fwd call. Must be called collectively
// on every PE in LIFO order vs. the fwd that produced them. After this,
// the buffers can be reused by a subsequent fwd. The torch::Tensor views
// returned by fwd remain valid pointers but will alias whatever lives in
// those slots after the next put.
void moe_pop_fwd() {
	auto& stack = global_symmetric_stack();
	// LIFO order: bottom pushes first. Reverse the put order in
	// allocate_moe_buffers (x_sorted, y_buf, all_expert_offsets).
	stack.pop(kSymmKeyAllExpertOffsets);
	stack.pop(kSymmKeyYBuf);
	stack.pop(kSymmKeyXSorted);
}

template <typename Config>
static MoeBuffers<Config> allocate_moe_buffers(
		BufferPool& pool,
		cudaStream_t stream,
		int grid_x,
		int num_blocks,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int top_k,
		int num_experts,
		int experts_per_pe,
		int num_pes,
		int total_slots,
		int num_m_tiles,
		// Caller-owned torch storage for the two sort outputs that bwd
		// needs to consume. Passed in so we don't lease buffer-pool
		// slots that the next fwd would overwrite.
		int* token_expert_slots_ptr,
		int* mlp_tile_expert_ids_ptr) {

	using Element = typename Config::Element;
	static constexpr int kTileM         = Config::kTileM;
	static constexpr int kZBufferSlots  = Config::kZBufferSlots;
	static constexpr int kCommNumStages = Config::kCommNumStages;
	static constexpr int kNC            = Config::kNC;

	// Staging is a flat ring of L = MC · kCommNumStages slots, where
	// MC = num_blocks / NC (num_blocks == launched gridDim.x). This MUST equal
	// the device-side comm producer's stride (mlp_comms.cuh: MC = n_comm/NC,
	// n_comm = (gridDim.x/NC)·NC) AND the MLP consumer's ring stride
	// (RemoteMlpTileIterator::init: mc = gridDim.x/NC), so every tile's
	// slot = g % L agrees across producer / consumer / allocation. Sizing this
	// on grid_x·NSplit (= n_gemm) instead under-sizes the signal arrays whenever
	// gridDim.x/NC > n_gemm/NC — i.e. num_sms % NSplit != 0, e.g. NSplit=8 on
	// 132 SMs (33 vs 32) — and the gridDim.x-based ring then indexes past the
	// end of src_ready / tile_expert_ids (OOB). num_blocks/NC == gridDim.x/NC
	// == n_gemm/NC in the divisible (legacy) case, so this is a no-op there.
	// Comm strides MC through NumStages per xc; MLP (ticket iterator) wraps mod L.
	int mc = num_blocks / kNC;
	int comm_l = mc * kCommNumStages;
	// Raw SM count — used as the device-independent upper bound on the comm
	// staging row count (mc <= sm_count) so the symmetric staging size is a
	// worst-case bound across configs (see staging sizing below).
	int sm_count = 0;
	{
		int dev = 0;
		cudaGetDevice(&dev);
		cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
	}

	MoeBuffers<Config> b;

	// Symmetric allocations use max sizes from MoeSymmConfig
	// so all PEs allocate the same sizes regardless of per-call config.
	auto& scfg = get_symm_config();
	LIGER_CHECK(scfg.initialized, "Call moe_configure_symmetric before moe_fused_fwd_bf16");

	// Persistent symmetric tensors (x_sorted, y_buf, all_expert_offsets) live
	// in the symmetric stack so they can survive across fwd calls into bwd.
	auto& sym_stack = global_symmetric_stack();

	// Z buffer (device).
	// size_t cast on b.z_rows: at NSplit=2 ZBuf=16 I=16384, the product
	// z_rows · intermediate_dim · 2 = ~4.4 GB which overflows int32 if
	// the multiply happens in int (b.z_rows and intermediate_dim are
	// both int). Cast first so the entire product is in size_t.
	int max_runtime_grid_x = (num_blocks + 1) / 2;  // min candidate NS = 2
	int z_m_tiles = max_runtime_grid_x * kZBufferSlots;
	b.z_rows = z_m_tiles * kTileM;
	size_t z_bytes = static_cast<size_t>(b.z_rows) * intermediate_dim * sizeof(Element);
	b.z_buf = reinterpret_cast<Element*>(pool.get_device("moe_z_buf", z_bytes));

	// Sorted X -- symmetric, dispatch writes here, remote PEs get from here.
	b.x_sorted = reinterpret_cast<Element*>(sym_stack.put(kSymmKeyXSorted));

	// Output Y -- symmetric, local MLP writes here, remote PEs put here.
	b.y_buf = reinterpret_cast<Element*>(sym_stack.put(kSymmKeyYBuf));
	size_t actual_y_bytes = static_cast<size_t>(total_slots) * hidden_dim * sizeof(Element);
	if (cudaError_t e = cudaMemsetAsync(b.y_buf, 0, actual_y_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(y_buf) failed: ", cudaGetErrorString(e));

	// Phase counters.
	size_t counter_bytes = max_runtime_grid_x * sizeof(int);
	b.phase_counter = reinterpret_cast<int*>(
		pool.get_device("moe_phase_counter", counter_bytes));
	if (cudaError_t e = cudaMemsetAsync(b.phase_counter, 0, counter_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(phase_counter) failed: ", cudaGetErrorString(e));

	// Flat staging list — L = grid_x * kCommNumStages slots, each
	// holding TileM rows × hidden_dim columns.
	// Size by the CONFIGURED MAX on every axis (scfg.hidden_dim, max_comm_stages,
	// max_tile_m), NOT the per-call/per-config values: the symmetric pool caches
	// by name and aborts on grow, so a small first config must not lock in a
	// staging buffer a later larger config can't grow into. This covers all three
	// independently-varying axes that the tuner sweeps:
	//   - hidden_dim:    D=512 then D=8192
	//   - CommNumStages: CS=2 then CS=8 (NS=16 config)
	//   - TileM:         64 then 128
	// The kernel indexes slots by the per-call (comm_l, tile_elems <= max), so an
	// over-sized buffer is always safe (symmetric memory may be larger than
	// needed, never smaller).
	size_t staging_rows_ub =
		(size_t)sm_count * scfg.max_comm_stages * scfg.max_tile_m;
	size_t src_bytes = staging_rows_ub * scfg.hidden_dim * sizeof(Element);
	size_t dst_bytes = staging_rows_ub * scfg.hidden_dim * sizeof(Element);
	b.src_staging = reinterpret_cast<Element*>(pool.get_symmetric("moe_src_staging", src_bytes));
	b.dst_staging = reinterpret_cast<Element*>(pool.get_symmetric("moe_dst_staging", dst_bytes));

	// Stage pipe signals — one entry per flat slot.
	size_t sig_bytes = (size_t)comm_l * sizeof(int);
	b.src_ready    = static_cast<int*>(pool.get_device("moe_src_ready",    sig_bytes));
	b.src_consumed = static_cast<int*>(pool.get_device("moe_src_consumed", sig_bytes));
	b.dst_ready    = static_cast<int*>(pool.get_device("moe_dst_ready",    sig_bytes));
	b.dst_consumed = static_cast<int*>(pool.get_device("moe_dst_consumed", sig_bytes));
	if (cudaError_t e = cudaMemsetAsync(b.src_ready, 0, sig_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(src_ready) failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaMemsetAsync(b.src_consumed, 0, sig_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(src_consumed) failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaMemsetAsync(b.dst_ready, 0, sig_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(dst_ready) failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaMemsetAsync(b.dst_consumed, 0, sig_bytes, stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(dst_consumed) failed: ", cudaGetErrorString(e));

	// Per-slot expert IDs — one entry per flat slot.
	size_t eid_bytes = (size_t)comm_l * sizeof(int);
	b.tile_expert_ids = static_cast<int*>(pool.get_device("moe_tile_expert_ids", eid_bytes));

	// All PEs' expert offsets (symmetric). Use max experts for stable sizing.
	b.all_expert_offsets = static_cast<int*>(sym_stack.put(kSymmKeyAllExpertOffsets));

	// CTA counter for local all-CTA barriers around NVSHMEM collectives.
	b.cta_counter = static_cast<int*>(
		pool.get_device("moe_cta_counter", sizeof(int)));

	// Sort buffers -- use scfg max sizes to avoid reallocation across configs.
	int max_sorted_slots = scfg.max_total_slots;
	int max_m_tiles = scfg.max_total_slots / kTileM;
	int max_tile_counts = max_m_tiles * scfg.max_num_experts;
	int max_token_slots_ub = scfg.max_total_slots;

	(void)max_token_slots_ub;
	b.tile_expert_counts = static_cast<int*>(
		pool.get_device("moe_sort_tile_counts", max_tile_counts * sizeof(int)));
	b.sorted_token_ids = static_cast<int*>(
		pool.get_device("moe_sort_sorted_ids", max_sorted_slots * sizeof(int)));
	// token_expert_slots: caller-owned torch tensor (passed in). Used by
	// the bwd path after this call returns — must not alias a pool slot.
	b.token_expert_slots = token_expert_slots_ptr;
	b.expert_offsets = static_cast<int*>(
		pool.get_symmetric("moe_sort_offsets", (scfg.max_num_experts + 1) * sizeof(int)));
	b.cta_done = static_cast<int*>(
		pool.get_device("moe_sort_cta_done", num_blocks * sizeof(int)));
	b.cta_sums = static_cast<int*>(
		pool.get_device("moe_sort_cta_sums", num_blocks * scfg.max_num_experts * sizeof(int)));

	// Per-tile expert IDs for MLP — caller-owned torch tensor.
	b.mlp_tile_expert_ids = mlp_tile_expert_ids_ptr;

	if (cudaError_t e = cudaMemsetAsync(b.cta_done, 0, num_blocks * sizeof(int), stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(cta_done) failed: ", cudaGetErrorString(e));
	if (cudaError_t e = cudaMemsetAsync(b.sorted_token_ids, 0xff,
			max_sorted_slots * sizeof(int), stream); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaMemsetAsync(sorted_token_ids) failed: ", cudaGetErrorString(e));

	return b;
}

// ============================================================================
// GET-via-TMA descriptor builder (raw CUtensorMap, FWD-only)
// ============================================================================
//
// Box = [box_rows, kGetKChunk] over a row-major [n_rows, hidden_dim] bf16
// tensor. SWIZZLE_NONE keeps the global bytes row-major, so the MLP's
// swizzled reader (tma_load_x_remote) sees correct data. Returns false on
// encode failure → caller disables the TMA-get path (falls back to getmem).
static inline bool make_get_tma_map(CUtensorMap& m, void* base,
		int hidden_dim, int n_rows, int box_rows) {
	uint64_t gdim[2]    = { (uint64_t)hidden_dim, (uint64_t)n_rows };
	uint64_t gstride[1] = { (uint64_t)hidden_dim * sizeof(bfloat16_t) };
	uint32_t bdim[2]    = { (uint32_t)liger::kGetKChunk, (uint32_t)box_rows };
	uint32_t estride[2] = { 1, 1 };
	CUresult r = cuTensorMapEncodeTiled(
		&m, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, base, gdim, gstride, bdim, estride,
		CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
		CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
	return r == CUDA_SUCCESS;
}

// ============================================================================
// Fused kernel: sort → dispatch → MLP + combine
// ============================================================================

template <typename Config,
          typename TmaLoadX, typename TmaLoadW1, typename TmaStoreZ,
          typename TmaLoadZ, typename TmaLoadW2, typename TmaStoreY,
          typename TmaStoreYCombine>
__global__ void __launch_bounds__(Config::kNumThreads)
moe_fused_kernel(
		__grid_constant__ TmaLoadX  const tma_load_x,
		__grid_constant__ TmaLoadW1 const tma_load_b,
		__grid_constant__ TmaLoadW1 const tma_load_c,
		__grid_constant__ TmaStoreZ const tma_store_z,
		__grid_constant__ TmaLoadZ  const tma_load_z,
		__grid_constant__ TmaLoadW2 const tma_load_a,
		__grid_constant__ TmaStoreY const tma_store_y,
		__grid_constant__ TmaLoadX  const tma_load_x_remote,
		__grid_constant__ TmaStoreY const tma_store_y_remote,
		__grid_constant__ TmaStoreYCombine const tma_store_y_combine,
		__grid_constant__ MlpDims   const mlp_dims,
		__grid_constant__ MoeParams<Config> const p,
		__grid_constant__ int       const static_nsplit,
		__grid_constant__ GetTmaDescs const get_descs,
		__grid_constant__ PeerYStoreDescs<TmaStoreY> const peer_y) {

	using Element       = typename Config::Element;
	using Traits1       = typename Config::Traits1;
	using Traits2       = typename Config::Traits2;
	using LocalIter     = typename Config::LocalIter;
	using RemoteIter    = typename Config::RemoteIter;
	constexpr int kTileM         = Config::kTileM;
	constexpr int kNumThreads    = Config::kNumThreads;
	constexpr int kZBufferSlots  = Config::kZBufferSlots;
	constexpr int kCommNumStages = Config::kCommNumStages;
	constexpr int kNC            = Config::kNC;

	// -- Sort ----------------------------------------------------------------
	{
		extern __shared__ int sort_smem[];
		sort_tokens<kTileM, kNumThreads>(
			sort_smem, p.expert_indices, p.sort,
			p.num_tokens, p.comm.num_experts, p.top_k, p.num_m_tiles);
	}
	__threadfence_system();  // device-scope (NVLink-only build; IB needs system scope)
	__syncthreads();

	// Target-based cross-CTA barrier — single instance per kernel call,
	// reused by every PeSync::barrier / signal_dispatch / signal_mlp
	// invocation below. Constructing a fresh barrier per call resets
	// the per-CTA target to 0 while the shared device counter keeps
	// growing → fast CTAs spin past the next-call target instantly.
	SyncThreadsCtaCounterBarrier pe_barrier(p.comm.pe_sync.cta_counter,
		(int)gridDim.x * (int)gridDim.y);

	// Cross-CTA barrier: sort writes sorted_token_ids across CTAs;
	// dispatch and combine read them from other CTAs.
	p.comm.pe_sync.barrier(pe_barrier);

	// -- Broadcast expert offsets to all PEs ----------------------------------
	broadcast_expert_offsets(p.comm);
	__syncthreads();

	// -- Dispatch -------------------------------------------------------------
	dispatch_tokens<Element, kTileM, kNumThreads>(
		p.tokens, p.dispatch_dst,
		p.sort.sorted_token_ids,
		p.comm.hidden_dim, p.total_slots);
	__syncthreads();

	// H2 (IB coherence): dispatch_tokens wrote dispatch_dst (= x_sorted) with
	// plain GPU stores; remote PEs RDMA-read it in do_get after wait_dispatch.
	// Flush those stores to HBM so the remote NIC reads fresh data — the quiet
	// below only orders this PE's OUTGOING nvshmem ops, not these local
	// stores, and __syncthreads is device-scope. The offset-broadcast source
	// is already covered by radix_sort's __threadfence_system. NVLink P2P
	// reads are L2-coherent, so single-host never exercised this.
	__threadfence_system();  // device-scope (NVLink-only build; IB needs system scope)

	p.comm.pe_sync.barrier_all(pe_barrier);

	// -- MLP (local + remote) ------------------------------------------------
	{
		extern __shared__ char raw_smem[];
		auto& smem = *reinterpret_cast<MoeSmem<Traits1, Traits2, Config::kCompute>*>(raw_smem);

		size_t offsets_region = sizeof(MoeSmem<Traits1, Traits2, Config::kCompute>);
		offsets_region = (offsets_region + sizeof(int) - 1) & ~(sizeof(int) - 1);
		smem.comm.remote_offsets = reinterpret_cast<int*>(raw_smem + offsets_region);

		// GET-via-TMA bounce region: placed after remote_offsets, 128-aligned
		// (TMA dst smem must be 128B-aligned). MUST match the host smem_size
		// computation (same sizeof(MoeSmem), num_pes, experts_per_pe). Only
		// touched by the comm warps' do_get when get_descs.enabled.
		char* get_bounce = nullptr;
		if (get_descs.enabled) {
			size_t boff = offsets_region
				+ (size_t)p.comm.num_pes() * (p.comm.experts_per_pe + 1) * sizeof(int);
			boff = (boff + 127) & ~((size_t)127);
			get_bounce = raw_smem + boff;
		}

		//if (threadIdx.x == 0)
		//	smem.comm.total_tiles = -1;

		// Flat-grid logical coordinates for this CTA. GEMM and comm gate to
		// the first n_gemm = floor_NS CTAs (the comm↔MLP staging ring needs
		// grid_x = n_gemm/static_nsplit and MC = n_gemm/NC consistent); the gap CTAs
		// [n_gemm, num_blocks) fall through to the shared pe_barrier only.
		int flat_id      = blockIdx.x;
		int n_gemm       = p.comm.n_gemm;
		int grid_x       = n_gemm / static_nsplit;
		int col          = flat_id / static_nsplit;
		int runtime_nsplit = static_nsplit;

		// gemm_active removed: comm warps self-gate on their precomputed tile
		// count (smem.comm.total_tiles) so a CTA can do comm-only work without
		// participating in GEMM (the point of the comm/GEMM grid relaxation).
		// Run the prologue unconditionally so every launched CTA has total_tiles
		// initialized; the MLP/GEMM path still gates on GEMM-grid membership
		// (flat_id < n_gemm) below.
		nvshmem_comm_prologue<kTileM, kNC>(smem.comm, p.comm, static_nsplit);
		__syncthreads();

		if (threadIdx.x == 0) {
			int total_m_tiles = smem.comm.global_total;
			runtime_nsplit = select_runtime_nsplit_from_tiles(
				total_m_tiles,
				mlp_dims.num_n_tiles_1,
				mlp_dims.num_n_tiles_2,
				mlp_dims.num_k_tiles_1,
				mlp_dims.num_k_tiles_2,
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

		// Diverge: comm warps handle NVSHMEM, MLP warps handle compute.
		// Separate scopes prevent CommBuffers from inflating MLP register budget.
		// Fwd uses warps 1-3 (2 getters + 1 putter).
		int warp_id = threadIdx.x / 32;
		bool is_comm_warp = (warp_id == kCommGetWarp || warp_id == kCommPutWarp);

		if (is_comm_warp) {
			// Comm warps: dedicated get/put — full CommBuffers access. Self-gate
			// on the precomputed comm tile count — no GEMM coupling, so a
			// comm-only CTA (no local GEMM work) still drives its comm tiles.
			if (smem.comm.total_tiles > 0) {
				nvshmem_comm_main<Element, kTileM, kCommNumStages, kNC>(
					smem.comm, p.comm, &get_descs, get_bounce);
			}
		} else if (flat_id < n_gemm) {
			// MLP warps: no CommBuffers in scope. Gap CTAs (flat_id >= n_gemm)
			// skip this entirely and fall through to the pe_barrier below.
			int local_e_start  = p.comm.my_pe() * p.comm.experts_per_pe;
			int local_e_end    = local_e_start + p.comm.experts_per_pe;
			int local_offset   = p.sort.expert_offsets[local_e_start];
			int local_end      = p.sort.expert_offsets[local_e_end];
			int local_m_tiles  = (local_end - local_offset) / kTileM;
			int local_tile_off = local_offset / kTileM;

			LocalIter local_iter;
			local_iter.init(
				p.dispatch_dst + local_offset * p.comm.hidden_dim,
				p.sort.tile_expert_ids + local_tile_off,
				local_m_tiles, p.comm.hidden_dim,
				/*col_stride=*/grid_x, /*start_col=*/col, local_tile_off);

			moe_fused_fwd<Traits1, Traits2, kZBufferSlots, kCommNumStages, kNC,
				Config::kSubTiles, Config::kCompute, LocalIter, RemoteIter>(
				smem, local_iter,
				p.comm.src_ready, p.comm.src_consumed,
				p.comm.dst_ready, p.comm.dst_consumed,
				p.comm.experts_per_pe, p.comm.num_pes(), p.comm.my_pe(),
				get_descs.gpus_per_node,
				tma_load_x, tma_load_b, tma_load_c, tma_store_z,
				tma_load_z, tma_load_a, tma_store_y,
				tma_load_x_remote, tma_store_y_remote,
				mlp_dims,
				local_m_tiles * kTileM, local_m_tiles,
				col, grid_x, /*split=*/flat_id % runtime_nsplit, runtime_nsplit,
				local_e_start,
				/*y_peer_desc=*/peer_y.enabled ? peer_y.desc : nullptr);
		}

		__threadfence();
		__syncthreads();
		p.comm.pe_sync.barrier_all(pe_barrier);
	}

	// -- Combine --------------------------------------------------------------
	// Reuse the kernel's dynamic SMEM region as CombineSmem. wait_mlp above
	// is a cross-CTA + cross-PE barrier that ends in its own __syncthreads, so
	// the CTA is already aligned and the MLP/Comm SMEM contents are dead by
	// this point — no extra __syncthreads needed before the reinterpret.
	{
		constexpr int kCombineNumWGs =
			kNumThreads / (kCombineWarpsPerWG * kCombineWarpSize);
		extern __shared__ char raw_smem[];
		auto& combine_smem =
			*reinterpret_cast<CombineSmem<Element, kCombineNumWGs>*>(raw_smem);
		combine_tokens_tma<Element, kNumThreads, TmaStoreYCombine>(
			reinterpret_cast<const Element*>(p.comm.local_output),
			p.expert_weights,
			p.sort.token_expert_slots,
			tma_store_y_combine,
			combine_smem,
			p.comm.hidden_dim,
			p.num_tokens,
			p.top_k);
	}

}

// ============================================================================
// Host wrapper
// ============================================================================

// MoeFwdArgs (the POD launch bundle) is defined in moe_launch.h, shared with the
// tuner. The runtime dispatch table holds a single function-pointer type
// (MoeFwdFn) while every tuned config is a distinct template instantiation.

template <
    typename Element_  = bfloat16_t,
    int TileN1         = 128,
    int TileK1         = 64,
    int Stages1        = 4,
    int TileN2         = 256,
    int TileK2         = 64,
    int Stages2        = 4,
    int ZBufferSlots   = 4,
    int CommNumStages  = 1,
    int EpiChunkN1     = 64,   // mlp1 epilogue chunk; must divide TileN1
    int EpiChunkN2     = 64,   // mlp2 epilogue chunk; must divide TileN2
    int TileM          = 128,  // COMM tile (staging/semaphores/NC); 128
    int GemmTileM      = TileM, // GEMM tile ∈ {64,128}; default == comm (no sub-tiling)
    int Compute        = 90>
void
moe_fused_fwd_bf16(const MoeFwdArgs& a, int static_nsplit) {

	// Unpack the POD argument bundle into the names the upstream body uses.
	const int num_experts = a.num_experts;
	const int top_k       = a.top_k;
	const nvshmem_team_t team = a.team;
	cudaStream_t stream   = a.stream;

	// NC = number of cooperating comm CTAs per (xc, yc) tile. With the
	// ticket-based GEMM-side iterator, runtime NS and NC no longer need to
	// share a divisibility relationship — the only runtime requirement
	// is NC | (gridDim.x · gridDim.y) so MC is integer.
	//
	// Wrapper param order (10 "core" args + trailing EpiChunkN + TileM):
	//   Element, TN1, TK1, S1, TN2, TK2, S2, ZBuf, CS,
	//   EpiChunkN1, EpiChunkN2, TileM
	// MoeFusedConfig interleaves TileM/EpiChunkN inside each phase:
	//   Element, TileM, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2,
	//   ZBuf, CS, NC
	// The remapping happens here. TileM is shared by MLP1 and MLP2.
	//
	// NC choice (verified via bench_mlp_comms BW sweep):
	//   - TileM=128: NC=4 gives ~2× the comm bandwidth of NC=2 because
	//     4 cooperating comm CTAs split each tile, deepening the comm
	//     pipeline (more in-flight slots = grid.x/NC).
	//   - TileM=64:  NC=2 already matches NC=4 throughput because tiles
	//     are half-size, so twice as many flow through the same pipeline.
	// COMM tile is FIXED at 128 (NC=4) for EVERY config. This keeps the
	// all-to-all dispatch padding (x_sorted / expert_offsets, padded to the
	// comm tile) uniform across PEs — required for correctness and what makes
	// ragged-token auto-dispatch work (a per-PE comm tile would misalign the
	// cross-PE get/put). `TileM` is now purely the GEMM-tile knob: it flows to
	// GemmTileM (which defaults to TileM), so a "TM64" config = comm128/gemm64.
	using Config = MoeFusedConfig<
		Element_, /*CommTileM=*/128,
		TileN1, TileK1, Stages1, EpiChunkN1,
		TileN2, TileK2, Stages2, EpiChunkN2,
		ZBufferSlots, CommNumStages,
		/*NC=*/4, /*GemmTileM=*/GemmTileM, /*Compute_=*/Compute>;

	using Element  = typename Config::Element;
	using Traits1  = typename Config::Traits1;
	using Traits2  = typename Config::Traits2;
	using TmaLoadAtom = TmaLoadAtomForCompute<Compute>;
	using TmaStoreAtom = TmaStoreAtomForCompute<Compute>;
	constexpr int kTileM      = Config::kTileM;
	constexpr int kNumThreads = Config::kNumThreads;

	// Iteration counter lives on the device and is incremented at the
	// end of the kernel — see PeSync::advance_iteration_counter() and
	// the comment block at the top of this file.
	int num_tokens       = a.num_tokens;
	int hidden_dim       = a.hidden_dim;
	int experts_per_pe   = a.experts_per_pe;      // local experts (from weight matrices)
	int intermediate_dim = a.intermediate_dim;

	// Upper bound for total_slots: T * K + E * TileM (TileM padding per expert).
	// Actual value is computed by sort (expert_offsets[E]) inside the kernel.
	int max_total_slots  = num_tokens * top_k + num_experts * kTileM;
	int num_m_tiles_in   = (num_tokens + kTileM - 1) / kTileM;  // sort input tiles

	if constexpr (Compute == 100) {
		LIGER_CHECK(hidden_dim % Traits1::TileK == 0,
			"hidden_dim (", hidden_dim, ") must be divisible by MLP1 TileK (", Traits1::TileK, ")");
		LIGER_CHECK(intermediate_dim % Traits1::TileN == 0,
			"intermediate_dim (", intermediate_dim, ") must be divisible by MLP1 TileN (", Traits1::TileN, ")");
		LIGER_CHECK(intermediate_dim % Traits2::TileK == 0,
			"intermediate_dim (", intermediate_dim, ") must be divisible by MLP2 TileK (", Traits2::TileK, ")");
		LIGER_CHECK(hidden_dim % Traits2::TileN == 0,
			"hidden_dim (", hidden_dim, ") must be divisible by MLP2 TileN (", Traits2::TileN, ")");
	}
	LIGER_CHECK(hidden_dim % 8 == 0,
		"hidden_dim (", hidden_dim, ") must be divisible by 8 (int4 vectorization in dispatch/combine)");
	LIGER_CHECK(intermediate_dim % 8 == 0,
		"intermediate_dim (", intermediate_dim, ") must be divisible by 8 (int4 vectorization)");
	LIGER_CHECK(top_k <= kCombineMaxTopK,
		"top_k (", top_k, ") exceeds kCombineMaxTopK (", kCombineMaxTopK,
		"). Bump the constant in combine.cuh.");

	int num_n_tiles_1 = (Compute == 90)
		? (intermediate_dim + Traits1::TileN - 1) / Traits1::TileN
		: intermediate_dim / Traits1::TileN;
	int num_n_tiles_2 = (Compute == 90)
		? (hidden_dim + Traits2::TileN - 1) / Traits2::TileN
		: hidden_dim / Traits2::TileN;

	// Relaxed: runtime NS > num_n_tiles is allowed; surplus split CTAs run
	// empty inner loops but still drive their TMA-load/store mbarriers so the
	// pipeline stays balanced across the column.

	LIGER_CHECK(a.weight_expert_stride >=
			static_cast<int64_t>(intermediate_dim) * hidden_dim,
		"weight expert stride (", a.weight_expert_stride,
		") must cover at least one [I,D] expert matrix");
	LIGER_CHECK(a.weight_expert_stride % hidden_dim == 0,
		"weight expert stride (", a.weight_expert_stride,
		") must contain complete hidden-dimension rows (D=", hidden_dim, ")");
	int weight_expert_stride_rows =
		static_cast<int>(a.weight_expert_stride / hidden_dim);
	LIGER_CHECK(weight_expert_stride_rows % Traits1::TileN == 0,
		"weight expert stride in rows (", weight_expert_stride_rows,
		") must be divisible by MLP1 TileN (", Traits1::TileN, ")");
	int total_n_rows_1 =
		(experts_per_pe - 1) * weight_expert_stride_rows + intermediate_dim;
	int total_n_rows_2 = experts_per_pe * hidden_dim;

	int num_pes = nvshmem_team_n_pes(team);
	int my_pe   = nvshmem_team_my_pe(team);

	// Grid — flat 1-D launch. static_nsplit only seeds n_gemm/grid_x before
	// the kernel selects the runtime split from the actual routed tile count.
	//
	// GEMM and comm gate to the first n_gemm = floor_NS CTAs: the comm↔MLP
	// staging ring couples producer and consumer (it needs grid_x = n_gemm/static_nsplit
	// and MC = n_gemm/NC to be consistent), so they share one count. The
	// sort/dispatch/combine grid-stride phases and the cross-CTA pe_barrier use
	// ALL num_blocks launched CTAs. The runtime selector may choose a different
	// NS inside the kernel and rewrites n_gemm/grid_x consistently for MLP.
	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, a.device);
	int floor_ns   = num_sms - num_sms % static_nsplit;
	int floor_nc   = num_sms - num_sms % Config::kNC;
	int n_gemm     = floor_ns;                                   // GEMM/comm-active CTAs
	int grid_x     = n_gemm / static_nsplit;                     // initial logical GEMM columns
	int num_blocks = (floor_ns > floor_nc) ? floor_ns : floor_nc; // launched CTAs
	dim3 grid(num_blocks);

	// stream comes from a.stream (the binding passes the current torch stream).
	auto& pool = global_buffer_pool();

	// The two sort outputs the bwd path consumes (token_expert_slots
	// [max_total_slots], tile_expert_ids [max_total_slots/kTileM]) are
	// caller-owned: the binding pre-allocated them at exactly these sizes and
	// passed their device pointers in a. They must NOT alias pool slots that the
	// next forward would overwrite.
	int* token_expert_slots_ptr = a.token_expert_slots;
	int* tile_expert_ids_ptr    = a.tile_expert_ids;

	// Allocate all buffers.
	auto buf = allocate_moe_buffers<Config>(pool, stream, grid_x, num_blocks,
		hidden_dim, intermediate_dim, num_tokens, top_k, num_experts,
		experts_per_pe, num_pes,
		max_total_slots, num_m_tiles_in,
		token_expert_slots_ptr,
		tile_expert_ids_ptr);
	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: CUDA error before TMA descriptor setup: ", cudaGetErrorString(e));

	// Final combine output [T, D] — caller-owned (binding-allocated) Y buffer.
	Element* output_ptr = reinterpret_cast<Element*>(a.Y);
	Element* y_buf_ptr = buf.y_buf;

	auto& scfg = get_symm_config();

	// ---- Build MoeParams -------------------------------------------------------
	MoeParams<Config> p;
	p.expert_indices      = a.expert_indices;
	p.expert_weights      = reinterpret_cast<const Element*>(a.expert_weights);

	p.sort.tile_expert_counts = buf.tile_expert_counts;
	p.sort.sorted_token_ids   = buf.sorted_token_ids;
	p.sort.token_expert_slots = buf.token_expert_slots;
	p.sort.expert_offsets     = buf.expert_offsets;
	p.sort.tile_expert_ids    = buf.mlp_tile_expert_ids;
	p.sort.cta_done           = buf.cta_done;
	p.sort.cta_sums           = buf.cta_sums;

	p.tokens              = reinterpret_cast<const Element*>(a.X);
	p.dispatch_dst        = buf.x_sorted;
	p.combine_output      = output_ptr;

	p.comm.src_staging         = buf.src_staging;
	p.comm.dst_staging         = buf.dst_staging;
	p.comm.src_ready           = buf.src_ready;
	p.comm.src_consumed        = buf.src_consumed;
	p.comm.dst_ready           = buf.dst_ready;
	p.comm.dst_consumed        = buf.dst_consumed;
	p.comm.expert_offsets      = buf.expert_offsets;
	p.comm.sorted_token_ids    = buf.sorted_token_ids;
	p.comm.local_tokens        = buf.x_sorted;
	p.comm.local_output        = y_buf_ptr;
	p.comm.tile_expert_ids     = buf.tile_expert_ids;
	p.comm.all_expert_offsets  = buf.all_expert_offsets;
	p.comm.pe_sync.cta_counter   = buf.cta_counter;
	p.comm.pe_sync.num_pes       = num_pes;
	p.comm.pe_sync.my_pe         = my_pe;
	p.comm.pe_sync.team          = team;
	p.comm.hidden_dim          = hidden_dim;
	p.comm.num_experts         = num_experts;
	p.comm.experts_per_pe      = experts_per_pe;
	p.comm.n_gemm              = n_gemm;  // GEMM/comm-active CTA count (floor_NS)

	p.num_tokens          = num_tokens;
	p.intermediate_dim    = intermediate_dim;
	p.top_k               = top_k;
	p.total_slots         = max_total_slots;
	p.num_m_tiles         = num_m_tiles_in;

	// ---- MLP invariant dims (passed as __grid_constant__) ----------------------
	MlpDims mlp_dims;
	mlp_dims.hidden_dim       = hidden_dim;
	mlp_dims.intermediate_dim = intermediate_dim;
	mlp_dims.num_experts      = experts_per_pe;
	mlp_dims.total_n_rows_1   = total_n_rows_1;
	mlp_dims.total_n_rows_2   = total_n_rows_2;
	mlp_dims.expert_n_stride_1 = weight_expert_stride_rows / Traits1::TileN;
	mlp_dims.num_n_tiles_1    = num_n_tiles_1;
	mlp_dims.num_n_tiles_2    = num_n_tiles_2;
	mlp_dims.num_k_tiles_1    = (Compute == 90)
		? (hidden_dim + Traits1::TileK - 1) / Traits1::TileK
		: hidden_dim / Traits1::TileK;
	mlp_dims.num_k_tiles_2    = (Compute == 90)
		? (intermediate_dim + Traits2::TileK - 1) / Traits2::TileK
		: intermediate_dim / Traits2::TileK;
	// Row-tile extent of the symmetric local_output (y_buf) for the
	// direct-to-peer Y store (num_y_m_tiles when storing at the token-tile row).
	mlp_dims.y_buf_m_tiles    = max_total_slots / kTileM;
	mlp_dims.phase_counter    = buf.phase_counter;

	// ---- MLP TMA descriptors -----------------------------------------------
	// X for MLP is x_sorted (dispatch destination, symmetric).
	// All shape/stride values are cast to int64_t — see the long comment
	// below the tma_load_c block for the rationale. B and C in particular
	// are large: their row count = experts_per_pe · intermediate_dim,
	// times hidden_dim → exceeds INT_MAX at production shapes.
	auto tma_load_x = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.x_sorted)),
			make_shape(static_cast<int64_t>(max_total_slots),
			           static_cast<int64_t>(hidden_dim)),
			make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
		typename Traits1::SmemLayoutX_1{});

	auto* ptr_B = reinterpret_cast<Element const*>(a.all_B);
	auto* ptr_C = reinterpret_cast<Element const*>(a.all_C);
	// Hopper keeps the expert as an explicit tensor mode. A tail TMA box can
	// therefore go OOB only within that expert (loads zero-fill) instead of
	// spilling into the next expert's flat rows.
	auto tma_load_b = [&]() {
		if constexpr (Compute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(ptr_B),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(hidden_dim),
						Int<1>{},
						a.weight_expert_stride)),
				typename Traits1::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(ptr_B),
					make_shape(static_cast<int64_t>(total_n_rows_1),
					           static_cast<int64_t>(hidden_dim)),
					make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
				typename Traits1::SmemLayoutW_1{});
		}
	}();
	auto tma_load_c = [&]() {
		if constexpr (Compute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(ptr_C),
					make_shape(static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(hidden_dim),
						Int<1>{},
						a.weight_expert_stride)),
				typename Traits1::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(ptr_C),
					make_shape(static_cast<int64_t>(total_n_rows_1),
					           static_cast<int64_t>(hidden_dim)),
					make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
				typename Traits1::SmemLayoutW_1{});
		}
	}();

	// New design's TMA store box = per-WG epi slot (AtomTileM, EpiChunkN).
	//
	// Shape / stride values are cast to int64_t so CUTE's layout math
	// (cosize, byte-offset, total-elements) uses 64-bit arithmetic. The
	// Z tensor in particular can exceed INT_MAX elements when
	// grid_x · ZBuf · TileM · intermediate_dim > 2^31 — e.g. NSplit=2
	// ZBuf=16 at intermediate_dim=16384 wraps to a negative stride and
	// IMAs the kernel. Casting one operand promotes the multiply to
	// int64_t; we cast both to be safe across CUTE versions.
	auto tma_store_z = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.z_buf),
			make_shape(static_cast<int64_t>(buf.z_rows),
			           static_cast<int64_t>(intermediate_dim)),
			make_stride(static_cast<int64_t>(intermediate_dim), Int<1>{})),
		typename Traits1::SmemLayoutStoreSlot{});

	auto tma_load_z = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.z_buf)),
			make_shape(static_cast<int64_t>(buf.z_rows),
			           static_cast<int64_t>(intermediate_dim)),
			make_stride(static_cast<int64_t>(intermediate_dim), Int<1>{})),
		typename Traits2::SmemLayoutZ_1{});

	auto* ptr_A = reinterpret_cast<Element const*>(a.all_A);
	// Same expert isolation for the down projection, including K-tail loads.
	auto tma_load_a = [&]() {
		if constexpr (Compute == 90) {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(ptr_A),
					make_shape(static_cast<int64_t>(hidden_dim),
					           static_cast<int64_t>(intermediate_dim),
					           static_cast<int64_t>(experts_per_pe)),
					make_stride(
						static_cast<int64_t>(intermediate_dim),
						Int<1>{},
						static_cast<int64_t>(hidden_dim) * intermediate_dim)),
				typename Traits2::SmemLayoutW_1{});
		} else {
			return make_tma_copy(TmaLoadAtom{},
				make_tensor(make_gmem_ptr(ptr_A),
					make_shape(static_cast<int64_t>(total_n_rows_2),
					           static_cast<int64_t>(intermediate_dim)),
					make_stride(static_cast<int64_t>(intermediate_dim), Int<1>{})),
				typename Traits2::SmemLayoutW_1{});
		}
	}();

	auto tma_store_y = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(y_buf_ptr),
			make_shape(static_cast<int64_t>(max_total_slots),
			           static_cast<int64_t>(hidden_dim)),
			make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
		typename Traits2::SmemLayoutStore_1{});

	// Remote staging is a flat list of L = MC · kCommNumStages slots,
	// each TileM rows tall. MC = num_blocks / NC (== gridDim.x / NC) — MUST
	// match the ring size everywhere else (allocate_moe_buffers, the per-launch
	// memset, the consumer's ring_len). Sizing this TMA descriptor on n_gemm
	// would leave the wrap-around slots outside its row extent (TMA OOB) when
	// num_blocks/NC > n_gemm/NC (num_sms % NSplit != 0).
	int mc_remote = num_blocks / Config::kNC;
	int remote_total_rows = mc_remote * Config::kCommNumStages * kTileM;

	auto tma_load_x_remote = make_tma_copy(TmaLoadAtom{},
		make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(buf.src_staging)),
			make_shape(static_cast<int64_t>(remote_total_rows),
			           static_cast<int64_t>(hidden_dim)),
			make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
		typename Traits1::SmemLayoutX_1{});

	auto tma_store_y_remote = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(buf.dst_staging),
			make_shape(static_cast<int64_t>(remote_total_rows),
			           static_cast<int64_t>(hidden_dim)),
			make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
		typename Traits2::SmemLayoutStore_1{});

	// ---- GET-via-TMA descriptors (FWD-only, raw CUtensorMap) -----------------
	// SAME-HOST source maps over nvshmem_ptr(x_sorted, peer) + one dst map over
	// local src_staging. TMA get is enabled only for same-host NVLink P2P peers
	// (cross-host peers fall back to warp getmem); also needs divisible
	// hidden_dim + smem budget (checked below).
	//
	// src_x_desc is indexed by INTRA-HOST local rank (pe % gpus_per_node), so the
	// array only ever needs gpus_per_node (≤ kGetMaxPes) entries regardless of
	// total PE count. Assumes host-contiguous, host-aligned ranks (true for a
	// WORLD-contiguous team); the do_get / RemoteIter is_local predicate uses the
	// same arithmetic.
	GetTmaDescs get_descs{};
	get_descs.enabled = 0;
	int gpus_per_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
	LIGER_CHECK(gpus_per_node >= 1 && gpus_per_node <= liger::kGetMaxPes,
		"gpus_per_node (", gpus_per_node, ") must be in [1, kGetMaxPes=",
		liger::kGetMaxPes, "]");
	get_descs.gpus_per_node = gpus_per_node;
	get_descs.my_pe         = my_pe;
	constexpr int kGetBoxRows =
		Config::kTileM / (liger::kNumGetWarpsFwd * Config::kNC);
	size_t get_bounce_bytes = (size_t)
		liger::GetBounce<Element, Config::kTileM, Config::kNC>::kTotalBytes;
	{
		bool ok = (hidden_dim % liger::kGetKChunk == 0);
		for (int tp = 0; ok && tp < num_pes; ++tp) {
			// Only same-host peers are P2P-reachable → only they get a TMA
			// descriptor (indexed by local rank). Cross-host peers skip it and
			// use getmem at runtime.
			bool same_host = (tp / gpus_per_node) == (my_pe / gpus_per_node);
			if (!same_host) continue;
			int gpe = nvshmem_team_translate_pe(team, tp, NVSHMEM_TEAM_WORLD);
			void* pp = nvshmem_ptr(buf.x_sorted, gpe);  // self → own ptr (non-null)
			if (pp == nullptr) { ok = false; break; }   // same-host but no P2P → disable TMA
			ok = make_get_tma_map(get_descs.src_x_desc[tp % gpus_per_node], pp,
				hidden_dim, max_total_slots, kGetBoxRows);
		}
		if (ok)
			ok = make_get_tma_map(get_descs.dst_staging_desc, buf.src_staging,
				hidden_dim, remote_total_rows, kGetBoxRows);
		get_descs.enabled = ok ? 1 : 0;
		// A/B toggle: LIGER_GET_TMA=0 forces the warp-getmem path at the same
		// config (no bounce smem), isolating the GET mechanism on the same pod.
		if (const char* e = std::getenv("LIGER_GET_TMA"))
			if (e[0] == '0') get_descs.enabled = 0;
	}

	// ---- Direct-to-peer Y store descriptors (FWD-only) -----------------------
	// Per-peer TMA-store maps over nvshmem_ptr(y_buf, peer) for SAME-HOST peers
	// (self included), indexed by intra-host local rank. When a tile is_local,
	// MLP2 stores Y straight into the destination's symmetric local_output via
	// its peer descriptor (token-tile row), bypassing dst_staging + the put warp
	// (which then only rubber-stamps). Same shape/layout as tma_store_y; only the
	// base pointer differs (the peer's y_buf). For single-PE the only peer is
	// self → desc[my_pe%gpus_per_node] == tma_store_y.
	PeerYStoreDescs<decltype(tma_store_y)> peer_y{};
	peer_y.enabled = 1;
	for (int tp = 0; tp < num_pes; ++tp) {
		bool same_host = (tp / gpus_per_node) == (my_pe / gpus_per_node);
		if (!same_host) continue;  // cross-host peers keep the dst_staging+putmem path
		int gpe = nvshmem_team_translate_pe(team, tp, NVSHMEM_TEAM_WORLD);
		void* py = nvshmem_ptr(buf.y_buf, gpe);  // same-host → P2P ptr (non-null on NVLink)
		LIGER_CHECK(py != nullptr,
			"direct-to-peer Y store: nvshmem_ptr(y_buf, same-host pe ", tp,
			") is null (no P2P) — unsupported for this build");
		peer_y.desc[tp % gpus_per_node] = make_tma_copy(TmaStoreAtom{},
			make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(py)),
				make_shape(static_cast<int64_t>(max_total_slots),
				           static_cast<int64_t>(hidden_dim)),
				make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
			typename Traits2::SmemLayoutStore_1{});
	}

	// TMA descriptor for the final combine output [num_tokens, hidden_dim].
	// Box shape matches combine_tokens_tma's WG tile: 32 tokens × 256 hidden.
	auto* ptr_combine_out = output_ptr;
	auto tma_store_y_combine = make_tma_copy(TmaStoreAtom{},
		make_tensor(make_gmem_ptr(ptr_combine_out),
			make_shape(static_cast<int64_t>(num_tokens),
			           static_cast<int64_t>(hidden_dim)),
			make_stride(static_cast<int64_t>(hidden_dim), Int<1>{})),
		CombineSmemLayoutSlot<Element>{});

	// ---- Dynamic smem --------------------------------------------------------
	int sort_num_warps = kNumThreads / 32;
	size_t smem_sort   = sort_num_warps * num_experts * sizeof(int);

	size_t smem_mlp    = sizeof(MoeSmem<Traits1, Traits2, Config::kCompute>);
	smem_mlp = (smem_mlp + sizeof(int) - 1) & ~(sizeof(int) - 1);
	smem_mlp += num_pes * (experts_per_pe + 1) * sizeof(int);  // remote_offsets

	// GET-via-TMA bounce: append a 128-aligned region (must match the kernel's
	// get_bounce offset). Disable the TMA-get path if it would exceed the
	// opt-in dynamic-smem cap (227 KiB) — that config just keeps warp getmem.
	if (get_descs.enabled) {
		size_t mlp_with_bounce = ((smem_mlp + 127) & ~((size_t)127)) + get_bounce_bytes;
		if (mlp_with_bounce <= (size_t)227 * 1024) smem_mlp = mlp_with_bounce;
		else get_descs.enabled = 0;
	}

	// Combine phase reinterprets the same SMEM region as CombineSmem.
	constexpr int kCombineNumWGs =
		Config::kNumThreads / (kCombineWarpsPerWG * kCombineWarpSize);
	size_t smem_combine = sizeof(CombineSmem<Element, kCombineNumWGs>);

	size_t smem_size = std::max({smem_sort, smem_mlp, smem_combine});

	// ---- Launch fused kernel -----------------------------------------------
	auto kernel_fn = moe_fused_kernel<Config,
		decltype(tma_load_x), decltype(tma_load_b), decltype(tma_store_z),
		decltype(tma_load_z), decltype(tma_load_a), decltype(tma_store_y),
		decltype(tma_store_y_combine)>;

	if (cudaError_t e = cudaFuncSetAttribute(kernel_fn,
			cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", smem_size,
			") failed: ", cudaGetErrorString(e));

	// Reset per-call buffers in a single fused-memset kernel — one CTA per
	// region. Replaces 11 cudaMemsetAsync calls that would otherwise
	// serialize on the stream.
	//
	// Notes:
	//   - dispatch_done / mlp_done are already reset upstream in
	//     allocate_moe_buffers (the original "monotonic, never reset" design
	//     was retired — wait_until could exit on stale residual signal
	//     values in multi-PE shape-transition workloads).
	//   - all_expert_offsets stays unreset (overwritten by
	//     broadcast_expert_offsets on every call).
	//   - y_buf stays unreset (slots referenced by token_expert_slots[t,k]
	//     are always written by local MLP or the source PE's NVSHMEM
	//     put-back; padding slots are never read by combine_tokens_tma).
	//   - src_staging / dst_staging / z_buf intentionally NOT zero'd (see
	//     comment block below).
	//
	// Per-region reasons:
	//   - cta_counter: PeSync.barrier() does `% num_ctas`. Residual values
	//     from a prior call with different num_ctas misalign to the new
	//     modulus, hanging barrier() forever.
	//   - tile_expert_ids (0xff): sentinel-fill so unfetched slots produce
	//     expert_id = -1; matches moe_bwd.cu's 0xff fill of
	//     tile_expert_ids_{x,dy}. Without this, stale slot values from prior
	//     calls can match a valid expert id and confuse the MLP-side remote
	//     iterator at higher PE counts.
	//   - src/dst_ready/consumed: atomically accumulated; stale ready/
	//     consumed values make the consumer-acquire wait fall through.
	//   - tile_expert_counts / cta_sums: atomicAdd'd during sort; must
	//     start at 0.
	{
		auto& scfg = get_symm_config();
		// L = MC · kCommNumStages, MC = num_blocks / NC (== gridDim.x / NC).
		// MUST match the ring size in allocate_moe_buffers and the device-side
		// comm/consumer stride — otherwise this memset under-covers the signal /
		// tile_expert_ids arrays (slots [n_gemm/NC·CS, gridDim.x/NC·CS) left
		// uninitialized) and the consumer reads stale src_ready / garbage expert
		// ids for those slots (see the OOB note in allocate_moe_buffers).
		int mc_local = num_blocks / Config::kNC;
		int comm_l = mc_local * Config::kCommNumStages;
		size_t sig_bytes = static_cast<size_t>(comm_l) * sizeof(int);
		int max_m_tiles = scfg.max_total_slots / kTileM;
		int max_tile_counts = max_m_tiles * scfg.max_num_experts;

		FusedMemsetArgs ma{};
		fused_memset_add(ma, buf.phase_counter,
			((num_blocks + 1) / 2) * sizeof(int), 0x00);
		fused_memset_add(ma, buf.cta_counter,
			sizeof(int), 0x00);
		fused_memset_add(ma, buf.cta_done,
			num_blocks * sizeof(int), 0x00);
		fused_memset_add(ma, buf.sorted_token_ids,
			static_cast<size_t>(max_total_slots) * sizeof(int), 0xff);
		fused_memset_add(ma, buf.src_ready,        sig_bytes, 0x00);
		fused_memset_add(ma, buf.src_consumed,     sig_bytes, 0x00);
		fused_memset_add(ma, buf.dst_ready,        sig_bytes, 0x00);
		fused_memset_add(ma, buf.dst_consumed,     sig_bytes, 0x00);
		fused_memset_add(ma, buf.tile_expert_ids,  sig_bytes, 0xff);
		fused_memset_add(ma, buf.tile_expert_counts,
			static_cast<size_t>(max_tile_counts) * sizeof(int), 0x00);
		fused_memset_add(ma, buf.cta_sums,
			static_cast<size_t>(num_blocks) * scfg.max_num_experts * sizeof(int), 0x00);
		fused_memset_launch(ma, stream);
	}

	// src_staging / dst_staging / z_buf intentionally NOT zero'd here:
	//
	//   src_staging: comm warps fill the slots THIS CTA fetches; unfetched
	//     slots are gated by tile_expert_ids = -1 (set via the 0xff memset
	//     above) and RemoteIter::has_next() walks only per_cta_tiles, so the
	//     remote MLP never reads unfetched slots. Stale prev-call contents
	//     are inert under both bounds checks.
	//
	//   dst_staging: write-then-read for routed slots only. Remote MLP
	//     writes Y for tiles RemoteIter returns; comm warps put back only
	//     those tiles. Padding/unused slots are never read at either end.
	//
	//   z_buf: per-tile write→read within a single MLP launch. LocalIter is
	//     range-bounded to [local_e_start, local_e_end) experts via
	//     set_range; the remote pass is bounded by per_cta_tiles. Tiles
	//     outside the current call's active range are never written and
	//     never read — buffer-pool size growth from previous (larger) calls
	//     is invisible to the kernel.
	//
	// The tile_expert_ids 0xff fill above is what makes this safe — keep
	// that.

	kernel_fn<<<grid, kNumThreads, smem_size, stream>>>(
		tma_load_x, tma_load_b, tma_load_c, tma_store_z,
		tma_load_z, tma_load_a, tma_store_y,
		tma_load_x_remote, tma_store_y_remote,
		tma_store_y_combine,
		mlp_dims, p, static_nsplit, get_descs, peer_y);
	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
		LIGER_FAIL_CUDA("moe_fwd: kernel launch failed for grid=", grid.x,
			" threads=", kNumThreads, " smem=", smem_size, ": ",
			cudaGetErrorString(e));

	// Post-launch global barrier: ensures all NVSHMEM operations (tile puts,
	// signal ops) across all PEs are complete before any PE starts the next
	// call. Without this, a fast PE could overwrite symmetric buffers
	// (x_sorted, y_buf, tile signals) while a slow PE is still reading them.
	//
	// Stream-ordered variant — capturable into a CUDA graph. The host-side
	// nvshmem_team_sync(team) would do an internal cudaStreamSynchronize and
	// therefore fail with "operation not permitted when stream is capturing".
	//if (num_pes > 1) {
	//	nvshmemx_barrier_on_stream(team, stream);
	//}

	// Hand the persistent symmetric buffers back to the ABI wrapper so it can
	// alias them as the *_out_symm views (shapes use the static MoeSymmConfig
	// max; the views do NOT own the storage — SymmetricMemoryStack does). Y and
	// the two sort outputs were written in place into the caller-owned buffers.
	*a.x_sorted_out            = buf.x_sorted;
	*a.y_buf_out               = y_buf_ptr;
	*a.all_expert_offsets_out  = buf.all_expert_offsets;
}

// ── Runtime dispatch table for tuned-config lookup ────────────────────

using MoeFwdFn = void (*)(const MoeFwdArgs&, int static_nsplit);
static constexpr int kFwdRuntimeNsplitSeed = 8;

struct DispatchEntry {
	int Compute;
	int TileN1, TileK1, Stages1, EpiChunkN1;
	int TileN2, TileK2, Stages2, EpiChunkN2;
	int ZBufferSlots, CommNumStages;
	int TileM;      // COMM tile (staging/semaphores/NC)
	int GemmTileM;  // GEMM tile ∈ {64,128}; == TileM for non-decoupled configs
	MoeFwdFn fn;
};

// 13-arg core: explicit GemmTileM. 12-arg shim sets GemmTileM = comm TileM.
#define LIGER_MOE_DISPATCH_ENTRY_G_C(Compute, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, GemmTileM) \
	{ Compute, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, GemmTileM, \
	  &moe_fused_fwd_bf16<bfloat16_t, TN1, TK1, S1, TN2, TK2, S2, ZBuf, CStages, EC1, EC2, TileM, GemmTileM, Compute> },
#define LIGER_MOE_DISPATCH_ENTRY_C(Compute, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM) \
	LIGER_MOE_DISPATCH_ENTRY_G_C(Compute, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, TileM)
#define LIGER_MOE_DISPATCH_ENTRY_G_SM90(TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, GemmTileM) \
	LIGER_MOE_DISPATCH_ENTRY_G_C(90, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, GemmTileM)
#define LIGER_MOE_DISPATCH_ENTRY_SM90(TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM) \
	LIGER_MOE_DISPATCH_ENTRY_C(90, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM)
#define LIGER_MOE_DISPATCH_ENTRY_G_SM100(TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, GemmTileM) \
	LIGER_MOE_DISPATCH_ENTRY_G_C(100, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM, GemmTileM)
#define LIGER_MOE_DISPATCH_ENTRY_SM100(TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM) \
	LIGER_MOE_DISPATCH_ENTRY_C(100, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TileM)

static const DispatchEntry kDispatchTable[] = {
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 90
	LIGER_MOE_FWD_DISPATCH_CONFIGS_SM90(LIGER_MOE_DISPATCH_ENTRY_SM90, LIGER_MOE_DISPATCH_ENTRY_G_SM90)
#endif
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 100
	LIGER_MOE_FWD_DISPATCH_CONFIGS_SM100(LIGER_MOE_DISPATCH_ENTRY_SM100, LIGER_MOE_DISPATCH_ENTRY_G_SM100)
#endif
};
#undef LIGER_MOE_DISPATCH_ENTRY_SM100
#undef LIGER_MOE_DISPATCH_ENTRY_G_SM100
#undef LIGER_MOE_DISPATCH_ENTRY_SM90
#undef LIGER_MOE_DISPATCH_ENTRY_G_SM90
#undef LIGER_MOE_DISPATCH_ENTRY_C
#undef LIGER_MOE_DISPATCH_ENTRY_G_C
static constexpr int kNumDispatchEntries =
	sizeof(kDispatchTable) / sizeof(kDispatchTable[0]);

} // namespace liger

// Joint fwd+bwd tuning table (auto-generated by benchmarks/tune_moe_fwd_bwd).
// We only consume the Fwd_* fields here; moe_bwd.cu consumes the same table's
// Bwd_* fields. Picking fwd and bwd from a single table guarantees the two
// directions are tuned as a pair (so their TileM agrees and their per-shape
// templates were measured together, not independently).
#include "moe_fwd_bwd_tuning_configs.cuh"

namespace liger {

// TunedConfig is the fwd-only projection of TunedConfigFwdBwd. The rest of
// the dispatch code (validity check, dispatch-entry match) was written
// against this struct and is reused unchanged.
struct TunedConfig {
	int Compute;
	int TK, TKE, D, I;
	int TileN1, TileK1, Stages1, EpiChunkN1;
	int TileN2, TileK2, Stages2, EpiChunkN2;
	int ZBufferSlots, CommNumStages;
	int TileM;      // comm tile
	int GemmTileM;  // gemm tile (auto path: == TileM; FORCE_CONFIG may set 64)
};

// Project a TunedConfigFwdBwd row down to its fwd template params. The tuned
// table has no GemmTileM column, so the auto path keeps gemm == comm (TileM).
static TunedConfig project_fwd(const TunedConfigFwdBwd& c, int compute) {
	return TunedConfig{
		compute,
		c.TK, c.TKE, c.D, c.I,
		c.Fwd_TileN1, c.Fwd_TileK1, c.Fwd_Stages1, c.Fwd_EpiChunkN1,
		c.Fwd_TileN2, c.Fwd_TileK2, c.Fwd_Stages2, c.Fwd_EpiChunkN2,
		c.Fwd_ZBufferSlots, c.Fwd_CommNumStages,
		c.Fwd_TileM, c.Fwd_TileM,
	};
}

// Find the dispatch table entry matching a TunedConfig's fields.
static const DispatchEntry* find_dispatch_entry(const TunedConfig& c) {
	for (int i = 0; i < kNumDispatchEntries; ++i) {
		const auto& e = kDispatchTable[i];
		if (e.Compute == c.Compute &&
		    e.TileN1 == c.TileN1 && e.TileK1 == c.TileK1 && e.Stages1 == c.Stages1 &&
		    e.EpiChunkN1 == c.EpiChunkN1 &&
		    e.TileN2 == c.TileN2 && e.TileK2 == c.TileK2 && e.Stages2 == c.Stages2 &&
		    e.EpiChunkN2 == c.EpiChunkN2 &&
		    e.ZBufferSlots == c.ZBufferSlots && e.CommNumStages == c.CommNumStages &&
		    e.TileM == c.TileM && e.GemmTileM == c.GemmTileM) {
			return &e;
		}
	}
	return nullptr;
}

// Mirrors shape_valid() in tune_moe.cu: structural constraints the kernel
// requires on (D, I) for a given config. Configs that violate these for
// the runtime shape are skipped during nearest-neighbor lookup so we
// don't dispatch to a template instantiation the kernel can't run.
//
// Cooperative 2-WG consumer layout switches on TileM:
//   - TileM=128 (M-split, Layout<_2,_1,_1>): each WG owns AtomTileM=64
//     rows × TileN cols. EpiChunkN must divide TileN.
//   - TileM=64  (N-split, Layout<_1,_2,_1>): each WG owns TileM=64 rows
//     × TileN/2 cols. EpiChunkN must divide TileN/2; TileN must be even.
//   - Hopper uses ceil-div tile counts; SM100 retains exact divisibility.
//   - store_buf is (AtomTileM=64, EpiChunkN) per WG in both modes.
static bool tuned_config_valid(int D, int I, const TunedConfig& c) {
	if (c.TileM != 64 && c.TileM != 128) return false;
	// The current Blackwell UMMA MLP1 consumer only supports the 128-row comm
	// tile. Keep mirrored SM90 TileM=64 tuning rows out of SM100 auto-dispatch
	// until that TMEM epilogue layout is implemented.
	if (c.Compute == 100 && c.TileM != 128) return false;
	if (c.Compute == 100) {
		if (D % c.TileK1 != 0) return false;
		if (I % c.TileN1 != 0) return false;
		if (I % c.TileK2 != 0) return false;
		if (D % c.TileN2 != 0) return false;
	}
	if (D % 8 != 0 || I % 8 != 0) return false;
	// NC selection mirrors moe_fused_fwd_bf16: TileM=128 → NC=4,
	// TileM=64 → NC=2. With the ticket-based GEMM iterator the only
	// runtime requirement is NC | (gridDim.x · gridDim.y); runtime NS /
	// CommNumStages no longer need to share any divisibility with NC.
	const int nc = (c.TileM == 128) ? 4 : 2;
	(void)nc;
	// EpiChunkN must divide the per-WG N extent (WgTileN).
	const int wg_tile_n1 = (c.TileM == 128) ? c.TileN1 : c.TileN1 / 2;
	const int wg_tile_n2 = (c.TileM == 128) ? c.TileN2 : c.TileN2 / 2;
	if (wg_tile_n1 <= 0 || wg_tile_n1 % c.EpiChunkN1 != 0) return false;
	if (wg_tile_n2 <= 0 || wg_tile_n2 % c.EpiChunkN2 != 0) return false;
	// N-split requires TileN to be evenly halvable (TileN/2 must be ≥ the
	// WGMMA atom-N min, which is 8 for bf16 but 64 for our smem layout).
	if (c.TileM == 64 && (c.TileN1 % 2 != 0 || c.TileN2 % 2 != 0)) return false;

	// SMEM cap (H100 = 228 KiB; budget 224 KiB inside MoeFusedConfig).
	// Per-CTA, bf16 (2 B/elem). TileM scales the X/Z staging:
	//   MLP1 = 2 * (TileM·TK1·S1 + 2·TN1·TK1·S1 + 2·64·EC1)
	//   MLP2 = 2 * (TileM·TK2·S2 +   TN2·TK2·S2 + 2·64·EC2)
	//   smem ≈ max(MLP1, MLP2)
	int mlp1_bytes = 2 * (c.TileM * c.TileK1 * c.Stages1
	                    + 2       * c.TileN1 * c.TileK1 * c.Stages1
	                    + 2       * 64       * c.EpiChunkN1);
	int mlp2_bytes = 2 * (c.TileM * c.TileK2 * c.Stages2
	                    +           c.TileN2 * c.TileK2 * c.Stages2
	                    + 2       * 64       * c.EpiChunkN2);
	int smem = mlp1_bytes > mlp2_bytes ? mlp1_bytes : mlp2_bytes;
	constexpr int kSmemBudget = 224 * 1024;
	if (smem > kSmemBudget) return false;
	return true;
}

static bool find_tuned_table(int compute, int n_pes,
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
// is the total routed-token count (the actual M-axis the kernel sees) and
// TKE = TK / E_local (E_local = E / n_pes) is the per-LOCAL-expert avg
// K-range. Adding TKE to the distance lets the dispatcher distinguish
// shapes whose mlp3/mlp4 work distribution differs even when (TK, D, I)
// is the same — e.g. same total tokens, different expert count or PE
// count.
// Configs that violate structural constraints for the runtime (D, I) are
// skipped — the nearest tuned point is not always valid for the actual
// shape (e.g. tuned at I=4096 with TileN1=256, runtime I=768).
// Out-param API instead of pointer-into-static-array because each candidate
// TunedConfig is projected on the fly from kTunedConfigsFwdBwd (no stable
// address). Returns true when a valid config was found.
static bool find_nearest_tuned(int compute, int TK, int TKE, int D, int I, int n_pes,
                               TunedConfig* out) {
	// World-size class: single-GPU (n_pes <= 1) and multi-GPU tables are tuned
	// and stored separately, since the comm path (and thus the optimal NSplit/
	// CommNumStages/GEMM-tile) differs sharply between one PE and many.
	const TunedConfigFwdBwd* table = nullptr;
	int table_n = 0;
	if (!find_tuned_table(compute, n_pes, &table, &table_n)) return false;
	bool found = false;
	double best_dist = 1e30;
	double lTK  = std::log((double)TK);
	double lTKE = std::log((double)TKE);
	double lD   = std::log((double)D);
	double lI   = std::log((double)I);
	for (int i = 0; i < table_n; ++i) {
		TunedConfig c = project_fwd(table[i], compute);
		if (!tuned_config_valid(D, I, c)) continue;
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
// Picks the nearest tuned config (log-space distance over (T, D, I)) from
// moe_fwd_bwd_tuning_configs.cuh and dispatches to the matching template
// instantiation in kDispatchTable. Both tables are populated by the
// tune_moe_fwd_bwd sweep —
// shapes outside the swept range error out instead of falling back to a
// heuristic. Add the missing shape to the sweep (or to LIGER_MOE_TUNE_CONFIGS
// directly) to extend coverage.

// Internal auto-dispatch: picks the nearest tuned config for the runtime shape
// (or honours LIGER_MOE_FORCE_CONFIG) and invokes the matching template
// instantiation via the dispatch table. Writes the chosen comm TileM to
// *chosen_tile_m. All shape info comes from the MoeFwdArgs bundle; the launcher
// writes the outputs in place.
void moe_fused_fwd_dispatch(const MoeFwdArgs& a, int* chosen_tile_m) {
	const int T  = a.num_tokens;
	const int D  = a.hidden_dim;
	const int I  = a.intermediate_dim;
	const int top_k = a.top_k;
	const int num_experts = a.num_experts;
	const int TK = T * top_k;  // total routed tokens — the M-axis the kernel sees
	// TKE = TK / E_local where E_local = num_experts / n_pes (per-LOCAL-expert
	// avg K-range). Matches the tuner's definition so the nearest-neighbor
	// lookup picks rows recorded at comparable per-PE load.
	const int n_pes = nvshmem_team_n_pes(a.team);
	const int E_local = (n_pes > 0) ? std::max(1, num_experts / n_pes) : num_experts;
	// The lookup operates in log space, so clamp the per-expert average to
	// the smallest representable workload for tiny but valid token batches.
	const int TKE = std::max(1, TK / E_local);
	const int compute = moe_detect_compute_dispatch_key(a.device, "moe_fused_fwd_bf16_auto");

	// ── Override path: LIGER_MOE_FORCE_CONFIG=TN1,TK1,S1,EC1,TN2,TK2,S2,EC2,ZB,CS,TM[,GemmTM]
	// Lets a test suite sweep every compiled config without rebuilding. The
	// optional final int is GemmTileM (comm/gemm tile decouple); omitted → == TM.
	if (const char* s = std::getenv("LIGER_MOE_FORCE_CONFIG")) {
		TunedConfig tc{};
		tc.Compute = compute;
		tc.TileM = 128;       // default when omitted (backward compat)
		tc.GemmTileM = -1;    // sentinel → default to TileM after parse
		int n = std::sscanf(s, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
			&tc.TileN1, &tc.TileK1, &tc.Stages1, &tc.EpiChunkN1,
			&tc.TileN2, &tc.TileK2, &tc.Stages2, &tc.EpiChunkN2,
			&tc.ZBufferSlots, &tc.CommNumStages,
			&tc.TileM, &tc.GemmTileM);
		LIGER_CHECK(n == 10 || n == 11 || n == 12,
			"LIGER_MOE_FORCE_CONFIG must have 10, 11 or 12 comma-separated ints "
			"(TN1,TK1,S1,EC1,TN2,TK2,S2,EC2,ZB,CS[,TM[,GemmTM]]); got '", s, "'");
		if (tc.GemmTileM <= 0) tc.GemmTileM = tc.TileM;  // omitted → gemm == comm
		const DispatchEntry* de = find_dispatch_entry(tc);
		LIGER_CHECK(de != nullptr,
			"LIGER_MOE_FORCE_CONFIG=", s, " is not in the compiled dispatch "
			"table. Add it to LIGER_MOE_TUNE_CONFIGS in moe_fwd_bwd_tune_configs.hpp.");
		de->fn(a, kFwdRuntimeNsplitSeed);
		*chosen_tile_m = tc.TileM;
		return;
	}

	TunedConfig tc{};
	LIGER_CHECK(find_nearest_tuned(compute, TK, TKE, D, I, n_pes, &tc),
		"moe_fused_fwd_bf16_auto: no valid tuned config for compute=", compute,
		" shape (T=", T,
		" K=", top_k, " TK=", TK, " TKE=", TKE, " D=", D, " I=", I,
		"). Either moe_fwd_bwd_tuning_configs.cuh is empty (run tune_moe_fwd_bwd), "
		"or every tuned entry violates structural constraints for this (D, I) "
		"(tile/layout compatibility and shared-memory limits). "
		"Add a compatible config to LIGER_MOE_TUNE_CONFIGS and re-tune.");

	const DispatchEntry* de = find_dispatch_entry(tc);
	LIGER_CHECK(de != nullptr,
		"moe_fused_fwd_bf16_auto: tuned config (Compute=", tc.Compute,
		" TN1=", tc.TileN1, "/", tc.TileK1, "/", tc.Stages1, "/EC", tc.EpiChunkN1,
		" TN2=", tc.TileN2, "/", tc.TileK2, "/", tc.Stages2, "/EC", tc.EpiChunkN2,
		" ZB=", tc.ZBufferSlots, " CS=", tc.CommNumStages,
		") for shape (T=", T, " K=", top_k, " TK=", TK, " D=", D, " I=", I,
		") is not in the compiled dispatch table. "
		"Add it to LIGER_MOE_TUNE_CONFIGS in moe_fwd_bwd_tune_configs.hpp.");

	de->fn(a, kFwdRuntimeNsplitSeed);
	*chosen_tile_m = tc.TileM;
}

// ─────────────────────────────────────────────────────────────────────
// Default (untuned) wrapper — single hardcoded config per TileM. Use
// this for the regular bench_moe / test_moe paths so they don't depend
// on the autotuner dispatch table. Returns the same 6-tuple as the auto
// wrapper.
//
// Each config below must have a matching explicit instantiation. The
// TileM=64 row is in LIGER_MOE_TUNE_CONFIGS (so LIGER_MOE_INSTANTIATE
// emits it automatically); the TileM=128 row is emitted as a direct
// `template` instantiation right below this function.
// ─────────────────────────────────────────────────────────────────────

// Returns the compiled MoE config table as a list of 12-int rows
// (Compute, TileN1, TileK1, Stages1, EpiChunkN1, TileN2, TileK2, Stages2,
// EpiChunkN2, ZBufferSlots, CommNumStages, TileM). Kept for the tuner / a future
// sweep binding so they stay in sync with the dispatch table without manual
// duplication. Internal (std::vector) — not part of the flat ABI.
std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int, int>>
moe_list_compiled_configs() {
	std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int, int>> out;
	out.reserve(kNumDispatchEntries);
	for (int i = 0; i < kNumDispatchEntries; ++i) {
		const auto& e = kDispatchTable[i];
		out.emplace_back(e.Compute, e.TileN1, e.TileK1, e.Stages1, e.EpiChunkN1,
			e.TileN2, e.TileK2, e.Stages2, e.EpiChunkN2,
			e.ZBufferSlots, e.CommNumStages, e.TileM);
	}
	return out;
}

}  // namespace liger

// ============================================================================
// Flat extern "C" control ABI (declared in liger_cute/moe.h)
// ============================================================================

namespace {

using liger_cute::detail::guarded;

}  // namespace

extern "C" {

liger_cute_status_t liger_cute_moe_get_symm_config(liger_cute_moe_symm_config_t* out) {
	return guarded([&]() -> liger_cute_status_t {
		LIGER_CHECK(out != nullptr, "moe_get_symm_config: out is null");
		const liger::MoeSymmConfig& cfg = liger::get_symm_config();
		out->max_total_slots = cfg.max_total_slots;
		out->max_num_experts = cfg.max_num_experts;
		out->hidden_dim      = cfg.hidden_dim;
		out->num_pes         = cfg.num_pes;
		out->experts_per_pe  = cfg.experts_per_pe;
		out->max_top_k       = cfg.max_top_k;
		out->initialized     = cfg.initialized ? 1 : 0;
		return LIGER_CUTE_OK;
	});
}

liger_cute_status_t liger_cute_moe_configure_symmetric(
		const int max_tokens, const int hidden_dim, const int max_num_experts,
		const int max_top_k, const int num_pes, const int num_hosts,
		const int gpus_per_host) {
	return guarded([&]() -> liger_cute_status_t {
		liger::moe_configure_symmetric(max_tokens, hidden_dim, max_num_experts,
		                               max_top_k, num_pes, num_hosts, gpus_per_host);
		return LIGER_CUTE_OK;
	});
}

liger_cute_status_t liger_cute_moe_pop_fwd(void) {
	return guarded([&]() -> liger_cute_status_t {
		liger::moe_pop_fwd();
		return LIGER_CUTE_OK;
	});
}

}  // extern "C"
