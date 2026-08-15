#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 2 Transpose (fused tile):  Y = Z @ A
// ═══════════════════════════════════════════════════════════════════
//
// Single M-tile version for embedding in the fused MLP kernel
// (mlp_bwd.cuh Phase 1b'). Matches the production design in mlp2_t.cuh:
//   - Cooperative 2-WG consumer; split axis selected by TileM (kMSplit
//     in Mlp2TTraits): TileM=128 → M-split, TileM=64 → N-split.
//   - One accumulator per WG covering (AtomTileM=64, AtomN)
//   - Per-WG store_buf with EpiChunkN-round epilogue
//   - K_PIPE_MMAS = 1 (CUTLASS-canonical)
//   - Single fused Z+W TMA pipe (1 mbarrier per stage, 2 TMA copies)
//
// Differs from mlp2_t.cuh only in:
//   - No outer for(m) loop — caller passes a single m (y_m for the
//     output buffer index, expert_k_offset for the input weights)
//   - Mlp2TFusedSmem omits pipeline storage (caller manages it)
//
// Mirrors mlp2_fused.cuh's single-tile style.

#include "mlp2_t.cuh"

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// Fused MLP2-T shared memory — no pipeline storage (managed externally)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp2TFusedSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_Z_size     = cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_W_size     = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element smem_W[smem_W_size];
	// Per-WG store buffer: WG1 uses store_buf[0..S-1], WG2 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	// SM100 (Compute=100) only: landing slot for tcgen05.alloc's granted TMEM
	// base address, plus the accumulator pipeline's barrier storage (UMMA→
	// epilogue handoff). The Hopper (Compute=90) path never touches these, and
	// keeping them here leaves the consumer signature identical across Compute
	// values (TMEM base is externally owned; the acc pipeline lives here).
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
	CUTE_DEVICE Element* W_data() { return &smem_W[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// Pipeline construction helper — single fused Z+W pipeline
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
__device__ __forceinline__ auto mlp2_t_make_pipe(
		typename Traits::MainloopPipeline::SharedStorage& storage) {

	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 3 && warp_id <= 11);  // warp 3 = UMMA producer

	typename Pipeline::Params pp;
	pp.transaction_bytes = Traits::TmaTransBytes;
	pp.num_producers = 1;
	pp.num_consumers = Traits::ConsumerThreads;
	if (is_producer) {
		pp.role = Category::Producer;
		pp.is_leader = (threadIdx.x == 0);
	} else if (is_consumer) {
		pp.role = Category::Consumer;
	} else {
		pp.role = Category::NonParticipant;
	}
	return Pipeline(storage, pp, Shape<_1, _1, _1>{},
		cute::true_type{}, cute::true_type{});
}

// SM100 (Compute=100) mainloop pipe. Mirrors mlp2_t_make_pipe but builds the
// UMMA-aware TMA pipeline (PipelineTmaUmmaAsync) whose consumer_release issues
// the UMMA-gated smem-buffer arrival. num_consumers=1: on Blackwell the buffer
// is released by a single umma_arrive per stage (the MMA warp), not per
// consumer thread. Params/PipelineState/SharedStorage alias the Hopper
// PipelineTmaAsync's, and the ctor arg pattern is identical, so the producer
// (mlp2_t_fused_producer) and the Compute=100 consumer drive it unchanged.
template <typename Traits>
__device__ __forceinline__ auto mlp2_t_make_pipe_umma(
		typename Traits::MainloopPipelineUmma::SharedStorage& storage) {

	using Pipeline = typename Traits::MainloopPipelineUmma;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	typename Pipeline::Params pp;
	pp.transaction_bytes = Traits::TmaTransBytes;
	pp.num_producers = 1;
	pp.num_consumers = 1;   // UMMA: one umma_arrive per stage releases the buffer
	if (is_producer) {
		pp.role = Category::Producer;
		pp.is_leader = (threadIdx.x == 0);
	} else if (is_consumer) {
		pp.role = Category::Consumer;
	} else {
		pp.role = Category::NonParticipant;
	}
	return Pipeline(storage, pp, Shape<_1, _1, _1>{},
		cute::true_type{}, cute::true_type{});
}

// ═══════════════════════════════════════════════════════════════════
// Fused producer — single (m), all (n) under blockIdx.y stride.
// Single fused Z + W TMA pipe per k-step.
//
// Per mlp2_t: the expert is contiguous along K (= hidden_dim), so the
// caller supplies expert_k_offset = expert · num_k_tiles. A is read
// through a column-major view (shape (intermediate_dim, total_k_cols))
// with N along intermediate_dim and K along E·hidden_dim.
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, bool Expert3D = false, typename Pipeline,
          typename TmaLoadZ, typename TmaLoadW>
__device__ __forceinline__ void mlp2_t_fused_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2TFusedSmem<Traits>& smem,
		TmaLoadZ const& tma_load_z,
		TmaLoadW const& tma_load_a,
		int m,
		int expert_or_k_offset,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
		int total_k_cols,        // num_experts · hidden_dim
		int num_n_tiles,
		int num_k_tiles,
		// Flat-grid N-split: split_idx = flat_id % NSplit, num_splits = NSplit.
		// Defaults (-1) fall back to blockIdx.y / gridDim.y for legacy callers.
		int split_idx = -1,
		int num_splits = -1) {

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

	// int64_t casts: num_tokens × hidden_dim and intermediate_dim ×
	// total_k_cols both exceed INT_MAX at production shapes.
	auto mZ = tma_load_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(hidden_dim)));
	auto mA = [&]() {
		if constexpr (Expert3D) {
			return tma_load_a.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_a.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(total_k_cols)));
		}
	}();

	auto cta_tma_z = tma_load_z.get_slice(Int<0>{});
	auto cta_tma_a = tma_load_a.get_slice(Int<0>{});

	auto tZsZ = cta_tma_z.partition_D(sZ);
	auto tWsW = cta_tma_a.partition_D(sW);

	auto gZ = local_tile(mZ,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tZgZ = cta_tma_z.partition_S(gZ);

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {
		auto gA = [&]() {
			if constexpr (Expert3D) {
				return local_tile(mA,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n, _, expert_or_k_offset));
			} else {
				return local_tile(mA,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n, _));
			}
		}();
		auto tAgA = cta_tma_a.partition_S(gA);

		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (threadIdx.x == 0) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_z.with(*bar, 0),
					tZgZ(_, _, _, k), tZsZ(_, _, _, state.index()));
				if constexpr (Expert3D) {
					copy(tma_load_a.with(*bar, 0),
						tAgA(_, _, _, k), tWsW(_, _, _, state.index()));
				} else {
					copy(tma_load_a.with(*bar, 0),
						tAgA(_, _, _, expert_or_k_offset + k),
						tWsW(_, _, _, state.index()));
				}
			}
			++state;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Fused consumer — architecture-specialized on `int Compute`
// ═══════════════════════════════════════════════════════════════════
//
// Primary template is undefined; one full specialization per supported compute
// capability provides a `run(...)` member (a function-template on
// Traits/Pipeline/TmaStoreY, since function templates can't be partially
// specialized). The free function `mlp2_t_fused_consumer` below forwards to the
// right specialization — call sites stay in the existing style.
//
//   Compute=90  → Hopper / WGMMA (cooperative 2-WG, single register acc)
//   Compute=100 → Blackwell / UMMA (single-warp issue, TMEM accumulator)
//
// y_m is the M-tile index into the output Y buffer.

template <int Compute>
struct Mlp2TFusedConsumerImpl;

// ───────────────────────────────────────────────────────────────────
// Compute=90 — Hopper. Split axis selected by TileM:
//   TileM=128 (M-split): WG_w writes rows [w*AtomTileM, (w+1)*AtomTileM).
//   TileM=64  (N-split): WG_w writes cols [w*WgTileN, (w+1)*WgTileN).
// (Verbatim from the original mlp2_t_fused_consumer.)
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp2TFusedConsumerImpl<90> {
template <typename Traits, typename Pipeline, typename TmaStoreY>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2TFusedSmem<Traits>& smem,
		TmaStoreY const& tma_store_y,
		int y_m,                // M-tile index into Y buffer
		int intermediate_dim,
		int num_y_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		// Flat-grid N-split (see producer). Defaults (-1) → blockIdx.y/gridDim.y.
		int split_idx,
		int num_splits) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

	auto tCsZ = thr_mma.partition_A(sZ);
	auto tCsW = thr_mma.partition_B(sW);

	auto acc = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});

	auto cC   = make_identity_tensor(make_shape(Int<Traits::TileM>{}, Int<Traits::TileN>{}));
	auto tCcC = thr_mma.partition_C(cC);

	const int my_wg = (threadIdx.x / Traits::WarpGroupSize) - 1;
	const int my_barrier_id = 1 + my_wg;
	const bool is_my_wg_leader = (tid_in_wg == 0);

	constexpr int store_slot_elems = Traits::AtomTileM * Traits::EpiChunkN;
	Element* my_store_ptr = smem.store_buf + my_wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	// Y view: [num_y_m_tiles · TileM, intermediate_dim] row-major.
	auto mY = tma_store_y.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_y_m_tiles) * Traits::TileM,
		static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_y = tma_store_y.get_slice(Int<0>{});

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;
	constexpr int K_PIPE_MMAS = 1;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		clear(acc);

		auto state_release = state;
		int prologue_count = (num_k_tiles < K_PIPE_MMAS) ? num_k_tiles : K_PIPE_MMAS;

		// Prologue
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc);
			warpgroup_arrive();
			gemm(tiled_mma, tCsZ(_, _, _, state.index()),
				tCsW(_, _, _, state.index()), acc);
			warpgroup_commit_batch();
			++state;
		}
		// Steady state
		for (int k = prologue_count; k < num_k_tiles; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc);
			warpgroup_arrive();
			gemm(tiled_mma, tCsZ(_, _, _, state.index()),
				tCsW(_, _, _, state.index()), acc);
			warpgroup_commit_batch();

			warpgroup_wait<K_PIPE_MMAS>();
			warpgroup_fence_operand(acc);
			pipe.consumer_release(state_release);
			++state;
			++state_release;
		}
		// Drain
		warpgroup_wait<0>();
		warpgroup_fence_operand(acc);
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_release(state_release);
			++state_release;
		}

		// ── Epilogue ──────────────────────────────────────
		// TileM=128 (M-split): WG_w owns rows [w*64, (w+1)*64), full N.
		// TileM=64  (N-split): WG_w owns full M (=64), cols [w*WgTileN, ...).
		CUTE_UNROLL
		for (int r = 0; r < Traits::NumEpiRounds; ++r) {
			if (store_in_flight)
				cute::tma_store_wait<0>();

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			int chunk_start = r * Traits::EpiChunkN;
			CUTE_UNROLL
			for (int i = 0; i < size(acc); ++i) {
				auto coord = tCcC(i);
				int m_loc = get<0>(coord);
				int n_loc = get<1>(coord);
				int m_local, n_local;
				if constexpr (Traits::kMSplit) {
					m_local = m_loc - my_wg * Traits::AtomTileM;
					n_local = n_loc;
				} else {
					m_local = m_loc;
					n_local = n_loc - my_wg * Traits::WgTileN;
				}
				if (n_local >= chunk_start &&
				    n_local <  chunk_start + Traits::EpiChunkN) {
					int chunk_n = n_local - chunk_start;
					sStore(m_local, chunk_n) = static_cast<Element>(acc(i));
				}
			}

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			if (is_my_wg_leader) {
				cute::tma_store_fence();
				int m_tile_idx, n_tile_idx;
				if constexpr (Traits::kMSplit) {
					// TileM=128: 2 row tiles per logical M tile.
					m_tile_idx = 2 * y_m + my_wg;
					n_tile_idx = n * Traits::NumEpiRounds + r;
				} else {
					// TileM=64: 1 row tile per logical M tile; WG_w handles
					// its N-half.
					m_tile_idx = y_m;
					n_tile_idx = n * (Traits::TileN / Traits::EpiChunkN)
					           + my_wg * Traits::NumEpiRounds + r;
				}
				auto gY = local_tile(mY,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(m_tile_idx, n_tile_idx));
				copy(tma_store_y, cta_tma_y.partition_S(sStore),
					cta_tma_y.partition_D(gY));
				cute::tma_store_arrive();
			}
			store_in_flight = true;
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}
};  // Mlp2TFusedConsumerImpl<90>

// ───────────────────────────────────────────────────────────────────
// Compute=100 — Blackwell / UMMA.
//
// No cooperative-warpgroup machinery (UMMA has no warpgroup concept):
//   - One 1SM UMMA atom with M=TileM covers the whole tile; the single
//     accumulator Y = Z·A lives in TMEM (no register/M-split).
//   - Warp 3 issues UMMA and remains epilogue-free; warps 4-11 run the
//     dual-WG epilogue.
//   - Epilogue: the two consumer warpgroups split TileN; each reads its
//     N-half from TMEM→regs in EpiChunkN chunks, casts to bf16, and stores
//     64-row (AtomTileM) tiles via the existing reg→SMEM→TMA path (so the host
//     TMA atom is unchanged). No activation fusion.
//
// mlp2_t-specific: operand B (the weight A) is consumed MN-major
// (Traits::TiledMmaUmma is SS<K,MN>) — the ONLY difference vs the mlp2_fused
// Compute=100 consumer (SS<K,K>). The epilogue is byte-for-byte identical.
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp2TFusedConsumerImpl<100> {
template <typename Traits, typename Pipeline, typename TmaStoreY>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2TFusedSmem<Traits>& smem,
		TmaStoreY const& tma_store_y,
		int y_m,
		int intermediate_dim,
		int num_y_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	constexpr int TileM     = Traits::TileM;
	constexpr int TileN     = Traits::TileN;
	constexpr int EpiChunkN = Traits::EpiChunkN;
	constexpr int AtomTileM = Traits::AtomTileM;     // 64-row TMA store tile
	static_assert(TileN % 2 == 0,
		"Blackwell consumer splits TileN across the two consumer warpgroups");
	constexpr int WgN         = TileN / 2;           // per-warpgroup N width
	static_assert(WgN % EpiChunkN == 0, "EpiChunkN must divide TileN/2");
	constexpr int NChunksHalf = WgN / EpiChunkN;     // n-chunks per warpgroup
	constexpr int MSub        = TileM / AtomTileM;   // 1 (TileM=64) or 2 (TileM=128)

	// ── Thread identity ─────────────────────────────────────
	// Warp 3 is the dedicated, epilogue-free UMMA producer. Warps 4..11 are
	// the two aligned epilogue WGs, mirroring the forward MLP2 SM100 path.
	const int  warp_id       = threadIdx.x / Traits::WarpSize;
	const bool is_mma_warp   = (warp_id == 3);
	const bool is_epilogue   = (warp_id >= 4 && warp_id <= 11);
	const int  tid_in_epi    = threadIdx.x - Traits::WarpGroupSize;  // warps 4..11 -> 0..255
	const int  wg            = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	const int  tid_wg        = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	const int  tmem_copy_tid = tid_wg;
	const bool is_wg_leader  = is_epilogue && (tid_wg == 0);
	const int  wg_barrier_id = 1 + wg;                               // 1 or 2
	constexpr int kEpilogueThreads = Traits::ConsumerThreads;
	constexpr int kMmaEpiThreads = kEpilogueThreads + Traits::WarpSize;
	static_assert(Traits::WarpGroupSize == 4 * Traits::WarpSize);
	static_assert(kEpilogueThreads == 8 * Traits::WarpSize);
	static_assert(kMmaEpiThreads == 9 * Traits::WarpSize);

	// ── TiledMMA: single 1SM UMMA atom, M=TileM, N=TileN, K-major A / MN-major
	//    B.  Operand A = Z (K-major), operand B = A (the weight, MN-major) — the
	//    mlp2_t transpose orientation. Traits::TiledMmaUmma is
	//    make_tiled_mma(SM100_MMA_F16BF16_SS<...,Major::K,Major::MN>{}). ──
	typename Traits::TiledMmaUmma tiled_mma;
	auto cta_mma = tiled_mma.get_slice(0);   // 1SM → single CTA, peer-coord 0

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

	// ── One TMEM accumulator: Y = Z·A ──
	auto cAccFull = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC     = cta_mma.partition_C(cAccFull);
	auto tCtAcc   = cta_mma.make_fragment_C(tCgC);

	// TMEM is allocated by the outer fused/standalone launcher once per CTA.

	// ── Accumulator pipeline: UMMA producer (warp 3) → epilogue consumers
	//    (warps 4..11, both WGs). AccStages TMEM stages let MMA(n+1) fill the
	//    alternate stage while epilogue drains/TMA-stores MMA(n). ──
	using AccPipe = typename Traits::AccumulatorPipeline;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp ? AccPipe::ThreadCategory::Producer
	                              : AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;          // one umma_arrive per commit
	acc_params.consumer_arv_count = 1;          // one elected releaser
	acc_params.initializing_warp  = 4;          // warp 4 inits the acc barriers
	AccPipe acc_pipe(smem.acc_pipe, acc_params,
		cute::Shape<cute::_1, cute::_1, cute::_1>{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
	const uint32_t tmem_base = smem.tmem_base;
	tCtAcc.data() = tmem_base;

	// ── Per-WG store slot (AtomTileM × EpiChunkN) — same smem as Hopper ──
	constexpr int store_slot_elems = AtomTileM * EpiChunkN;
	Element* my_store_ptr = smem.store_buf + wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	// Y view: [num_y_m_tiles · TileM, intermediate_dim] row-major. int64_t cast
	// keeps CUTE's layout math in 64-bit at production shapes.
	auto mY = tma_store_y.get_tma_tensor(
		make_shape(static_cast<int64_t>(num_y_m_tiles) * TileM,
		           static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_y = tma_store_y.get_slice(Int<0>{});

	// ── Epilogue TMEM→reg copy plumbing (built once; sliced per WG-local tid).
	//    flat_divide (not zipped_divide) keeps the epi tile as flat (M,N) modes,
	//    which is what make_tmem_copy's cotiled builder requires. ──
	auto epi_tile  = make_tile(Int<TileM>{}, Int<EpiChunkN>{});
	auto acc_mn    = tCtAcc(make_coord(_, _), _0{}, _0{});   // (TileM,TileN)
	auto tAcc_epi  = flat_divide(acc_mn, epi_tile);   // (TileM,EpiChunkN,1,TileN/EpiChunkN)
	auto t2r       = make_tmem_copy(TmemLoadOp<EpiChunkN>{}, tAcc_epi(_, _, _0{}, _0{}));
	auto thr_t2r   = t2r.get_slice(tmem_copy_tid);
	auto tTR_tAcc  = thr_t2r.partition_S(tAcc_epi);     // (Cpy,Cpy_M,Cpy_N,1,nTiles)
	auto cChunk    = make_identity_tensor(make_shape(Int<TileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);       // (Cpy,Cpy_M,Cpy_N)
	// Register fragment sized from the DEST (partition_D) shape, not partition_S.
	auto tTR_rAcc  = make_tensor<float>(shape(tTR_cChunk));   // f32 regs

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		// ── Mainloop (warp 3 only): bracket the k-loop with the accumulator
		//    pipeline. Each k-stage waits/releases the TMA→UMMA mainloop
		//    pipeline; consumer_release issues the UMMA-gated smem-buffer
		//    arrival. ──
		if (is_mma_warp) {
			acc_pipe.producer_acquire(acc_prod_state);   // TMEM acc free (prev epilogue done)
			int acc_stage = acc_prod_state.index();
			tCtAcc.data() = tmem_base + uint32_t(acc_stage * TileN);
			for (int k = 0; k < num_k_tiles; ++k) {
				pipe.consumer_wait(state);
				auto tCsZ = cta_mma.partition_A(sZ(_, _, state.index()));
				auto tCsW = cta_mma.partition_B(sW(_, _, state.index()));
				auto tCrZ = cta_mma.make_fragment_A(tCsZ);
				auto tCrW = cta_mma.make_fragment_B(tCsW);
				CUTE_UNROLL
				for (int kb = 0; kb < size<2>(tCrZ); ++kb) {
					// First k-block of this n-tile clears TMEM; the rest accumulate.
					tiled_mma.accumulate_ = (k == 0 && kb == 0)
						? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
					gemm(tiled_mma, tCrZ(_, _, kb), tCrW(_, _, kb), tCtAcc);
				}
				pipe.consumer_release(state);   // UMMA-gated smem-buffer release
				++state;
			}
			acc_pipe.producer_commit(acc_prod_state);    // signal acc ready (umma_arrive)
			++acc_prod_state;
		}

		// ── Epilogue: wait for the accumulator, then this WG processes its
		//    n-chunks [wg·NChunksHalf, +NChunksHalf). ──
		if (is_epilogue) {
			acc_pipe.consumer_wait(acc_cons_state);
			int acc_stage = acc_cons_state.index();
			tCtAcc.data() = tmem_base + uint32_t(acc_stage * TileN);
			auto acc_mn_stage   = tCtAcc(make_coord(_, _), _0{}, _0{});
			auto tAcc_epi_stage = flat_divide(acc_mn_stage, epi_tile);
			auto tTR_tAcc_stage = thr_t2r.partition_S(tAcc_epi_stage);

			CUTE_UNROLL
			for (int r = 0; r < NChunksHalf; ++r) {
				int chunk = wg * NChunksHalf + r;            // absolute n-chunk index

				// TMEM → registers (this chunk, full TileM rows).
				copy(t2r, tTR_tAcc_stage(_, _, _, _0{}, chunk), tTR_rAcc);

				// Store as MSub × (AtomTileM=64)-row TMA tiles (1 for TileM=64, 2 for 128).
				CUTE_UNROLL
				for (int ms = 0; ms < MSub; ++ms) {
					if (store_in_flight)
						cute::tma_store_wait<0>();

					cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, wg_barrier_id);
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rAcc); ++i) {
						int m_row = get<0>(tTR_cChunk(i));   // 0..TileM
						int n_col = get<1>(tTR_cChunk(i));   // 0..EpiChunkN
						if (m_row >= ms * AtomTileM && m_row < (ms + 1) * AtomTileM)
							sStore(m_row - ms * AtomTileM, n_col) =
								static_cast<Element>(tTR_rAcc(i));
					}
					cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, wg_barrier_id);

					if (is_wg_leader) {
						cute::tma_store_fence();
						int m_tile_idx = MSub * y_m + ms;
						int n_tile_idx = n * (TileN / EpiChunkN) + chunk;
						auto gY = local_tile(mY,
							make_tile(Int<AtomTileM>{}, Int<EpiChunkN>{}),
							make_coord(m_tile_idx, n_tile_idx));
						copy(tma_store_y, cta_tma_y.partition_S(sStore),
							cta_tma_y.partition_D(gY));
						cute::tma_store_arrive();
					}
					store_in_flight = true;
				}
			}

			// All epilogue TMEM reads are done; release the accumulator so the next
			// n-tile's MMA may reuse it (one elected consumer thread arrives).
			cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
			if (tid_in_epi == 0)
				acc_pipe.consumer_release(acc_cons_state);
			++acc_cons_state;
		}
	}
	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();

	// Keep the final MMA+epilogue barrier so the outer launcher can safely free
	// TMEM after this consumer returns.
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
#else
	// The Compute=100 specialization is never dispatched on non-SM100 targets
	// (host picks Compute=90 there). Trap if it is ever reached.
	__trap();
#endif
}
};  // Mlp2TFusedConsumerImpl<100>

// ───────────────────────────────────────────────────────────────────
// Forwarder — keeps the existing call style. `Compute` defaults to 90 so
// current call sites (mlp2_t_fused_consumer<Traits>(...)) are unchanged; pass
// mlp2_t_fused_consumer<Traits, 100>(...) to select the Blackwell path.
// ───────────────────────────────────────────────────────────────────

template <typename Traits, int Compute = 90, typename Pipeline, typename TmaStoreY>
__device__ __forceinline__ void mlp2_t_fused_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2TFusedSmem<Traits>& smem,
		TmaStoreY const& tma_store_y,
		int y_m,
		int intermediate_dim,
		int num_y_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx = -1,
		int num_splits = -1) {
	Mlp2TFusedConsumerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_store_y, y_m, intermediate_dim,
		num_y_m_tiles, num_n_tiles, num_k_tiles, split_idx, num_splits);
}

} // namespace liger
