#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 1 (fused tile + silu-backward coefficients):
//   U  = B @ X
//   V  = C @ X
//   V' = silu(U)           (coefficient for dV = dZ * V' later)
//   U' = V * silu'(U)      (coefficient for dU = dZ * U' later)
//   Z  = V' * V = silu(U) * V   (forward output, consumed by mlp3)
// ═══════════════════════════════════════════════════════════════════
//
// Cooperative 2-WG consumer producing three outputs (U', V', Z) per
// (TileM, TileN) tile. Split axis selected by TileM (kMSplit pattern,
// same as mlp1_fused.cuh / mlp2_t_fused.cuh / mlp5_fused.cuh):
//   TileM=128 (M-split, Layout<_2,_1,_1>): each WG owns AtomTileM=64
//     rows × full TileN. WG_w writes rows [w*64, (w+1)*64).
//   TileM=64  (N-split, Layout<_1,_2,_1>): each WG owns full M (=64)
//     × WgTileN = TileN/2. WG_w writes cols [w*WgTileN, (w+1)*WgTileN).
// Dual acc_B/acc_C from a single fused X+W1+W2 TMA pipeline,
// EpiChunkN-wide per-WG epilogue rounds — each round writes three
// smem buffers and issues three TMA stores.
//
// Smem layout (per Mlp1Traits): three per-WG store slots. WG_w uses
// indices [w·S, (w+1)·S) within each buf_{u,v,z}, where S =
// AtomTileM·EpiChunkN. Stages=3 default (vs Stages=4 in fwd mlp1) to
// fit the extra two output buffers within the 228 KiB SM smem cap.
//
// Single-tile API: caller passes (m, expert_n_offset) and a single
// fused X+B+C pipeline. Used by the standalone mlp1_act launcher.

#include "mlp1_fused.cuh"

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// Shared memory — three per-WG store buffers (U', V', Z)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp1FusedActSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_X_size     = cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int smem_W_size     = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_X[smem_X_size];
	alignas(128) Element smem_W1[smem_W_size];
	alignas(128) Element smem_W2[smem_W_size];
	// 3 outputs × 2 WGs of per-WG (AtomTileM, EpiChunkN) slots.
	alignas(128) Element store_buf_u[2 * smem_store_size];
	alignas(128) Element store_buf_v[2 * smem_store_size];
	alignas(128) Element store_buf_z[2 * smem_store_size];

	// SM100 (Compute=100) only: tcgen05.alloc landing slot + accumulator
	// pipeline storage (see Mlp1FusedSmem). Untouched by the Hopper path.
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

	CUTE_DEVICE Element* X_data()  { return &smem_X[0]; }
	CUTE_DEVICE Element* W1_data() { return &smem_W1[0]; }
	CUTE_DEVICE Element* W2_data() { return &smem_W2[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// Producer — single (m), all (n) under split_idx/num_splits stride.
// Single fused X + W1 + W2 TMA pipe per k-step.
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, bool Expert3D = false, typename Pipeline,
          typename TmaLoadX, typename TmaLoadW>
__device__ __forceinline__ void mlp1_fused_act_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedActSmem<Traits>& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		int m,
		int expert_or_n_offset,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
		int total_n_rows,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {

	auto sX  = make_tensor(make_smem_ptr(smem.X_data()),  typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	// int64_t cast for production shapes where E·I·D > 2^31.
	auto mX = tma_load_x.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(hidden_dim)));
	auto mB = [&]() {
		if constexpr (Expert3D) {
			return tma_load_b.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_b.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_n_rows),
				static_cast<int64_t>(hidden_dim)));
		}
	}();
	auto mC = [&]() {
		if constexpr (Expert3D) {
			return tma_load_c.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_c.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_n_rows),
				static_cast<int64_t>(hidden_dim)));
		}
	}();

	auto cta_tma_x = tma_load_x.get_slice(Int<0>{});
	auto cta_tma_b = tma_load_b.get_slice(Int<0>{});
	auto cta_tma_c = tma_load_c.get_slice(Int<0>{});

	auto tXsX   = cta_tma_x.partition_D(sX);
	auto tW1sW1 = cta_tma_b.partition_D(sW1);
	auto tW2sW2 = cta_tma_c.partition_D(sW2);

	auto gX = local_tile(mX,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tXgX = cta_tma_x.partition_S(gX);

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {
		auto gB = [&]() {
			if constexpr (Expert3D) {
				return local_tile(mB,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n, _, expert_or_n_offset));
			} else {
				return local_tile(mB,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(expert_or_n_offset + n, _));
			}
		}();
		auto gC = [&]() {
			if constexpr (Expert3D) {
				return local_tile(mC,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n, _, expert_or_n_offset));
			} else {
				return local_tile(mC,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(expert_or_n_offset + n, _));
			}
		}();
		auto tBgB = cta_tma_b.partition_S(gB);
		auto tCgC = cta_tma_c.partition_S(gC);

		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (threadIdx.x == 0) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_x.with(*bar, 0),
					tXgX(_, _, _, k), tXsX(_, _, _, state.index()));
				copy(tma_load_b.with(*bar, 0),
					tBgB(_, _, _, k), tW1sW1(_, _, _, state.index()));
				copy(tma_load_c.with(*bar, 0),
					tCgC(_, _, _, k), tW2sW2(_, _, _, state.index()));
			}
			++state;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Consumer — architecture-specialized on `int Compute` (see
// Mlp1FusedConsumerImpl in mlp1_fused.cuh for the dispatch rationale).
//   Compute=90  → Hopper / WGMMA (cooperative 2-WG)
//   Compute=100 → Blackwell / UMMA (single-warp issue, TMEM accumulators)
// ═══════════════════════════════════════════════════════════════════

template <int Compute>
struct Mlp1FusedActConsumerImpl;

// ───────────────────────────────────────────────────────────────────
// Compute=90 — Hopper. z_m is the M-tile index into the output buffers.
// Split axis selected by TileM (kMSplit pattern):
//   TileM=128 (M-split): WG_w writes rows [w*AtomTileM, (w+1)*AtomTileM)
//   TileM=64  (N-split): WG_w writes cols [w*WgTileN, (w+1)*WgTileN)
// Each WG issues NumEpiRounds × 3 TMA stores per N-tile (U', V', Z,
// each at (AtomTileM, EpiChunkN)). (Verbatim from the original consumer.)
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp1FusedActConsumerImpl<90> {
template <typename Traits, typename Pipeline, typename TmaStore>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedActSmem<Traits>& smem,
		TmaStore const& tma_store_du_coef,
		TmaStore const& tma_store_dv_coef,
		TmaStore const& tma_store_z,
		int z_m,
		int intermediate_dim,
		int num_z_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;   // 0..255
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;    // 0..127
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sX  = make_tensor(make_smem_ptr(smem.X_data()),  typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	auto tCsX  = thr_mma.partition_A(sX);
	auto tCsW1 = thr_mma.partition_B(sW1);
	auto tCsW2 = thr_mma.partition_B(sW2);

	auto acc_B = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});
	auto acc_C = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});

	// Identity coords for per-thread acc → store_buf mapping.
	auto cC   = make_identity_tensor(make_shape(Int<Traits::TileM>{}, Int<Traits::TileN>{}));
	auto tCcC = thr_mma.partition_C(cC);

	const int my_wg = (threadIdx.x / Traits::WarpGroupSize) - 1;  // 0 or 1
	const int my_barrier_id = 1 + my_wg;                          // 1 or 2
	const bool is_my_wg_leader = (tid_in_wg == 0);

	constexpr int store_slot_elems = Traits::AtomTileM * Traits::EpiChunkN;
	Element* my_store_u = smem.store_buf_u + my_wg * store_slot_elems;
	Element* my_store_v = smem.store_buf_v + my_wg * store_slot_elems;
	Element* my_store_z = smem.store_buf_z + my_wg * store_slot_elems;
	auto sStoreU = make_tensor(make_smem_ptr(my_store_u),
		typename Traits::SmemLayoutStoreSlot{});
	auto sStoreV = make_tensor(make_smem_ptr(my_store_v),
		typename Traits::SmemLayoutStoreSlot{});
	auto sStoreZ = make_tensor(make_smem_ptr(my_store_z),
		typename Traits::SmemLayoutStoreSlot{});

	// int64_t cast: num_z_m_tiles · TileM · intermediate_dim can exceed INT_MAX.
	auto out_shape = make_shape(
		static_cast<int64_t>(num_z_m_tiles) * Traits::TileM,
		static_cast<int64_t>(intermediate_dim));
	auto mU = tma_store_du_coef.get_tma_tensor(out_shape);
	auto mV = tma_store_dv_coef.get_tma_tensor(out_shape);
	auto mZ = tma_store_z.get_tma_tensor(out_shape);
	auto cta_tma_u = tma_store_du_coef.get_slice(Int<0>{});
	auto cta_tma_v = tma_store_dv_coef.get_slice(Int<0>{});
	auto cta_tma_z = tma_store_z.get_slice(Int<0>{});

	bool store_in_flight = false;
	constexpr int K_PIPE_MMAS = 1;  // CUTLASS-matching

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {

		clear(acc_B);
		clear(acc_C);

		auto state_release = state;
		int prologue_count = (num_k_tiles < K_PIPE_MMAS) ? num_k_tiles : K_PIPE_MMAS;

		// Prologue
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc_B);
			warpgroup_fence_operand(acc_C);
			warpgroup_arrive();
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW1(_, _, _, state.index()), acc_B);
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW2(_, _, _, state.index()), acc_C);
			warpgroup_commit_batch();
			++state;
		}
		// Steady state
		for (int k = prologue_count; k < num_k_tiles; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc_B);
			warpgroup_fence_operand(acc_C);
			warpgroup_arrive();
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW1(_, _, _, state.index()), acc_B);
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW2(_, _, _, state.index()), acc_C);
			warpgroup_commit_batch();

			warpgroup_wait<K_PIPE_MMAS>();
			warpgroup_fence_operand(acc_B);
			warpgroup_fence_operand(acc_C);
			pipe.consumer_release(state_release);
			++state;
			++state_release;
		}
		// Drain
		warpgroup_wait<0>();
		warpgroup_fence_operand(acc_B);
		warpgroup_fence_operand(acc_C);
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_release(state_release);
			++state_release;
		}

		// ── Epilogue: 3-store per-WG rounds ─────────────
		// Per round each WG: write U', V', Z into 3 separate per-WG smem
		// slots, then issue 3 TMA stores. No cross-WG sync — each WG
		// owns its strip independently.
		// TileM=128 (M-split): WG_w owns rows [w*AtomTileM, (w+1)*AtomTileM),
		//   full N. Store at row_tile = 2*z_m + my_wg, col_tile = n*NER + r.
		// TileM=64 (N-split): WG_w owns full M (=64), cols
		//   [w*WgTileN, (w+1)*WgTileN). Store at row_tile = z_m,
		//   col_tile = n * (TileN/EpiChunkN) + my_wg*NER + r.
		CUTE_UNROLL
		for (int r = 0; r < Traits::NumEpiRounds; ++r) {
			if (store_in_flight)
				cute::tma_store_wait<0>();

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			int chunk_start = r * Traits::EpiChunkN;
			CUTE_UNROLL
			for (int i = 0; i < size(acc_B); ++i) {
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
					float u = acc_B(i);
					float v = acc_C(i);
					float sig    = fast_sigmoid(u);
					float vprime = u * sig;                          // silu(U)
					float sil_d  = sig + vprime * (1.0f - sig);      // silu'(U)
					float uprime = v * sil_d;                        // V · silu'(U)
					float z      = vprime * v;                       // Z = V' * V
					sStoreU(m_local, chunk_n) = static_cast<Element>(uprime);
					sStoreV(m_local, chunk_n) = static_cast<Element>(vprime);
					sStoreZ(m_local, chunk_n) = static_cast<Element>(z);
				}
			}

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			if (is_my_wg_leader) {
				cute::tma_store_fence();
				int row_tile, col_tile;
				if constexpr (Traits::kMSplit) {
					// TileM=128: 2 row tiles per logical M tile.
					row_tile = 2 * z_m + my_wg;
					col_tile = n * Traits::NumEpiRounds + r;
				} else {
					// TileM=64: 1 row tile per logical M tile; WG_w handles
					// its N-half.
					row_tile = z_m;
					col_tile = n * (Traits::TileN / Traits::EpiChunkN)
					         + my_wg * Traits::NumEpiRounds + r;
				}
				auto gU = local_tile(mU,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(row_tile, col_tile));
				auto gV = local_tile(mV,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(row_tile, col_tile));
				auto gZ = local_tile(mZ,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(row_tile, col_tile));
				copy(tma_store_du_coef, cta_tma_u.partition_S(sStoreU),
					cta_tma_u.partition_D(gU));
				cute::tma_store_arrive();
				copy(tma_store_dv_coef, cta_tma_v.partition_S(sStoreV),
					cta_tma_v.partition_D(gV));
				cute::tma_store_arrive();
				copy(tma_store_z, cta_tma_z.partition_S(sStoreZ),
					cta_tma_z.partition_D(gZ));
				cute::tma_store_arrive();
			}
			store_in_flight = true;
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}
};  // Mlp1FusedActConsumerImpl<90>

// ───────────────────────────────────────────────────────────────────
// Compute=100 — Blackwell / UMMA. Same structure as the fused
// Mlp1FusedConsumerImpl<100>, but the epilogue derives the three act
// outputs from (U, V) and issues three TMA stores per 64-row tile:
//   V' = silu(U),  U' = V·silu'(U),  Z = V'·V.
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp1FusedActConsumerImpl<100> {
template <typename Traits, typename Pipeline, typename TmaStore>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedActSmem<Traits>& smem,
		TmaStore const& tma_store_du_coef,
		TmaStore const& tma_store_dv_coef,
		TmaStore const& tma_store_z,
		int z_m,
		int intermediate_dim,
		int num_z_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	constexpr int TileM     = Traits::TileM;
	constexpr int TileN     = Traits::TileN;
	constexpr int EpiChunkN = Traits::EpiChunkN;
	constexpr int AtomTileM = Traits::AtomTileM;
	static_assert(TileN % 2 == 0,
		"Blackwell consumer splits TileN across the two consumer warpgroups");
	constexpr int WgN         = TileN / 2;
	static_assert(WgN % EpiChunkN == 0, "EpiChunkN must divide TileN/2");
	constexpr int NChunksHalf = WgN / EpiChunkN;
	constexpr int MSub        = TileM / AtomTileM;

	const int  warp_id       = threadIdx.x / Traits::WarpSize;
	const bool is_mma_warp   = (warp_id == 3);
	const bool is_epilogue   = (warp_id >= 4 && warp_id <= 11);
	const int  tid_in_epi    = threadIdx.x - Traits::WarpGroupSize;  // warps 4..11 -> 0..255
	const int  wg            = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	const int  tid_wg        = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	const int  tmem_copy_tid = tid_wg;
	const bool is_wg_leader  = is_epilogue && (tid_wg == 0);
	const int  wg_barrier_id = 1 + wg;
	constexpr int kEpilogueThreads = Traits::ConsumerThreads;
	constexpr int kMmaEpiThreads = kEpilogueThreads + Traits::WarpSize;
	static_assert(Traits::WarpGroupSize == 4 * Traits::WarpSize);
	static_assert(kEpilogueThreads == 8 * Traits::WarpSize);
	static_assert(kMmaEpiThreads == 9 * Traits::WarpSize);

	auto tiled_mma = make_tiled_mma(
		SM100_MMA_F16BF16_SS<Element, Element, float, TileM, TileN,
		                     UMMA::Major::K, UMMA::Major::K>{});
	auto cta_mma = tiled_mma.get_slice(0);

	auto sX  = make_tensor(make_smem_ptr(smem.X_data()),  typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	auto cAccFull = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC     = cta_mma.partition_C(cAccFull);
	auto tCtAccU  = cta_mma.make_fragment_C(tCgC);   // U = X·B
	auto tCtAccV  = cta_mma.make_fragment_C(tCgC);   // V = X·C

	// TMEM is allocated by the outer fused/standalone launcher once per CTA.

	// Accumulator pipeline: UMMA producer (warp 3) → epilogue consumers
	// (warps 4..11, both WGs). AccStages TMEM stages let MMA(n+1) fill the
	// alternate stage while both epilogue WGs drain MMA(n).
	using AccPipe = typename Traits::AccumulatorPipeline;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp ? AccPipe::ThreadCategory::Producer
	                              : AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = 1;
	acc_params.initializing_warp  = 4;
	AccPipe acc_pipe(smem.acc_pipe, acc_params,
		cute::Shape<cute::_1, cute::_1, cute::_1>{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
	const uint32_t tmem_base = smem.tmem_base;
	tCtAccU.data() = tmem_base;
	tCtAccV.data() = tmem_base + uint32_t(TileN);

	// Three per-WG store slots (U', V', Z) — same smem as Hopper.
	constexpr int store_slot_elems = AtomTileM * EpiChunkN;
	auto sStoreU = make_tensor(make_smem_ptr(smem.store_buf_u + wg * store_slot_elems),
		typename Traits::SmemLayoutStoreSlot{});
	auto sStoreV = make_tensor(make_smem_ptr(smem.store_buf_v + wg * store_slot_elems),
		typename Traits::SmemLayoutStoreSlot{});
	auto sStoreZ = make_tensor(make_smem_ptr(smem.store_buf_z + wg * store_slot_elems),
		typename Traits::SmemLayoutStoreSlot{});

	auto out_shape = make_shape(
		static_cast<int64_t>(num_z_m_tiles) * TileM,
		static_cast<int64_t>(intermediate_dim));
	auto mU = tma_store_du_coef.get_tma_tensor(out_shape);
	auto mV = tma_store_dv_coef.get_tma_tensor(out_shape);
	auto mZ = tma_store_z.get_tma_tensor(out_shape);
	auto cta_tma_u = tma_store_du_coef.get_slice(Int<0>{});
	auto cta_tma_v = tma_store_dv_coef.get_slice(Int<0>{});
	auto cta_tma_z = tma_store_z.get_slice(Int<0>{});

	auto epi_tile  = make_tile(Int<TileM>{}, Int<EpiChunkN>{});
	// Flat (M,N) view of the UMMA C-fragment before tiling (see mlp1_fused.cuh).
	auto accU_mn   = tCtAccU(make_coord(_, _), _0{}, _0{});   // (TileM,TileN)
	auto accV_mn   = tCtAccV(make_coord(_, _), _0{}, _0{});
	auto tAccU_epi = flat_divide(accU_mn, epi_tile);   // (TileM,EpiChunkN,1,TileN/EpiChunkN)
	auto tAccV_epi = flat_divide(accV_mn, epi_tile);
	auto t2r       = make_tmem_copy(TmemLoadOp<EpiChunkN>{}, tAccU_epi(_, _, _0{}, _0{}));
	auto thr_t2r   = t2r.get_slice(tmem_copy_tid);
	auto tTR_tAccU = thr_t2r.partition_S(tAccU_epi);   // (Cpy,Cpy_M,Cpy_N,1,nTiles)
	auto tTR_tAccV = thr_t2r.partition_S(tAccV_epi);
	auto cChunk    = make_identity_tensor(make_shape(Int<TileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);     // (Cpy,Cpy_M,Cpy_N)
	// Register fragments sized from the DEST (partition_D) per-thread shape, not
	// partition_S(tmem) (warp-collective). See mlp1_fused.cuh for the rationale.
	auto tTR_rU = make_tensor<float>(shape(tTR_cChunk));   // f32 regs (U)
	auto tTR_rV = make_tensor<float>(shape(tTR_cChunk));   // f32 regs (V)

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		// Mainloop (warp 3 only), bracketed by the accumulator pipeline.
		if (is_mma_warp) {
			acc_pipe.producer_acquire(acc_prod_state);
			int acc_stage = acc_prod_state.index();
			uint32_t stage_base = tmem_base + uint32_t(acc_stage * (2 * TileN));
			tCtAccU.data() = stage_base;
			tCtAccV.data() = stage_base + uint32_t(TileN);
			for (int k = 0; k < num_k_tiles; ++k) {
				pipe.consumer_wait(state);
				auto tCsX  = cta_mma.partition_A(sX (_, _, state.index()));
				auto tCsW1 = cta_mma.partition_B(sW1(_, _, state.index()));
				auto tCsW2 = cta_mma.partition_B(sW2(_, _, state.index()));
				auto tCrX  = cta_mma.make_fragment_A(tCsX);
				auto tCrW1 = cta_mma.make_fragment_B(tCsW1);
				auto tCrW2 = cta_mma.make_fragment_B(tCsW2);
				CUTE_UNROLL
				for (int kb = 0; kb < size<2>(tCrX); ++kb) {
					tiled_mma.accumulate_ = (k == 0 && kb == 0)
						? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
					gemm(tiled_mma, tCrX(_, _, kb), tCrW1(_, _, kb), tCtAccU);
					gemm(tiled_mma, tCrX(_, _, kb), tCrW2(_, _, kb), tCtAccV);
				}
				pipe.consumer_release(state);
				++state;
			}
			acc_pipe.producer_commit(acc_prod_state);
			++acc_prod_state;
		}

		if (is_epilogue) {
			acc_pipe.consumer_wait(acc_cons_state);
			int acc_stage = acc_cons_state.index();
			uint32_t stage_base = tmem_base + uint32_t(acc_stage * (2 * TileN));
			tCtAccU.data() = stage_base;
			tCtAccV.data() = stage_base + uint32_t(TileN);
			auto accU_mn_stage   = tCtAccU(make_coord(_, _), _0{}, _0{});
			auto accV_mn_stage   = tCtAccV(make_coord(_, _), _0{}, _0{});
			auto tAccU_epi_stage = flat_divide(accU_mn_stage, epi_tile);
			auto tAccV_epi_stage = flat_divide(accV_mn_stage, epi_tile);
			auto tTR_tAccU_stage = thr_t2r.partition_S(tAccU_epi_stage);
			auto tTR_tAccV_stage = thr_t2r.partition_S(tAccV_epi_stage);

			CUTE_UNROLL
			for (int r = 0; r < NChunksHalf; ++r) {
				int chunk = wg * NChunksHalf + r;

				// U and V stay in registers (tTR_rU, tTR_rV); the three act
				// outputs are derived and cast to Element inline at the smem write
				// below — no extra register sets. Each reg's row falls in exactly
				// one MSub slice, so this evaluates each element once.
				copy(t2r, tTR_tAccU_stage(_, _, _, _0{}, chunk), tTR_rU);
				copy(t2r, tTR_tAccV_stage(_, _, _, _0{}, chunk), tTR_rV);

				CUTE_UNROLL
				for (int ms = 0; ms < MSub; ++ms) {
					if (store_in_flight)
						cute::tma_store_wait<0>();

					cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, wg_barrier_id);
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rU); ++i) {
						int m_row = get<0>(tTR_cChunk(i));
						int n_col = get<1>(tTR_cChunk(i));
						if (m_row >= ms * AtomTileM && m_row < (ms + 1) * AtomTileM) {
							int m_loc = m_row - ms * AtomTileM;
							float u = tTR_rU(i);
							float v = tTR_rV(i);
							float sig    = fast_sigmoid(u);
							float vprime = u * sig;                      // silu(U)
							float sil_d  = sig + vprime * (1.0f - sig);  // silu'(U)
							sStoreU(m_loc, n_col) = static_cast<Element>(v * sil_d);  // V·silu'(U)
							sStoreV(m_loc, n_col) = static_cast<Element>(vprime);     // silu(U)
							sStoreZ(m_loc, n_col) = static_cast<Element>(vprime * v); // V'·V
						}
					}
					cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, wg_barrier_id);

					if (is_wg_leader) {
						cute::tma_store_fence();
						int row_tile = MSub * z_m + ms;
						int col_tile = n * (TileN / EpiChunkN) + chunk;
						auto gU = local_tile(mU,
							make_tile(Int<AtomTileM>{}, Int<EpiChunkN>{}),
							make_coord(row_tile, col_tile));
						auto gV = local_tile(mV,
							make_tile(Int<AtomTileM>{}, Int<EpiChunkN>{}),
							make_coord(row_tile, col_tile));
						auto gZ = local_tile(mZ,
							make_tile(Int<AtomTileM>{}, Int<EpiChunkN>{}),
							make_coord(row_tile, col_tile));
						copy(tma_store_du_coef, cta_tma_u.partition_S(sStoreU),
							cta_tma_u.partition_D(gU));
						cute::tma_store_arrive();
						copy(tma_store_dv_coef, cta_tma_v.partition_S(sStoreV),
							cta_tma_v.partition_D(gV));
						cute::tma_store_arrive();
						copy(tma_store_z, cta_tma_z.partition_S(sStoreZ),
							cta_tma_z.partition_D(gZ));
						cute::tma_store_arrive();
					}
					store_in_flight = true;
				}
			}

			// Epilogue done reading TMEM; release the accumulator for the next
			// n-tile's MMA (one elected consumer thread arrives).
			cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
			if (tid_in_epi == 0)
				acc_pipe.consumer_release(acc_cons_state);
			++acc_cons_state;
		}
	}
	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();

	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
#else
	__trap();
#endif
}
};  // Mlp1FusedActConsumerImpl<100>

// Forwarder — `Compute` defaults to 90 (existing call sites unchanged);
// pass <Traits, 100> for the Blackwell path.
template <typename Traits, int Compute = 90, typename Pipeline, typename TmaStore>
__device__ __forceinline__ void mlp1_fused_act_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedActSmem<Traits>& smem,
		TmaStore const& tma_store_du_coef,
		TmaStore const& tma_store_dv_coef,
		TmaStore const& tma_store_z,
		int z_m,
		int intermediate_dim,
		int num_z_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {
	Mlp1FusedActConsumerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_store_du_coef, tma_store_dv_coef, tma_store_z,
		z_m, intermediate_dim, num_z_m_tiles, num_n_tiles, num_k_tiles,
		split_idx, num_splits);
}

} // namespace liger
