#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 5 (single-tile API for fused use)
// ═══════════════════════════════════════════════════════════════════
//
// Same kernel body as mlp5.cuh's nested-loop default, exposed as a
// single-(m)-tile producer + consumer so a parent kernel can drive the
// outer m loop itself (e.g. mlp_bwd.cuh composing Phase 1d).
//
// Mirrors how mlp1_fused.cuh exposes single-tile entry points: callers
// own the Pipeline construction (use
// mlp5_make_pipe from mlp5.cuh) and the Mlp5Smem allocation; this
// header just provides the producer/consumer body parameterized for
// ONE m-tile.
//
// Params unique to fused use:
//   - m               : the m-tile this invocation processes
//   - expert_k_offset : expert · num_k_tiles (caller resolves the
//                       expert lookup once per m)
//   - split_idx       : the column index within an N-split (caller
//                       picks; analogous to blockIdx.y in the
//                       standalone driver)
//   - num_splits      : total columns (analogous to gridDim.y)
//
// Output store goes to a global dX tensor view of shape
// (num_m_tiles · TileM, hidden_dim) (same as the standalone consumer).
// ═══════════════════════════════════════════════════════════════════

#include "mlp5.cuh"  // Mlp5Traits, Mlp5Smem, mlp5_make_pipe

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// Producer (single m-tile) — fills 2·num_k_tiles slots per (m, n)
// across n ∈ {split_idx, split_idx+num_splits, ...}.
//   Phase 1 (k=0..num_k_tiles-1):  Z=dU, W=B
//   Phase 2 (k=num_k_tiles..2K-1): Z=dV, W=C
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, bool Expert3D = false, typename Pipeline,
          typename TmaLoadZ, typename TmaLoadW>
__device__ __forceinline__ void mlp5_fused_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaLoadZ const& tma_load_du,
		TmaLoadZ const& tma_load_dv,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		int m,
		int expert_or_k_offset,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
		int total_k_cols,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {

	auto sZ = make_tensor(make_smem_ptr(smem.smem_Z), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.smem_W), typename Traits::SmemLayoutW{});

	// int64_t cast for production shapes where T · I or E · I · H can
	// exceed 2^31 elements (same rationale as in mlp1_fused_act.cuh and
	// the int64 commit on the fwd path).
	auto mDU = tma_load_du.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(intermediate_dim)));
	auto mDV = tma_load_dv.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(intermediate_dim)));
	auto mB = [&]() {
		if constexpr (Expert3D) {
			return tma_load_b.get_tma_tensor(make_shape(
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_b.get_tma_tensor(make_shape(
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(total_k_cols)));
		}
	}();
	auto mC = [&]() {
		if constexpr (Expert3D) {
			return tma_load_c.get_tma_tensor(make_shape(
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_c.get_tma_tensor(make_shape(
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(total_k_cols)));
		}
	}();

	auto cta_tma_du = tma_load_du.get_slice(Int<0>{});
	auto cta_tma_b  = tma_load_b.get_slice(Int<0>{});

	auto tZsZ = cta_tma_du.partition_D(sZ);
	auto tWsW = cta_tma_b.partition_D(sW);

	auto gB_all = [&]() {
		if constexpr (Expert3D) {
			return local_tile(mB,
				make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
				make_coord(_, _, expert_or_k_offset));
		} else {
			return local_tile(mB,
				make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
				make_coord(_, _));
		}
	}();
	auto gC_all = [&]() {
		if constexpr (Expert3D) {
			return local_tile(mC,
				make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
				make_coord(_, _, expert_or_k_offset));
		} else {
			return local_tile(mC,
				make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
				make_coord(_, _));
		}
	}();
	auto tBgB_all = cta_tma_b.partition_S(gB_all);
	auto tCgC_all = cta_tma_b.partition_S(gC_all);

	auto gDU = local_tile(mDU,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tZgDU = cta_tma_du.partition_S(gDU);
	auto gDV = local_tile(mDV,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tZgDV = cta_tma_du.partition_S(gDV);

	const bool is_leader = (threadIdx.x == 0);

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {
		// Phase 1: dU @ B
		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (is_leader) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_du.with(*bar, 0),
					tZgDU(_, _, _, k), tZsZ(_, _, _, state.index()));
				if constexpr (Expert3D) {
					copy(tma_load_b.with(*bar, 0),
						tBgB_all(_, _, _, n, k), tWsW(_, _, _, state.index()));
				} else {
					copy(tma_load_b.with(*bar, 0),
						tBgB_all(_, _, _, n, expert_or_k_offset + k),
						tWsW(_, _, _, state.index()));
				}
			}
			++state;
		}
		// Phase 2: dV @ C
		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (is_leader) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_dv.with(*bar, 0),
					tZgDV(_, _, _, k), tZsZ(_, _, _, state.index()));
				if constexpr (Expert3D) {
					copy(tma_load_c.with(*bar, 0),
						tCgC_all(_, _, _, n, k), tWsW(_, _, _, state.index()));
				} else {
					copy(tma_load_c.with(*bar, 0),
						tCgC_all(_, _, _, n, expert_or_k_offset + k),
						tWsW(_, _, _, state.index()));
				}
			}
			++state;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Consumer (single m-tile) — architecture-specialized on `int Compute`
// ═══════════════════════════════════════════════════════════════════
//
// Primary template is undefined; one full specialization per supported
// compute capability provides a `run(...)` member (a function-template on
// Traits/Pipeline/TmaStoreDX, since function templates can't be partially
// specialized). The free function `mlp5_fused_consumer` below forwards to
// the right specialization — call sites stay in the existing style.
//
//   Compute=90  → Hopper / WGMMA (cooperative 2-WG, single register acc,
//                 cross-phase sum accumulated across 2·num_k_tiles MMAs)
//   Compute=100 → Blackwell / UMMA (single-warp issue, ONE TMEM accumulator,
//                 cross-phase accumulate via the per-instruction ScaleOut bit)

template <int Compute>
struct Mlp5FusedConsumerImpl;

// ───────────────────────────────────────────────────────────────────
// Compute=90 — Hopper. The two WGs split (TileM,TileN); the single register
// accumulator carries dU·B (phase 1, k=0..K-1) + dV·C (phase 2, k=K..2K-1)
// via cute::gemm's running accumulation over the whole 2·num_k_tiles k-loop.
// (Verbatim from the original mlp5_fused_consumer.)
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp5FusedConsumerImpl<90> {
template <typename Traits, typename Pipeline, typename TmaStoreDX>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaStoreDX const& tma_store_dx,
		int m,
		int hidden_dim,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sZ = make_tensor(make_smem_ptr(smem.smem_Z), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.smem_W), typename Traits::SmemLayoutW{});

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
		typename Traits::SmemLayoutStore{});

	auto mDX = tma_store_dx.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_m_tiles) * Traits::TileM,
		static_cast<int64_t>(hidden_dim)));
	auto cta_tma_dx = tma_store_dx.get_slice(Int<0>{});

	int total_k = 2 * num_k_tiles;
	constexpr int K_PIPE_MMAS = 1;
	bool store_in_flight = false;

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {

		clear(acc);
		auto state_release = state;
		int prologue_count = (total_k < K_PIPE_MMAS) ? total_k : K_PIPE_MMAS;

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
		for (int k = prologue_count; k < total_k; ++k) {
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
					m_tile_idx = 2 * m + my_wg;
					n_tile_idx = n * Traits::NumEpiRounds + r;
				} else {
					m_tile_idx = m;
					n_tile_idx = n * (Traits::TileN / Traits::EpiChunkN)
					           + my_wg * Traits::NumEpiRounds + r;
				}
				auto gDX = local_tile(mDX,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(m_tile_idx, n_tile_idx));
				copy(tma_store_dx, cta_tma_dx.partition_S(sStore),
					cta_tma_dx.partition_D(gDX));
				cute::tma_store_arrive();
			}
			store_in_flight = true;
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}
};  // Mlp5FusedConsumerImpl<90>

// ───────────────────────────────────────────────────────────────────
// Compute=100 — Blackwell / UMMA.
//
// One 1SM UMMA atom with M=TileM covers the whole tile; the single
// accumulator dX = dU·B + dV·C lives in TMEM. Operand A (dU/dV) is K-major;
// operand B (B/C) is MN-major — reusing mlp2_t's column-major weight view
// (SmemLayoutAtomW = Layout_MN_SW128_Atom, already in Mlp5Traits) with the
// UMMA atom's b_major = UMMA::Major::MN (the descriptor builder reads the
// MN-major stride from the smem layout, so no transpose is needed).
//
// Cross-phase accumulate (the crux): the producer fills 2·num_k_tiles pipe
// stages — B in phase 1 (k<K), C in phase 2 (k≥K) — into the same W slot. The
// consumer runs one continuous k-loop over all 2·num_k_tiles stages into ONE
// TMEM accumulator, setting the per-instruction accumulate bit
// (UMMA::ScaleOut) to Zero on the very first MMA only (writes/clears the acc)
// and One on every subsequent MMA — INCLUDING the phase-1→phase-2 boundary.
// The bit is NOT reset when phase 2 starts, so the dU·B term is preserved.
//
// Epilogue (reused from mlp2_fused's single-accumulator UMMA epilogue): the
// two consumer warpgroups split TileN; each reads its N-half from TMEM→regs in
// EpiChunkN=64 chunks, casts to bf16, and stores 64-row (AtomTileM) tiles via
// the existing reg→SMEM→TMA path (host TMA atom unchanged).
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp5FusedConsumerImpl<100> {
template <typename Traits, typename Pipeline, typename TmaStoreDX>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaStoreDX const& tma_store_dx,
		int m,
		int hidden_dim,
		int num_m_tiles,
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
	// the two aligned epilogue WGs.
	const int  warp_id       = threadIdx.x / Traits::WarpSize;
	const bool is_mma_warp   = (warp_id == 3);
	const bool is_epilogue   = (warp_id >= 4 && warp_id <= 11);
	const int  tid_in_epi    = threadIdx.x - Traits::WarpGroupSize;  // warps 4..11 -> 0..255
	const int  wg            = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	const int  tid_wg        = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	const bool is_wg_leader  = is_epilogue && (tid_wg == 0);
	const int  wg_barrier_id = 1 + wg;                               // 1 or 2
	constexpr int kEpilogueThreads = Traits::ConsumerThreads;
	constexpr int kMmaEpiThreads = kEpilogueThreads + Traits::WarpSize;
	static_assert(Traits::WarpGroupSize == 4 * Traits::WarpSize);
	static_assert(kEpilogueThreads == 8 * Traits::WarpSize);
	static_assert(kMmaEpiThreads == 9 * Traits::WarpSize);

	// ── TiledMMA: single 1SM UMMA atom, M=TileM, N=TileN, SS.
	//    Operand A = dU/dV → K-major; operand B = B/C → MN-major (the
	//    column-major weight view, matching Mlp5Traits::SmemLayoutW). ──
	auto tiled_mma = make_tiled_mma(
		SM100_MMA_F16BF16_SS<Element, Element, float, TileM, TileN,
		                     UMMA::Major::K, UMMA::Major::MN>{});
	auto cta_mma = tiled_mma.get_slice(0);   // 1SM → single CTA, peer-coord 0

	auto sZ = make_tensor(make_smem_ptr(smem.smem_Z), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.smem_W), typename Traits::SmemLayoutW{});

	// ── One TMEM accumulator: dX = dU·B + dV·C ──
	auto cAccFull = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC     = cta_mma.partition_C(cAccFull);
	auto tCtAcc   = cta_mma.make_fragment_C(tCgC);

	// TMEM is allocated by the outer fused/standalone launcher once per CTA.

	// ── Accumulator pipeline: UMMA producer (warp 3) → epilogue consumers
	//    (warps 4..11, both WGs). AccStages stages let MMA for the next n-tile
	//    overlap the previous n-tile's epilogue.
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
		typename Traits::SmemLayoutStore{});

	// dX view: [num_m_tiles·TileM, hidden_dim] row-major. int64_t cast keeps
	// CUTE's layout math in 64-bit for large output buffers.
	auto mDX = tma_store_dx.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_m_tiles) * TileM,
		static_cast<int64_t>(hidden_dim)));
	auto cta_tma_dx = tma_store_dx.get_slice(Int<0>{});

	// ── Epilogue TMEM→reg copy plumbing (built once; sliced per WG-local tid).
	//    flat_divide (not zipped_divide) keeps the epi tile as flat (M,N) modes,
	//    which is what make_tmem_copy's cotiled builder requires. The register
	//    fragment is sized from partition_D (the DEST coords), NOT partition_S,
	//    so it excludes the warp-collective datapath-lane dim. ──
	auto epi_tile  = make_tile(Int<TileM>{}, Int<EpiChunkN>{});
	auto acc_mn    = tCtAcc(make_coord(_, _), _0{}, _0{});   // (TileM,TileN)
	auto tAcc_epi  = flat_divide(acc_mn, epi_tile);   // (TileM,EpiChunkN,1,TileN/EpiChunkN)
	auto t2r       = make_tmem_copy(TmemLoadOp<EpiChunkN>{}, tAcc_epi(_, _, _0{}, _0{}));
	auto thr_t2r   = t2r.get_slice(tid_wg);
	auto tTR_tAcc  = thr_t2r.partition_S(tAcc_epi);     // (Cpy,Cpy_M,Cpy_N,1,nTiles)
	auto cChunk    = make_identity_tensor(make_shape(Int<TileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);       // (Cpy,Cpy_M,Cpy_N)
	auto tTR_rAcc  = make_tensor<float>(shape(tTR_cChunk));   // f32 regs

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;

	const int total_k = 2 * num_k_tiles;   // phase 1 (dU·B) + phase 2 (dV·C)

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		// ── Mainloop (warp 3 only): one continuous k-loop over ALL
		//    2·num_k_tiles stages into the single TMEM accumulator. The
		//    accumulate bit is Zero on the very first MMA (clears the acc) and
		//    One thereafter — including across the phase-1→phase-2 boundary, so
		//    dX = dU·B + dV·C accumulates without dropping the dU·B term. ──
		if (is_mma_warp) {
			acc_pipe.producer_acquire(acc_prod_state);   // TMEM acc free (prev epilogue done)
			int acc_stage = acc_prod_state.index();
			tCtAcc.data() = tmem_base + uint32_t(acc_stage * TileN);
			for (int k = 0; k < total_k; ++k) {
				pipe.consumer_wait(state);
				auto tCsZ = cta_mma.partition_A(sZ(_, _, state.index()));
				auto tCsW = cta_mma.partition_B(sW(_, _, state.index()));
				auto tCrZ = cta_mma.make_fragment_A(tCsZ);
				auto tCrW = cta_mma.make_fragment_B(tCsW);
				CUTE_UNROLL
				for (int kb = 0; kb < size<2>(tCrZ); ++kb) {
					// First MMA of the whole 2K loop clears TMEM; every other
					// MMA (incl. the phase boundary) accumulates. Do NOT reset.
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
						int m_tile_idx = MSub * m + ms;
						int n_tile_idx = n * (TileN / EpiChunkN) + chunk;
						auto gDX = local_tile(mDX,
							make_tile(Int<AtomTileM>{}, Int<EpiChunkN>{}),
							make_coord(m_tile_idx, n_tile_idx));
						copy(tma_store_dx, cta_tma_dx.partition_S(sStore),
							cta_tma_dx.partition_D(gDX));
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

	// The TMEM allocation is owned by the launcher (freed once per CTA after the
	// m-loop). Do NOT relinquish/free here — this consumer runs once per m-tile
	// and freeing per-tile then re-allocating next tile trips the tcgen05
	// "phase invalid during alloc" guardrail. Keep this final 288-thread
	// MMA+epilogue barrier so the next m-tile's accumulator-pipeline init is
	// isolated from this tile's in-flight handshake.
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
#else
	// The Compute=100 specialization is never dispatched on non-SM100 targets
	// (host picks Compute=90 there). Trap if it is ever reached.
	__trap();
#endif
}
};  // Mlp5FusedConsumerImpl<100>

// ───────────────────────────────────────────────────────────────────
// Forwarder — keeps the existing call style. `Compute` defaults to 90 so
// current call sites (mlp5_fused_consumer<Traits>(...)) are unchanged; pass
// mlp5_fused_consumer<Traits, 100>(...) to select the Blackwell path.
// ───────────────────────────────────────────────────────────────────

template <typename Traits, int Compute = 90, typename Pipeline, typename TmaStoreDX>
__device__ __forceinline__ void mlp5_fused_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaStoreDX const& tma_store_dx,
		int m,
		int hidden_dim,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {
	Mlp5FusedConsumerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_store_dx, m, hidden_dim,
		num_m_tiles, num_n_tiles, num_k_tiles, split_idx, num_splits);
}

} // namespace liger
