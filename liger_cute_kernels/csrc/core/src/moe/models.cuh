#pragma once

// #ifndef NDEBUG
// #define NDEBUG
// #endif

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// GMMA atom selection
// ═══════════════════════════════════════════════════════════════════

template <typename Element, int N>
struct GmmaSelector;

static constexpr auto GmmaK  = GMMA::Major::K;
static constexpr auto GmmaMN = GMMA::Major::MN;

template <> struct GmmaSelector<bfloat16_t, 8>  { using Atom = SM90_64x8x16_F32BF16BF16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<bfloat16_t, 16> { using Atom = SM90_64x16x16_F32BF16BF16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<bfloat16_t, 32> { using Atom = SM90_64x32x16_F32BF16BF16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<bfloat16_t, 64>  { using Atom = SM90_64x64x16_F32BF16BF16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<bfloat16_t, 128> { using Atom = SM90_64x128x16_F32BF16BF16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<bfloat16_t, 192> { using Atom = SM90_64x192x16_F32BF16BF16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<bfloat16_t, 256> { using Atom = SM90_64x256x16_F32BF16BF16_SS<GmmaK, GmmaK>; };

template <> struct GmmaSelector<half_t, 8>  { using Atom = SM90_64x8x16_F32F16F16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<half_t, 16> { using Atom = SM90_64x16x16_F32F16F16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<half_t, 32> { using Atom = SM90_64x32x16_F32F16F16_SS<GmmaK, GmmaK>; };
template <> struct GmmaSelector<half_t, 64> { using Atom = SM90_64x64x16_F32F16F16_SS<GmmaK, GmmaK>; };

// ═══════════════════════════════════════════════════════════════════
// GMMA atom selection — A=K-major, B=MN-major  (SS<K, MN>)
// Used when B operand has MN-contiguous smem (e.g. mlp2_t_fused).
// ═══════════════════════════════════════════════════════════════════

template <typename Element, int N>
struct GmmaSelectorKMN;

template <> struct GmmaSelectorKMN<bfloat16_t, 8>   { using Atom = SM90_64x8x16_F32BF16BF16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<bfloat16_t, 16>  { using Atom = SM90_64x16x16_F32BF16BF16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<bfloat16_t, 32>  { using Atom = SM90_64x32x16_F32BF16BF16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<bfloat16_t, 64>  { using Atom = SM90_64x64x16_F32BF16BF16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<bfloat16_t, 128> { using Atom = SM90_64x128x16_F32BF16BF16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<bfloat16_t, 256> { using Atom = SM90_64x256x16_F32BF16BF16_SS<GmmaK, GmmaMN>; };

template <> struct GmmaSelectorKMN<half_t, 8>   { using Atom = SM90_64x8x16_F32F16F16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<half_t, 16>  { using Atom = SM90_64x16x16_F32F16F16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<half_t, 32>  { using Atom = SM90_64x32x16_F32F16F16_SS<GmmaK, GmmaMN>; };
template <> struct GmmaSelectorKMN<half_t, 64>  { using Atom = SM90_64x64x16_F32F16F16_SS<GmmaK, GmmaMN>; };

// Backward compat alias
template <typename Element, int N>
struct GmmaSelectorMN : GmmaSelectorKMN<Element, N> {};

// ═══════════════════════════════════════════════════════════════════
// GMMA atom selection — both MN-major  (SS<MN, MN>)
// Used when both operands have MN-contiguous smem (e.g. mlp3).
// ═══════════════════════════════════════════════════════════════════

template <typename Element, int N>
struct GmmaSelectorMNMN;

template <> struct GmmaSelectorMNMN<bfloat16_t, 8>   { using Atom = SM90_64x8x16_F32BF16BF16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<bfloat16_t, 16>  { using Atom = SM90_64x16x16_F32BF16BF16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<bfloat16_t, 32>  { using Atom = SM90_64x32x16_F32BF16BF16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<bfloat16_t, 64>  { using Atom = SM90_64x64x16_F32BF16BF16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<bfloat16_t, 128> { using Atom = SM90_64x128x16_F32BF16BF16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<bfloat16_t, 256> { using Atom = SM90_64x256x16_F32BF16BF16_SS<GmmaMN, GmmaMN>; };

template <> struct GmmaSelectorMNMN<half_t, 8>   { using Atom = SM90_64x8x16_F32F16F16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<half_t, 16>  { using Atom = SM90_64x16x16_F32F16F16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<half_t, 32>  { using Atom = SM90_64x32x16_F32F16F16_SS<GmmaMN, GmmaMN>; };
template <> struct GmmaSelectorMNMN<half_t, 64>  { using Atom = SM90_64x64x16_F32F16F16_SS<GmmaMN, GmmaMN>; };

// ═══════════════════════════════════════════════════════════════════
// FusedMoETraits
// ═══════════════════════════════════════════════════════════════════

template <
	typename Element_,
	int TileN_,
	int TileK_    = 64,
	int Stages_   = 2,
	int MaxTopK_  = 8
>
struct FusedMoETraits {
	using Element      = Element_;
	using ElementAccum = float;

	static constexpr int TileM    = 64;
	static constexpr int TileN    = TileN_;
	static constexpr int TileK    = TileK_;
	static constexpr int Stages   = Stages_;
	static constexpr int MaxTopK  = MaxTopK_;

	using GmmaAtom = typename GmmaSelector<Element, TileN>::Atom;
	using TiledMma  = TiledMMA<MMA_Atom<GmmaAtom>, Layout<Shape<_1, _1, _1>>>;

	using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;

	using SmemLayoutA_1 = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileM>, Int<TileK>>{}));
	using SmemLayoutB_1 = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileN>, Int<TileK>>{}));

	using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileM>, Int<TileK>, Int<Stages>>{}));
	using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileN>, Int<TileK>, Int<Stages>>{}));

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	static constexpr int TmaTransactionBytesA = static_cast<int>(size(SmemLayoutA_1{}) * sizeof(Element));
	static constexpr int TmaTransactionBytesB = static_cast<int>(size(SmemLayoutB_1{}) * sizeof(Element));
	static constexpr int TmaTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;

	static constexpr int WarpSize        = 32;
	static constexpr int WarpGroupSize   = 128;
	static constexpr int NumThreads      = 384;
	static constexpr int EpilogueThreads = 2 * WarpSize;  // warps 1+2 (64 threads)
	static constexpr int EpilogueBarrier = 3;              // NamedBarrier ID for epilogue sync
	static constexpr int SortWarpId      = 3;              // warp 3 (threads 96-127)
};

// ═══════════════════════════════════════════════════════════════════
// RouterSmem — shared memory layout for the fused MoE kernel
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct RouterSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_A_size = cosize_v<typename Traits::SmemLayoutA>;
	static constexpr int smem_B_size = cosize_v<typename Traits::SmemLayoutB>;

	// Consumer A smem (even K-tiles)
	alignas(128) Element smem_A_a[smem_A_size];
	alignas(128) Element smem_B_a[smem_B_size];
	// Consumer B smem (odd K-tiles)
	alignas(128) Element smem_A_b[smem_A_size];
	alignas(128) Element smem_B_b[smem_B_size];

	// Two TMA pipelines (one per consumer)
	typename Traits::MainloopPipeline::SharedStorage pipe_a_storage;
	typename Traits::MainloopPipeline::SharedStorage pipe_b_storage;

	alignas(16) float score_strip[2][Traits::TileM][Traits::TileN];

	alignas(16) float running_max[Traits::TileM];
	alignas(16) float running_sum[Traits::TileM];
	alignas(16) float top_scores[Traits::TileM][Traits::MaxTopK];
	alignas(16) int   top_indices[Traits::TileM][Traits::MaxTopK];

	alignas(16) int strip_ready[2];
	alignas(16) int strip_consumed[2];

	alignas(8) uint64_t mbar_debug;

	CUTE_DEVICE Element* A_a() { return &smem_A_a[0]; }
	CUTE_DEVICE Element* B_a() { return &smem_B_a[0]; }
	CUTE_DEVICE Element* A_b() { return &smem_A_b[0]; }
	CUTE_DEVICE Element* B_b() { return &smem_B_b[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// SortBuffers — global memory pointers for the sort phase
// ═══════════════════════════════════════════════════════════════════

struct SortBuffers {
	int* tile_expert_counts;  // [num_m_tiles * E]  — per-tile histograms,
	                          //                      modified in-place for prefix sums
	int* sorted_token_ids;    // [total_slots]       — final output: slot → token
	int* token_expert_slots;  // [T * K]             — final output: (token, k) → slot
	int* expert_offsets;      // [E + 1]             — aligned expert boundaries (symmetric)
	int* tile_expert_ids;     // [max_total_tiles]   — expert ID per M-tile (nullable)
	int* cta_done;            // [num_blocks]         — per-CTA completion flags (Blelloch tree)
	int* cta_sums;            // [num_blocks * E]    — per-CTA expert totals for prefix sum
};

} // namespace liger
