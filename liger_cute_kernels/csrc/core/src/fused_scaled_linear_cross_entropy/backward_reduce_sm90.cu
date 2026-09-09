#include "backward_reduce_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device/nvshmemx_collective_launch_apis.h>

#include <cstddef>
#include <cstdint>

#include "backward_gemm_mainloop_sm90.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/local_reduce.cuh"
#include "workspace.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

using HostConfig = BackwardGemmConfigSm90<90>;
using HostLaunch = BackwardGemmLaunchSm90<90>;
using HostTraits = BackwardGemmTraitsSm90<90>;
using HostElement = HostTraits::Element;

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy backward reduction: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

template <class Layout, class Element>
auto make_row_major_load(const Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_LOAD{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{});
}

template <class Layout, class Element>
auto make_row_major_store(Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_STORE{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{});
}

template <class Layout, class Element>
auto make_row_major_reduce_add(Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_REDUCE_ADD{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{});
}

template <class Layout, class Element>
auto make_col_major_load(
		const Element* ptr, int64_t rows, int64_t cols, int64_t leading) {
	return make_tma_copy(
		SM90_TMA_LOAD{},
		backward_col_major_tensor(ptr, rows, cols, leading),
		Layout{});
}

template <class CommConfig, int Compute>
__device__ void direct_peer_communicate(
		BackwardDxDwSmemSm90<Compute>& smem,
		const BackwardGemmParamsSm90<Compute>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::DirectPeerReduceView& mapping,
		int num_n_tiles,
		int cta,
		int num_ctas,
		int wave,
		int groups_per_wave) {
	using Traits = BackwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;

	int comm_warp =
		static_cast<int>(threadIdx.x) / Traits::kWarpSize -
		CommConfig::kFirstCommWarp;
	int lane = static_cast<int>(threadIdx.x) % Traits::kWarpSize;
	auto* grad_input = static_cast<__nv_bfloat16*>(params.grad_input);
	int next_index = 0;

	for (int unit = cta; unit < groups_per_wave; unit += num_ctas) {
		int group_id = wave * groups_per_wave + unit;
		DxTileGroup group =
			dx_unit_to_group<CommConfig>(unit, num_n_tiles, group_id);
		DxCtaGroupSlot slot =
			dx_cta_group_slot<CommConfig>(next_index);

		std::uint64_t ready_epoch =
			static_cast<std::uint64_t>(group_id) + 1;
		if (lane == 0) {
			dx_ready_wait_acquire(
				&smem.dx_ready_epoch[comm_warp][slot.stage],
				ready_epoch);
		}
		__syncwarp();

		std::size_t message_rows =
			static_cast<std::size_t>(group.num_tiles) * Traits::kTileM;
		std::size_t first_row =
			message_rows * static_cast<std::size_t>(comm_warp) /
			kDxCommWarpsPerChannel;
		std::size_t last_row =
			message_rows * static_cast<std::size_t>(comm_warp + 1) /
			kDxCommWarpsPerChannel;
		std::size_t segment_elements =
			(last_row - first_row) * Traits::kTileN;
		std::size_t segment_begin = first_row * Traits::kTileN;
		std::size_t base =
			dx_slot_offset<CommConfig>(cta, slot.comm_warp, slot.stage);
		std::size_t ready_offset = dx_sync_offset<CommConfig>(
			cta,
			comm_warp,
			slot.stage,
			kDxReadyPhase,
			mapping.size);
		std::size_t complete_offset = dx_sync_offset<CommConfig>(
			cta,
			comm_warp,
			slot.stage,
			kDxCompletePhase,
			mapping.size);
		std::uint64_t epoch =
			dx_epoch_base(comm) |
			(static_cast<std::uint64_t>(wave + 1) << 16) |
			static_cast<std::uint64_t>(slot.pass + 1);

		constexpr std::size_t kValuesPerVector = 4;
		std::size_t segment_vectors =
			segment_elements / kValuesPerVector;
		float* local_reduced = comm.reduced + base + segment_begin;
		liger_cute::detail::LocalReduceContext<
			liger_cute::detail::LocalReduceBackend::kDirectPeer,
			float> context{
			mapping.peer_partial,
			base + segment_begin,
			comm.sync + ready_offset,
			mapping.peer_sync,
			ready_offset,
			comm.sync + complete_offset,
			complete_offset,
			mapping.rank,
			mapping.size};
		liger_cute::detail::local_all_reduce<
			liger_cute::detail::LocalReduceBackend::kDirectPeer,
			liger_cute::detail::ReduceOp::kSum>(
			context,
			local_reduced,
			comm.partial + base + segment_begin,
			segment_elements,
			epoch);

		constexpr std::size_t kVectorsPerTile =
			Traits::kTileM * Traits::kTileN / kValuesPerVector;
		std::size_t segment_vector_begin =
			segment_begin / kValuesPerVector;
		for (std::size_t segment_vector = lane;
				segment_vector < segment_vectors;
				segment_vector += Traits::kWarpSize) {
			std::size_t message_vector =
				segment_vector_begin + segment_vector;
			int tile = static_cast<int>(
				message_vector / kVectorsPerTile);
			int vector_in_tile = static_cast<int>(
				message_vector % kVectorsPerTile);
			int row = vector_in_tile / (Traits::kTileN / 4);
			int column =
				(vector_in_tile % (Traits::kTileN / 4)) * 4;
			int output_row = wave * Config::kWaveRows +
				group.m_tile * Traits::kTileM + row;
			int output_column =
				(group.first_n_tile + tile) * Traits::kTileN + column;
			if (output_row < params.tokens &&
				output_column < params.hidden) {
				float4 values =
					reinterpret_cast<const float4*>(
						local_reduced)[segment_vector];
				typename DxPhaseSm90<
					CommConfig, Compute>::Bfloat16x4 packed{
					__floats2bfloat162_rn(values.x, values.y),
					__floats2bfloat162_rn(values.z, values.w)};
				auto* output = reinterpret_cast<
					typename DxPhaseSm90<
						CommConfig, Compute>::Bfloat16x4*>(
					grad_input +
					static_cast<std::size_t>(output_row) *
						params.hidden +
					output_column);
				*output = packed;
			}
		}
		__syncwarp();
		if (lane == 0) {
			cute::arrive_barrier(
				smem.dx_consumed_barrier[comm_warp][slot.stage]);
		}
		++next_index;
	}
}

template <
	bool RunDx,
	bool RunDw,
	bool ReturnEntropy,
	int Compute,
	class CommConfig,
	class Bundle>
__global__ __launch_bounds__(BackwardGemmTraitsSm90<Compute>::kNumThreads, 1)
void backward_dx_dw_direct_peer_wave_kernel_sm90(
		__grid_constant__ const Bundle tma,
		__grid_constant__ const BackwardGemmParamsSm90<Compute> params,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const liger_cute::detail::DirectPeerReduceView mapping,
		int wave) {
	using Traits = BackwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Launch = BackwardGemmLaunchSm90<Compute>;
	using Smem = BackwardDxDwSmemSm90<Compute>;
	using PipelineTraits = typename Traits::PipelineTraits;
	using ClusterShape = typename Traits::ClusterShape;
	using Registers = sm90::WarpSpecializedRegistersSm90<
		Config::kDxDwProducerRegisters, Config::kDxDwMmaRegisters, Compute>;

	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);
	int warp_group = cutlass::canonical_warp_group_idx();
	int warp = cutlass::canonical_warp_idx_sync();

	PipelineTraits::init_barriers(
		smem.mainloop_pipeline,
		Traits::kTmaTransBytes,
		Traits::kConsumerThreads,
		ClusterShape{});
	if (threadIdx.x == 0) {
		CUTE_UNROLL
		for (int comm_warp = 0;
				comm_warp < kDxCommWarpsPerChannel;
				++comm_warp) {
			CUTE_UNROLL
			for (int stage = 0; stage < kDxRingStages; ++stage) {
				smem.dx_ready_epoch[comm_warp][stage] = 0;
				cute::initialize_barrier(
					smem.dx_consumed_barrier[comm_warp][stage], 1);
			}
		}
	}
	sm90::cluster_pipeline_init_sm90<Compute>();

	int num_ctas = static_cast<int>(gridDim.x);
	int cta = static_cast<int>(blockIdx.x);
	if (warp_group == 0) {
		Registers::producer();
		if (warp == 0) {
			int padded_vocab = Launch::padded_vocab(params.local_vocab);
			auto state = PipelineTraits::producer_start_state();
			auto pipe = PipelineTraits::make_producer(
				smem.mainloop_pipeline,
				Traits::kTmaTransBytes,
				Traits::kConsumerThreads,
				threadIdx.x == 0,
				ClusterShape{});
			if constexpr (RunDx) {
			int dx_n_tiles = Launch::num_dx_n_tiles(params.hidden);
			int dx_k_tiles =
				Launch::num_dx_k_tiles(params.local_vocab);
			int dx_groups_per_wave = Config::kMTilesPerWave *
				dx_groups_per_m_tile<CommConfig>(dx_n_tiles);
			DxPhaseSm90<CommConfig, Compute>::
				template produce<ReturnEntropy>(
					pipe,
					state,
					smem,
					tma.dz_load,
					tma.wt,
					params,
					padded_vocab,
					dx_n_tiles,
					dx_k_tiles,
					dx_groups_per_wave,
					cta,
					num_ctas);
			}

			if constexpr (RunDx && RunDw) {
			backward_compute_barrier_sm90<Compute>();
			}

			if constexpr (RunDw) {
			int dw_n_tiles = Launch::num_dw_n_tiles(params.hidden);
			constexpr int kDwKTiles = Launch::num_dw_k_tiles();
			int dw_total =
				Launch::num_dw_m_tiles(params.local_vocab) * dw_n_tiles;
			int dw_tiles = dw_total > cta
				? ceil_div(dw_total - cta, num_ctas)
				: 0;
			DwPhaseSm90<Compute>::template produce<ReturnEntropy>(
				pipe,
				state,
				smem,
				tma.dzt,
				tma.xt,
				params,
				padded_vocab,
				wave,
				dw_n_tiles,
				kDwKTiles,
				dw_tiles,
				cta,
				num_ctas);
			}
		} else if (
				warp >= CommConfig::kFirstCommWarp &&
				warp <= CommConfig::kLastCommWarp) {
			if constexpr (RunDx) {
			int dx_n_tiles = Launch::num_dx_n_tiles(params.hidden);
			int dx_groups_per_wave = Config::kMTilesPerWave *
				dx_groups_per_m_tile<CommConfig>(dx_n_tiles);
			direct_peer_communicate<CommConfig, Compute>(
				smem,
				params,
				comm,
				mapping,
				dx_n_tiles,
				cta,
				num_ctas,
				wave,
				dx_groups_per_wave);
			}
		}
	} else {
		Registers::consumer();
		typename Traits::PipelineState read_state;
		typename Traits::PipelineState release_state;
		auto pipe = PipelineTraits::make_consumer(
			smem.mainloop_pipeline,
			Traits::kTmaTransBytes,
			Traits::kConsumerThreads,
			ClusterShape{});
		if constexpr (RunDx) {
		int dx_n_tiles = Launch::num_dx_n_tiles(params.hidden);
		int dx_k_tiles =
			Launch::num_dx_k_tiles(params.local_vocab);
		int dx_groups_per_wave = Config::kMTilesPerWave *
			dx_groups_per_m_tile<CommConfig>(dx_n_tiles);
		int next_index = 0;
		DxPhaseSm90<CommConfig, Compute>::
			template consume<ReturnEntropy>(
				pipe,
				read_state,
				release_state,
				smem,
				comm,
				dx_n_tiles,
				dx_k_tiles,
				wave,
				dx_groups_per_wave,
				cta,
				num_ctas,
				next_index);
		}

		if constexpr (RunDx && RunDw) {
		backward_compute_barrier_sm90<Compute>();
		}

		if constexpr (RunDw) {
		int dw_n_tiles = Launch::num_dw_n_tiles(params.hidden);
		constexpr int kDwKTiles = Launch::num_dw_k_tiles();
		int dw_total =
			Launch::num_dw_m_tiles(params.local_vocab) * dw_n_tiles;
		int dw_tiles = dw_total > cta
			? ceil_div(dw_total - cta, num_ctas)
			: 0;
		DwPhaseSm90<Compute>::template consume<ReturnEntropy>(
			pipe,
			read_state,
			release_state,
			smem,
			tma.dw_store,
			tma.dw_add,
			params,
			wave,
			dw_n_tiles,
			kDwKTiles,
			dw_tiles,
			cta,
			num_ctas);
		}
	}

	__syncthreads();
	sm90::cluster_exit_sm90<Compute>();
}

template <int Compute>
__global__ void cluster_local_allgather_scatter_kernel(
		__grid_constant__ const BackwardGemmParamsSm90<Compute> params,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const liger_cute::detail::NvlsReduceView mapping,
		int grid_ctas,
		int wave,
		int tiles_per_wave,
		int num_n_tiles) {
	using Traits = BackwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using CommConfig = DxCommConfig<Config, kDxRingStages, 1, Compute>;

	int unit = static_cast<int>(blockIdx.x);
	if (unit >= tiles_per_wave) return;
	int comm_warp = static_cast<int>(threadIdx.x) / Traits::kWarpSize;
	int lane = static_cast<int>(threadIdx.x) % Traits::kWarpSize;
	if (comm_warp >= kDxCommWarpsPerChannel) return;

	constexpr std::size_t kTileElements = CommConfig::kTileElements;
	constexpr std::size_t kSegmentElements =
		kTileElements / kDxCommWarpsPerChannel;
	std::size_t packed_tile_elements =
		kTileElements / mapping.size;
	std::size_t packed_segment_elements =
		kSegmentElements / mapping.size;
	const float* packed_source = mapping.reduced_shard +
		(static_cast<std::size_t>(wave) * tiles_per_wave + unit) *
			packed_tile_elements +
		static_cast<std::size_t>(comm_warp) * packed_segment_elements;

	int cta = unit % grid_ctas;
	int next_index = unit / grid_ctas;
	DxCtaGroupSlot slot = dx_cta_group_slot<CommConfig>(next_index);
	const float* values = packed_source;
	if (mapping.size > 1) {
		std::size_t ready_offset = dx_sync_offset<CommConfig>(
			cta,
			comm_warp,
			slot.stage,
			kDxReadyPhase,
			mapping.size);
		std::size_t complete_offset = dx_sync_offset<CommConfig>(
			cta,
			comm_warp,
			slot.stage,
			kDxCompletePhase,
			mapping.size);
		std::uint64_t epoch =
			dx_epoch_base(comm) | 0x20000000u |
			(static_cast<std::uint64_t>(wave) << 16) |
			static_cast<std::uint64_t>(slot.pass + 1);
		liger_cute::detail::nvls_barrier_warp(
			comm.sync + ready_offset,
			mapping.multicast_sync + ready_offset,
			mapping.rank,
			mapping.size,
			epoch);
		std::size_t full_tile_offset =
			(static_cast<std::size_t>(wave) * tiles_per_wave + unit) *
			kTileElements;
		std::size_t segment_begin =
			static_cast<std::size_t>(comm_warp) * kSegmentElements;
		liger_cute::detail::nvls_allgather_warp(
			mapping.multicast_reduced +
				full_tile_offset + segment_begin,
			packed_source,
			kSegmentElements,
			mapping.rank,
			mapping.size);
		liger_cute::detail::nvls_barrier_warp(
			comm.sync + complete_offset,
			mapping.multicast_sync + complete_offset,
			mapping.rank,
			mapping.size,
			epoch);
		values = comm.reduced + full_tile_offset + segment_begin;
	}

	auto* grad_input = static_cast<__nv_bfloat16*>(params.grad_input);
	std::size_t segment_begin =
		static_cast<std::size_t>(comm_warp) * kSegmentElements;
	int m_tile = unit / num_n_tiles;
	int n_tile = unit % num_n_tiles;
	constexpr std::size_t kSegmentVectors = kSegmentElements / 4;
	for (std::size_t vector = lane;
			vector < kSegmentVectors;
			vector += Traits::kWarpSize) {
		std::size_t tile_vector = segment_begin / 4 + vector;
		int row = static_cast<int>(
			tile_vector / (Traits::kTileN / 4));
		int column = static_cast<int>(
			(tile_vector % (Traits::kTileN / 4)) * 4);
		int output_row = wave * Config::kWaveRows +
			m_tile * Traits::kTileM + row;
		int output_column = n_tile * Traits::kTileN + column;
		if (output_row < params.tokens &&
			output_column < params.hidden) {
			float4 value =
				reinterpret_cast<const float4*>(values)[vector];
			typename DxPhaseSm90<CommConfig, Compute>::Bfloat16x4 packed{
				__floats2bfloat162_rn(value.x, value.y),
				__floats2bfloat162_rn(value.z, value.w)};
			auto* output = reinterpret_cast<
				typename DxPhaseSm90<
					CommConfig, Compute>::Bfloat16x4*>(
				grad_input +
					static_cast<std::size_t>(output_row) *
						params.hidden +
					output_column);
			*output = packed;
		}
	}
}

template <bool RequiresRemote>
void finalize_cluster_typed(
		const BackwardGemmParamsSm90<90>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& mapping,
		const liger_cute::detail::RemoteReduceView& remote,
		int grid,
		int num_waves,
		cudaStream_t stream) {
	using CommConfig = DxCommConfig<HostConfig, kDxRingStages, 1, 90>;
	int num_n_tiles = HostLaunch::num_dx_n_tiles(params.hidden);
	int tiles_per_wave = HostConfig::kMTilesPerWave * num_n_tiles;
	std::size_t full_elements =
		static_cast<std::size_t>(num_waves) *
		static_cast<std::size_t>(tiles_per_wave) *
		CommConfig::kTileElements;
	std::size_t packed_elements =
		full_elements / static_cast<std::size_t>(mapping.size);
	LIGER_CHECK(
		full_elements * sizeof(float) <=
			backward_dx_configured_durable_bytes(),
		"fused_scaled_linear_cross_entropy backward: cluster dX durable "
		"workspace capacity exceeded");
	if constexpr (RequiresRemote) {
		liger_cute::detail::launch_remote_reduce(
			remote,
			comm.launch_epoch,
			packed_elements,
			stream);
	}

	auto* kernel = &cluster_local_allgather_scatter_kernel<90>;
	for (int wave = 0; wave < num_waves; ++wave) {
		void* args[] = {
			const_cast<BackwardGemmParamsSm90<90>*>(&params),
			const_cast<DxReduceWorkspace<float>*>(&comm),
			const_cast<liger_cute::detail::NvlsReduceView*>(&mapping),
			&grid,
			&wave,
			&tiles_per_wave,
			&num_n_tiles};
		int status = nvshmemx_collective_launch(
			reinterpret_cast<const void*>(kernel),
			dim3(static_cast<unsigned>(tiles_per_wave), 1, 1),
			dim3(kDxCommWarpsPerChannel * kWarpSize, 1, 1),
			args,
			0,
			stream);
		LIGER_CHECK(
			status == 0,
			"fused_scaled_linear_cross_entropy backward: cluster local "
			"all-gather/scatter launch failed with status ",
			status);
		check_cuda(
			cudaGetLastError(),
			"cluster_local_allgather_scatter_kernel launch");
	}
}

template <
	int TilesPerReduce,
	bool ReturnEntropy,
	bool RunDx = true,
	bool RunDw = true>
void launch_typed(
		const BackwardGemmParamsSm90<90>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::DirectPeerReduceView& mapping,
		int grid,
		cudaStream_t stream,
		int wave) {
	using CommConfig =
		DxCommConfig<HostConfig, kDxRingStages, TilesPerReduce, 90>;
	using Smem = BackwardDxDwSmemSm90<90>;

	auto* w = static_cast<const HostElement*>(params.weight);
	auto* dz = static_cast<HostElement*>(params.dz_workspace);
	auto* dw = static_cast<HostElement*>(params.grad_weight);
	int padded_vocab = HostLaunch::padded_vocab(params.local_vocab);

	auto tma_dz_load =
		make_row_major_load<typename HostTraits::SmemAKMajor::Single>(
			dz, HostConfig::kWaveRows, padded_vocab);
	auto tma_wt = make_col_major_load<
		typename HostTraits::SmemBMnMajor::Single>(
			w, params.hidden, params.local_vocab, params.hidden);
	auto tma_dzt = make_col_major_load<
		typename HostTraits::SmemAMnMajor::Single>(
			dz, padded_vocab, HostConfig::kWaveRows, padded_vocab);
	auto tma_xt = make_col_major_load<
		typename HostTraits::SmemBMnMajor::Single>(
			static_cast<const HostElement*>(params.x),
			params.hidden,
			params.tokens,
			params.hidden);
	auto tma_dw_store =
		make_row_major_store<typename HostTraits::DwSmemStore1::Single>(
			dw, params.local_vocab, params.hidden);
	auto tma_dw_add =
		make_row_major_reduce_add<typename HostTraits::DwSmemStore1::Single>(
			dw, params.local_vocab, params.hidden);
	using Bundle = BackwardDxDwTmaBundleSm90<
		decltype(tma_dz_load),
		decltype(tma_wt),
		decltype(tma_dzt),
		decltype(tma_xt),
		decltype(tma_dw_store),
		decltype(tma_dw_add)>;
	Bundle bundle{
		tma_dz_load,
		tma_wt,
		tma_dzt,
		tma_xt,
		tma_dw_store,
		tma_dw_add};
	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));
	static_assert(
		2 * kSmemBytes > kBackwardMaxSmemBytes,
		"the direct-peer grid assumes at most one resident CTA per SM");
	auto* kernel =
		&backward_dx_dw_direct_peer_wave_kernel_sm90<
			RunDx,
			RunDw,
			ReturnEntropy,
			90,
			CommConfig,
			Bundle>;
	check_cuda(
		cudaFuncSetAttribute(
			kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			kSmemBytes),
		"cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
	void* args[] = {
		&bundle,
		const_cast<BackwardGemmParamsSm90<90>*>(&params),
		const_cast<DxReduceWorkspace<float>*>(&comm),
		const_cast<liger_cute::detail::DirectPeerReduceView*>(&mapping),
		&wave};
	int status = nvshmemx_collective_launch(
		reinterpret_cast<const void*>(kernel),
		dim3(static_cast<unsigned>(grid), 1, 1),
		dim3(HostConfig::kNumThreads, 1, 1),
		args,
		static_cast<std::size_t>(kSmemBytes),
		stream);
	LIGER_CHECK(
		status == 0,
		"fused_scaled_linear_cross_entropy backward: "
		"non-NVLS collective launch failed with status ",
		status);
	check_cuda(
		cudaGetLastError(),
		"backward_dx_dw_direct_peer_wave_kernel_sm90 launch");
}

}  // namespace

void launch_backward_dx_dw_direct_peer_wave_sm90(
		bool return_entropy,
		int tiles_per_reduce,
		const BackwardGemmParamsSm90<90>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::DirectPeerReduceView& mapping,
		int grid,
		cudaStream_t stream,
		int wave) {
	LIGER_CHECK(
			mapping.size > 1,
			"fused_scaled_linear_cross_entropy backward: invalid direct-peer team "
		"metadata");
	LIGER_CHECK(
			mapping.available != 0,
		"fused_scaled_linear_cross_entropy backward: NVLS is unavailable and "
		"one or more tensor-parallel peers do not have a direct symmetric "
		"mapping");
	if (return_entropy) {
		switch (tiles_per_reduce) {
			case 1:
				launch_typed<1, true>(
					params, comm, mapping, grid, stream, wave);
				return;
			case 2:
				launch_typed<2, true>(
					params, comm, mapping, grid, stream, wave);
				return;
			default:
				launch_typed<4, true>(
					params, comm, mapping, grid, stream, wave);
				return;
		}
	}
	switch (tiles_per_reduce) {
		case 1:
			launch_typed<1, false>(
				params, comm, mapping, grid, stream, wave);
			return;
		case 2:
			launch_typed<2, false>(
				params, comm, mapping, grid, stream, wave);
			return;
		default:
			launch_typed<4, false>(
				params, comm, mapping, grid, stream, wave);
			return;
	}
}

template <bool RequiresRemote>
void finalize_backward_dx_cluster_sm90(
		const BackwardGemmParamsSm90<90>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& mapping,
		const liger_cute::detail::RemoteReduceView& remote,
		int grid,
		int num_waves,
		cudaStream_t stream) {
	finalize_cluster_typed<RequiresRemote>(
		params, comm, mapping, remote, grid, num_waves, stream);
}

template void finalize_backward_dx_cluster_sm90<false>(
		const BackwardGemmParamsSm90<90>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& mapping,
		const liger_cute::detail::RemoteReduceView& remote,
		int grid,
		int num_waves,
		cudaStream_t stream);
template void finalize_backward_dx_cluster_sm90<true>(
		const BackwardGemmParamsSm90<90>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& mapping,
		const liger_cute::detail::RemoteReduceView& remote,
		int grid,
		int num_waves,
		cudaStream_t stream);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
