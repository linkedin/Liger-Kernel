// SM100 fused scaled linear cross entropy backward — contract tests.
//
// Macro guarded: the executable mainloop is only included when the translation
// unit is compiled for the Blackwell family, so the same target can be
// configured (and skipped) on a Hopper build.

#include <gtest/gtest.h>

#include <type_traits>

#include "backward_gemm_sm100.cuh"
#include "backward_gemm_sm90.cuh"
#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_TEST_MAINLOOP)
#include "backward_gemm_mainloop_sm100.cuh"
#endif

namespace fslce = liger::fused_scaled_linear_cross_entropy;

using Config = fslce::BackwardGemmConfigSm100<>;
using Launch = fslce::BackwardGemmLaunchSm100<>;
using CommConfig =
	fslce::DxCommConfig<Config, fslce::kDxRingStages, 1, 100>;

#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_TEST_MAINLOOP)
__global__ void dx_stage_handshake_contract_kernel(
		int* staging, int* finalized, int* violations) {
	__shared__ std::uint64_t ready[fslce::kDxRingStages];
	__shared__ std::uint64_t consumed[fslce::kDxRingStages];
	if (threadIdx.x == 0) {
		for (int stage = 0; stage < fslce::kDxRingStages; ++stage) {
			cute::initialize_barrier(ready[stage], 1);
			cute::initialize_barrier(consumed[stage], 1);
		}
	}
	cutlass::arch::fence_barrier_init();
	__syncthreads();

	int warp = static_cast<int>(threadIdx.x) / fslce::kWarpSize;
	int lane = static_cast<int>(threadIdx.x) & (fslce::kWarpSize - 1);
	constexpr int kItems = 3 * fslce::kDxRingStages;
	if (lane != 0) return;

	if (warp == Config::kFirstEpilogueWarp) {
		for (int item = 0; item < kItems; ++item) {
			int stage = item % fslce::kDxRingStages;
			int pass = item / fslce::kDxRingStages;
			if (pass > 0) {
				cute::wait_barrier(consumed[stage], (pass - 1) & 1);
				if (atomicAdd(finalized + item - fslce::kDxRingStages, 0) !=
					1) {
					atomicAdd(violations, 1);
				}
			}
			staging[stage] = item;
			__threadfence_block();
			cute::arrive_barrier(ready[stage]);
		}
	} else if (warp == Config::kDxLocalReduceWarp) {
		for (int item = 0; item < kItems; ++item) {
			int stage = item % fslce::kDxRingStages;
			int pass = item / fslce::kDxRingStages;
			cute::wait_barrier(ready[stage], pass & 1);
			int value = staging[stage];
			finalized[item] = value == item ? 1 : -1;
			__threadfence_block();
			cute::arrive_barrier(consumed[stage]);
		}
	}
}
#endif

TEST(TensorParallelFusedScaledLinearCrossEntropyBackwardSm100, PinsWarpPlan) {
	EXPECT_EQ(
		fslce::backward_warp_role_sm100(0),
		fslce::BackwardWarpRoleSm100::kDxLocalReduce);
	EXPECT_EQ(
		fslce::backward_warp_role_sm100(1),
		fslce::BackwardWarpRoleSm100::kRemoteCommunication);
	EXPECT_EQ(
		fslce::backward_warp_role_sm100(2),
		fslce::BackwardWarpRoleSm100::kTmaProducer);
	EXPECT_EQ(
		fslce::backward_warp_role_sm100(3),
		fslce::BackwardWarpRoleSm100::kUmmaProducer);
	for (int warp = 4; warp < fslce::kNumWarps; ++warp) {
		EXPECT_EQ(
			fslce::backward_warp_role_sm100(warp),
			fslce::BackwardWarpRoleSm100::kEpilogue);
	}
	EXPECT_EQ(
		fslce::backward_warp_role_sm100(fslce::kNumWarps),
		fslce::BackwardWarpRoleSm100::kInactive);
	using Plan = fslce::BackwardWarpPlanSm100<>;
	EXPECT_EQ(Plan::kTmemOwnerWarp, Config::kFirstEpilogueWarp);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	KeepsCommunicationWarpsOutOfComputeBarriers) {
	// Every compute-side named barrier must be strictly narrower than the CTA,
	// otherwise it degenerates into __syncthreads() and drags warps 0 and 1
	// into the GEMM rendezvous.
	EXPECT_LT(Config::kComputeThreads, Config::kNumThreads);
	EXPECT_LT(Config::kMmaEpilogueThreads, Config::kNumThreads);
	EXPECT_LT(Config::kEpilogueThreads, Config::kNumThreads);
	EXPECT_EQ(Config::kComputeThreads, 320);
	EXPECT_EQ(Config::kMmaEpilogueThreads, 288);
	EXPECT_EQ(Config::kEpilogueThreads, 256);
	// The compute barrier starts at warp 2, so warps 0 and 1 are excluded.
	EXPECT_EQ(
		Config::kComputeThreads,
		(Config::kLastEpilogueWarp - Config::kTmaWarp + 1) *
			fslce::kWarpSize);
	EXPECT_GT(Config::kTmaWarp, Config::kRemoteCommunicationWarp);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	SharesOneTmemAllocationAcrossPhases) {
	EXPECT_EQ(Config::kCompute, 100);
	EXPECT_EQ(Config::kClusterM, 2);
	EXPECT_EQ(Config::kTileM, 256);
	EXPECT_EQ(Config::kCtaTileM, 128);
	EXPECT_EQ(Config::kTileN, 256);
	EXPECT_EQ(Config::kTileK, 64);
	EXPECT_EQ(Config::kAccumulatorStages, 2);
	EXPECT_EQ(Config::kTmemStageColumns, 256);
	// Max over phases: dZ, dX and dW all use the same N256 accumulator, so a
	// single 512-column allocation is reused by every phase.
	EXPECT_EQ(Config::kTmemColumns, 512);
	EXPECT_EQ(Config::kDzTileN, Config::kTileN);
	EXPECT_EQ(Config::kDzMainloopStages, 5);
	EXPECT_EQ(Config::kDxTileN, Config::kTileN);
	EXPECT_EQ(Config::kDxMainloopStages, 4);
	EXPECT_EQ(Config::kDwTileN, Config::kTileN);
	EXPECT_EQ(Config::kDwMainloopStages, 6);
	EXPECT_EQ(Config::kMainloopStages, LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES);
	EXPECT_EQ(fslce::DzGemmContractSm100<>::kStages, 5);
	EXPECT_EQ(fslce::DxGemmContractSm100<>::kStages, 4);
	EXPECT_EQ(fslce::DwGemmContractSm100<>::kStages, 6);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	UsesTheConfiguredWaveGeometry) {
	EXPECT_EQ(
		Config::kWaveRows,
		LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS);
	EXPECT_EQ(Config::kMTilesPerWave, Config::kWaveRows / 128);
	EXPECT_EQ(Config::kMPairsPerWave, Config::kWaveRows / 256);
	EXPECT_EQ(Config::kVocabAlign, 64);
	EXPECT_EQ(Config::kDwKTiles, Config::kWaveRows / 64);

	EXPECT_EQ(Launch::num_waves(1), 1);
	EXPECT_EQ(Launch::num_waves(Config::kWaveRows), 1);
	EXPECT_EQ(Launch::num_waves(Config::kWaveRows + 1), 2);
	EXPECT_EQ(Launch::padded_vocab(1), 64);
	EXPECT_EQ(Launch::padded_vocab(64), 64);
	EXPECT_EQ(Launch::padded_vocab(65), 128);
	EXPECT_EQ(Launch::num_dx_k_tiles(129), 3);
	EXPECT_EQ(Launch::num_dz_n_tiles(300), 2);
	EXPECT_EQ(Launch::num_dx_n_tiles(2048), 8);
	EXPECT_EQ(Launch::num_dw_n_tiles(2048), 8);
	EXPECT_EQ(Launch::num_dw_m_pairs(512), 2);
	EXPECT_EQ(Launch::num_dw_cluster_pairs(2048, 512), 8);
	EXPECT_EQ(
		Launch::dx_tiles_per_wave(2048),
		Config::kMTilesPerWave * 8);
	EXPECT_EQ(
		Launch::dz_workspace_bytes(300),
		static_cast<std::size_t>(Config::kDzWorkspaceSlots) *
			static_cast<std::size_t>(Config::kWaveRows) * 320u * 2u);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	ReusesTheCtaOwnedStagingContract) {
	// dx_reduce.cuh is generalized, not forked: the SM100 staging tile, slot
	// stride and signal prefixes are byte-identical to SM90's.
	using Sm90Config = fslce::BackwardGemmConfigSm90<90>;
	using Sm90CommConfig =
		fslce::DxCommConfig<Sm90Config, fslce::kDxRingStages, 1, 90>;
	EXPECT_EQ(CommConfig::kTileM, Sm90CommConfig::kTileM);
	EXPECT_EQ(CommConfig::kTileN, Sm90CommConfig::kTileN);
	EXPECT_EQ(CommConfig::kTileElements, Sm90CommConfig::kTileElements);
	EXPECT_EQ(CommConfig::kNumStages, fslce::kDxRingStages);
	EXPECT_EQ(fslce::kDxRingStages, 4);
	// SM100 gives the whole message to warp 0 and keeps warp 1 remote-only.
	EXPECT_EQ(CommConfig::kFirstCommWarp, 0);
	EXPECT_EQ(CommConfig::kNumCommWarps, 1);
	EXPECT_EQ(CommConfig::kProducerWarp, Config::kTmaWarp);
	EXPECT_EQ(
		fslce::dx_slot_offset<CommConfig>(0, 0, 1),
		static_cast<std::size_t>(CommConfig::kTileElements));
	EXPECT_EQ(
		fslce::dx_slot_offset<CommConfig>(1, 0, 0),
		static_cast<std::size_t>(
			fslce::kDxCommWarpsPerChannel * fslce::kDxRingStages) *
			CommConfig::kTileElements);

	// TilesPerReduce is retained for capacity/API parity.
	using Tiles2 = fslce::DxCommConfig<Config, fslce::kDxRingStages, 2, 100>;
	using Tiles4 = fslce::DxCommConfig<Config, fslce::kDxRingStages, 4, 100>;
	EXPECT_EQ(Tiles2::kTilesPerReduce, 2);
	EXPECT_EQ(Tiles4::kTilesPerReduce, 4);
	EXPECT_EQ(Tiles2::kGroupElements, 2 * CommConfig::kTileElements);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	KeepsTp1CommunicationWarpsIdle) {
	EXPECT_FALSE(fslce::kBackwardTp1ComputeOnlySm100);
	EXPECT_TRUE(fslce::kBackwardNodeLocalReduceSm100);
	EXPECT_TRUE(fslce::kBackwardRemoteReduceSm100);
	EXPECT_TRUE(fslce::kBackwardDxConsumedAfterFinalStoreSm100);
	EXPECT_EQ(Config::kDxLocalReduceWarp, 0);
	EXPECT_EQ(Config::kRemoteCommunicationWarp, 1);
	EXPECT_EQ(Config::kTmaWarp, 2);
	EXPECT_EQ(Config::kUmmaWarp, 3);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	KeepsWaveEpochsDisjoint) {
	constexpr std::uint64_t kLaunch = 0x1234ull << 32;
	std::uint64_t scatter = fslce::backward_wave_epoch_sm100(
		kLaunch, fslce::kBackwardDxScatterEpochSuffixSm100, 3);
	std::uint64_t allgather = fslce::backward_wave_epoch_sm100(
		kLaunch, fslce::kBackwardDxAllgatherEpochSuffixSm100, 3);
	std::uint64_t remote = fslce::backward_wave_epoch_sm100(
		kLaunch, fslce::kBackwardDxRemoteEpochSuffixSm100, 3);
	EXPECT_NE(scatter, allgather);
	EXPECT_NE(scatter, remote);
	EXPECT_NE(allgather, remote);
	// Ring epochs are strictly increasing in the wave index so warp 0 can wait
	// with a monotone comparison.
	EXPECT_LT(
		fslce::backward_wave_epoch_sm100(
			kLaunch, fslce::kBackwardDxRemoteEpochSuffixSm100, 3),
		fslce::backward_wave_epoch_sm100(
			kLaunch, fslce::kBackwardDxRemoteEpochSuffixSm100, 4));
	EXPECT_TRUE(fslce::backward_wave_count_supported_sm100(1));
	EXPECT_TRUE(
		fslce::backward_wave_count_supported_sm100(
			fslce::kBackwardMaxWavesSm100));
	EXPECT_FALSE(
		fslce::backward_wave_count_supported_sm100(
			fslce::kBackwardMaxWavesSm100 + 1));
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	DistributesRemoteMergeAcrossEveryResidentCta) {
	constexpr int kGridCtas = 148;
	EXPECT_EQ(
		fslce::backward_remote_merge_workers_sm100(kGridCtas),
		kGridCtas * fslce::kWarpSize);
	EXPECT_EQ(
		fslce::backward_remote_merge_worker_sm100(0, 0),
		0);
	EXPECT_EQ(
		fslce::backward_remote_merge_worker_sm100(
			kGridCtas - 1, fslce::kWarpSize - 1),
		kGridCtas * fslce::kWarpSize - 1);
	EXPECT_EQ(
		fslce::backward_remote_merge_target_sm100(0, kGridCtas),
		148ull);
	EXPECT_EQ(
		fslce::backward_remote_merge_target_sm100(1, kGridCtas),
		296ull);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	PublishesTheFusedLauncherContract) {
	fslce::BackwardTpParamsSm100<> params;
	EXPECT_EQ(params.tiles_per_reduce, 2);
	EXPECT_EQ(params.gemm.ignore_index, -100);
	EXPECT_FLOAT_EQ(params.gemm.inverse_temperature, 1.0f);
	// lse and entropy are ordinary saved tensors, never symmetric memory.
	static_assert(std::is_same_v<
		decltype(params.gemm.lse), const float*>);
	static_assert(std::is_same_v<
		decltype(params.gemm.entropy), const float*>);
	fslce::BackwardWaveWorkspaceSm100<> workspace;
	EXPECT_EQ(workspace.grid_barrier, nullptr);
	EXPECT_EQ(workspace.dx_scatter_ready, nullptr);
	EXPECT_EQ(workspace.dx_remote_received, nullptr);
	EXPECT_EQ(workspace.dx_remote_ready, nullptr);
	EXPECT_EQ(workspace.dx_remote_merge_arrived, nullptr);
	EXPECT_LT(
		fslce::kBackwardSignalRemoteReceived,
		fslce::kBackwardSignalEntries);
	EXPECT_LT(
		fslce::kBackwardSignalRemoteBase,
		fslce::kBackwardSignalEntries);
	EXPECT_LT(
		fslce::kBackwardSignalRemoteMergeArrived,
		fslce::kBackwardSignalEntries);
	EXPECT_NE(
		fslce::kBackwardSignalRemoteReceived,
		fslce::kBackwardSignalRemoteBase);
	EXPECT_NE(
		fslce::kBackwardSignalRemoteBase,
		fslce::kBackwardSignalRemoteMergeArrived);
	EXPECT_EQ(fslce::kBackwardSignalEntries, 24);
}

#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_TEST_MAINLOOP)
TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	DxStageHandshakePublishesConsumedAfterFinalStore) {
	if (!fslce::kBackwardDxConsumedAfterFinalStoreSm100) {
		FAIL() << "the production contract must publish consumed last";
	}
	constexpr int kItems = 3 * fslce::kDxRingStages;
	int* staging = nullptr;
	int* finalized = nullptr;
	int* violations = nullptr;
	ASSERT_EQ(cudaMalloc(&staging, fslce::kDxRingStages * sizeof(int)),
		cudaSuccess);
	ASSERT_EQ(cudaMalloc(&finalized, kItems * sizeof(int)), cudaSuccess);
	ASSERT_EQ(cudaMalloc(&violations, sizeof(int)), cudaSuccess);
	ASSERT_EQ(cudaMemset(finalized, 0, kItems * sizeof(int)), cudaSuccess);
	ASSERT_EQ(cudaMemset(violations, 0, sizeof(int)), cudaSuccess);
	dx_stage_handshake_contract_kernel<<<1, Config::kNumThreads>>>(
		staging, finalized, violations);
	ASSERT_EQ(cudaGetLastError(), cudaSuccess);
	ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
	int host_violations = -1;
	ASSERT_EQ(
		cudaMemcpy(
			&host_violations,
			violations,
			sizeof(int),
			cudaMemcpyDeviceToHost),
		cudaSuccess);
	EXPECT_EQ(host_violations, 0);
	ASSERT_EQ(cudaFree(violations), cudaSuccess);
	ASSERT_EQ(cudaFree(finalized), cudaSuccess);
	ASSERT_EQ(cudaFree(staging), cudaSuccess);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	AliasesOneOperandArenaAcrossPhases) {
	using Traits = fslce::BackwardGemmTraitsSm100<100>;
	using DxTraits = fslce::BackwardDxTraitsSm100;
	using DwTraits = fslce::BackwardDwTraitsSm100<100>;
	EXPECT_EQ(Traits::kStages, 5);
	EXPECT_EQ(DxTraits::kStages, 4);
	EXPECT_EQ(DwTraits::kStages, 6);
	EXPECT_EQ(
		Traits::kTmaTransBytes,
		Config::kClusterM *
			(Traits::kTmaTransBytesA + Traits::kTmaTransBytesB));
	using SmemNoEntropy = fslce::BackwardGemmSmemSm100<100, false>;
	using SmemEntropy = fslce::BackwardGemmSmemSm100<100, true>;
	EXPECT_EQ(
		sizeof(((SmemNoEntropy*)nullptr)->dx_ready) /
			sizeof(std::uint64_t),
		static_cast<std::size_t>(fslce::kDxRingStages));
	EXPECT_EQ(
		sizeof(((SmemNoEntropy*)nullptr)->dx_consumed) /
			sizeof(std::uint64_t),
		static_cast<std::size_t>(fslce::kDxRingStages));
	EXPECT_LE(static_cast<int>(sizeof(SmemNoEntropy)), 227 * 1024);
	EXPECT_LE(static_cast<int>(sizeof(SmemEntropy)), 227 * 1024);
}

TEST(
	TensorParallelFusedScaledLinearCrossEntropyBackwardSm100,
	RastersClusterPairsDeterministically) {
	fslce::BackwardPairCoordSm100 coord =
		fslce::backward_pair_m_fast_sm100(5, 4);
	EXPECT_EQ(coord.m_pair, 1);
	EXPECT_EQ(coord.n_tile, 1);
	coord = fslce::backward_pair_n_fast_sm100(6, 4);
	EXPECT_EQ(coord.m_pair, 1);
	EXPECT_EQ(coord.n_tile, 2);
	fslce::BackwardDwPairCoordSm100 dw_coord =
		fslce::backward_dw_pair_coord_sm100(5, 4);
	EXPECT_EQ(dw_coord.m_pair, 1);
	EXPECT_EQ(dw_coord.n_tile_begin, 2);
	EXPECT_EQ(fslce::backward_items_for_cluster_sm100(10, 0, 4), 3);
	EXPECT_EQ(fslce::backward_items_for_cluster_sm100(10, 3, 4), 2);
	EXPECT_EQ(fslce::backward_items_for_cluster_sm100(2, 3, 4), 0);
	EXPECT_EQ(
		fslce::backward_dx_durable_tile_sm100(1, 64, 2, 3, 8),
		static_cast<std::size_t>(64 + 2 * 8 + 3));
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
