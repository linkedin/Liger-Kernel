#include <gtest/gtest.h>

#include <type_traits>

#include "forward_gemm_mainloop_sm100.cuh"
#include "forward_reduction.cuh"

namespace fslce = liger::fused_scaled_linear_cross_entropy;

using Config = fslce::ForwardGemmConfigSm100<>;

// C++17 equivalent of remove_cvref_t for the parenthesized member expression.
using ForwardOutputMember = std::remove_cv_t<std::remove_reference_t<
	decltype((fslce::ForwardGemmParamsSm100<>{}.output))>>;
static_assert(std::is_same_v<
	ForwardOutputMember,
	fslce::ForwardLocalStatsBuffers>);

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, UsesSharedReductionState) {
	EXPECT_EQ(fslce::forward_reduced_fields<false>(), 2);
	EXPECT_EQ(fslce::forward_reduced_fields<true>(), 3);
	EXPECT_EQ(fslce::forward_reduced_state_fields<false>(), 3);
	EXPECT_EQ(fslce::forward_reduced_state_fields<true>(), 4);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, PreservesWarpRoles) {
	EXPECT_EQ(
		fslce::forward_warp_role_sm100(0),
		fslce::ForwardWarpRoleSm100::kLocalReduceControl);
	EXPECT_EQ(
		fslce::forward_warp_role_sm100(1),
		fslce::ForwardWarpRoleSm100::kRemoteCommunication);
	EXPECT_EQ(
		fslce::forward_warp_role_sm100(2),
		fslce::ForwardWarpRoleSm100::kTmaProducer);
	EXPECT_EQ(
		fslce::forward_warp_role_sm100(3),
		fslce::ForwardWarpRoleSm100::kUmmaProducer);
	for (int warp = 4; warp < fslce::kNumWarps; ++warp) {
		EXPECT_EQ(
			fslce::forward_warp_role_sm100(warp),
			fslce::ForwardWarpRoleSm100::kEpilogue);
	}
	EXPECT_EQ(
		fslce::forward_warp_role_sm100(fslce::kNumWarps),
		fslce::ForwardWarpRoleSm100::kInactive);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, DefinesDoubleBufferedTmem) {
	EXPECT_EQ(Config::kCompute, 100);
	EXPECT_EQ(Config::kClusterM, 2);
	EXPECT_EQ(Config::kTileM, 256);
	EXPECT_EQ(Config::kCtaTileM, 128);
	EXPECT_EQ(Config::kUmmaTileN, 256);
	EXPECT_EQ(Config::kLogicalTileN, 256);
	EXPECT_EQ(Config::kAccumulatorStages, 2);
	EXPECT_EQ(Config::kAccumulatorPanels, 1);
	EXPECT_EQ(Config::kTmemStageColumns, 256);
	EXPECT_EQ(Config::kTmemColumns, 512);
	EXPECT_EQ(
		Config::kMainloopStages,
		LIGER_CUTE_FSLCE_SM100_STAGES);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, SplitsEpilogueNBetweenWarpgroups) {
	EXPECT_EQ(Config::kEpilogueWarpgroups, 2);
	EXPECT_EQ(Config::kWarpgroupTileN, 128);
	EXPECT_EQ(Config::kEpilogueChunkN, 64);
	EXPECT_EQ(Config::kChunksPerWarpgroup, 2);
	EXPECT_EQ(Config::kEpilogueThreads, 256);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, UsesCtaLocalMGeometryForPartials) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	EXPECT_EQ(Launch::num_m_tiles(1), 1);
	EXPECT_EQ(Launch::num_m_tiles(128), 1);
	EXPECT_EQ(Launch::num_m_tiles(129), 2);
	EXPECT_EQ(Launch::num_m_pairs(129), 1);
	EXPECT_EQ(Launch::num_m_pairs(257), 2);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, KeepsNLoopInsideEachSplitCluster) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	EXPECT_EQ(Launch::num_logical_n_tiles(1025), 5);

	fslce::ForwardGemmTuningSm100<> tuning;
	tuning.split_n = 2;
	auto split = Launch::resolve_split(
		tuning, 257, 4096, /*max_active_clusters=*/16);
	EXPECT_EQ(split.num_m_tiles, 3);
	EXPECT_EQ(split.num_m_pairs, 2);
	EXPECT_EQ(split.split_n, 2);
	EXPECT_EQ(split.num_cluster_pairs, 4);
	EXPECT_LT(
		split.num_cluster_pairs,
		split.num_m_pairs * split.num_logical_n_tiles);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, ResolvesUnevenSplitsDeterministically) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	fslce::ForwardGemmTuningSm100<> tuning;
	auto split = Launch::resolve_split(
		tuning, 513, 8192, /*max_active_clusters=*/10);
	ASSERT_EQ(split.num_m_pairs, 3);
	EXPECT_EQ(split.base_split_n, 3);
	EXPECT_EQ(split.extra_m_pairs, 1);
	EXPECT_EQ(split.split_n, 4);
	EXPECT_EQ(split.num_cluster_pairs, 10);
	EXPECT_EQ(split.split_count_for_pair(0), 4);
	EXPECT_EQ(split.split_count_for_pair(1), 3);
	EXPECT_EQ(split.split_count_for_pair(2), 3);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, DecodesWMajorWorkWithUnevenTail) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	fslce::ForwardGemmTuningSm100<> tuning;
	auto split = Launch::resolve_split(
		tuning, 513, 8192, /*max_active_clusters=*/10);
	ASSERT_EQ(split.num_cluster_pairs, 10);

	constexpr int expected_m_pair[] = {
		0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
	constexpr int expected_split[] = {
		0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
	for (int work_id = 0;
			work_id < split.num_cluster_pairs;
			++work_id) {
		auto work = fslce::forward_gemm_assign_work_sm100(
			split, work_id, /*cluster_rank=*/0);
		EXPECT_EQ(work.m_pair, expected_m_pair[work_id]);
		EXPECT_EQ(work.split_id, expected_split[work_id]);
		EXPECT_EQ(
			work.split_count,
			split.split_count_for_pair(work.m_pair));
	}
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, MapsVocabularyTilesToWaves) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	constexpr int kWaveColumns =
		fslce::kForwardWaveNTilesSm100 *
		Config::kLogicalTileN;
	EXPECT_EQ(
		fslce::forward_wave_count_sm100(
			Launch::num_logical_n_tiles(kWaveColumns)),
		1);
	EXPECT_EQ(
		fslce::forward_wave_count_sm100(
			Launch::num_logical_n_tiles(kWaveColumns + 1)),
		2);
	EXPECT_EQ(
		fslce::forward_wave_begin_tile_sm100(1),
		fslce::kForwardWaveNTilesSm100);
	EXPECT_EQ(
		fslce::forward_wave_end_tile_sm100(
			1, fslce::kForwardWaveNTilesSm100 + 3),
		fslce::kForwardWaveNTilesSm100 + 3);
	EXPECT_EQ(
		fslce::forward_wave_tile_valid_columns_sm100(
			kWaveColumns + 1, 1, 0),
		1);
	EXPECT_EQ(
		fslce::forward_wave_tile_valid_columns_sm100(
			kWaveColumns + 1, 1, 1),
		0);
	EXPECT_FALSE(
		fslce::forward_wave_tile_is_neutral_sm100(
			kWaveColumns + 1, 1, 0));
	EXPECT_TRUE(
		fslce::forward_wave_tile_is_neutral_sm100(
			kWaveColumns + 1, 1, 1));
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, SplitsEveryWaveDeterministically) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	constexpr int kLogicalTiles =
		fslce::kForwardWaveNTilesSm100 + 3;
	fslce::ForwardGemmTuningSm100<> tuning;
	tuning.split_n = 3;
	auto split = Launch::resolve_split(
		tuning,
		256,
		kLogicalTiles * Config::kLogicalTileN,
		/*max_active_clusters=*/3);
	ASSERT_EQ(split.num_waves, 2);
	int first_wave_tiles = 0;
	int second_wave_tiles = 0;
	for (int split_id = 0; split_id < 3; ++split_id) {
		first_wave_tiles +=
			fslce::forward_wave_split_tile_count_sm100(
				split, 0, split_id, 3);
		second_wave_tiles +=
			fslce::forward_wave_split_tile_count_sm100(
				split, 1, split_id, 3);
		int first_tail =
			fslce::forward_wave_split_first_tile_sm100(
				split, 1, split_id, 3);
		EXPECT_GE(
			first_tail,
			fslce::kForwardWaveNTilesSm100);
		EXPECT_EQ(first_tail % 3, split_id);
	}
	EXPECT_EQ(
		first_wave_tiles,
		fslce::kForwardWaveNTilesSm100);
	EXPECT_EQ(second_wave_tiles, 3);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, ReusesFourSourceSlotsByExactEpoch) {
	constexpr std::uint64_t launch_epoch =
		std::uint64_t{9} << 32;
	for (int wave = 0; wave < 12; ++wave) {
		EXPECT_EQ(
			fslce::forward_wave_slot_sm100(wave),
			wave & 3);
		EXPECT_EQ(
			fslce::forward_wave_reused_wave_sm100(wave),
			wave < 4 ? -1 : wave - 4);
		std::uint64_t epoch =
			fslce::forward_wave_epoch_sm100(
				launch_epoch, wave);
		EXPECT_EQ(epoch >> 32, launch_epoch >> 32);
		EXPECT_EQ(
			epoch & fslce::kForwardWaveEpochMaskSm100,
			static_cast<std::uint64_t>(wave + 1) <<
				fslce::kForwardWaveEpochShiftSm100);
		EXPECT_EQ(
			epoch &
				liger_cute::detail::kRemoteRingStepMask,
			0u);
		std::uint64_t operation =
			fslce::forward_wave_operation_suffix_sm100(
				liger_cute::detail::
					kForwardRemoteEpochSuffix,
				wave);
		EXPECT_EQ(
			operation &
				liger_cute::detail::kRemoteRingStepMask,
			0u);
		EXPECT_EQ(
			operation &
				fslce::kForwardWaveEpochMaskSm100,
			epoch &
				fslce::kForwardWaveEpochMaskSm100);
	}
	EXPECT_TRUE(
		fslce::forward_wave_count_supported_sm100(
			fslce::kForwardMaxWavesSm100));
	EXPECT_FALSE(
		fslce::forward_wave_count_supported_sm100(
			fslce::kForwardMaxWavesSm100 + 1));
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, SizesWorkspaceForCtaRows) {
	using Launch = fslce::ForwardGemmLaunchSm100<>;
	std::size_t expected_rows =
		fslce::kForwardWaveSourceSlotsSm100 *
		2u * 3u * 128u;
	EXPECT_EQ(
		Launch::workspace_bytes(
			129, 3 * Config::kLogicalTileN, true),
		expected_rows * 4u * sizeof(float));
	EXPECT_EQ(
		Launch::wave_partial_ready_entries(
			129, 3 * Config::kLogicalTileN),
		fslce::kForwardWaveSourceSlotsSm100 * 2u * 3u);
	EXPECT_EQ(
		Launch::wave_tile_ready_entries(129),
		fslce::kForwardWaveSourceSlotsSm100 * 2u);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, UsesNativePairedCtaUmmaTraits) {
	using Traits = fslce::ForwardGemmTraitsSm100<>;
	EXPECT_EQ(cute::size(typename Traits::TiledMma::AtomThrID{}), 2);
	EXPECT_EQ(Traits::kTileM, 256);
	EXPECT_EQ(Traits::kTileN, 256);
	EXPECT_EQ(Traits::kTileK, 64);
	EXPECT_GT(Traits::kTmaTransBytes, 0);
}

TEST(TensorParallelFusedScaledLinearCrossEntropySm100, FitsConfiguredStageSharedMemoryBudget) {
	using Smem = fslce::ForwardGemmSmemSm100<100, true>;
	constexpr std::size_t kBlackwellOptinBudget = 228u * 1024u;
	EXPECT_EQ(
		Config::kMainloopStages,
		LIGER_CUTE_FSLCE_SM100_STAGES);
	EXPECT_GE(Config::kMainloopStages, 3);
	EXPECT_LE(Config::kMainloopStages, 6);
	EXPECT_LE(sizeof(Smem), kBlackwellOptinBudget);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
