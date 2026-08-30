#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "backward_gemm_mainloop_sm90.cuh"
#include "fused_scaled_linear_cross_entropy.cuh"

namespace fslce = liger::fused_scaled_linear_cross_entropy;

using ForwardConfig = fslce::ForwardGemmConfigSm90<90>;
using BackwardConfig = fslce::BackwardGemmConfigSm90<90>;
using CommConfig = fslce::DxCommConfig<BackwardConfig, 4, 2>;

static_assert(ForwardConfig::kCompute == 90);
static_assert(BackwardConfig::kCompute == 90);
static_assert(fslce::BackwardDzHandoffTraitsSm90<90>::kStages == 3);
static_assert(fslce::BackwardDxCluster2TraitsSm90<90>::kStages == 4);
static_assert(fslce::BackwardDxCluster2TraitsSm90<90>::kStoreTileN == 32);
static_assert(sizeof(fslce::BackwardDxCluster2SmemSm90<90>) == 230400);
static_assert(fslce::backward_dx_split_k(2048, 256) == 2);
static_assert(fslce::backward_dx_split_k(2048, 320) == 1);
static_assert(fslce::backward_dx_split_k(4096, 256) == 1);

TEST(TensorParallelFusedScaledLinearCrossEntropyConfig, PreservesHopperWarpRoles) {
	EXPECT_EQ(
		fslce::backward_warp_role(0),
		fslce::BackwardWarpRole::kProducer);
	EXPECT_EQ(
		fslce::backward_warp_role(1),
		fslce::BackwardWarpRole::kDxCommunication);
	EXPECT_EQ(
		fslce::backward_warp_role(2),
		fslce::BackwardWarpRole::kDxCommunication);
	EXPECT_EQ(
		fslce::backward_warp_role(3),
		fslce::BackwardWarpRole::kReserved);
	for (int warp = 4; warp < fslce::kNumWarps; ++warp) {
		EXPECT_EQ(
			fslce::backward_warp_role(warp),
			fslce::BackwardWarpRole::kConsumer);
	}
	EXPECT_EQ(
		fslce::backward_warp_role(fslce::kNumWarps),
		fslce::BackwardWarpRole::kInactive);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyConfig, CoalescesTwoDxTilesByDefault) {
	EXPECT_EQ(fslce::dx_groups_per_m_tile<CommConfig>(5), 3);

	fslce::DxTileGroup first =
		fslce::make_dx_tile_group<CommConfig>(3, 0, 5, 7);
	EXPECT_EQ(first.m_tile, 3);
	EXPECT_EQ(first.first_n_tile, 0);
	EXPECT_EQ(first.num_tiles, 2);
	EXPECT_EQ(first.sequence, 7);

	fslce::DxTileGroup tail =
		fslce::make_dx_tile_group<CommConfig>(3, 2, 5, 9);
	EXPECT_EQ(tail.first_n_tile, 4);
	EXPECT_EQ(tail.num_tiles, 1);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyOnlineSoftmax, MatchesGlobalReference) {
	fslce::OnlineSoftmaxState rank0;
	rank0.fold<true>(1.0f, true, false);
	rank0.fold<true>(3.0f, true, false);

	fslce::OnlineSoftmaxState rank1;
	rank1.fold<true>(2.0f, true, true);
	rank1.fold<true>(4.0f, true, false);

	float global_max = fmaxf(rank0.max_value, rank1.max_value);
	fslce::CorrectedSoftmaxStats corrected0 =
		fslce::correct_for_global_max<true>(rank0, global_max);
	fslce::CorrectedSoftmaxStats corrected1 =
		fslce::correct_for_global_max<true>(rank1, global_max);

	float global_sum = corrected0.exp_sum + corrected1.exp_sum;
	float global_weighted =
		corrected0.exp_weighted_sum + corrected1.exp_weighted_sum;
	float global_target =
		corrected0.target_logit + corrected1.target_logit;
	fslce::FinalizedSoftmax result = fslce::finalize_softmax<true>(
		global_max,
		global_sum,
		global_target,
		global_weighted,
		false);

	float direct_sum =
		expf(1.0f - global_max) +
		expf(2.0f - global_max) +
		expf(3.0f - global_max) +
		expf(4.0f - global_max);
	float expected_lse = global_max + logf(direct_sum);
	float direct_weighted =
		expf(1.0f - global_max) * 1.0f +
		expf(2.0f - global_max) * 2.0f +
		expf(3.0f - global_max) * 3.0f +
		expf(4.0f - global_max) * 4.0f;

	EXPECT_NEAR(result.lse, expected_lse, 1.0e-6f);
	EXPECT_NEAR(result.nll, expected_lse - 2.0f, 1.0e-6f);
	EXPECT_NEAR(
		result.entropy,
		expected_lse - direct_weighted / direct_sum,
		1.0e-6f);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyOnlineSoftmax, ChunkMergeMatchesSingleFold) {
	fslce::OnlineSoftmaxState lhs;
	lhs.fold<true>(-2.0f, true, false);
	lhs.fold<true>(0.5f, true, false);

	fslce::OnlineSoftmaxState rhs;
	rhs.fold<true>(3.0f, true, true);
	rhs.fold<true>(1.0f, true, false);

	fslce::OnlineSoftmaxState merged =
		fslce::merge_online_softmax<true>(lhs, rhs);
	fslce::OnlineSoftmaxState direct;
	direct.fold<true>(-2.0f, true, false);
	direct.fold<true>(0.5f, true, false);
	direct.fold<true>(3.0f, true, true);
	direct.fold<true>(1.0f, true, false);

	EXPECT_NEAR(merged.max_value, direct.max_value, 1.0e-7f);
	EXPECT_NEAR(merged.exp_sum, direct.exp_sum, 1.0e-6f);
	EXPECT_NEAR(
		merged.exp_weighted_sum,
		direct.exp_weighted_sum,
		1.0e-6f);
	EXPECT_EQ(merged.has_target, 1);
	EXPECT_FLOAT_EQ(merged.target_logit, 3.0f);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyOnlineSoftmax, CompilesOutEntropyMoment) {
	fslce::OnlineSoftmaxState state;
	state.fold<false>(2.0f, true, false);
	state.fold<false>(4.0f, true, false);
	EXPECT_FLOAT_EQ(state.exp_weighted_sum, 0.0f);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyOnlineSoftmax, ZerosIgnoredEntropy) {
	fslce::FinalizedSoftmax result = fslce::finalize_softmax<true>(
		4.0f,
		2.0f,
		0.0f,
		6.0f,
		true);
	EXPECT_FLOAT_EQ(result.nll, 0.0f);
	EXPECT_FLOAT_EQ(result.entropy, 0.0f);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyBackward, AppliesTemperatureChainRule) {
	float temperature = 2.0f;
	float scaled_logit = 0.25f;
	float lse = 1.25f;
	float scale = 0.75f;
	float probability = expf(scaled_logit - lse);

	float gradient = fslce::scaled_cross_entropy_dlogit(
		scaled_logit,
		lse,
		true,
		scale,
		temperature);
	float expected = scale * (probability - 1.0f) / temperature;
	EXPECT_NEAR(gradient, expected, 1.0e-7f);
}

TEST(TensorParallelFusedScaledLinearCrossEntropyBackward, ZerosIgnoredRows) {
	float gradient = fslce::scaled_cross_entropy_dlogit(
		0.25f,
		1.25f,
		true,
		1.0f,
		2.0f,
		0.5f,
		1.0f,
		true);
	EXPECT_FLOAT_EQ(gradient, 0.0f);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
