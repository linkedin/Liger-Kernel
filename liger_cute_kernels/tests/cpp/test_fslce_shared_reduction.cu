#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "forward_reduction.cuh"
#include "liger_cute/detail/tp_reduce.cuh"

namespace fslce = liger::fused_scaled_linear_cross_entropy;

TEST(TensorParallelRemoteRing, TranslatesMatchingLocalRankNeighbors) {
	constexpr std::array<int, 4> local_rank_zero_world{
		0, 8, 16, 24};
	constexpr std::array<int, 4> local_rank_one_world{
		1, 9, 17, 25};
	auto translate = [](const auto& team, int team_rank) {
		return team[static_cast<std::size_t>(team_rank)];
	};

	int rank = 1;
	int previous = liger_cute::detail::remote_ring_previous_rank(rank, 4);
	int next = liger_cute::detail::remote_ring_next_rank(rank, 4);
	EXPECT_EQ(translate(local_rank_zero_world, previous), 0);
	EXPECT_EQ(translate(local_rank_zero_world, next), 16);
	EXPECT_EQ(translate(local_rank_one_world, previous), 1);
	EXPECT_EQ(translate(local_rank_one_world, next), 17);

	rank = 3;
	next = liger_cute::detail::remote_ring_next_rank(rank, 4);
	EXPECT_EQ(translate(local_rank_zero_world, next), 0);
	EXPECT_EQ(translate(local_rank_one_world, next), 1);
}

TEST(TensorParallelRemoteRing, UsesHostCountMinusOneSteps) {
	EXPECT_EQ(liger_cute::detail::kRemoteRingWarpSize, 32);
	EXPECT_EQ(
		liger_cute::detail::kRemoteRingFullWarpMask,
		0xffffffffu);
	EXPECT_EQ(liger_cute::detail::remote_ring_step_count(1), 0);
	EXPECT_EQ(liger_cute::detail::remote_ring_step_count(2), 1);
	EXPECT_EQ(liger_cute::detail::remote_ring_step_count(3), 2);
	EXPECT_EQ(liger_cute::detail::remote_ring_step_count(4), 3);
	EXPECT_FALSE(liger_cute::detail::remote_ring_uses_consumed(2));
	EXPECT_TRUE(liger_cute::detail::remote_ring_uses_consumed(3));
	for (int sequence = 0; sequence < 8; ++sequence) {
		EXPECT_EQ(
			liger_cute::detail::remote_ring_transport_slot(
				2, 0, sequence),
			sequence & 1);
	}
	EXPECT_EQ(
		liger_cute::detail::remote_ring_transport_slot(4, 0, 7),
		0);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_transport_slot(4, 1, 7),
		1);
}

TEST(TensorParallelRemoteRing, ModelsTheTwoHostTp16Topology) {
	liger_cute::detail::RemoteReduceView rank_zero{
		reinterpret_cast<float*>(0x1000),
		1024,
		reinterpret_cast<float*>(0x2000),
		reinterpret_cast<std::uint64_t*>(0x3000),
		reinterpret_cast<std::uint64_t*>(0x4000),
		1024,
		0,
		2,
		8,
		8,
		0,
		17};
	liger_cute::detail::RemoteReduceView rank_one = rank_zero;
	rank_one.rank = 1;
	rank_one.previous_world = 0;
	rank_one.next_world = 0;

	EXPECT_TRUE(rank_zero.enabled());
	EXPECT_TRUE(rank_one.enabled());
	rank_one.team_handle = -1;
	EXPECT_FALSE(rank_one.enabled());
	rank_one.team_handle = 17;
	EXPECT_EQ(
		liger_cute::detail::remote_ring_previous_rank(0, 2),
		1);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_next_rank(0, 2),
		1);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_previous_rank(1, 2),
		0);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_next_rank(1, 2),
		0);
}

TEST(TensorParallelRemoteRing, SelectsUniformParentRelativeTopologies) {
	EXPECT_FALSE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			1, 1, 1));
	EXPECT_FALSE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			2, 2, 1));
	EXPECT_FALSE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			4, 4, 1));
	EXPECT_FALSE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			8, 8, 1));
	EXPECT_TRUE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			2, 1, 2));
	EXPECT_TRUE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			4, 2, 2));
	EXPECT_TRUE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			8, 4, 2));
	EXPECT_TRUE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			16, 8, 2));
	EXPECT_FALSE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			16, 6, 2));
	EXPECT_TRUE(
		liger_cute::detail::tp_reduce_uses_remote_ring(
			16, 4, 4));
}

TEST(TensorParallelRemoteRing, MapsStridedWorldTeamAsHostMajor2DGrid) {
	// Parent-team ranks for world ranks {0, 4, 8, 12}. xrange=2 gives
	// local rows {0,1}/{2,3} and matching-rank columns {0,2}/{1,3}.
	EXPECT_EQ(liger_cute::detail::tp_reduce_host_rank(0, 2), 0);
	EXPECT_EQ(liger_cute::detail::tp_reduce_host_rank(1, 2), 0);
	EXPECT_EQ(liger_cute::detail::tp_reduce_host_rank(2, 2), 1);
	EXPECT_EQ(liger_cute::detail::tp_reduce_host_rank(3, 2), 1);
	EXPECT_EQ(liger_cute::detail::tp_reduce_local_rank(0, 2), 0);
	EXPECT_EQ(liger_cute::detail::tp_reduce_local_rank(1, 2), 1);
	EXPECT_EQ(liger_cute::detail::tp_reduce_local_rank(2, 2), 0);
	EXPECT_EQ(liger_cute::detail::tp_reduce_local_rank(3, 2), 1);
	EXPECT_EQ(liger_cute::detail::tp_reduce_parent_rank(0, 0, 2), 0);
	EXPECT_EQ(liger_cute::detail::tp_reduce_parent_rank(0, 1, 2), 1);
	EXPECT_EQ(liger_cute::detail::tp_reduce_parent_rank(1, 0, 2), 2);
	EXPECT_EQ(liger_cute::detail::tp_reduce_parent_rank(1, 1, 2), 3);
}

TEST(TensorParallelRemoteRing, DefinesPerBlockAndGridWorkerContracts) {
	EXPECT_EQ(
		liger_cute::detail::
			remote_ring_worker_threads_per_block<1>(),
		32);
	EXPECT_EQ(
		liger_cute::detail::
			remote_ring_worker_threads_per_block<8>(),
		256);
	EXPECT_EQ(
		liger_cute::detail::
			remote_ring_worker_threads_per_block<32>(),
		1024);
	EXPECT_FALSE(
		liger_cute::detail::remote_ring_uses_grid_sync<1>());
	EXPECT_TRUE(
		liger_cute::detail::remote_ring_uses_grid_sync<8>());
	EXPECT_EQ(
		liger_cute::detail::remote_ring_global_worker_index(
			3, 256, 17),
		785);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_global_worker_count(
			12, 256),
		3072);
}

TEST(TensorParallelRemoteRing, CirculatesOriginalContributionsToEveryHost) {
	for (int hosts = 2; hosts <= 4; ++hosts) {
		std::array<int, 4> accumulator{};
		std::array<int, 4> forwarded{};
		for (int rank = 0; rank < hosts; ++rank) {
			accumulator[static_cast<std::size_t>(rank)] = rank + 1;
			forwarded[static_cast<std::size_t>(rank)] = rank + 1;
		}
		for (int step = 0;
				step < liger_cute::detail::remote_ring_step_count(hosts);
				++step) {
			std::array<int, 4> received{};
			for (int rank = 0; rank < hosts; ++rank) {
				int previous =
					liger_cute::detail::remote_ring_previous_rank(
						rank, hosts);
				received[static_cast<std::size_t>(rank)] =
					forwarded[static_cast<std::size_t>(previous)];
			}
			for (int rank = 0; rank < hosts; ++rank) {
				accumulator[static_cast<std::size_t>(rank)] +=
					received[static_cast<std::size_t>(rank)];
				forwarded[static_cast<std::size_t>(rank)] =
					received[static_cast<std::size_t>(rank)];
			}
		}
		int expected = hosts * (hosts + 1) / 2;
		for (int rank = 0; rank < hosts; ++rank) {
			EXPECT_EQ(
				accumulator[static_cast<std::size_t>(rank)],
				expected);
		}
	}
}

TEST(TensorParallelRemoteRing, ReusesSlotsOnlyAfterAcknowledgement) {
	for (int hosts = 2; hosts <= 4; ++hosts) {
		int steps = liger_cute::detail::remote_ring_step_count(hosts);
		std::array<int, 3> acknowledgement_count{};
		for (int step = 0; step < steps; ++step) {
			EXPECT_EQ(
				liger_cute::detail::remote_ring_slot(step),
				step & 1);
			int reused =
				liger_cute::detail::remote_ring_reused_step(step);
			if (reused >= 0) {
				EXPECT_EQ(
					liger_cute::detail::remote_ring_slot(reused),
					liger_cute::detail::remote_ring_slot(step));
				EXPECT_EQ(reused, step - 2);
			}
			int forwarded =
				liger_cute::detail::remote_ring_forwarded_step(step);
			if (forwarded >= 0) {
				++acknowledgement_count[
					static_cast<std::size_t>(forwarded)];
			}
		}
		++acknowledgement_count[
			static_cast<std::size_t>(steps - 1)];
		for (int step = 0; step < steps; ++step) {
			EXPECT_EQ(
				acknowledgement_count[
					static_cast<std::size_t>(step)],
				1);
		}
		EXPECT_EQ(
			liger_cute::detail::remote_ring_drain_begin_step(hosts),
			steps > 2 ? steps - 2 : 0);
	}
}

TEST(TensorParallelRemoteRing, CarriesFinalSlotAcknowledgementsAcrossWaves) {
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(2, 0),
		0);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(2, 1),
		-1);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(3, 0),
		0);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(3, 1),
		1);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(4, 0),
		2);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(4, 1),
		1);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(5, 0),
		2);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_last_step_for_slot(5, 1),
		3);
}

TEST(TensorParallelRemoteRing, SizesPingPongPayloadAndSignals) {
	constexpr std::size_t payload_bytes = 4096;
	EXPECT_EQ(
		liger_cute::detail::remote_ring_inbox_bytes(payload_bytes),
		2 * payload_bytes);
	EXPECT_EQ(
		liger_cute::detail::remote_ring_signal_bytes(),
		4 * sizeof(std::uint64_t));

	constexpr std::uint64_t launch_epoch = std::uint64_t{7} << 32;
	std::uint64_t step0 = liger_cute::detail::remote_ring_signal_value(
		launch_epoch,
		liger_cute::detail::kForwardRemoteEpochSuffix,
		0);
	std::uint64_t step1 = liger_cute::detail::remote_ring_signal_value(
		launch_epoch,
		liger_cute::detail::kForwardRemoteEpochSuffix,
		1);
	std::uint64_t next_launch =
		liger_cute::detail::remote_ring_signal_value(
			launch_epoch + (std::uint64_t{1} << 32),
			liger_cute::detail::kRemoteSumEpochSuffix,
			0);
	EXPECT_LT(step0, step1);
	EXPECT_LT(step1, next_launch);
	EXPECT_EQ(step0 >> 32, launch_epoch >> 32);
	EXPECT_EQ(step0 & 0xffffu, 1u);
	EXPECT_EQ(step1 & 0xffffu, 2u);
}

TEST(TensorParallelForwardReduction, SharesPackedStateLayout) {
	EXPECT_EQ(fslce::forward_reduced_fields<false>(), 2);
	EXPECT_EQ(fslce::forward_reduced_fields<true>(), 3);
	EXPECT_EQ(fslce::forward_reduced_state_fields<false>(), 3);
	EXPECT_EQ(fslce::forward_reduced_state_fields<true>(), 4);

	float packed[4] = {};
	fslce::ReducedSoftmaxState state{4.0f, 2.0f, 3.0f, 6.0f};
	fslce::store_forward_reduced_state<true>(packed, state);
	fslce::ReducedSoftmaxState loaded =
		fslce::load_forward_reduced_state<true>(packed);
	EXPECT_FLOAT_EQ(loaded.max_value, state.max_value);
	EXPECT_FLOAT_EQ(loaded.exp_sum, state.exp_sum);
	EXPECT_FLOAT_EQ(loaded.target_logit, state.target_logit);
	EXPECT_FLOAT_EQ(
		loaded.exp_weighted_sum, state.exp_weighted_sum);
}

TEST(TensorParallelForwardReduction, MergesTwoThreeAndFourHostsStably) {
	for (int hosts = 2; hosts <= 4; ++hosts) {
		fslce::ReducedSoftmaxState merged{
			1.0f, 1.0f, 0.0f, 1.0f};
		for (int host = 1; host < hosts; ++host) {
			float logit = static_cast<float>(host + 1);
			fslce::ReducedSoftmaxState contribution{
				logit,
				1.0f,
				host == 1 ? logit : 0.0f,
				logit};
			merged = fslce::merge_reduced_softmax<true>(
				merged, contribution);
		}

		float global_max = static_cast<float>(hosts);
		float expected_sum = 0.0f;
		float expected_weighted = 0.0f;
		for (int host = 0; host < hosts; ++host) {
			float logit = static_cast<float>(host + 1);
			float weight = expf(logit - global_max);
			expected_sum += weight;
			expected_weighted += weight * logit;
		}
		EXPECT_FLOAT_EQ(merged.max_value, global_max);
		EXPECT_NEAR(merged.exp_sum, expected_sum, 1.0e-6f);
		EXPECT_FLOAT_EQ(merged.target_logit, 2.0f);
		EXPECT_NEAR(
			merged.exp_weighted_sum,
			expected_weighted,
			1.0e-6f);
	}
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
