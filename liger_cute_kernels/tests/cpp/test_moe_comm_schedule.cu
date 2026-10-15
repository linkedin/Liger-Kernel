#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <vector>

#include "liger_cute/detail/comm_schedule.cuh"

namespace {
using liger_cute::detail::comm_peer;

__global__ void schedule_kernel(int* output, int num_pes) {
	int offset = threadIdx.x;
	if (offset < num_pes)
		output[offset] = comm_peer(1, offset, num_pes);
}
}  // namespace

TEST(MoeCommSchedule, VisitsEveryPeerInTeamRankOrder) {
	for (int size : {1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512}) {
		for (int rank = 0; rank < size; ++rank) {
			EXPECT_EQ(comm_peer(rank, 0, size), rank);
			std::vector<bool> visited(size);
			int previous = rank;
			for (int offset = 0; offset < size; ++offset) {
				int peer = comm_peer(rank, offset, size);
				ASSERT_GE(peer, 0);
				ASSERT_LT(peer, size);
				ASSERT_FALSE(visited[peer]);
				visited[peer] = true;
				if (offset > 0) EXPECT_EQ(peer, (previous + 1) % size);
				previous = peer;
			}
		}
	}
}

TEST(MoeCommSchedule, CapturedGroupSizeSurvivesOtherLaunches) {
	int devices = 0;
	if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
		GTEST_SKIP() << "requires CUDA";
	int* output = nullptr;
	cudaStream_t stream;
	cudaGraph_t graph;
	cudaGraphExec_t executable;
	ASSERT_EQ(cudaMalloc(&output, 8 * sizeof(int)), cudaSuccess);
	ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
	ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
	schedule_kernel<<<1, 32, 0, stream>>>(output, 8);
	ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
	ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
	for (int iteration = 0; iteration < 2; ++iteration) {
		schedule_kernel<<<1, 32, 0, stream>>>(output, 4);
		ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
		ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
		int actual[8];
		ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
		for (int offset = 0; offset < 8; ++offset)
			EXPECT_EQ(actual[offset], comm_peer(1, offset, 8));
	}
	EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
	EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
	EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
	EXPECT_EQ(cudaFree(output), cudaSuccess);
}
