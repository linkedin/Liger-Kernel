#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <vector>

#include "liger_cute/detail/comm_schedule.cuh"

namespace {
using liger_cute::detail::comm_peer;

__global__ void schedule_kernel(int* output, int hosts, int local_pes) {
	int offset = threadIdx.x;
	if (offset < hosts * local_pes)
		output[offset] = comm_peer(1, offset, hosts * local_pes, hosts);
}
}  // namespace

TEST(MoeCommSchedule, MatchesHostMajorPermutation) {
	for (int hosts : {1, 2, 3, 4, 8, 16, 32, 64}) {
		for (int local_pes : {1, 2, 4, 8}) {
			int size = hosts * local_pes;
			std::vector<int> destinations, positions(size);
			for (int local = 0; local < local_pes; ++local) {
				for (int host = 0; host < hosts; ++host) {
					int rank = host * local_pes + local;
					positions[rank] = destinations.size();
					destinations.push_back(rank);
				}
			}
			for (int rank = 0; rank < size; ++rank) {
				for (int offset = 0; offset < size; ++offset) {
					ASSERT_EQ(comm_peer(rank, offset, size, hosts),
						destinations[(positions[rank] + offset) % size]);
				}
			}
		}
	}
}

TEST(MoeCommSchedule, CapturedTopologySurvivesOtherLaunches) {
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
	schedule_kernel<<<1, 32, 0, stream>>>(output, 2, 4);
	ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
	ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
	for (int iteration = 0; iteration < 2; ++iteration) {
		schedule_kernel<<<1, 32, 0, stream>>>(output, 1, 8);
		ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
		ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
		int actual[8];
		ASSERT_EQ(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
		for (int offset = 0; offset < 8; ++offset)
			EXPECT_EQ(actual[offset], comm_peer(1, offset, 8, 2));
	}
	EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
	EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
	EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
	EXPECT_EQ(cudaFree(output), cudaSuccess);
}
