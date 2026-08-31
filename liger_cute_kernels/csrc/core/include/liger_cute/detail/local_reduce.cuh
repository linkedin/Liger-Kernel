#pragma once

// Device-side local reduction algorithms for TpReducePlan backends.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "liger_cute/detail/nvls.cuh"
#include "liger_cute/detail/tp_reduce.cuh"

namespace liger_cute {
namespace detail {

template <LocalReduceBackend Backend, typename Element>
struct LocalReduceContext;

template <typename Element>
struct LocalReduceContext<LocalReduceBackend::kNvls, Element> {
	std::uint64_t* local_ready;
	std::uint64_t* multicast_ready;
	std::uint64_t* local_complete;
	std::uint64_t* multicast_complete;
	int rank;
	int size;
};

template <typename Element>
struct LocalReduceContext<LocalReduceBackend::kDirectPeer, Element> {
	const Element* const* peer_sources;
	std::size_t source_offset;
	std::uint64_t* local_ready;
	std::uint64_t* const* peer_sync;
	std::size_t ready_offset;
	std::uint64_t* local_complete;
	std::size_t complete_offset;
	int rank;
	int size;
};

__device__ __forceinline__ void direct_peer_barrier_warp(
		std::uint64_t* local_signals,
		std::uint64_t* const* peer_signals,
		std::size_t signal_offset,
		int rank,
		int size,
		std::uint64_t epoch) {
	unsigned int lane = nvls_lane_id();
	for (int peer = static_cast<int>(lane); peer < size; peer += 32) {
		asm volatile(
			"st.release.sys.global.u64 [%0], %1;"
			:
			: "l"(peer_signals[peer] + signal_offset + rank), "l"(epoch)
			: "memory");
	}
	__syncwarp();

	for (int peer = static_cast<int>(lane); peer < size; peer += 32) {
		std::uint64_t observed = 0;
		do {
			asm volatile(
				"ld.acquire.sys.global.u64 %0, [%1];"
				: "=l"(observed)
				: "l"(local_signals + peer)
				: "memory");
		} while (observed != epoch);
	}
	__syncwarp();
}

template <LocalReduceBackend Backend, typename Element>
__device__ __forceinline__ void local_all_reduce(
		const LocalReduceContext<Backend, Element>& context,
		Element* destination,
		const Element* source,
		std::size_t count,
		std::uint64_t epoch) {
	if constexpr (Backend == LocalReduceBackend::kNvls) {
		static_assert(
			std::is_same_v<Element, float>,
			"NVLS two-shot currently supports FP32 SUM");
		nvls_barrier_warp(
			context.local_ready,
			context.multicast_ready,
			context.rank,
			context.size,
			epoch);
		nvls_sum_reduce_warp_twoshot(
			destination,
			source,
			count,
			context.rank,
			context.size);
		nvls_barrier_warp(
			context.local_complete,
			context.multicast_complete,
			context.rank,
			context.size,
			epoch);
	} else {
		static_assert(
			std::is_same_v<Element, float>,
			"direct-peer all-reduce currently supports FP32 SUM");
		direct_peer_barrier_warp(
			context.local_ready,
			context.peer_sync,
			context.ready_offset,
			context.rank,
			context.size,
			epoch);

		unsigned int lane = nvls_lane_id();
		std::size_t vectors = count / 4;
		for (std::size_t vector = lane; vector < vectors; vector += 32) {
			float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
			for (int peer = 0; peer < context.size; ++peer) {
				const auto* peer_source =
					reinterpret_cast<const float4*>(
						context.peer_sources[peer] +
						context.source_offset);
				float4 value = peer_source[vector];
				sum.x += value.x;
				sum.y += value.y;
				sum.z += value.z;
				sum.w += value.w;
			}
			reinterpret_cast<float4*>(destination)[vector] = sum;
		}
		for (std::size_t index = vectors * 4 + lane;
				index < count;
				index += 32) {
			float sum = 0.0f;
			for (int peer = 0; peer < context.size; ++peer) {
				sum += context.peer_sources[peer][
					context.source_offset + index];
			}
			destination[index] = sum;
		}
		__syncwarp();

		direct_peer_barrier_warp(
			context.local_complete,
			context.peer_sync,
			context.complete_offset,
			context.rank,
			context.size,
			epoch);
	}
}

}  // namespace detail
}  // namespace liger_cute
