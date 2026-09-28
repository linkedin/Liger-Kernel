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

__device__ __forceinline__ std::uint32_t ordered_float_bits(float value) {
	std::uint32_t bits = __float_as_uint(value);
	std::uint32_t mask =
		static_cast<std::int32_t>(bits) < 0
		? 0xffffffffu
		: 0x80000000u;
	return bits ^ mask;
}

__device__ __forceinline__ float ordered_bits_float(std::uint32_t ordered) {
	std::uint32_t mask =
		(ordered & 0x80000000u) != 0
		? 0x80000000u
		: 0xffffffffu;
	return __uint_as_float(ordered ^ mask);
}

__device__ __forceinline__ void encode_ordered_float_warp(
		float* values, std::size_t count) {
	unsigned int lane = nvls_lane_id();
	for (std::size_t index = lane; index < count; index += 32) {
		reinterpret_cast<std::uint32_t*>(values)[index] =
			ordered_float_bits(values[index]);
	}
	__syncwarp();
}

__device__ __forceinline__ void decode_ordered_float_warp(
		float* values, std::size_t count) {
	unsigned int lane = nvls_lane_id();
	for (std::size_t index = lane; index < count; index += 32) {
		std::uint32_t ordered =
			reinterpret_cast<std::uint32_t*>(values)[index];
		values[index] = ordered_bits_float(ordered);
	}
	__syncwarp();
}

template <LocalReduceBackend Backend, typename Element>
struct LocalReduceContext;

template <typename Element>
struct LocalReduceContext<LocalReduceBackend::kNvls, Element> {
	Element* local_source;
	Element* local_destination;
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

template <LocalReduceBackend Backend, ReduceOp Op, typename Element>
__device__ __forceinline__ void local_all_reduce(
		const LocalReduceContext<Backend, Element>& context,
		Element* destination,
		const Element* source,
		std::size_t count,
		std::uint64_t epoch) {
	if constexpr (Backend == LocalReduceBackend::kNvls) {
		static_assert(
			std::is_same_v<Element, float>,
			"NVLS two-shot currently supports FP32 reductions");
		if constexpr (Op == ReduceOp::kMax) {
			encode_ordered_float_warp(
				context.local_source, count);
			publish_local_reduce_source();
		}
		nvls_barrier_warp(
			context.local_ready,
			context.multicast_ready,
			context.rank,
			context.size,
			epoch);
		if constexpr (Op == ReduceOp::kSum) {
			nvls_sum_reduce_warp_twoshot(
				destination,
				source,
				count,
				context.rank,
				context.size);
		} else {
			nvls_max_reduce_warp_twoshot(
				destination,
				source,
				count,
				context.rank,
				context.size);
		}
		nvls_barrier_warp(
			context.local_complete,
			context.multicast_complete,
			context.rank,
			context.size,
			epoch);
		if constexpr (Op == ReduceOp::kMax) {
			decode_ordered_float_warp(
				context.local_destination, count);
		}
	} else {
		static_assert(
			std::is_same_v<Element, float>,
			"direct-peer all-reduce currently supports FP32 reductions");
		if constexpr (Op == ReduceOp::kMax) {
			encode_ordered_float_warp(
				const_cast<Element*>(source), count);
			publish_local_reduce_source();
		}
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
			float4 reduced = {0.0f, 0.0f, 0.0f, 0.0f};
			std::uint32_t reduced_bits[4] = {0, 0, 0, 0};
			for (int peer = 0; peer < context.size; ++peer) {
				if constexpr (Op == ReduceOp::kSum) {
					const auto* peer_source =
						reinterpret_cast<const float4*>(
							context.peer_sources[peer] +
							context.source_offset);
					float4 value = peer_source[vector];
					reduced.x += value.x;
					reduced.y += value.y;
					reduced.z += value.z;
					reduced.w += value.w;
				} else {
					const auto* peer_source =
						reinterpret_cast<const uint4*>(
							context.peer_sources[peer] +
							context.source_offset);
					uint4 value_bits = peer_source[vector];
					reduced_bits[0] = reduced_bits[0] > value_bits.x
						? reduced_bits[0]
						: value_bits.x;
					reduced_bits[1] = reduced_bits[1] > value_bits.y
						? reduced_bits[1]
						: value_bits.y;
					reduced_bits[2] = reduced_bits[2] > value_bits.z
						? reduced_bits[2]
						: value_bits.z;
					reduced_bits[3] = reduced_bits[3] > value_bits.w
						? reduced_bits[3]
						: value_bits.w;
				}
			}
			if constexpr (Op == ReduceOp::kSum) {
				reinterpret_cast<float4*>(destination)[vector] = reduced;
			} else {
				auto* destination_bits =
					reinterpret_cast<std::uint32_t*>(destination) +
					vector * 4;
				destination_bits[0] = reduced_bits[0];
				destination_bits[1] = reduced_bits[1];
				destination_bits[2] = reduced_bits[2];
				destination_bits[3] = reduced_bits[3];
			}
		}
		for (std::size_t index = vectors * 4 + lane;
				index < count;
				index += 32) {
			if constexpr (Op == ReduceOp::kSum) {
				float reduced = 0.0f;
				for (int peer = 0; peer < context.size; ++peer) {
					float value = context.peer_sources[peer][
						context.source_offset + index];
					reduced += value;
				}
				destination[index] = reduced;
			} else {
				std::uint32_t reduced = 0;
				for (int peer = 0; peer < context.size; ++peer) {
					const auto* peer_source =
						reinterpret_cast<const std::uint32_t*>(
							context.peer_sources[peer] +
							context.source_offset);
					reduced = reduced > peer_source[index]
						? reduced
						: peer_source[index];
				}
				reinterpret_cast<std::uint32_t*>(destination)[index] =
					reduced;
			}
		}
		__syncwarp();

		direct_peer_barrier_warp(
			context.local_complete,
			context.peer_sync,
			context.complete_offset,
			context.rank,
			context.size,
			epoch);
		if constexpr (Op == ReduceOp::kMax) {
			decode_ordered_float_warp(destination, count);
		}
	}
}

}  // namespace detail
}  // namespace liger_cute
