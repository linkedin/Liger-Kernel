#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace liger {

// Fused memset: a single kernel launch that fills N small regions, one CTA
// per region. Replaces sequences of cudaMemsetAsync calls that would otherwise
// serialize on the stream and pay ~5 us launch overhead each.
//
// Use when:
//   - all regions are small enough that one CTA can saturate the write in
//     a few iterations (KB-scale, not GB-scale).
//   - patterns are byte-broadcast (cudaMemset semantics).
//
// One CTA per region: gridDim.x = num_regions. Within a CTA, threads issue
// uint4 (16-byte) stores in a thread-strided loop; bytes < 16 of tail are
// handled by per-byte writes from the same threads.

constexpr int kFusedMemsetMaxRegions = 16;
constexpr int kFusedMemsetThreads    = 256;

struct FusedMemsetArgs {
	void*    ptr[kFusedMemsetMaxRegions];
	uint64_t bytes[kFusedMemsetMaxRegions];
	uint32_t pattern[kFusedMemsetMaxRegions];  // byte broadcast to 32 bits
	int      num_regions;
};

__global__ void fused_memset_kernel(
		__grid_constant__ const FusedMemsetArgs args) {
	const int r = blockIdx.x;
	if (r >= args.num_regions) return;

	uint8_t* dst         = reinterpret_cast<uint8_t*>(args.ptr[r]);
	const uint64_t bytes = args.bytes[r];
	const uint32_t pat   = args.pattern[r];

	uint4 v;
	v.x = v.y = v.z = v.w = pat;

	// Main loop: uint4 (16 B) stores, thread-strided.
	const uint64_t main_bytes = bytes & ~uint64_t{15};
	const uint64_t stride_b   = (uint64_t)blockDim.x * 16;
	for (uint64_t off = (uint64_t)threadIdx.x * 16;
			off < main_bytes; off += stride_b) {
		*reinterpret_cast<uint4*>(dst + off) = v;
	}

	// Tail (< 16 B). Single byte per thread; loop handles any size.
	for (uint64_t i = main_bytes + threadIdx.x; i < bytes; i += blockDim.x) {
		dst[i] = (uint8_t)(pat & 0xff);
	}
}

inline void fused_memset_add(
		FusedMemsetArgs& args,
		void* ptr,
		size_t bytes,
		uint8_t byte_pattern) {
	if (bytes == 0) return;
	int i = args.num_regions++;
	args.ptr[i]   = ptr;
	args.bytes[i] = bytes;
	uint32_t p = byte_pattern;
	args.pattern[i] = (p << 24) | (p << 16) | (p << 8) | p;
}

inline void fused_memset_launch(
		const FusedMemsetArgs& args,
		cudaStream_t stream) {
	if (args.num_regions == 0) return;
	fused_memset_kernel<<<args.num_regions, kFusedMemsetThreads, 0, stream>>>(args);
}

} // namespace liger
