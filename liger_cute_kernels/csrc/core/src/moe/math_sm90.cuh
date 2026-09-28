#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// Sigmoid and SiLU (Swish) — exact, matches torch.sigmoid / F.silu
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ float fast_sigmoid(float x) {
	// if (x >= 4.0f) return 1.0f;
	// if (x <= -4.0f) return 0.0f;
	// const float x2 = x * x;
	// return 0.5f + x * (0.23897898f + x2 * (-0.01301264f + x2 * 0.00035572f));
	return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float fast_silu(float x) {
	return x * fast_sigmoid(x);
}

__device__ __forceinline__ float grad_fast_silu(float x) {
	const float f = fast_sigmoid(x);
	return f * (1.0f + x * (1.0f - f));
}

// ═══════════════════════════════════════════════════════════════════
// In-place smem tile transpose — single warp group (128 threads)
// ═══════════════════════════════════════════════════════════════════
//
// Transposes a square [N, N] smem tile in place by swapping upper-
// triangle elements: for each (r, c) with r < c, swap tensor(r, c)
// with tensor(c, r). Diagonal elements are untouched.
//
// Race-free: each (r, c) pair is assigned to exactly one thread.
// No pair can collide — if thread A swaps (r, c) ↔ (c, r) with
// r < c, no other thread touches either element because that would
// require c < r (contradiction).
//
// CuTe's operator() handles swizzled addressing automatically.
//
// Usage:
//   auto sW = make_tensor(make_smem_ptr(ptr), SmemLayoutW_stage{});
//   smem_transpose_inplace_wg<128>(sW);
//
// Requires: square tile (dim 0 == dim 1), called by one warp group.

template <int N, typename Tensor>
__device__ __forceinline__ void smem_transpose_inplace_wg(Tensor& tensor) {
	constexpr int kWarpGroupSize = 128;
	int tid = threadIdx.x % kWarpGroupSize;

	for (int r = 0; r < N - 1; ++r) {
		for (int c = r + 1 + tid; c < N; c += kWarpGroupSize) {
			auto tmp = tensor(r, c);
			tensor(r, c) = tensor(c, r);
			tensor(c, r) = tmp;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// float → Element conversion (template specialization per type)
// ═══════════════════════════════════════════════════════════════════

template <typename Element>
__device__ __forceinline__ Element from_float(float x);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
	return __float2bfloat16(x);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
	return __float2half(x);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t from_float<cutlass::bfloat16_t>(float x) {
	__nv_bfloat16 tmp = __float2bfloat16(x);
	return reinterpret_cast<const cutlass::bfloat16_t&>(tmp);
}

template <>
__device__ __forceinline__ cutlass::half_t from_float<cutlass::half_t>(float x) {
	__half tmp = __float2half(x);
	return reinterpret_cast<const cutlass::half_t&>(tmp);
}

// ═══════════════════════════════════════════════════════════════════
// Element → float conversion (template specialization per type)
// ═══════════════════════════════════════════════════════════════════

template <typename Element>
__device__ __forceinline__ float to_float(Element v);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
	return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
	return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<cutlass::bfloat16_t>(cutlass::bfloat16_t v) {
	return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(v));
}

template <>
__device__ __forceinline__ float to_float<cutlass::half_t>(cutlass::half_t v) {
	return __half2float(reinterpret_cast<const __half&>(v));
}

// ═══════════════════════════════════════════════════════════════════
// Cached global load (__ldg) for 2-byte element types
// ═══════════════════════════════════════════════════════════════════
//
// CUDA provides __ldg for __nv_bfloat16 / __half directly; the cutlass
// wrappers (cutlass::bfloat16_t / cutlass::half_t) require a reinterpret
// since the byte layout is identical.

template <typename T>
__device__ __forceinline__ T ldg_elem(const T* p);

template <>
__device__ __forceinline__ __nv_bfloat16 ldg_elem<__nv_bfloat16>(const __nv_bfloat16* p) {
	return __ldg(p);
}

template <>
__device__ __forceinline__ __half ldg_elem<__half>(const __half* p) {
	return __ldg(p);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t ldg_elem<cutlass::bfloat16_t>(const cutlass::bfloat16_t* p) {
	__nv_bfloat16 raw = __ldg(reinterpret_cast<const __nv_bfloat16*>(p));
	return reinterpret_cast<const cutlass::bfloat16_t&>(raw);
}

template <>
__device__ __forceinline__ cutlass::half_t ldg_elem<cutlass::half_t>(const cutlass::half_t* p) {
	__half raw = __ldg(reinterpret_cast<const __half*>(p));
	return reinterpret_cast<const cutlass::half_t&>(raw);
}

} // namespace liger
