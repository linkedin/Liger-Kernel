#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy — TMA store completion support.
//
// Transport selection and collective mappings live in liger_cute/detail.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

// ───────────────────────────────────────────────────────────────────────────
// TMA store completion helpers used by the dZ and dW epilogues.
// ───────────────────────────────────────────────────────────────────────────

// cp.async.bulk.wait_group N, always the full form and never `.read`.
//
// `.read` only waits for the SMEM source of a bulk store to be re-usable, and
// it *retires the group from the pending set*. A staging pipeline that drains
// with `.read` and then asks for `wait_group 0` therefore gets no guarantee at
// all about the already-retired groups' global writes. Every tile here is
// published to another warp the moment it is stored, so the epilogue has to
// wait for completion, not merely for the source to be free.
template <int Pending>
__device__ __forceinline__ void wait_tma_store() {
	asm volatile("cp.async.bulk.wait_group %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ void wait_tma_store_complete() {
	wait_tma_store<0>();
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
