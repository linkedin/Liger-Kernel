#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cutlass/arch/barrier.h>

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// CtaCounterBarrier — target-based cross-CTA monotonic barrier
// ═══════════════════════════════════════════════════════════════════
//
// Synchronizes a group of CTAs (or column of CTAs) by incrementing a
// shared device-memory counter and waiting until it has accumulated
// one increment from every participant of the current round.
//
// ── Why not modulo? ────────────────────────────────────────────────
// The naive `while (counter % stride != 0)` variant deadlocks under
// CTA-execution skew:
//   1. counter advances 0 → stride as all CTAs add 1 each.
//   2. A fast CTA polls at counter==stride, sees %==0, exits.
//   3. It runs post-barrier work AND returns for round 2's add →
//      counter==stride+1.
//   4. A slow CTA that hadn't yet polled at counter==stride now reads
//      counter==stride+1. `1 % stride != 0`, keeps spinning.
//   5. As other fast CTAs each add for round 2 the counter walks up
//      to 2*stride − 1 (everyone except the slow one). `(2*stride−1)
//      % stride == stride−1`, never 0. Deadlock.
//
// ── Why not derive the target from atomicAdd's return? ─────────────
// Snapshotting `prev = atomicAdd(counter, 1)` and computing
// `target = ((prev/stride) + 1)*stride` looks similar but is racy:
// two CTAs entering the SAME conceptual round can land on different
// sides of a stride boundary (one gets prev=stride−1, the other
// prev=stride), and they then wait for DIFFERENT targets — one for
// round K, the other for round K+1. The "round" assignment depends
// on the order in which CTAs win the atomicAdd, which is exactly
// the kind of skew the barrier is supposed to mask.
//
// ── The fix ────────────────────────────────────────────────────────
// Each barrier instance owns a per-CTA `target` in register memory.
// The constructor initializes target=0; every wait() bumps it by
// `stride` BEFORE the atomicAdd. Each CTA always knows exactly which
// round it is in (its own call count × stride), independent of the
// race between CTAs.
//
// ── Usage ──────────────────────────────────────────────────────────
//   liger::CtaCounterBarrier<kNumThreads, kBarrierId> barrier(
//       counter_ptr, stride);
//   barrier.wait();   // round 1
//   ... work ...
//   barrier.wait();   // round 2
//
// Each CTA must construct the barrier ONCE and reuse it for the
// matching sequence of waits — constructing a fresh barrier per
// wait() resets target=0 and is equivalent to the broken behavior
// above.
//
// `kNumThreads` and `kBarrierId` are template parameters for the
// intra-CTA NamedBarrier (so MLP-only barriers can exclude comm warps
// via a smaller NumThreads). The counter holds device-memory data
// and must be zeroed at kernel-launch time.
//
template <int NumThreads, int BarrierId>
struct CtaCounterBarrier {
	int* counter;  // device memory
	int  stride;   // number of CTAs that contribute per wait()
	int  target;   // per-CTA monotonic target — register memory

	// ── Pre-condition: the host MUST cudaMemsetAsync `counter` to 0
	// before every kernel launch. The constructor sets target=0 and
	// each wait() then advances target by `stride` while atomically
	// incrementing counter. If the caller skips the zero-out and
	// counter still holds a residual value from the previous launch,
	// the very first wait() will see counter >> target and exit
	// without actually waiting for anyone — silent corruption.
	//
	// DO NOT read counter here to "derive" the starting target. That
	// read races with any thread/CTA that has begun wait() — there
	// is no implicit sync between CTAs at kernel prologue. The only
	// race-free contract is: host zeros the counter, kernel
	// constructs barrier with target=0. Keep it that way.
	__device__ __forceinline__ CtaCounterBarrier(int* counter_, int stride_)
		: counter(counter_), stride(stride_), target(0) {}

	__device__ __forceinline__ void wait() {
		// Bump the per-CTA target by `stride` first, so we know
		// which round this wait() corresponds to independent of
		// the race between CTAs at the atomicAdd below.
		target += stride;
		__threadfence();
		cutlass::arch::NamedBarrier::sync(NumThreads, BarrierId);
		if (threadIdx.x == 0) {
			atomicAdd(counter, 1);
			cuda::atomic_ref<int, cuda::thread_scope_device> r(*counter);
			while (r.load(cuda::memory_order_relaxed) < target) {}
		}
		cutlass::arch::NamedBarrier::sync(NumThreads, BarrierId);
	}
};

// ═══════════════════════════════════════════════════════════════════
// SyncThreadsCtaCounterBarrier — same idea, intra-CTA via __syncthreads
// ═══════════════════════════════════════════════════════════════════
//
// Variant for callers that need to sync ALL threads in the CTA (not
// just an MLP-warp subset). Used by PeSync::barrier where every thread
// — including comm warps — must be at the barrier before the atomic
// increments and the cross-PE signal_op fire.
struct SyncThreadsCtaCounterBarrier {
	int* counter;
	int  stride;
	int  target;

	// Same pre-condition as CtaCounterBarrier: the host MUST zero
	// `counter` before every kernel launch. Constructor sets
	// target=0 — DO NOT read counter here to derive the starting
	// value, that races with any thread that already entered wait().
	__device__ __forceinline__ SyncThreadsCtaCounterBarrier(int* counter_, int stride_)
		: counter(counter_), stride(stride_), target(0) {}

	__device__ __forceinline__ void wait() {
		target += stride;
		__threadfence();
		__syncthreads();
		if (threadIdx.x == 0) {
			atomicAdd(counter, 1);
			cuda::atomic_ref<int, cuda::thread_scope_device> r(*counter);
			while (r.load(cuda::memory_order_relaxed) < target) {}
		}
		__syncthreads();
	}
};

} // namespace liger
