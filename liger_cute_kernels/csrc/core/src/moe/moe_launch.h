// moe_launch.h — internal POD argument bundles for the templated MoE launchers.
//
// Shared by moe.cu (forward), moe_bwd.cu (backward), and the offline tuner
// (src/moe/tune/tune_moe_fwd_bwd.cu). The templated launchers
//   liger::moe_fused_fwd_bf16<Config>(const MoeFwdArgs&, int static_nsplit)
//   liger::moe_bwd_fwd_bf16_tuned<...>(const MoeBwdArgs&)
// take these structs; the extern "C" ABI wrappers and the tuner both build them.
//
// Deliberately self-contained (no <nvshmem.h>): the team handle is carried as a
// plain int (NVSHMEM's nvshmem_team_t is `int`), so a consumer that must avoid
// the NVSHMEM device headers — like the tuner, whose RDC + kernel function
// pointers trip nvlink's double-link bug — can include this without pulling them.
// moe.cu / moe_bwd.cu pass `args.team` straight to nvshmem_team_* (int → team).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace liger {

// Forward launcher inputs. bf16 payloads are carried as void* and reinterpret_cast
// to Element inside the launcher.
struct MoeFwdArgs {
	const void* X;                 // [T, D] bf16, symmetric
	const int*  expert_indices;    // [T, K] int32
	const void* expert_weights;    // [T, K] bf16
	const void* all_B;             // [epp, I, D] bf16
	const void* all_C;             // [epp, I, D] bf16
	const void* all_A;             // [epp, D, I] bf16
	int64_t weight_expert_stride;  // all_B/all_C expert stride in bf16 elements
	int num_tokens, hidden_dim, intermediate_dim, experts_per_pe;
	int num_experts, top_k;
	int team;                      // NVSHMEM team id (nvshmem_team_t == int)
	cudaStream_t stream;
	int device;                    // CUDA device of the inputs (SM-count query)
	// Caller-owned outputs (pre-allocated; the kernel writes them).
	void* Y;                       // [T, D] bf16
	int*  token_expert_slots;      // [max_total_slots] int32
	int*  tile_expert_ids;         // [max_total_slots / kTileM] int32
	// Symmetric buffers allocated inside the launcher; the addresses are written
	// back so the caller can alias / thread them into the backward.
	void** x_sorted_out;
	void** y_buf_out;
	void** all_expert_offsets_out;
};

// Backward launcher inputs. dB/dC/dA/dW are gradient ACCUMULATION targets — the
// launcher zeroes them before the kernel.
struct MoeBwdArgs {
	const void* dY;                 // [T, D] bf16
	const void* Y_fwd;              // [total_slots, D] bf16 (fwd output buffer)
	void*       x_sorted;           // [total_slots, D] bf16 (cast to Element*, non-const)
	int*        token_expert_slots; // [T*top_k] int32
	int*        tile_expert_ids;    // [max_m_tiles] int32
	int*        expert_offsets;     // [num_pes, num_experts+1] int32 (fwd's all_*)
	int*        expert_indices;     // [T, K] int32
	const void* expert_weights;     // [T, K] bf16
	const void* all_B;              // [epp, I, D] bf16
	const void* all_C;              // [epp, I, D] bf16
	const void* all_A;              // [epp, D, I] bf16
	int num_tokens, hidden_dim, intermediate_dim, experts_per_pe;
	int num_experts, top_k;
	int team;                       // NVSHMEM team id (nvshmem_team_t == int)
	cudaStream_t stream;
	int device;
	// Caller-owned gradient outputs (binding-allocated; the launcher writes them).
	void* dX;   // [T, D] bf16
	void* dB;   // [epp, I, D] bf16 (zeroed + accumulated)
	void* dC;   // [epp, I, D] bf16 (zeroed + accumulated)
	void* dA;   // [epp, D, I] bf16 (zeroed + accumulated)
	void* dW;   // [T, K] bf16     (zeroed + accumulated)
};

}  // namespace liger
