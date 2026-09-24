#pragma once

#include <cuda.h>
#include <nvshmem.h>

#include <cstdlib>
#include <cstring>

#include "liger_cute/check.h"

namespace liger {

inline bool sm90_nonrdc_moe_supports_benchmark_shape(
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
		int top_k,
		int num_pes) {
#if defined(LIGER_CUTE_SM90_NONRDC_BENCHMARK_ALL_CONFIGS)
	return num_tokens > 0 && hidden_dim > 0 && intermediate_dim > 0 &&
		num_experts > 0 && top_k > 0 &&
		(num_pes == 1 || num_pes == 2 || num_pes == 4 ||
			num_pes == 8 || num_pes == 16);
#else
	if (num_tokens != 8192 || num_pes != 8)
		return false;
	return
		hidden_dim == 4096 && intermediate_dim == 14336 &&
		num_experts == 8 && top_k == 2;
#endif
}

#if defined(LIGER_CUTE_HAS_SM90_NONRDC_MOE)
bool sm90_nonrdc_moe_requested();
bool sm90_nonrdc_moe_team_uses_ib(nvshmem_team_t team);
void configure_sm90_nonrdc_moe(int num_hosts, int gpus_per_host);
CUfunction resolve_sm90_nonrdc_moe(
	const char* kernel_name,
	const void* fallback_kernel,
	bool use_ib_transport,
	bool allow_lazy_resolution);
void finalize_sm90_nonrdc_moe();
#else
inline bool sm90_nonrdc_moe_requested() {
	return false;
}

inline bool sm90_nonrdc_moe_team_uses_ib(nvshmem_team_t) {
	return true;
}

inline void configure_sm90_nonrdc_moe(int, int) {
	const char* value = std::getenv("LIGER_MOE_SM90_NONRDC");
	LIGER_CHECK(
		value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0,
		"LIGER_MOE_SM90_NONRDC=1 requested the external SM90 MoE module, "
		"but this native core was built without "
		"LIGER_CUTE_ENABLE_SM90_NONRDC_MOE");
}

inline CUfunction resolve_sm90_nonrdc_moe(
		const char*, const void*, bool, bool) {
	return nullptr;
}

inline void finalize_sm90_nonrdc_moe() {}
#endif

} // namespace liger
