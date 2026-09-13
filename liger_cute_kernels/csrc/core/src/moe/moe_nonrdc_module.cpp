#include "moe_nonrdc_module.h"

#include <cuda_runtime.h>
#include <host/nvshmemx_api.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

#include "liger_cute/check.h"
#include "liger_cute/detail/comm_schedule.cuh"
#include "liger_cute/detail/status.h"

#ifndef LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT
#define LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT 0ULL
#endif

namespace liger {
namespace {

constexpr const char* kEnableEnv = "LIGER_MOE_SM90_NONRDC";
constexpr const char* kCubinPathEnv = "LIGER_MOE_SM90_NONRDC_CUBIN";
constexpr const char* kDestTableSymbol =
	"_ZN10liger_cute6detail12g_dest_tableE";
constexpr const char* kRankTableSymbol =
	"_ZN10liger_cute6detail12g_rank_tableE";
constexpr const char* kFingerprintSymbol =
	"liger_cute_sm90_nonrdc_build_fingerprint";
constexpr const char* kTransportModeSymbol =
	"liger_cute_sm90_nonrdc_transport_mode";

char g_module_path_anchor = 0;

struct ModuleState {
	CUcontext context = nullptr;
	int device = -1;
	int num_hosts = -1;
	int gpus_per_host = -1;
	CUmodule module = nullptr;
	bool registered_with_nvshmem = false;
	std::string path;
	std::map<std::string, CUfunction> functions;
};

struct PendingModule {
	CUmodule module = nullptr;
	bool registered_with_nvshmem = false;

	PendingModule() = default;
	PendingModule(const PendingModule&) = delete;
	PendingModule& operator=(const PendingModule&) = delete;

	~PendingModule() {
		if (module != nullptr) {
			if (registered_with_nvshmem)
				nvshmemx_cumodule_finalize(module);
			cuModuleUnload(module);
		}
	}
};

std::mutex g_module_mutex;
ModuleState g_module;

std::string driver_error(CUresult result) {
	const char* name = nullptr;
	const char* text = nullptr;
	cuGetErrorName(result, &name);
	cuGetErrorString(result, &text);
	std::string message = name != nullptr ? name : "CUDA_ERROR_UNKNOWN";
	if (text != nullptr) {
		message += ": ";
		message += text;
	}
	return message;
}

[[noreturn]] void fail_driver(CUresult result, const char* what) {
	LIGER_FAIL_CUDA(what, " failed: ", driver_error(result));
}

bool parse_enable_env() {
	const char* value = std::getenv(kEnableEnv);
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0)
		return false;
	LIGER_CHECK(
		std::strcmp(value, "1") == 0,
		kEnableEnv, " must be 0 or 1, got '", value, "'");
	return true;
}

std::string default_cubin_path() {
#if defined(LIGER_CUTE_HAS_SM90_NONRDC_MOE)
	Dl_info info = {};
	LIGER_CHECK(
		dladdr(&g_module_path_anchor, &info) != 0 &&
			info.dli_fname != nullptr,
		"failed to locate libliger_cute_kernels.so for the non-RDC MoE cubin");
	std::string library_path(info.dli_fname);
	std::size_t slash = library_path.find_last_of('/');
	std::string directory =
		slash == std::string::npos ? "." : library_path.substr(0, slash);
	return directory + "/" + LIGER_CUTE_SM90_NONRDC_MOE_CUBIN_NAME;
#else
	return {};
#endif
}

std::string configured_cubin_path() {
	if (const char* override_path = std::getenv(kCubinPathEnv);
			override_path != nullptr && override_path[0] != '\0')
		return override_path;
	return default_cubin_path();
}

CUcontext current_context() {
	CUcontext context = nullptr;
	CUresult result = cuCtxGetCurrent(&context);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuCtxGetCurrent");
	LIGER_CHECK(
		context != nullptr,
		"the non-RDC MoE module requires an active CUDA context");
	return context;
}

int current_device() {
	int device = -1;
	cudaError_t error = cudaGetDevice(&device);
	if (error != cudaSuccess)
		LIGER_FAIL_CUDA("cudaGetDevice failed: ", cudaGetErrorString(error));
	return device;
}

void validate_module_fingerprint(CUmodule module) {
	CUdeviceptr fingerprint_ptr = 0;
	std::size_t fingerprint_bytes = 0;
	CUresult result = cuModuleGetGlobal(
		&fingerprint_ptr,
		&fingerprint_bytes,
		module,
		kFingerprintSymbol);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuModuleGetGlobal(non-RDC build fingerprint)");
	LIGER_CHECK(
		fingerprint_bytes == sizeof(std::uint64_t),
		"non-RDC MoE fingerprint has unexpected size ",
		fingerprint_bytes);
	std::uint64_t fingerprint = 0;
	result = cuMemcpyDtoH(
		&fingerprint, fingerprint_ptr, sizeof(fingerprint));
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuMemcpyDtoH(non-RDC build fingerprint)");
	LIGER_CHECK(
		fingerprint ==
			static_cast<std::uint64_t>(
				LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT),
		"non-RDC MoE cubin does not match this native core build");
}

void validate_module_transport(CUmodule module) {
	CUdeviceptr mode_ptr = 0;
	std::size_t mode_bytes = 0;
	CUresult result = cuModuleGetGlobal(
		&mode_ptr, &mode_bytes, module, kTransportModeSymbol);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuModuleGetGlobal(non-RDC transport mode)");
	LIGER_CHECK(
		mode_bytes == sizeof(int),
		"non-RDC MoE transport mode has unexpected size ", mode_bytes);
	int mode = 0;
	result = cuMemcpyDtoH(&mode, mode_ptr, sizeof(mode));
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuMemcpyDtoH(non-RDC transport mode)");
	LIGER_CHECK(
		mode == 3,
		"non-RDC MoE cubin transport mismatch: expected dual local/IB "
		"template support, module reports mode ", mode);
}

void validate_parameter_abi(
		const void* fallback_kernel,
		CUfunction external_kernel,
		const char* kernel_name) {
	constexpr std::size_t kMaxParameters = 64;
	for (std::size_t index = 0; index < kMaxParameters; ++index) {
		std::size_t runtime_offset = 0;
		std::size_t runtime_size = 0;
		std::size_t driver_offset = 0;
		std::size_t driver_size = 0;
		cudaError_t runtime_result = cudaFuncGetParamInfo(
			fallback_kernel, index, &runtime_offset, &runtime_size);
		CUresult driver_result = cuFuncGetParamInfo(
			external_kernel, index, &driver_offset, &driver_size);
		const bool runtime_end = runtime_result == cudaErrorInvalidValue;
		const bool driver_end = driver_result == CUDA_ERROR_INVALID_VALUE;
		if (runtime_end || driver_end) {
			cudaGetLastError();
			LIGER_CHECK(
				runtime_end && driver_end,
				"kernel parameter count mismatch for ", kernel_name,
				" at parameter ", index);
			return;
		}
		if (runtime_result != cudaSuccess) {
			LIGER_FAIL_CUDA(
				"cudaFuncGetParamInfo failed for ", kernel_name,
				" parameter ", index, ": ",
				cudaGetErrorString(runtime_result));
		}
		if (driver_result != CUDA_SUCCESS) {
			fail_driver(
				driver_result,
				"cuFuncGetParamInfo(non-RDC MoE backward)");
		}
		LIGER_CHECK(
			runtime_offset == driver_offset && runtime_size == driver_size,
			"kernel parameter ABI mismatch for ", kernel_name,
			" parameter ", index, ": fallback(offset=", runtime_offset,
			", size=", runtime_size, "), external(offset=", driver_offset,
			", size=", driver_size, ")");
	}
	LIGER_CHECK(
		false,
		"non-RDC MoE kernel exceeds the supported parameter count for ",
		kernel_name);
}

void copy_schedule(
		CUmodule module,
		const std::array<int, liger_cute::detail::kMaxPEs>& destinations,
		const std::array<int, liger_cute::detail::kMaxPEs>& ranks,
		int num_pes) {
	CUdeviceptr destination_ptr = 0;
	CUdeviceptr rank_ptr = 0;
	std::size_t destination_bytes = 0;
	std::size_t rank_bytes = 0;

	CUresult result = cuModuleGetGlobal(
		&destination_ptr, &destination_bytes, module, kDestTableSymbol);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuModuleGetGlobal(g_dest_table)");
	result = cuModuleGetGlobal(
		&rank_ptr, &rank_bytes, module, kRankTableSymbol);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuModuleGetGlobal(g_rank_table)");

	const std::size_t bytes = static_cast<std::size_t>(num_pes) * sizeof(int);
	LIGER_CHECK(
		destination_bytes >= bytes && rank_bytes >= bytes,
		"non-RDC MoE communication schedule capacity is smaller than ",
		num_pes, " PEs");

	result = cuMemcpyHtoD(destination_ptr, destinations.data(), bytes);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuMemcpyHtoD(g_dest_table)");
	result = cuMemcpyHtoD(rank_ptr, ranks.data(), bytes);
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuMemcpyHtoD(g_rank_table)");
}

void configure_module(
		CUcontext context,
		int device,
		const std::array<int, liger_cute::detail::kMaxPEs>& destinations,
		const std::array<int, liger_cute::detail::kMaxPEs>& ranks,
		int num_pes,
		int num_hosts,
		int gpus_per_host) {
	const std::string path = configured_cubin_path();
	LIGER_CHECK(!path.empty(), "non-RDC MoE cubin path is empty");
	std::ifstream cubin(path, std::ios::binary);
	LIGER_CHECK(
		cubin.good(),
		"non-RDC MoE cubin is missing or unreadable: ", path);

	ModuleState& module_state = g_module;
	if (module_state.module != nullptr) {
		LIGER_CHECK(
			module_state.context == context && module_state.device == device,
			"the non-RDC MoE path supports one CUDA context/device per "
			"process");
		LIGER_CHECK(
			module_state.path == path,
			"non-RDC MoE cubin path changed after module initialization: ",
			module_state.path, " -> ", path);
		LIGER_CHECK(
			module_state.num_hosts == num_hosts &&
				module_state.gpus_per_host == gpus_per_host,
			"non-RDC MoE topology changed after module initialization: "
			"configured (num_hosts=", module_state.num_hosts,
			", gpus_per_host=", module_state.gpus_per_host,
			"), requested (num_hosts=", num_hosts,
			", gpus_per_host=", gpus_per_host, ")");
		if (!module_state.registered_with_nvshmem) {
			const int init_status =
				nvshmemx_cumodule_init(module_state.module);
			if (init_status != 0) {
				LIGER_FAIL_NVSHMEM(
					"nvshmemx_cumodule_init(non-RDC MoE cubin) failed "
					"with status ", init_status);
			}
			module_state.registered_with_nvshmem = true;
		}
		copy_schedule(module_state.module, destinations, ranks, num_pes);
		return;
	}

	PendingModule pending;
	CUresult result = cuModuleLoad(&pending.module, path.c_str());
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuModuleLoad(non-RDC MoE cubin)");
	const CUmodule module = pending.module;
	validate_module_fingerprint(module);
	validate_module_transport(module);

	const int init_status = nvshmemx_cumodule_init(module);
	if (init_status != 0) {
		LIGER_FAIL_NVSHMEM(
			"nvshmemx_cumodule_init(non-RDC MoE cubin) failed with status ",
			init_status);
	}
	pending.registered_with_nvshmem = true;
	copy_schedule(module, destinations, ranks, num_pes);

	ModuleState state;
	state.context = context;
	state.device = device;
	state.num_hosts = num_hosts;
	state.gpus_per_host = gpus_per_host;
	state.module = module;
	state.registered_with_nvshmem = true;
	state.path = path;
	module_state = std::move(state);
	pending.module = nullptr;
}

} // namespace

bool sm90_nonrdc_moe_requested() {
	return parse_enable_env();
}

bool sm90_nonrdc_moe_team_uses_ib(nvshmem_team_t team) {
	const int team_size = nvshmem_team_n_pes(team);
	LIGER_CHECK(
		team_size > 0,
		"cannot determine SM90 MoE transport for an invalid or empty "
		"NVSHMEM team");
	LIGER_CHECK(
		nvshmem_team_my_pe(team) >= 0,
		"the current PE is not a member of the configured SM90 MoE team");
	for (int pe = 0; pe < team_size; ++pe) {
		if (nvshmem_team_translate_pe(
				team, pe, NVSHMEMX_TEAM_NODE) < 0)
			return true;
	}
	return false;
}

void configure_sm90_nonrdc_moe(int num_hosts, int gpus_per_host) {
	if (!parse_enable_env())
		return;

#if !defined(LIGER_CUTE_HAS_SM90_NONRDC_MOE)
	LIGER_CHECK(
		false,
		kEnableEnv,
		"=1 requested the SM90 non-RDC MoE path, but this native "
		"core was built without LIGER_CUTE_ENABLE_SM90_NONRDC_MOE");
#else
	const int num_pes = num_hosts * gpus_per_host;
	LIGER_CHECK(
		num_hosts > 0 && gpus_per_host > 0 &&
			num_pes <= liger_cute::detail::kMaxPEs,
		"invalid non-RDC MoE topology: num_hosts=", num_hosts,
		", gpus_per_host=", gpus_per_host);

	std::array<int, liger_cute::detail::kMaxPEs> destinations = {};
	std::array<int, liger_cute::detail::kMaxPEs> ranks = {};
	liger_cute::detail::build_comm_schedule(
		num_hosts, gpus_per_host, destinations.data(), ranks.data());

	const CUcontext context = current_context();
	const int device = current_device();
	std::lock_guard<std::mutex> lock(g_module_mutex);
	if (g_module.module != nullptr) {
		LIGER_CHECK(
			g_module.context == context && g_module.device == device,
			"the non-RDC MoE path supports one CUDA context/device per "
			"process");
	}
	configure_module(
		context, device, destinations, ranks, num_pes,
		num_hosts, gpus_per_host);
#endif
}

CUfunction resolve_sm90_nonrdc_moe(
		const char* kernel_name,
		const void* fallback_kernel,
		bool use_ib_transport,
		bool allow_lazy_resolution) {
	LIGER_CHECK(
		parse_enable_env(),
		"resolve_sm90_nonrdc_moe called while ", kEnableEnv,
		" is disabled");
	LIGER_CHECK(
		kernel_name != nullptr && kernel_name[0] != '\0',
		"non-RDC MoE kernel name is empty");
	LIGER_CHECK(fallback_kernel != nullptr, "fallback MoE kernel is null");

	const CUcontext context = current_context();
	const int device = current_device();
	std::lock_guard<std::mutex> lock(g_module_mutex);
	ModuleState& module_state = g_module;
	LIGER_CHECK(
		module_state.module != nullptr &&
			module_state.registered_with_nvshmem,
		"non-RDC MoE ",
		use_ib_transport ? "IB-capable" : "node-local",
		" template module is not configured for the current CUDA context; "
		"call moe_configure_symmetric after enabling ", kEnableEnv);
	LIGER_CHECK(
		module_state.context == context && module_state.device == device,
		"the current CUDA context/device differs from the configured non-RDC "
		"MoE context/device");

	auto existing = module_state.functions.find(kernel_name);
	if (existing != module_state.functions.end())
		return existing->second;

	CUfunction function = nullptr;
	CUresult result = cuModuleGetFunction(
		&function, module_state.module, kernel_name);
	if (result == CUDA_ERROR_NOT_FOUND)
		return nullptr;
	if (result != CUDA_SUCCESS)
		fail_driver(result, "cuModuleGetFunction(non-RDC MoE)");
	LIGER_CHECK(
		allow_lazy_resolution,
		"non-RDC MoE specialization ", kernel_name,
		" was first requested during CUDA graph capture; warm up this shape "
		"before capture");
	validate_parameter_abi(fallback_kernel, function, kernel_name);
	module_state.functions.emplace(kernel_name, function);
	return function;
}

void finalize_sm90_nonrdc_moe() {
	std::lock_guard<std::mutex> lock(g_module_mutex);
	if (g_module.module == nullptr || !g_module.registered_with_nvshmem)
		return;

	const CUcontext context = current_context();
	const int device = current_device();
	cudaError_t sync_error = cudaDeviceSynchronize();
	if (sync_error != cudaSuccess)
		LIGER_FAIL_CUDA(
			"cudaDeviceSynchronize before non-RDC MoE finalization failed: ",
			cudaGetErrorString(sync_error));
	LIGER_CHECK(
		g_module.context == context && g_module.device == device,
		"NVSHMEM must be finalized on the CUDA context/device that "
		"configured the non-RDC MoE module");
	const int finalize_status =
		nvshmemx_cumodule_finalize(g_module.module);
	if (finalize_status != 0) {
		LIGER_FAIL_NVSHMEM(
			"nvshmemx_cumodule_finalize(non-RDC MoE cubin) failed with "
			"status ", finalize_status);
	}
	g_module.registered_with_nvshmem = false;
	g_module.functions.clear();
}

} // namespace liger
