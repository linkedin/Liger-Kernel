#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "moe_dispatch_configs_sm90.cuh"
#include "moe_dispatch_configs_sm100.cuh"

namespace {

namespace ffi = tvm::ffi;

#ifndef LIGER_CUTE_TEST_MODULE_PATH
#define LIGER_CUTE_TEST_MODULE_PATH "libliger_cute_kernels.so"
#endif

constexpr int kTokens = 256;
constexpr int kHiddenDim = 512;
constexpr int kIntermediateDim = 2048;
constexpr int kTopK = 1;
constexpr int kCommTileM = 128;

DLDataType bf16_dtype() { return DLDataType{kDLBfloat, 16, 1}; }
DLDataType i32_dtype() { return DLDataType{kDLInt, 32, 1}; }
DLDataType i64_dtype() { return DLDataType{kDLInt, 64, 1}; }
DLDevice cuda_device(int device_id) { return DLDevice{kDLCUDA, device_id}; }
DLDevice cpu_device() { return DLDevice{kDLCPU, 0}; }

struct FwdConfig {
	int compute;
	int tn1, tk1, s1, ec1, tn2, tk2, s2, ec2, zb, cs, tm, gtm;
};

struct BwdConfig {
	int compute;
	int ns2, tn1, tk1, s1, tm3, tn3, tk3, s3, en1, en25, en34, cs, tm, gtm;
};

#define FWD_CONFIG_C(C, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZB, CS, TM) \
	FwdConfig{C, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZB, CS, TM, TM},
#define FWD_CONFIG_G_C(C, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZB, CS, TM, GTM) \
	FwdConfig{C, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZB, CS, TM, GTM},
#define FWD_CONFIG_SM90(...) FWD_CONFIG_C(90, __VA_ARGS__)
#define FWD_CONFIG_G_SM90(...) FWD_CONFIG_G_C(90, __VA_ARGS__)
#define FWD_CONFIG_SM100(...) FWD_CONFIG_C(100, __VA_ARGS__)
#define FWD_CONFIG_G_SM100(...) FWD_CONFIG_G_C(100, __VA_ARGS__)

std::vector<FwdConfig> all_fwd_configs() {
	return {
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 90
		LIGER_MOE_FWD_DISPATCH_CONFIGS_SM90(FWD_CONFIG_SM90, FWD_CONFIG_G_SM90)
#endif
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 100
		LIGER_MOE_FWD_DISPATCH_CONFIGS_SM100(FWD_CONFIG_SM100, FWD_CONFIG_G_SM100)
#endif
	};
}

#undef FWD_CONFIG_SM100
#undef FWD_CONFIG_G_SM100
#undef FWD_CONFIG_SM90
#undef FWD_CONFIG_G_SM90
#undef FWD_CONFIG_C
#undef FWD_CONFIG_G_C

#define BWD_CONFIG_C(C, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	BwdConfig{C, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, TM},
#define BWD_CONFIG_G_C(C, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM) \
	BwdConfig{C, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, GTM},
#define BWD_CONFIG_SM90(...) BWD_CONFIG_C(90, __VA_ARGS__)
#define BWD_CONFIG_G_SM90(...) BWD_CONFIG_G_C(90, __VA_ARGS__)
#define BWD_CONFIG_SM100(...) BWD_CONFIG_C(100, __VA_ARGS__)
#define BWD_CONFIG_G_SM100(...) BWD_CONFIG_G_C(100, __VA_ARGS__)

std::vector<BwdConfig> all_bwd_configs() {
	return {
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 90
		LIGER_MOE_BWD_DISPATCH_CONFIGS_SM90(BWD_CONFIG_SM90, BWD_CONFIG_G_SM90)
#endif
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 100
		LIGER_MOE_BWD_DISPATCH_CONFIGS_SM100(BWD_CONFIG_SM100, BWD_CONFIG_G_SM100)
#endif
	};
}

#undef BWD_CONFIG_SM100
#undef BWD_CONFIG_G_SM100
#undef BWD_CONFIG_SM90
#undef BWD_CONFIG_G_SM90
#undef BWD_CONFIG_C
#undef BWD_CONFIG_G_C

std::string force_string(const FwdConfig& c) {
	std::ostringstream os;
	os << c.tn1 << ',' << c.tk1 << ',' << c.s1 << ',' << c.ec1
	   << ',' << c.tn2 << ',' << c.tk2 << ',' << c.s2 << ',' << c.ec2
	   << ',' << c.zb << ',' << c.cs << ',' << c.tm << ',' << c.gtm;
	return os.str();
}

std::string force_string(const BwdConfig& c) {
	std::ostringstream os;
	os << c.ns2 << ',' << c.tn1 << ',' << c.tk1 << ',' << c.s1
	   << ',' << c.tm3 << ',' << c.tn3 << ',' << c.tk3 << ',' << c.s3
	   << ',' << c.en1 << ',' << c.en25 << ',' << c.en34 << ',' << c.cs
	   << ',' << c.tm << ',' << c.gtm;
	return os.str();
}

int compute_dispatch_key() {
	int dev = 0;
	cudaDeviceProp prop{};
	if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
		return 0;
	if (prop.major == 9) return 90;
	if (prop.major == 10) return 100;
	return 0;
}

int initial_device() {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return 0;
	const char* rank_envs[] = {
		"LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_RANK",
	};
	for (const char* name : rank_envs) {
		if (const char* value = std::getenv(name)) {
			return std::atoi(value) % count;
		}
	}
	return 0;
}

void check_cuda(cudaError_t status, const char* what) {
	ASSERT_EQ(status, cudaSuccess) << what << " failed: " << cudaGetErrorString(status);
}

ffi::Module& ffi_module() {
	static ffi::Module module = ffi::Module::LoadFromFile(LIGER_CUTE_TEST_MODULE_PATH);
	return module;
}

ffi::Function module_func(const char* name) {
	auto opt = ffi_module()->GetFunction(ffi::String(name));
	EXPECT_TRUE(opt.has_value()) << "missing TVM FFI function: " << name;
	return opt.value();
}

struct TensorArg {
	DLTensor dl{};
	std::vector<int64_t> shape;
	std::vector<int64_t> strides;

	TensorArg() = default;
	TensorArg(void* data, std::vector<int64_t> dims, DLDataType dtype, DLDevice device)
	    : shape(std::move(dims)) {
		strides.resize(shape.size());
		int64_t stride = 1;
		for (size_t i = shape.size(); i-- > 0;) {
			strides[i] = stride;
			stride *= shape[i];
		}
		dl.data = data;
		dl.device = device;
		dl.ndim = static_cast<int32_t>(shape.size());
		dl.dtype = dtype;
		dl.shape = shape.data();
		dl.strides = strides.data();
		dl.byte_offset = 0;
	}

	ffi::TensorView view() const { return ffi::TensorView(&dl); }
};

template <typename T>
struct HostBuffer {
	std::vector<T> data;
	explicit HostBuffer(size_t n) : data(n) {}
	TensorArg tensor(DLDataType dtype) {
		return TensorArg(data.data(), {static_cast<int64_t>(data.size())}, dtype, cpu_device());
	}
};

struct DeviceBuffer {
	void* ptr = nullptr;
	size_t bytes = 0;

	DeviceBuffer() = default;
	explicit DeviceBuffer(size_t nbytes) : bytes(nbytes) {
		check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
		check_cuda(cudaMemset(ptr, 0, bytes), "cudaMemset");
	}
	~DeviceBuffer() {
		if (ptr) cudaFree(ptr);
	}
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct MoeBuffers {
	int n_pes;
	int num_experts;
	int experts_per_pe;
	int max_total_slots;
	int tile_id_slots;
	int device;

	DeviceBuffer X;
	DeviceBuffer dY;
	DeviceBuffer expert_indices;
	DeviceBuffer expert_weights;
	DeviceBuffer all_B;
	DeviceBuffer all_C;
	DeviceBuffer all_A;
	DeviceBuffer Y;
	DeviceBuffer token_expert_slots;
	DeviceBuffer tile_expert_ids;
	DeviceBuffer dX;
	DeviceBuffer dB;
	DeviceBuffer dC;
	DeviceBuffer dA;
	DeviceBuffer dW;
	HostBuffer<int64_t> symm_meta;

	MoeBuffers(int pes, int device_id)
	    : n_pes(pes),
	      num_experts(pes > 0 ? pes : 1),
	      experts_per_pe(num_experts / n_pes),
	      max_total_slots(kTokens * kTopK + num_experts * kCommTileM),
	      tile_id_slots(max_total_slots / kCommTileM),
	      device(device_id),
	      X(sizeof(uint16_t) * kTokens * kHiddenDim),
	      dY(sizeof(uint16_t) * kTokens * kHiddenDim),
	      expert_indices(sizeof(int32_t) * kTokens * kTopK),
	      expert_weights(sizeof(uint16_t) * kTokens * kTopK),
	      all_B(sizeof(uint16_t) * experts_per_pe * kIntermediateDim * kHiddenDim),
	      all_C(sizeof(uint16_t) * experts_per_pe * kIntermediateDim * kHiddenDim),
	      all_A(sizeof(uint16_t) * experts_per_pe * kHiddenDim * kIntermediateDim),
	      Y(sizeof(uint16_t) * kTokens * kHiddenDim),
	      token_expert_slots(sizeof(int32_t) * max_total_slots),
	      tile_expert_ids(sizeof(int32_t) * ((max_total_slots + kCommTileM - 1) / kCommTileM)),
	      dX(sizeof(uint16_t) * kTokens * kHiddenDim),
	      dB(sizeof(uint16_t) * experts_per_pe * kIntermediateDim * kHiddenDim),
	      dC(sizeof(uint16_t) * experts_per_pe * kIntermediateDim * kHiddenDim),
	      dA(sizeof(uint16_t) * experts_per_pe * kHiddenDim * kIntermediateDim),
	      dW(sizeof(uint16_t) * kTokens * kTopK),
	      symm_meta(13) {
		std::vector<int32_t> h_indices(kTokens * kTopK);
		for (int t = 0; t < kTokens; ++t) h_indices[t] = t % num_experts;
		check_cuda(cudaMemcpy(expert_indices.ptr, h_indices.data(),
		                      h_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
		           "cudaMemcpy expert_indices");
	}

	DLDevice cuda_dev() const { return cuda_device(device); }

	TensorArg X_tensor() { return TensorArg(X.ptr, {kTokens, kHiddenDim}, bf16_dtype(), cuda_dev()); }
	TensorArg dY_tensor() { return TensorArg(dY.ptr, {kTokens, kHiddenDim}, bf16_dtype(), cuda_dev()); }
	TensorArg expert_indices_tensor() {
		return TensorArg(expert_indices.ptr, {kTokens, kTopK}, i32_dtype(), cuda_dev());
	}
	TensorArg expert_weights_tensor() {
		return TensorArg(expert_weights.ptr, {kTokens, kTopK}, bf16_dtype(), cuda_dev());
	}
	TensorArg B_tensor() {
		return TensorArg(all_B.ptr, {experts_per_pe, kIntermediateDim, kHiddenDim}, bf16_dtype(), cuda_dev());
	}
	TensorArg C_tensor() {
		return TensorArg(all_C.ptr, {experts_per_pe, kIntermediateDim, kHiddenDim}, bf16_dtype(), cuda_dev());
	}
	TensorArg A_tensor() {
		return TensorArg(all_A.ptr, {experts_per_pe, kHiddenDim, kIntermediateDim}, bf16_dtype(), cuda_dev());
	}
	TensorArg Y_tensor() { return TensorArg(Y.ptr, {kTokens, kHiddenDim}, bf16_dtype(), cuda_dev()); }
	TensorArg token_expert_slots_tensor() {
		return TensorArg(token_expert_slots.ptr, {max_total_slots}, i32_dtype(), cuda_dev());
	}
	TensorArg tile_expert_ids_tensor() {
		return TensorArg(tile_expert_ids.ptr, {tile_id_slots}, i32_dtype(), cuda_dev());
	}
	TensorArg dX_tensor() { return TensorArg(dX.ptr, {kTokens, kHiddenDim}, bf16_dtype(), cuda_dev()); }
	TensorArg dB_tensor() {
		return TensorArg(dB.ptr, {experts_per_pe, kIntermediateDim, kHiddenDim}, bf16_dtype(), cuda_dev());
	}
	TensorArg dC_tensor() {
		return TensorArg(dC.ptr, {experts_per_pe, kIntermediateDim, kHiddenDim}, bf16_dtype(), cuda_dev());
	}
	TensorArg dA_tensor() {
		return TensorArg(dA.ptr, {experts_per_pe, kHiddenDim, kIntermediateDim}, bf16_dtype(), cuda_dev());
	}
	TensorArg dW_tensor() { return TensorArg(dW.ptr, {kTokens, kTopK}, bf16_dtype(), cuda_dev()); }
	TensorArg symm_meta_tensor() { return symm_meta.tensor(i64_dtype()); }
};

void run_fwd(MoeBuffers& b, int64_t team, const ffi::Function& fwd_fn) {
	auto X = b.X_tensor();
	auto ei = b.expert_indices_tensor();
	auto ew = b.expert_weights_tensor();
	auto B = b.B_tensor();
	auto C = b.C_tensor();
	auto A = b.A_tensor();
	auto Y = b.Y_tensor();
	auto slots = b.token_expert_slots_tensor();
	auto tile_ids = b.tile_expert_ids_tensor();
	auto meta = b.symm_meta_tensor();
	fwd_fn(X.view(), ei.view(), ew.view(), B.view(), C.view(), A.view(),
	       static_cast<int64_t>(b.num_experts), static_cast<int64_t>(kTopK), team,
	       Y.view(), slots.view(), tile_ids.view(), meta.view());
}

void run_bwd(MoeBuffers& b, int64_t team, const ffi::Function& bwd_fn) {
	auto dY = b.dY_tensor();
	auto meta = b.symm_meta_tensor();
	auto slots = b.token_expert_slots_tensor();
	auto tile_ids = b.tile_expert_ids_tensor();
	auto ei = b.expert_indices_tensor();
	auto ew = b.expert_weights_tensor();
	auto B = b.B_tensor();
	auto C = b.C_tensor();
	auto A = b.A_tensor();
	auto dX = b.dX_tensor();
	auto dB = b.dB_tensor();
	auto dC = b.dC_tensor();
	auto dA = b.dA_tensor();
	auto dW = b.dW_tensor();
	bwd_fn(dY.view(), meta.view(), slots.view(), tile_ids.view(), ei.view(), ew.view(),
	       B.view(), C.view(), A.view(),
	       static_cast<int64_t>(b.num_experts), static_cast<int64_t>(kTopK), team,
	       dX.view(), dB.view(), dC.view(), dA.view(), dW.view());
}

struct NvshmemSession {
	bool initialized = false;
	~NvshmemSession() {
		if (initialized) {
			module_func("finalize")();
		}
	}
};

void run_dispatch_coverage_for_compute(int target_compute) {
	int dev_count = 0;
	if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0)
		GTEST_SKIP() << "requires CUDA";
	const int device = initial_device();
	check_cuda(cudaSetDevice(device), "cudaSetDevice");

	const int compute = compute_dispatch_key();
	if (compute != 90 && compute != 100)
		GTEST_SKIP() << "requires Hopper-family or Blackwell-family GPU";
	if (compute != target_compute)
		GTEST_SKIP() << "dispatch compute " << target_compute
		             << " is not supported by this GPU family (detected "
		             << compute << ")";

	auto uniqueid_nbytes = module_func("uniqueid_nbytes");
	auto get_uniqueid = module_func("get_uniqueid");
	auto init_with_uniqueid = module_func("init_with_uniqueid");
	auto n_pes_fn = module_func("n_pes");
	auto team_world_fn = module_func("team_world");
	auto configure_fn = module_func("moe_configure_symmetric");
	auto pop_fwd_fn = module_func("moe_pop_fwd");
	auto pool_clear_fn = module_func("pool_clear_all");
	auto fwd_fn = module_func("moe_fused_fwd_bf16");
	auto bwd_fn = module_func("moe_fused_bwd_bf16");

	HostBuffer<int64_t> uid_nbytes_buf(1);
	auto uid_nbytes_tensor = uid_nbytes_buf.tensor(i64_dtype());
	uniqueid_nbytes(uid_nbytes_tensor.view());
	std::vector<unsigned char> uid(static_cast<size_t>(uid_nbytes_buf.data[0]));
	get_uniqueid(static_cast<int64_t>(reinterpret_cast<uintptr_t>(uid.data())));
	NvshmemSession nvshmem;
	init_with_uniqueid(static_cast<int64_t>(0), static_cast<int64_t>(1),
	                   static_cast<int64_t>(reinterpret_cast<uintptr_t>(uid.data())));
	nvshmem.initialized = true;

	HostBuffer<int32_t> n_pes_buf(1);
	auto n_pes_tensor = n_pes_buf.tensor(i32_dtype());
	n_pes_fn(n_pes_tensor.view());
	const int n_pes = n_pes_buf.data[0];

	HostBuffer<int64_t> team_buf(1);
	auto team_tensor = team_buf.tensor(i64_dtype());
	team_world_fn(team_tensor.view());
	const int64_t team = team_buf.data[0];

	MoeBuffers buffers(n_pes, device);
	configure_fn(static_cast<int64_t>(kTokens), static_cast<int64_t>(kHiddenDim),
	             static_cast<int64_t>(buffers.num_experts), static_cast<int64_t>(kTopK),
	             static_cast<int64_t>(n_pes), static_cast<int64_t>(1), static_cast<int64_t>(n_pes));

	std::vector<FwdConfig> fwd_configs;
	for (const auto& c : all_fwd_configs())
		if (c.compute == target_compute) fwd_configs.push_back(c);
	std::vector<BwdConfig> bwd_configs;
	for (const auto& c : all_bwd_configs())
		if (c.compute == target_compute) bwd_configs.push_back(c);
	ASSERT_FALSE(fwd_configs.empty());
	ASSERT_FALSE(bwd_configs.empty());

	unsetenv("LIGER_MOE_BWD_FORCE_CONFIG");
	for (const auto& cfg : fwd_configs) {
		const std::string force = force_string(cfg);
		setenv("LIGER_MOE_FORCE_CONFIG", force.c_str(), 1);
		SCOPED_TRACE("fwd " + force);
		run_fwd(buffers, team, fwd_fn);
		check_cuda(cudaDeviceSynchronize(), "fwd sync");
		pop_fwd_fn();
	}

	const std::string fwd_force = force_string(fwd_configs.front());
	setenv("LIGER_MOE_FORCE_CONFIG", fwd_force.c_str(), 1);
	for (const auto& cfg : bwd_configs) {
		const std::string force = force_string(cfg);
		setenv("LIGER_MOE_BWD_FORCE_CONFIG", force.c_str(), 1);
		SCOPED_TRACE("bwd " + force);
		run_fwd(buffers, team, fwd_fn);
		run_bwd(buffers, team, bwd_fn);
		check_cuda(cudaDeviceSynchronize(), "bwd sync");
		pop_fwd_fn();
	}

	unsetenv("LIGER_MOE_FORCE_CONFIG");
	unsetenv("LIGER_MOE_BWD_FORCE_CONFIG");
	pool_clear_fn();
	module_func("finalize")();
	nvshmem.initialized = false;
}

}  // namespace

TEST(MoeApiDispatchSm90, RunsEveryForwardAndBackwardDispatchEntry) {
	run_dispatch_coverage_for_compute(90);
}

TEST(MoeApiDispatchSm100, RunsEveryForwardAndBackwardDispatchEntry) {
	run_dispatch_coverage_for_compute(100);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
