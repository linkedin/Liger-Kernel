#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>

#include "backward_gemm_sm90.cuh"
#include "forward_gemm_sm90.cuh"
#include "forward_reduce.cuh"
#include "workspace.cuh"
#include "moe_launch.h"

namespace liger {
void moe_fused_fwd_dispatch(const MoeFwdArgs& a, int* chosen_tile_m);
void moe_bwd_dispatch(const MoeBwdArgs& a, int fwd_tile_m);
}  // namespace liger

namespace {
namespace ffi = tvm::ffi;

enum liger_cute_status_t {
  LIGER_CUTE_OK = 0,
};

extern "C" {
const char* liger_cute_status_string(liger_cute_status_t status);
const char* liger_cute_last_error_string(void);

liger_cute_status_t liger_cute_nvshmem_uniqueid_nbytes(size_t* out);
liger_cute_status_t liger_cute_nvshmem_get_uniqueid(void* out);
liger_cute_status_t liger_cute_nvshmem_init_with_uniqueid(int rank, int nranks, const void* uid);
liger_cute_status_t liger_cute_nvshmem_init_pmi(void);
liger_cute_status_t liger_cute_nvshmem_finalize(void);
liger_cute_status_t liger_cute_nvshmem_my_pe(int* out);
liger_cute_status_t liger_cute_nvshmem_n_pes(int* out);
liger_cute_status_t liger_cute_nvshmem_team_world(int64_t* out);
liger_cute_status_t liger_cute_nvshmem_team_split_strided(
    int64_t parent_handle, int start, int stride, int size, int64_t* out);
liger_cute_status_t liger_cute_nvshmem_team_destroy(int64_t team_handle);
liger_cute_status_t liger_cute_nvshmem_team_my_pe(int64_t team_handle, int* out);
liger_cute_status_t liger_cute_nvshmem_team_n_pes(int64_t team_handle, int* out);
liger_cute_status_t liger_cute_nvshmem_team_translate_pe(
    int64_t src_team_handle, int src_pe, int64_t dst_team_handle, int* out);
liger_cute_status_t liger_cute_pool_clear_all(void);
liger_cute_status_t liger_cute_pool_clear_buffers(void);

typedef struct liger_cute_moe_symm_config_t {
  int32_t max_total_slots;
  int32_t max_num_experts;
  int32_t hidden_dim;
  int32_t num_pes;
  int32_t experts_per_pe;
  int32_t max_top_k;
  int32_t initialized;
} liger_cute_moe_symm_config_t;

liger_cute_status_t liger_cute_moe_get_symm_config(liger_cute_moe_symm_config_t* out);
liger_cute_status_t liger_cute_moe_configure_symmetric(
    int max_tokens, int hidden_dim, int max_num_experts, int max_top_k, int num_pes,
    int num_hosts, int gpus_per_host);
liger_cute_status_t liger_cute_moe_pop_fwd(void);
}

void CheckStatus(liger_cute_status_t status, const char* what) {
  if (status != LIGER_CUTE_OK) {
    TVM_FFI_THROW(RuntimeError) << "liger_cute: " << what << " failed ("
                                << liger_cute_status_string(status)
                                << "): " << liger_cute_last_error_string();
  }
}

void CheckNccl(ncclResult_t status, const char* what) {
  if (status != ncclSuccess) {
    TVM_FFI_THROW(RuntimeError) << "liger_cute: " << what << " failed: "
                                << ncclGetErrorString(status);
  }
}

void RequireTensor(ffi::TensorView tensor, int ndim, DLDataType dtype, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ndim) << name;
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name;
}

void RequireCudaTensor(ffi::TensorView tensor, int ndim, DLDataType dtype, const char* name) {
  RequireTensor(tensor, ndim, dtype, name);
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name;
}

void RequireSameCudaDevice(ffi::TensorView tensor, ffi::TensorView reference, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, reference.device().device_type)
      << name << " must be on the same device as x";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, reference.device().device_id)
      << name << " must be on the same device as x";
}

void RequirePositiveInverseTemperature(double inverse_temperature) {
  TVM_FFI_ICHECK(std::isfinite(inverse_temperature) && inverse_temperature > 0.0)
      << "inverse_temperature must be finite and positive";
}

void RequireCudaMoeWeight(ffi::TensorView tensor, DLDataType dtype, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3) << name;
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name;
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name;
  TVM_FFI_ICHECK_EQ(tensor.stride(2), 1)
      << name << " hidden dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(tensor.stride(1), tensor.size(2))
      << name << " intermediate rows must be contiguous";
  TVM_FFI_ICHECK_GE(tensor.stride(0), tensor.size(1) * tensor.size(2))
      << name << " expert stride overlaps adjacent experts";
  TVM_FFI_ICHECK_EQ(tensor.stride(0) % tensor.size(2), 0)
      << name << " expert stride must contain complete hidden-dimension rows";
}

void RequireRank(ffi::TensorView tensor, int ndim, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ndim) << name;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name;
}

void RequireCpuInt64(ffi::TensorView tensor, int64_t size) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), (DLDataType{kDLInt, 64, 1}));
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 1);
  TVM_FFI_ICHECK_EQ(tensor.size(0), size);
  TVM_FFI_ICHECK(tensor.IsContiguous());
}

void RequireCpuInt32(ffi::TensorView tensor, int64_t size) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), (DLDataType{kDLInt, 32, 1}));
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 1);
  TVM_FFI_ICHECK_EQ(tensor.size(0), size);
  TVM_FFI_ICHECK(tensor.IsContiguous());
}

void WriteMeta(ffi::TensorView meta, int offset, void* ptr, int64_t size0, int64_t size1, int64_t dtype) {
  int64_t* data = static_cast<int64_t*>(meta.data_ptr());
  data[offset] = reinterpret_cast<int64_t>(ptr);
  data[offset + 1] = size0;
  data[offset + 2] = size1;
  data[offset + 3] = dtype;
}

struct Meta2 {
  void* ptr;
  int64_t size0;
  int64_t size1;
};

Meta2 ReadMeta(ffi::TensorView meta, int offset) {
  const int64_t* data = static_cast<const int64_t*>(meta.data_ptr());
  return {reinterpret_cast<void*>(data[offset]), data[offset + 1], data[offset + 2]};
}

void ThrowCoreError(const char* what, const std::exception& error) {
  TVM_FFI_THROW(RuntimeError) << "liger_cute: " << what << " failed: " << error.what();
}

void uniqueid_nbytes(ffi::TensorView out) {
  RequireCpuInt64(out, 1);
  size_t n = 0;
  CheckStatus(liger_cute_nvshmem_uniqueid_nbytes(&n), "uniqueid_nbytes");
  static_cast<int64_t*>(out.data_ptr())[0] = static_cast<int64_t>(n);
}

void get_uniqueid(int64_t buf_ptr) {
  CheckStatus(liger_cute_nvshmem_get_uniqueid(reinterpret_cast<void*>(buf_ptr)), "get_uniqueid");
}

void init_with_uniqueid(int64_t rank, int64_t nranks, int64_t buf_ptr) {
  CheckStatus(
      liger_cute_nvshmem_init_with_uniqueid(static_cast<int>(rank), static_cast<int>(nranks),
                                            reinterpret_cast<const void*>(buf_ptr)),
      "init_with_uniqueid");
}

void init_pmi() { CheckStatus(liger_cute_nvshmem_init_pmi(), "init_pmi"); }
void finalize() { CheckStatus(liger_cute_nvshmem_finalize(), "finalize"); }
void pool_clear_all() { CheckStatus(liger_cute_pool_clear_all(), "pool_clear_all"); }
void pool_clear_buffers() { CheckStatus(liger_cute_pool_clear_buffers(), "pool_clear_buffers"); }

void nccl_unique_id_nbytes(ffi::TensorView out) {
  RequireCpuInt64(out, 1);
  static_cast<int64_t*>(out.data_ptr())[0] = sizeof(ncclUniqueId);
}

void nccl_get_unique_id(ffi::TensorView out) {
  DLDataType u8{kDLUInt, 8, 1};
  RequireTensor(out, 1, u8, "out");
  TVM_FFI_ICHECK_EQ(out.device().device_type, kDLCPU);
  TVM_FFI_ICHECK_EQ(out.size(0), static_cast<int64_t>(sizeof(ncclUniqueId)));
  ncclUniqueId unique_id;
  CheckNccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
  std::memcpy(out.data_ptr(), &unique_id, sizeof(unique_id));
}

void nccl_comm_init_rank(
    int64_t rank, int64_t nranks, ffi::TensorView unique_id, ffi::TensorView out) {
  DLDataType u8{kDLUInt, 8, 1};
  RequireTensor(unique_id, 1, u8, "unique_id");
  TVM_FFI_ICHECK_EQ(unique_id.device().device_type, kDLCPU);
  TVM_FFI_ICHECK_EQ(
      unique_id.size(0), static_cast<int64_t>(sizeof(ncclUniqueId)));
  RequireCpuInt64(out, 1);
  TVM_FFI_ICHECK_GT(nranks, 0);
  TVM_FFI_ICHECK_GE(rank, 0);
  TVM_FFI_ICHECK_LT(rank, nranks);
  ncclUniqueId id;
  std::memcpy(&id, unique_id.data_ptr(), sizeof(id));
  ncclComm_t comm = nullptr;
  CheckNccl(
      ncclCommInitRank(
          &comm, static_cast<int>(nranks), id, static_cast<int>(rank)),
      "ncclCommInitRank");
  static_cast<int64_t*>(out.data_ptr())[0] =
      reinterpret_cast<int64_t>(comm);
}

void nccl_comm_destroy(int64_t comm_handle) {
  if (comm_handle == 0) {
    return;
  }
  CheckNccl(
      ncclCommDestroy(reinterpret_cast<ncclComm_t>(comm_handle)),
      "ncclCommDestroy");
}

void my_pe(ffi::TensorView out) {
  RequireCpuInt32(out, 1);
  CheckStatus(liger_cute_nvshmem_my_pe(static_cast<int*>(out.data_ptr())), "my_pe");
}

void n_pes(ffi::TensorView out) {
  RequireCpuInt32(out, 1);
  CheckStatus(liger_cute_nvshmem_n_pes(static_cast<int*>(out.data_ptr())), "n_pes");
}

void team_world(ffi::TensorView out) {
  RequireCpuInt64(out, 1);
  CheckStatus(liger_cute_nvshmem_team_world(static_cast<int64_t*>(out.data_ptr())), "team_world");
}

void team_split_strided(int64_t parent, int64_t start, int64_t stride, int64_t size, ffi::TensorView out) {
  RequireCpuInt64(out, 1);
  CheckStatus(
      liger_cute_nvshmem_team_split_strided(parent, static_cast<int>(start), static_cast<int>(stride),
                                            static_cast<int>(size), static_cast<int64_t*>(out.data_ptr())),
      "team_split_strided");
}

void team_destroy(int64_t team_handle) {
  CheckStatus(liger_cute_nvshmem_team_destroy(team_handle), "team_destroy");
}

void team_my_pe(int64_t team_handle, ffi::TensorView out) {
  RequireCpuInt32(out, 1);
  CheckStatus(liger_cute_nvshmem_team_my_pe(team_handle, static_cast<int*>(out.data_ptr())), "team_my_pe");
}

void team_n_pes(int64_t team_handle, ffi::TensorView out) {
  RequireCpuInt32(out, 1);
  CheckStatus(liger_cute_nvshmem_team_n_pes(team_handle, static_cast<int*>(out.data_ptr())), "team_n_pes");
}

void team_translate_pe(int64_t src_team, int64_t src_pe, int64_t dst_team, ffi::TensorView out) {
  RequireCpuInt32(out, 1);
  CheckStatus(
      liger_cute_nvshmem_team_translate_pe(src_team, static_cast<int>(src_pe), dst_team,
                                           static_cast<int*>(out.data_ptr())),
      "team_translate_pe");
}

void moe_get_symm_config(ffi::TensorView out) {
  RequireCpuInt32(out, 7);
  liger_cute_moe_symm_config_t cfg;
  CheckStatus(liger_cute_moe_get_symm_config(&cfg), "moe_get_symm_config");
  int32_t* data = static_cast<int32_t*>(out.data_ptr());
  data[0] = cfg.max_total_slots;
  data[1] = cfg.max_num_experts;
  data[2] = cfg.hidden_dim;
  data[3] = cfg.num_pes;
  data[4] = cfg.experts_per_pe;
  data[5] = cfg.max_top_k;
  data[6] = cfg.initialized;
}

void moe_configure_symmetric(
    int64_t max_tokens, int64_t hidden_dim, int64_t max_num_experts, int64_t max_top_k,
    int64_t num_pes, int64_t num_hosts, int64_t gpus_per_host) {
  CheckStatus(
      liger_cute_moe_configure_symmetric(
          static_cast<int>(max_tokens), static_cast<int>(hidden_dim),
          static_cast<int>(max_num_experts), static_cast<int>(max_top_k), static_cast<int>(num_pes),
          static_cast<int>(num_hosts), static_cast<int>(gpus_per_host)),
      "moe_configure_symmetric");
}

void moe_pop_fwd() { CheckStatus(liger_cute_moe_pop_fwd(), "moe_pop_fwd"); }

void moe_fused_fwd_bf16(
    ffi::TensorView X, ffi::TensorView expert_indices, ffi::TensorView expert_weights,
    ffi::TensorView all_B, ffi::TensorView all_C, ffi::TensorView all_A, int64_t num_experts,
    int64_t top_k, int64_t team_handle, ffi::TensorView Y, ffi::TensorView token_expert_slots,
    ffi::TensorView tile_expert_ids, ffi::TensorView symm_meta) {
  RequireCpuInt64(symm_meta, 17);
  DLDataType bf16{kDLBfloat, 16, 1};
  DLDataType i32{kDLInt, 32, 1};
  RequireCudaTensor(X, 2, bf16, "X");
  RequireCudaTensor(expert_indices, 2, i32, "expert_indices");
  RequireCudaTensor(expert_weights, 2, bf16, "expert_weights");
  RequireCudaMoeWeight(all_B, bf16, "all_B");
  RequireCudaMoeWeight(all_C, bf16, "all_C");
  RequireCudaTensor(all_A, 3, bf16, "all_A");
  RequireCudaTensor(Y, 2, bf16, "Y");
  RequireCudaTensor(token_expert_slots, 1, i32, "token_expert_slots");
  RequireCudaTensor(tile_expert_ids, 1, i32, "tile_expert_ids");

  liger_cute_moe_symm_config_t cfg;
  CheckStatus(liger_cute_moe_get_symm_config(&cfg), "moe_get_symm_config");
  TVM_FFI_ICHECK_NE(cfg.initialized, 0) << "call moe_configure_symmetric first";

  const int64_t num_tokens = X.size(0);
  const int64_t hidden_dim = X.size(1);
  const int64_t intermediate_dim = all_B.size(1);
  const int64_t experts_per_pe = all_B.size(0);
  TVM_FFI_ICHECK_EQ(all_B.size(2), hidden_dim)
      << "all_B hidden dimension must match X";
  TVM_FFI_ICHECK_EQ(all_C.size(0), experts_per_pe);
  TVM_FFI_ICHECK_EQ(all_C.size(1), intermediate_dim);
  TVM_FFI_ICHECK_EQ(all_C.size(2), hidden_dim);
  TVM_FFI_ICHECK_EQ(all_B.stride(0), all_C.stride(0))
      << "all_B and all_C must use the same expert stride";
  TVM_FFI_ICHECK_EQ(all_A.size(0), experts_per_pe);
  TVM_FFI_ICHECK_EQ(all_A.size(1), hidden_dim);
  TVM_FFI_ICHECK_EQ(all_A.size(2), intermediate_dim);
  TVM_FFI_ICHECK_EQ(hidden_dim, cfg.hidden_dim);
  TVM_FFI_ICHECK_EQ(num_experts, cfg.max_num_experts);
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= cfg.max_top_k);
  TVM_FFI_ICHECK_EQ(expert_indices.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(expert_indices.size(1), top_k);
  TVM_FFI_ICHECK_LE(num_tokens * top_k, cfg.max_total_slots);

  int chosen_tile_m = 0;
  int64_t stream_handle =
      reinterpret_cast<int64_t>(TVMFFIEnvGetStream(X.device().device_type, X.device().device_id));
  int device = 0;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&device), cudaSuccess);
  void* x_sorted = nullptr;
  void* y_buf = nullptr;
  void* all_expert_offsets = nullptr;
  void* all_expert_counts = nullptr;
  liger::MoeFwdArgs args{};
  args.X = X.data_ptr();
  args.expert_indices = static_cast<const int*>(expert_indices.data_ptr());
  args.expert_weights = expert_weights.data_ptr();
  args.all_B = all_B.data_ptr();
  args.all_C = all_C.data_ptr();
  args.all_A = all_A.data_ptr();
  args.weight_expert_stride = all_B.stride(0);
  args.num_tokens = static_cast<int>(num_tokens);
  args.hidden_dim = static_cast<int>(hidden_dim);
  args.intermediate_dim = static_cast<int>(intermediate_dim);
  args.experts_per_pe = static_cast<int>(experts_per_pe);
  args.num_experts = static_cast<int>(num_experts);
  args.top_k = static_cast<int>(top_k);
  args.team = static_cast<int>(team_handle);
  args.stream = reinterpret_cast<cudaStream_t>(stream_handle);
  args.device = device;
  args.Y = Y.data_ptr();
  args.token_expert_slots = static_cast<int*>(token_expert_slots.data_ptr());
  args.tile_expert_ids = static_cast<int*>(tile_expert_ids.data_ptr());
  args.x_sorted_out = &x_sorted;
  args.y_buf_out = &y_buf;
  args.all_expert_offsets_out = &all_expert_offsets;
  args.all_expert_counts_out = &all_expert_counts;
  try {
    liger::moe_fused_fwd_dispatch(args, &chosen_tile_m);
  } catch (const std::exception& e) {
    ThrowCoreError("moe_fused_fwd_bf16", e);
  }
  WriteMeta(symm_meta, 0, x_sorted, cfg.max_total_slots, cfg.hidden_dim, 3);
  WriteMeta(symm_meta, 4, y_buf, cfg.max_total_slots, cfg.hidden_dim, 3);
  WriteMeta(symm_meta, 8, all_expert_offsets, cfg.num_pes, cfg.max_num_experts + 1, 7);
  WriteMeta(symm_meta, 12, all_expert_counts, cfg.num_pes, cfg.max_num_experts, 7);
  static_cast<int64_t*>(symm_meta.data_ptr())[16] = chosen_tile_m;
}

void moe_fused_bwd_bf16(
    ffi::TensorView dY, ffi::TensorView symm_meta, ffi::TensorView token_expert_slots,
    ffi::TensorView tile_expert_ids, ffi::TensorView expert_indices, ffi::TensorView expert_weights,
    ffi::TensorView all_B, ffi::TensorView all_C, ffi::TensorView all_A, int64_t num_experts,
    int64_t top_k, int64_t team_handle, ffi::TensorView dX, ffi::TensorView dB,
    ffi::TensorView dC, ffi::TensorView dA, ffi::TensorView dW) {
  RequireCpuInt64(symm_meta, 17);
  DLDataType bf16{kDLBfloat, 16, 1};
  DLDataType i32{kDLInt, 32, 1};
  RequireCudaTensor(dY, 2, bf16, "dY");
  RequireCudaTensor(token_expert_slots, 1, i32, "token_expert_slots");
  RequireCudaTensor(tile_expert_ids, 1, i32, "tile_expert_ids");
  RequireCudaTensor(expert_indices, 2, i32, "expert_indices");
  RequireCudaTensor(expert_weights, 2, bf16, "expert_weights");
  RequireCudaTensor(all_B, 3, bf16, "all_B");
  RequireCudaTensor(all_C, 3, bf16, "all_C");
  RequireCudaTensor(all_A, 3, bf16, "all_A");
  RequireCudaTensor(dX, 2, bf16, "dX");
  RequireCudaTensor(dB, 3, bf16, "dB");
  RequireCudaTensor(dC, 3, bf16, "dC");
  RequireCudaTensor(dA, 3, bf16, "dA");
  RequireCudaTensor(dW, 2, bf16, "dW");
  Meta2 x_sorted = ReadMeta(symm_meta, 0);
  Meta2 y_buf = ReadMeta(symm_meta, 4);
  Meta2 expert_offsets = ReadMeta(symm_meta, 8);
  Meta2 expert_counts = ReadMeta(symm_meta, 12);
  int64_t stream_handle =
      reinterpret_cast<int64_t>(TVMFFIEnvGetStream(dY.device().device_type, dY.device().device_id));
  int device = 0;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&device), cudaSuccess);
  liger::MoeBwdArgs args{};
  args.dY = dY.data_ptr();
  args.Y_fwd = y_buf.ptr;
  args.x_sorted = x_sorted.ptr;
  args.token_expert_slots = static_cast<int*>(token_expert_slots.data_ptr());
  args.tile_expert_ids = static_cast<int*>(tile_expert_ids.data_ptr());
  args.expert_offsets = static_cast<int*>(expert_offsets.ptr);
  args.expert_counts = static_cast<int*>(expert_counts.ptr);
  args.expert_indices = static_cast<int*>(expert_indices.data_ptr());
  args.expert_weights = expert_weights.data_ptr();
  args.all_B = all_B.data_ptr();
  args.all_C = all_C.data_ptr();
  args.all_A = all_A.data_ptr();
  args.num_tokens = static_cast<int>(dY.size(0));
  args.hidden_dim = static_cast<int>(dY.size(1));
  args.intermediate_dim = static_cast<int>(all_B.size(1));
  args.experts_per_pe = static_cast<int>(all_B.size(0));
  args.num_experts = static_cast<int>(num_experts);
  args.top_k = static_cast<int>(top_k);
  args.team = static_cast<int>(team_handle);
  args.stream = reinterpret_cast<cudaStream_t>(stream_handle);
  args.device = device;
  args.dX = dX.data_ptr();
  args.dB = dB.data_ptr();
  args.dC = dC.data_ptr();
  args.dA = dA.data_ptr();
  args.dW = dW.data_ptr();
  try {
    liger::moe_bwd_dispatch(args, static_cast<int>(static_cast<const int64_t*>(symm_meta.data_ptr())[16]));
  } catch (const std::exception& e) {
    ThrowCoreError("moe_fused_bwd_bf16", e);
  }
}

void fused_linear_scaled_cross_entropy_configure_forward(
    int64_t max_tokens, int64_t max_local_vocab) {
  TVM_FFI_ICHECK_GT(max_tokens, 0);
  TVM_FFI_ICHECK_GT(max_local_vocab, 0);
  try {
    liger::fused_scaled_linear_cross_entropy::configure_forward_tp_workspace(
        static_cast<int>(max_tokens), static_cast<int>(max_local_vocab));
  } catch (const std::exception& e) {
    ThrowCoreError("fused_linear_scaled_cross_entropy_configure_forward", e);
  }
}

void fused_linear_scaled_cross_entropy_configure_backward(
    int64_t max_tokens, int64_t max_hidden, int64_t max_local_vocab,
    int64_t max_tiles_per_reduce, int64_t team_handle) {
  TVM_FFI_ICHECK_GT(max_tokens, 0);
  TVM_FFI_ICHECK_GT(max_hidden, 0);
  TVM_FFI_ICHECK_GT(max_local_vocab, 0);
  TVM_FFI_ICHECK(
      max_tiles_per_reduce == 1 || max_tiles_per_reduce == 2 ||
      max_tiles_per_reduce == 4);
  try {
    liger::fused_scaled_linear_cross_entropy::configure_backward_tp_symmetric(
        static_cast<int>(max_tokens), static_cast<int>(max_hidden),
        static_cast<int>(max_local_vocab), static_cast<int>(max_tiles_per_reduce), 1, team_handle);
  } catch (const std::exception& e) {
    ThrowCoreError("fused_linear_scaled_cross_entropy_configure_backward", e);
  }
}

void fused_linear_scaled_cross_entropy_forward(
    ffi::TensorView x, ffi::TensorView weight, ffi::TensorView target,
    int64_t vocab_start, int64_t ignore_index, double inverse_temperature,
    int64_t nccl_comm_handle, bool return_entropy, ffi::TensorView nll,
    ffi::TensorView lse, ffi::TensorView entropy) {
  DLDataType bf16{kDLBfloat, 16, 1};
  DLDataType f32{kDLFloat, 32, 1};
  DLDataType i64{kDLInt, 64, 1};
  RequireCudaTensor(x, 2, bf16, "x");
  RequireCudaTensor(weight, 2, bf16, "weight");
  RequireCudaTensor(target, 1, i64, "target");
  RequireCudaTensor(nll, 1, f32, "nll");
  RequireCudaTensor(lse, 1, f32, "lse");
  RequireCudaTensor(entropy, 1, f32, "entropy");
  RequireSameCudaDevice(weight, x, "weight");
  RequireSameCudaDevice(target, x, "target");
  RequireSameCudaDevice(nll, x, "nll");
  RequireSameCudaDevice(lse, x, "lse");
  RequireSameCudaDevice(entropy, x, "entropy");

  int64_t tokens = x.size(0);
  int64_t hidden = x.size(1);
  int64_t local_vocab = weight.size(0);
  TVM_FFI_ICHECK_EQ(weight.size(1), hidden);
  TVM_FFI_ICHECK_EQ(target.size(0), tokens);
  TVM_FFI_ICHECK_EQ(nll.size(0), tokens);
  TVM_FFI_ICHECK_EQ(lse.size(0), tokens);
  TVM_FFI_ICHECK_EQ(entropy.size(0), tokens);
  TVM_FFI_ICHECK_GE(vocab_start, 0);
  TVM_FFI_ICHECK_NE(nccl_comm_handle, 0);
  RequirePositiveInverseTemperature(inverse_temperature);

  using namespace liger::fused_scaled_linear_cross_entropy;
  ForwardTpParamsSm90<90> params;
  params.gemm.x = x.data_ptr();
  params.gemm.weight = weight.data_ptr();
  params.gemm.target = static_cast<const int64_t*>(target.data_ptr());
  params.gemm.tokens = static_cast<int>(tokens);
  params.gemm.hidden = static_cast<int>(hidden);
  params.gemm.local_vocab = static_cast<int>(local_vocab);
  params.gemm.vocab_start = vocab_start;
  params.gemm.ignore_index = ignore_index;
  params.gemm.inverse_temperature = static_cast<float>(inverse_temperature);
  params.nll = static_cast<float*>(nll.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());
  params.entropy = static_cast<float*>(entropy.data_ptr());
  params.nccl_comm_handle = nccl_comm_handle;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  try {
    if (return_entropy) {
      liger::fused_scaled_linear_cross_entropy::
          fused_linear_scaled_cross_entropy_forward<true, 90>(params, stream);
    } else {
      liger::fused_scaled_linear_cross_entropy::
          fused_linear_scaled_cross_entropy_forward<false, 90>(params, stream);
    }
  } catch (const std::exception& e) {
    ThrowCoreError("fused_linear_scaled_cross_entropy_forward", e);
  }
}

void fused_linear_scaled_cross_entropy_backward(
    ffi::TensorView grad_output, ffi::TensorView entropy_grad, ffi::TensorView x,
    ffi::TensorView weight, ffi::TensorView target, ffi::TensorView lse,
    ffi::TensorView entropy, int64_t vocab_start, int64_t ignore_index,
    double inverse_temperature, int64_t team_handle, int64_t tiles_per_reduce,
    bool return_entropy, ffi::TensorView grad_input, ffi::TensorView grad_weight) {
  DLDataType bf16{kDLBfloat, 16, 1};
  DLDataType f32{kDLFloat, 32, 1};
  DLDataType i64{kDLInt, 64, 1};
  RequireCudaTensor(grad_output, 1, f32, "grad_output");
  RequireCudaTensor(entropy_grad, 1, f32, "entropy_grad");
  RequireCudaTensor(x, 2, bf16, "x");
  RequireCudaTensor(weight, 2, bf16, "weight");
  RequireCudaTensor(target, 1, i64, "target");
  RequireCudaTensor(lse, 1, f32, "lse");
  RequireCudaTensor(entropy, 1, f32, "entropy");
  RequireCudaTensor(grad_input, 2, bf16, "grad_input");
  RequireCudaTensor(grad_weight, 2, bf16, "grad_weight");
  RequireSameCudaDevice(entropy_grad, x, "entropy_grad");
  RequireSameCudaDevice(weight, x, "weight");
  RequireSameCudaDevice(target, x, "target");
  RequireSameCudaDevice(grad_output, x, "grad_output");
  RequireSameCudaDevice(lse, x, "lse");
  RequireSameCudaDevice(entropy, x, "entropy");
  RequireSameCudaDevice(grad_input, x, "grad_input");
  RequireSameCudaDevice(grad_weight, x, "grad_weight");

  int64_t tokens = x.size(0);
  int64_t hidden = x.size(1);
  int64_t local_vocab = weight.size(0);
  TVM_FFI_ICHECK_EQ(weight.size(1), hidden);
  TVM_FFI_ICHECK_EQ(target.size(0), tokens);
  TVM_FFI_ICHECK_EQ(grad_output.size(0), tokens);
  TVM_FFI_ICHECK_EQ(entropy_grad.size(0), tokens);
  TVM_FFI_ICHECK_EQ(lse.size(0), tokens);
  TVM_FFI_ICHECK_EQ(entropy.size(0), tokens);
  TVM_FFI_ICHECK_EQ(grad_input.size(0), tokens);
  TVM_FFI_ICHECK_EQ(grad_input.size(1), hidden);
  TVM_FFI_ICHECK_EQ(grad_weight.size(0), local_vocab);
  TVM_FFI_ICHECK_EQ(grad_weight.size(1), hidden);
  TVM_FFI_ICHECK(
      tiles_per_reduce == 1 || tiles_per_reduce == 2 || tiles_per_reduce == 4);
  RequirePositiveInverseTemperature(inverse_temperature);

  using namespace liger::fused_scaled_linear_cross_entropy;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  try {
    TVM_FFI_ICHECK_EQ(team_handle, backward_dx_team_handle())
        << "team_handle must match the configured tensor-parallel team";
    BackwardScratch scratch = reserve_backward_scratch(static_cast<int>(local_vocab));
    BackwardTpParamsSm90<90> params;
    params.gemm.x = x.data_ptr();
    params.gemm.weight = weight.data_ptr();
    params.gemm.target = static_cast<const int64_t*>(target.data_ptr());
    params.gemm.grad_output = static_cast<const float*>(grad_output.data_ptr());
    params.gemm.lse = static_cast<const float*>(lse.data_ptr());
    params.gemm.entropy = static_cast<const float*>(entropy.data_ptr());
    params.gemm.entropy_grad = static_cast<const float*>(entropy_grad.data_ptr());
    params.gemm.grad_input = grad_input.data_ptr();
    params.gemm.grad_weight = grad_weight.data_ptr();
    params.gemm.dz_workspace = scratch.dz_workspace;
    params.gemm.dz_workspace_bytes = scratch.dz_workspace_bytes;
    params.gemm.tokens = static_cast<int>(tokens);
    params.gemm.hidden = static_cast<int>(hidden);
    params.gemm.local_vocab = static_cast<int>(local_vocab);
    params.gemm.vocab_start = vocab_start;
    params.gemm.ignore_index = ignore_index;
    params.gemm.inverse_temperature = static_cast<float>(inverse_temperature);
    params.tiles_per_reduce = static_cast<int>(tiles_per_reduce);
    params.num_comm_channels = 1;
    if (return_entropy) {
      liger::fused_scaled_linear_cross_entropy::
          fused_linear_scaled_cross_entropy_backward<true, 90>(params, stream);
    } else {
      liger::fused_scaled_linear_cross_entropy::
          fused_linear_scaled_cross_entropy_backward<false, 90>(params, stream);
    }
  } catch (const std::exception& e) {
    ThrowCoreError("fused_linear_scaled_cross_entropy_backward", e);
  }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(uniqueid_nbytes, uniqueid_nbytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_uniqueid, get_uniqueid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_with_uniqueid, init_with_uniqueid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_pmi, init_pmi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(finalize, finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nccl_unique_id_nbytes, nccl_unique_id_nbytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nccl_get_unique_id, nccl_get_unique_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nccl_comm_init_rank, nccl_comm_init_rank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nccl_comm_destroy, nccl_comm_destroy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_pe, my_pe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(n_pes, n_pes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(team_world, team_world);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(team_split_strided, team_split_strided);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(team_destroy, team_destroy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(team_my_pe, team_my_pe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(team_n_pes, team_n_pes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(team_translate_pe, team_translate_pe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pool_clear_all, pool_clear_all);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pool_clear_buffers, pool_clear_buffers);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_get_symm_config, moe_get_symm_config);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_configure_symmetric, moe_configure_symmetric);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_pop_fwd, moe_pop_fwd);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_fused_fwd_bf16, moe_fused_fwd_bf16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_fused_bwd_bf16, moe_fused_bwd_bf16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    fused_linear_scaled_cross_entropy_configure_forward,
    fused_linear_scaled_cross_entropy_configure_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    fused_linear_scaled_cross_entropy_configure_backward,
    fused_linear_scaled_cross_entropy_configure_backward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    fused_linear_scaled_cross_entropy_forward,
    fused_linear_scaled_cross_entropy_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    fused_linear_scaled_cross_entropy_backward,
    fused_linear_scaled_cross_entropy_backward);
