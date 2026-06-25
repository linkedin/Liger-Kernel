#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstddef>
#include <cstdint>

namespace {
namespace ffi = tvm::ffi;

enum liger_cute_dtype_t {
  LIGER_CUTE_DTYPE_INVALID = 0,
  LIGER_CUTE_DTYPE_FLOAT32 = 1,
  LIGER_CUTE_DTYPE_FLOAT16 = 2,
  LIGER_CUTE_DTYPE_BFLOAT16 = 3,
  LIGER_CUTE_DTYPE_FLOAT64 = 4,
  LIGER_CUTE_DTYPE_INT8 = 5,
  LIGER_CUTE_DTYPE_INT16 = 6,
  LIGER_CUTE_DTYPE_INT32 = 7,
  LIGER_CUTE_DTYPE_INT64 = 8,
  LIGER_CUTE_DTYPE_UINT8 = 9,
  LIGER_CUTE_DTYPE_BOOL = 10,
};

enum liger_cute_status_t {
  LIGER_CUTE_OK = 0,
};

namespace liger_cute {
template <int N>
struct TensorView {
  void* data;
  int64_t sizes[N];
  liger_cute_dtype_t dtype;
};
}  // namespace liger_cute

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
liger_cute_status_t liger_cute_moe_fused_fwd_bf16_auto(
    liger_cute::TensorView<2> X, liger_cute::TensorView<2> expert_indices,
    liger_cute::TensorView<2> expert_weights, liger_cute::TensorView<3> all_B,
    liger_cute::TensorView<3> all_C, liger_cute::TensorView<3> all_A, int num_experts,
    int top_k, int64_t team_handle, int64_t stream_handle, liger_cute::TensorView<2>* Y_out,
    liger_cute::TensorView<1>* token_expert_slots_out,
    liger_cute::TensorView<1>* tile_expert_ids_out,
    liger_cute::TensorView<2>* x_sorted_out_symm, liger_cute::TensorView<2>* y_buf_out_symm,
    liger_cute::TensorView<2>* all_expert_offsets_out_symm, int* chosen_tile_m_out);
liger_cute_status_t liger_cute_moe_fused_bwd_bf16_auto(
    liger_cute::TensorView<2> dY, liger_cute::TensorView<2> Y_fwd,
    liger_cute::TensorView<2> x_sorted, liger_cute::TensorView<1> token_expert_slots,
    liger_cute::TensorView<1> tile_expert_ids, liger_cute::TensorView<2> expert_offsets,
    liger_cute::TensorView<2> expert_indices, liger_cute::TensorView<2> expert_weights,
    liger_cute::TensorView<3> all_B, liger_cute::TensorView<3> all_C,
    liger_cute::TensorView<3> all_A, int num_experts, int top_k, int64_t team_handle,
    int64_t stream_handle, int fwd_tile_m, liger_cute::TensorView<2>* dX_out,
    liger_cute::TensorView<3>* dB_out, liger_cute::TensorView<3>* dC_out,
    liger_cute::TensorView<3>* dA_out, liger_cute::TensorView<2>* dW_out);
}

void CheckStatus(liger_cute_status_t status, const char* what) {
  if (status != LIGER_CUTE_OK) {
    TVM_FFI_THROW(RuntimeError) << "liger_cute: " << what << " failed ("
                                << liger_cute_status_string(status)
                                << "): " << liger_cute_last_error_string();
  }
}

liger_cute_dtype_t DTypeFromDL(DLDataType dtype) {
  if (dtype.code == kDLFloat && dtype.bits == 32) return LIGER_CUTE_DTYPE_FLOAT32;
  if (dtype.code == kDLFloat && dtype.bits == 16) return LIGER_CUTE_DTYPE_FLOAT16;
  if (dtype.code == kDLBfloat && dtype.bits == 16) return LIGER_CUTE_DTYPE_BFLOAT16;
  if (dtype.code == kDLFloat && dtype.bits == 64) return LIGER_CUTE_DTYPE_FLOAT64;
  if (dtype.code == kDLInt && dtype.bits == 8) return LIGER_CUTE_DTYPE_INT8;
  if (dtype.code == kDLInt && dtype.bits == 16) return LIGER_CUTE_DTYPE_INT16;
  if (dtype.code == kDLInt && dtype.bits == 32) return LIGER_CUTE_DTYPE_INT32;
  if (dtype.code == kDLInt && dtype.bits == 64) return LIGER_CUTE_DTYPE_INT64;
  if (dtype.code == kDLUInt && dtype.bits == 8) return LIGER_CUTE_DTYPE_UINT8;
  if (dtype.code == kDLBool) return LIGER_CUTE_DTYPE_BOOL;
  TVM_FFI_THROW(TypeError) << "Unsupported DLPack dtype: " << dtype;
}

template <int N>
liger_cute::TensorView<N> ToTensorView(ffi::TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), N);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  liger_cute::TensorView<N> out;
  out.data = tensor.data_ptr();
  for (int i = 0; i < N; ++i) {
    out.sizes[i] = tensor.size(i);
  }
  out.dtype = DTypeFromDL(tensor.dtype());
  return out;
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

template <int N>
void WriteMeta(ffi::TensorView meta, int offset, const liger_cute::TensorView<N>& view) {
  int64_t* data = static_cast<int64_t*>(meta.data_ptr());
  data[offset] = reinterpret_cast<int64_t>(view.data);
  for (int i = 0; i < N; ++i) {
    data[offset + 1 + i] = view.sizes[i];
  }
  data[offset + 1 + N] = static_cast<int64_t>(view.dtype);
}

template <int N>
liger_cute::TensorView<N> ReadMeta(ffi::TensorView meta, int offset) {
  const int64_t* data = static_cast<const int64_t*>(meta.data_ptr());
  liger_cute::TensorView<N> out;
  out.data = reinterpret_cast<void*>(data[offset]);
  for (int i = 0; i < N; ++i) {
    out.sizes[i] = data[offset + 1 + i];
  }
  out.dtype = static_cast<liger_cute_dtype_t>(data[offset + 1 + N]);
  return out;
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
  RequireCpuInt64(symm_meta, 13);
  auto Y_v = ToTensorView<2>(Y);
  auto slots_v = ToTensorView<1>(token_expert_slots);
  auto tiles_v = ToTensorView<1>(tile_expert_ids);
  liger_cute::TensorView<2> x_sorted_v{};
  liger_cute::TensorView<2> y_buf_v{};
  liger_cute::TensorView<2> offsets_v{};
  int chosen_tile_m = 0;
  int64_t stream_handle =
      reinterpret_cast<int64_t>(TVMFFIEnvGetStream(X.device().device_type, X.device().device_id));
  CheckStatus(
      liger_cute_moe_fused_fwd_bf16_auto(
          ToTensorView<2>(X), ToTensorView<2>(expert_indices), ToTensorView<2>(expert_weights),
          ToTensorView<3>(all_B), ToTensorView<3>(all_C), ToTensorView<3>(all_A),
          static_cast<int>(num_experts), static_cast<int>(top_k), team_handle, stream_handle,
          &Y_v, &slots_v, &tiles_v, &x_sorted_v, &y_buf_v, &offsets_v, &chosen_tile_m),
      "moe_fused_fwd_bf16");
  WriteMeta<2>(symm_meta, 0, x_sorted_v);
  WriteMeta<2>(symm_meta, 4, y_buf_v);
  WriteMeta<2>(symm_meta, 8, offsets_v);
  static_cast<int64_t*>(symm_meta.data_ptr())[12] = chosen_tile_m;
}

void moe_fused_bwd_bf16(
    ffi::TensorView dY, ffi::TensorView symm_meta, ffi::TensorView token_expert_slots,
    ffi::TensorView tile_expert_ids, ffi::TensorView expert_indices, ffi::TensorView expert_weights,
    ffi::TensorView all_B, ffi::TensorView all_C, ffi::TensorView all_A, int64_t num_experts,
    int64_t top_k, int64_t team_handle, ffi::TensorView dX, ffi::TensorView dB,
    ffi::TensorView dC, ffi::TensorView dA, ffi::TensorView dW) {
  RequireCpuInt64(symm_meta, 13);
  int64_t stream_handle =
      reinterpret_cast<int64_t>(TVMFFIEnvGetStream(dY.device().device_type, dY.device().device_id));
  auto dX_v = ToTensorView<2>(dX);
  auto dB_v = ToTensorView<3>(dB);
  auto dC_v = ToTensorView<3>(dC);
  auto dA_v = ToTensorView<3>(dA);
  auto dW_v = ToTensorView<2>(dW);
  CheckStatus(
      liger_cute_moe_fused_bwd_bf16_auto(
          ToTensorView<2>(dY), ReadMeta<2>(symm_meta, 4), ReadMeta<2>(symm_meta, 0),
          ToTensorView<1>(token_expert_slots), ToTensorView<1>(tile_expert_ids),
          ReadMeta<2>(symm_meta, 8), ToTensorView<2>(expert_indices), ToTensorView<2>(expert_weights),
          ToTensorView<3>(all_B), ToTensorView<3>(all_C), ToTensorView<3>(all_A),
          static_cast<int>(num_experts), static_cast<int>(top_k), team_handle, stream_handle,
          static_cast<int>(static_cast<const int64_t*>(symm_meta.data_ptr())[12]),
          &dX_v, &dB_v, &dC_v, &dA_v, &dW_v),
      "moe_fused_bwd_bf16");
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(uniqueid_nbytes, uniqueid_nbytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_uniqueid, get_uniqueid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_with_uniqueid, init_with_uniqueid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_pmi, init_pmi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(finalize, finalize);
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
