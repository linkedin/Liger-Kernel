// nvshmem.cu — NVSHMEM bootstrap / team / pool entry points.
//
// Implements the flat extern "C" boundary declared in nvshmem.h. This is the
// ONLY core TU (besides the eventual kernels) that includes <nvshmem.h>; the
// flat header keeps the device runtime off the binding's include path. Every
// entry point runs through liger_cute::detail::guarded() so a failed
// LIGER_CHECK / a non-zero NVSHMEM return code becomes a liger_cute_status_t +
// a last-error string instead of unwinding across the .so boundary.
//
// Adapted from LigerCuteKernels' NVSHMEM helpers; the throw-based error model
// (std::runtime_error) becomes the status-code model, and the explicit
// clear_global_pools()/finalize ordering is preserved.
#define LIGER_CUTE_BUILDING 1

#include "liger_cute/nvshmem.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstring>

#include "liger_cute/check.h"
#include "liger_cute/detail/status.h"
#include "liger_cute/detail/symmetric_memory.h"
#include "fused_scaled_linear_cross_entropy/state.h"
#include "moe_nonrdc_module.h"
#include "moe_context.h"

namespace liger_cute {
namespace detail {

// Drain the caches that own NVSHMEM allocations. Must run BEFORE
// nvshmem_finalize(): otherwise the singleton destructors run at program exit,
// after NVSHMEM has been torn down, and nvshmem_free aborts with "API called
// before NVSHMEM initialization has completed". std::map iteration order is
// deterministic across PEs so the collective nvshmem_free calls stay in
// lockstep.
void clear_global_pools() {
  liger::reset_moe_configuration();
  liger::fused_scaled_linear_cross_entropy::reset_fslce_tp_configuration();
  global_symmetric_stack().clear();
  global_buffer_pool().clear();
}

// Cast an int64 handle back to an nvshmem_team_t in one place.
inline nvshmem_team_t as_team(int64_t handle) {
  return static_cast<nvshmem_team_t>(handle);
}

}  // namespace detail
}  // namespace liger_cute

extern "C" {

// ─── Bootstrap / init ────────────────────────────────────────────────────────

liger_cute_status_t liger_cute_nvshmem_uniqueid_nbytes(size_t* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_uniqueid_nbytes: out is null");
    *out = sizeof(nvshmemx_uniqueid_t);
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_get_uniqueid(void* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_get_uniqueid: out is null");
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    int rc = nvshmemx_get_uniqueid(&id);
    if (rc != 0) {
      LIGER_FAIL_NVSHMEM("nvshmemx_get_uniqueid failed (rc=", rc, ")");
    }
    std::memcpy(out, &id, sizeof(id));
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_init_with_uniqueid(int rank, int nranks,
                                                          const void* uid) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(uid != nullptr, "nvshmem_init_with_uniqueid: uid is null");
    LIGER_CHECK(nranks > 0 && rank >= 0 && rank < nranks,
                "nvshmem_init_with_uniqueid: rank (", rank, ") out of range [0, ", nranks,
                ")");
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    std::memcpy(&id, uid, sizeof(id));

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int rc = nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
    if (rc != 0) {
      LIGER_FAIL_NVSHMEM("nvshmemx_set_attr_uniqueid_args failed (rc=", rc, ")");
    }
    rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (rc != 0) {
      LIGER_FAIL_NVSHMEM("nvshmemx_init_attr(UNIQUEID) failed (rc=", rc, ")");
    }
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_init_pmi(void) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    // Mirror of the static-inline nvshmem_init() in <host/nvshmem_api.h>, for
    // PMI-launched (srun/mpirun) runs.
    nvshmem_init();
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_finalize(void) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    liger_cute::detail::clear_global_pools();  // BEFORE finalize — see header
    liger::finalize_sm90_nonrdc_moe();
    nvshmem_finalize();
    return LIGER_CUTE_OK;
  });
}

// ─── PE queries (world) ──────────────────────────────────────────────────────

liger_cute_status_t liger_cute_nvshmem_my_pe(int* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_my_pe: out is null");
    *out = nvshmem_my_pe();
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_n_pes(int* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_n_pes: out is null");
    *out = nvshmem_n_pes();
    return LIGER_CUTE_OK;
  });
}

// ─── Teams ───────────────────────────────────────────────────────────────────

liger_cute_status_t liger_cute_nvshmem_team_world(int64_t* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_team_world: out is null");
    *out = static_cast<int64_t>(NVSHMEM_TEAM_WORLD);
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_team_split_strided(int64_t parent_handle, int start,
                                                          int stride, int size, int64_t* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_team_split_strided: out is null");
    nvshmem_team_t parent = liger_cute::detail::as_team(parent_handle);
    nvshmem_team_t team = NVSHMEM_TEAM_INVALID;
    nvshmem_team_config_t cfg{};
    int rc = nvshmem_team_split_strided(parent, start, stride, size, &cfg, 0, &team);
    if (rc != 0) {
      LIGER_FAIL_NVSHMEM("nvshmem_team_split_strided failed (rc=", rc, ")");
    }
    // Non-member ranks get NVSHMEM_TEAM_INVALID — surface it as -1.
    *out = (team == NVSHMEM_TEAM_INVALID) ? -1 : static_cast<int64_t>(team);
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_team_destroy(int64_t team_handle) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    nvshmem_team_t team = liger_cute::detail::as_team(team_handle);
    // Never destroy the predefined world team or an invalid handle.
    if (team == NVSHMEM_TEAM_WORLD || team == NVSHMEM_TEAM_INVALID) {
      return LIGER_CUTE_OK;
    }
    liger::release_moe_team(team_handle);
    liger::fused_scaled_linear_cross_entropy::release_fslce_tp_team(team_handle);
    nvshmem_team_destroy(team);
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_team_my_pe(int64_t team_handle, int* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_team_my_pe: out is null");
    *out = nvshmem_team_my_pe(liger_cute::detail::as_team(team_handle));
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_team_n_pes(int64_t team_handle, int* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_team_n_pes: out is null");
    *out = nvshmem_team_n_pes(liger_cute::detail::as_team(team_handle));
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_nvshmem_team_translate_pe(int64_t src_team_handle, int src_pe,
                                                         int64_t dst_team_handle, int* out) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "nvshmem_team_translate_pe: out is null");
    *out = nvshmem_team_translate_pe(liger_cute::detail::as_team(src_team_handle), src_pe,
                                     liger_cute::detail::as_team(dst_team_handle));
    return LIGER_CUTE_OK;
  });
}

// ─── Symmetric / device memory pools ─────────────────────────────────────────

liger_cute_status_t liger_cute_pool_clear_all(void) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    liger_cute::detail::clear_global_pools();
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_pool_clear_buffers(void) {
  return liger_cute::detail::guarded([&]() -> liger_cute_status_t {
    liger::fused_scaled_linear_cross_entropy::reset_fslce_tp_configuration();
    liger::reset_moe_configuration(false);
    liger_cute::detail::global_buffer_pool().clear();
    return LIGER_CUTE_OK;
  });
}

}  // extern "C"
