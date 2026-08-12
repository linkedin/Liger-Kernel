// nvshmem_helpers.cuh — device-side comm-schedule shim for the ported MoE kernels.
//
// The upstream kernels (mlp_comms.cuh, ...) read the communication schedule as
// liger::g_dest_table / liger::g_rank_table / liger::comm_slot_of. In this repo
// those tables live in liger_cute::detail (csrc/core/include/liger_cute/detail/
// comm_schedule.cuh), populated by the flat-ABI init path. Rather than edit the
// large kernel headers, this shim aliases the canonical symbols into namespace
// liger so the upstream device code compiles unchanged and future re-syncs stay
// trivial. There is exactly ONE definition of each table (in nvshmem.cu); these
// using-declarations bind to it, so no symbol is duplicated.
#pragma once

#include "liger_cute/detail/comm_schedule.cuh"

namespace liger {

using liger_cute::detail::kMaxPEs;
using liger_cute::detail::g_dest_table;
using liger_cute::detail::g_rank_table;

#ifdef __CUDACC__
using liger_cute::detail::comm_slot_of;
#endif

}  // namespace liger
