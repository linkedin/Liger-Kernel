// buffer_pool.cuh — infra shim for the ported MoE host launchers.
//
// Upstream moe.cu / moe_bwd.cu use liger::BufferPool / liger::global_buffer_pool
// and liger::SymmetricMemoryStack / liger::global_symmetric_stack. In this repo
// that machinery is the torch-free liger_cute::detail implementation in
// csrc/core/include/liger_cute/detail/symmetric_memory.h (LIGER_CHECK instead of
// throw/abort, otherwise API-identical). Alias it into namespace liger so the
// ported host code compiles unchanged and there is a single pool/stack singleton.
#pragma once

#include "liger_cute/detail/symmetric_memory.h"

namespace liger {

using liger_cute::detail::BufferPool;
using liger_cute::detail::global_buffer_pool;
using liger_cute::detail::SymmetricMemoryStack;
using liger_cute::detail::global_symmetric_stack;

}  // namespace liger
