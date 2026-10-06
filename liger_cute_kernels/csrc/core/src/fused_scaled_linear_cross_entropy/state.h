#pragma once
#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

// Invalidates cached FSLCE capacities, mappings, and duplicated teams before
// the shared BufferPool is cleared or NVSHMEM is finalized.
void reset_fslce_tp_configuration();
void release_fslce_tp_team(std::int64_t team_handle);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
