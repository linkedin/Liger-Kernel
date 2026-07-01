# ═══════════════════════════════════════════════════════════════════════════════
# C++ unit tests (gtest). Torch-free: tests include the device-function headers
# from csrc/core/src/moe directly and define their own launcher kernels, so they
# only need CUTLASS (header-only), CUDA, and gtest — not libliger_cute_kernels
# or NVSHMEM.
#
# Enabled via -DLIGER_CUTE_BUILD_TESTS=ON. Global CMAKE_CUDA_FLAGS already sets
# the sm_90a gencode required by the WGMMA/TMA device functions.
# ═══════════════════════════════════════════════════════════════════════════════

find_package(GTest REQUIRED)
include(GoogleTest)

# CUTLASS::CUTLASS is defined when the core is built from source (top-level
# find_package(CUTLASS)). Fall back to CUTLASS_HOME include dirs otherwise.
add_executable(test_mlp1_fused test_mlp1_fused.cu)
target_link_libraries(test_mlp1_fused PRIVATE
    GTest::gtest
    CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp1_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp1_fused SYSTEM PRIVATE
        "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp1_fused PRIVATE
    "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp1_fused DISCOVERY_MODE PRE_TEST)
