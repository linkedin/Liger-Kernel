# ═══════════════════════════════════════════════════════════════════════════════
# C++ unit tests (gtest). Torch-free: tests include the device-function headers
# from csrc/core/src/moe directly and define their own launcher kernels, so they
# only need CUTLASS (header-only), CUDA, and gtest. The lck.so API dispatch
# coverage test links the core library when it is built in this CMake run.
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

# ── Blackwell (SM100a) port targets — pre-staged by the orchestrator so the
# three parallel port subagents each build ONLY their own already-registered
# target and never edit this shared file. Each test .cu starts as a minimal stub
# (int main) and is overwritten with the real cloned test by its owning subagent.
add_executable(test_mlp2_fused test_mlp2_fused.cu)
target_link_libraries(test_mlp2_fused PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp2_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp2_fused SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp2_fused PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp2_fused DISCOVERY_MODE PRE_TEST)

add_executable(test_mlp2_t_fused test_mlp2_t_fused.cu)
target_link_libraries(test_mlp2_t_fused PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp2_t_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp2_t_fused SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp2_t_fused PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp2_t_fused DISCOVERY_MODE PRE_TEST)

add_executable(test_mlp5_fused test_mlp5_fused.cu)
target_link_libraries(test_mlp5_fused PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp5_fused PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp5_fused SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp5_fused PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp5_fused DISCOVERY_MODE PRE_TEST)

# ── mlp3 (dA = dYᵀ·Z) and mlp4 (dB = dUᵀ·X, dC = dVᵀ·X) Blackwell port targets.
# Pre-staged so the two parallel port pipelines each build ONLY their own
# already-registered target and never edit this shared file. Each .cu starts as a
# minimal stub and is overwritten with the real cloned test by its owning pipeline.
add_executable(test_mlp3 test_mlp3.cu)
target_link_libraries(test_mlp3 PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp3 PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp3 SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp3 PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp3 DISCOVERY_MODE PRE_TEST)

add_executable(test_mlp4 test_mlp4.cu)
target_link_libraries(test_mlp4 PRIVATE GTest::gtest CUDA::cudart)
if(TARGET CUTLASS::CUTLASS)
    target_link_libraries(test_mlp4 PRIVATE CUTLASS::CUTLASS)
else()
    target_include_directories(test_mlp4 SYSTEM PRIVATE "${CUTLASS_HOME}/include")
endif()
target_include_directories(test_mlp4 PRIVATE "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
gtest_discover_tests(test_mlp4 DISCOVERY_MODE PRE_TEST)

if(TARGET liger_cute_kernels)
    add_executable(test_moe_api_dispatch test_moe_api_dispatch.cpp)
    target_compile_options(test_moe_api_dispatch PRIVATE ${TVM_FFI_CFLAGS})
    target_compile_definitions(test_moe_api_dispatch PRIVATE
        LIGER_CUTE_TEST_MODULE_PATH=\"$<TARGET_FILE:liger_cute_kernels>\")
    target_link_libraries(test_moe_api_dispatch PRIVATE
        GTest::gtest
        CUDA::cudart
        liger_cute_kernels
        ${TVM_FFI_LDFLAGS}
        tvm_ffi)
    target_include_directories(test_moe_api_dispatch PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/core/include"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/moe")
    gtest_discover_tests(test_moe_api_dispatch DISCOVERY_MODE PRE_TEST)
endif()
