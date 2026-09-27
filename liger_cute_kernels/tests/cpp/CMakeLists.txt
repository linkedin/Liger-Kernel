# ═══════════════════════════════════════════════════════════════════════════════
# C++ unit tests (gtest). Torch-free: tests include the device-function headers
# from csrc/core/src/moe directly and define their own launcher kernels, so they
# only need CUTLASS (header-only), CUDA, and gtest. The native core API dispatch
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

add_executable(
    test_fslce_shared_reduction
    test_fslce_shared_reduction.cu)
target_link_libraries(test_fslce_shared_reduction PRIVATE
    GTest::gtest
    CUDA::cudart)
target_include_directories(test_fslce_shared_reduction PRIVATE
    "${CMAKE_SOURCE_DIR}/csrc/core/include"
    "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
gtest_discover_tests(
    test_fslce_shared_reduction
    DISCOVERY_MODE PRE_TEST)

add_executable(
    test_fused_scaled_linear_cross_entropy
    test_fused_scaled_linear_cross_entropy.cu)
target_link_libraries(test_fused_scaled_linear_cross_entropy PRIVATE
    GTest::gtest
    CUTLASS::CUTLASS
    CUDA::cudart)
target_include_directories(test_fused_scaled_linear_cross_entropy PRIVATE
    "${CMAKE_SOURCE_DIR}/csrc/core/include"
    "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
gtest_discover_tests(
    test_fused_scaled_linear_cross_entropy
    DISCOVERY_MODE PRE_TEST)

add_executable(
    test_fused_scaled_linear_cross_entropy_sm100
    test_fused_scaled_linear_cross_entropy_sm100.cu)
target_link_libraries(
    test_fused_scaled_linear_cross_entropy_sm100 PRIVATE
    GTest::gtest
    CUTLASS::CUTLASS
    CUDA::cudart)
target_compile_definitions(
    test_fused_scaled_linear_cross_entropy_sm100 PRIVATE
    LIGER_CUTE_FSLCE_SM100_STAGES=${LIGER_CUTE_FSLCE_SM100_STAGES}
    LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES=${LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES})
target_include_directories(
    test_fused_scaled_linear_cross_entropy_sm100 PRIVATE
    "${CMAKE_SOURCE_DIR}/csrc/core/include"
    "${CMAKE_SOURCE_DIR}/csrc/core/src/moe"
    "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
gtest_discover_tests(
    test_fused_scaled_linear_cross_entropy_sm100
    DISCOVERY_MODE PRE_TEST)

# SM100 fused backward contract test. The executable mainloop (2SM UMMA / TMEM /
# paired TMA) only compiles for the Blackwell family, so the mainloop-dependent
# cases are macro guarded and enabled only on a 100/103 build.
add_executable(
    test_fused_scaled_linear_cross_entropy_backward_sm100
    test_fused_scaled_linear_cross_entropy_backward_sm100.cu)
target_link_libraries(
    test_fused_scaled_linear_cross_entropy_backward_sm100 PRIVATE
    GTest::gtest
    CUTLASS::CUTLASS
    CUDA::cudart)
target_compile_definitions(
    test_fused_scaled_linear_cross_entropy_backward_sm100 PRIVATE
    LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES=${LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES}
    LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS=${LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS}
    LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT=${LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT})
if(LIGER_CUTE_CUDA_ARCH MATCHES "^(100|103)")
    target_compile_definitions(
        test_fused_scaled_linear_cross_entropy_backward_sm100 PRIVATE
        LIGER_CUTE_FSLCE_SM100_BACKWARD_TEST_MAINLOOP=1)
endif()
target_include_directories(
    test_fused_scaled_linear_cross_entropy_backward_sm100 PRIVATE
    "${CMAKE_SOURCE_DIR}/csrc/core/include"
    "${CMAKE_SOURCE_DIR}/csrc/core/src/moe"
    "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
gtest_discover_tests(
    test_fused_scaled_linear_cross_entropy_backward_sm100
    DISCOVERY_MODE PRE_TEST)

if(LIGER_CUTE_CUDA_ARCH MATCHES "^(100|103)")
    add_executable(
        backward_dz_sm100_bench
        "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy/backward_dz_sm100_bench.cu")
    target_link_libraries(
        backward_dz_sm100_bench PRIVATE
        CUTLASS::CUTLASS
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt)
    target_compile_options(
        backward_dz_sm100_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    target_include_directories(
        backward_dz_sm100_bench PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/core/src/moe"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")

    add_executable(
        test_fslce_dx_gemm_sm100
        test_fslce_dx_gemm_sm100.cu)
    target_link_libraries(
        test_fslce_dx_gemm_sm100 PRIVATE
        GTest::gtest
        CUTLASS::CUTLASS
        CUDA::cudart)
    target_compile_options(
        test_fslce_dx_gemm_sm100 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    target_include_directories(
        test_fslce_dx_gemm_sm100 PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/core/src/moe"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
    gtest_discover_tests(
        test_fslce_dx_gemm_sm100
        DISCOVERY_MODE PRE_TEST)

    add_executable(
        benchmark_fslce_dx_cublas_sm100
        benchmark_fslce_dx_cublas_sm100.cu)
    target_link_libraries(
        benchmark_fslce_dx_cublas_sm100 PRIVATE
        CUTLASS::CUTLASS
        CUDA::cudart
        CUDA::cublasLt)

    add_executable(
        benchmark_fused_scaled_linear_cross_entropy_backward_gemm_sm100
        benchmark_fused_scaled_linear_cross_entropy_backward_gemm_sm100.cu)
    target_link_libraries(
        benchmark_fused_scaled_linear_cross_entropy_backward_gemm_sm100 PRIVATE
        CUTLASS::CUTLASS
        CUDA::cudart)
    target_include_directories(
        benchmark_fused_scaled_linear_cross_entropy_backward_gemm_sm100 PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/core/src/moe"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
    target_compile_options(
        benchmark_fused_scaled_linear_cross_entropy_backward_gemm_sm100 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    add_executable(
        test_fslce_dw_sm100
        test_fslce_dw_sm100.cu)
    target_link_libraries(
        test_fslce_dw_sm100 PRIVATE
        CUTLASS::CUTLASS
        CUDA::cudart
        CUDA::cublas)
    target_include_directories(
        test_fslce_dw_sm100 PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/core/include"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/moe"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")

    add_executable(
        test_fslce_dw_cutlass_sm100
        test_fslce_dw_cutlass_sm100.cu)
    target_link_libraries(
        test_fslce_dw_cutlass_sm100 PRIVATE
        CUTLASS::CUTLASS
        CUDA::cudart
        CUDA::cublas)
    target_include_directories(
        test_fslce_dw_cutlass_sm100 PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/core/include"
        "${CMAKE_SOURCE_DIR}/csrc/core/src/fused_scaled_linear_cross_entropy")
endif()

# ── Blackwell (SM100 family) port targets — pre-staged by the orchestrator so the
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
