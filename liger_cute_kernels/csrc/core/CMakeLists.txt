# ─────────────────────────────────────────────────────────────────────────────
# liger_cute_kernels — torch-free CUTLASS + NVSHMEM core.
#
# Built ONCE; reused by every torch wheel. ABI-agnostic because:
#   * its public boundary (include/liger_cute/liger_cute.h) is flat `extern "C"`;
#   * symbols default to hidden, so CuTe/libstdc++ internals stay local;
#   * a linker version script exports only the liger_cute_* C API;
#   * libstdc++/libgcc are linked statically, so the core's internal CXX11 ABI
#     choice is invisible to whatever torch the binding was built against.
# ─────────────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE LIGER_CUTE_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
# src/moe/tune/ holds the standalone offline autotuner (tune_moe_fwd_bwd) — its
# own CMake project, a torch-linking executable that is NOT a core lib source.
# The recursive glob above would otherwise sweep it into the torch-free core and
# break the build on its `#include <torch/torch.h>`. Keep it out.
list(FILTER LIGER_CUTE_CORE_SOURCES EXCLUDE REGEX "/src/moe/tune/")
# The legacy TensorView ABI is no longer part of libliger_cute_kernels.so.
# Tensor-carrying entry points are exported through TVM FFI.
list(FILTER LIGER_CUTE_CORE_SOURCES EXCLUDE REGEX "/src/tensor_view\\.cpp$")
# Whole-program external module source; including it in the core would duplicate
# the ordinary moe.cu / moe_bwd.cu definitions.
list(FILTER LIGER_CUTE_CORE_SOURCES EXCLUDE REGEX "/src/moe/moe_nonrdc_sm90\\.cu$")
set(LIGER_CUTE_NONRDC_MOE_MODULE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/moe_nonrdc_module.cpp")
list(REMOVE_ITEM LIGER_CUTE_CORE_SOURCES
    "${LIGER_CUTE_NONRDC_MOE_MODULE_SOURCE}")
if(LIGER_CUTE_ENABLE_SM90_NONRDC_MOE)
    list(APPEND LIGER_CUTE_CORE_SOURCES
        "${LIGER_CUTE_NONRDC_MOE_MODULE_SOURCE}")
endif()

# SM90 WGMMA/TMA translation units stay non-RDC. SM100 inlines the warp-scoped
# NVSHMEM forward ring, but the installed NVSHMEM ABI exposes device-state
# symbols only to RDC compilation; the target is switched below only for that
# architecture. Generic large-payload collectives remain isolated in
# detail/remote_all_reduce.cu.
set(LIGER_CUTE_FSLCE_FORWARD_SM90_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_forward.cu")
set(LIGER_CUTE_FSLCE_FORWARD_SM100_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_forward_sm100.cu")
set(LIGER_CUTE_FSLCE_FORWARD_REDUCE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/forward_reduce.cu")
set(LIGER_CUTE_FSLCE_BACKWARD_SM90_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_backward.cu")
set(LIGER_CUTE_FSLCE_BACKWARD_SM100_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_backward_sm100.cu")
set(LIGER_CUTE_FSLCE_BACKWARD_BENCH_SM100_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_backward_bench_sm100.cu")
set(LIGER_CUTE_FSLCE_BACKWARD_REDUCE_SM90_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/backward_reduce_sm90.cu")
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/workspace.cu"
    PROPERTIES COMPILE_DEFINITIONS
    "LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS=${LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS};LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT=${LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT}")
list(REMOVE_ITEM LIGER_CUTE_CORE_SOURCES
    "${LIGER_CUTE_FSLCE_FORWARD_SM90_SOURCE}"
    "${LIGER_CUTE_FSLCE_FORWARD_SM100_SOURCE}"
    "${LIGER_CUTE_FSLCE_FORWARD_REDUCE_SOURCE}"
    "${LIGER_CUTE_FSLCE_BACKWARD_SM90_SOURCE}"
    "${LIGER_CUTE_FSLCE_BACKWARD_SM100_SOURCE}"
    "${LIGER_CUTE_FSLCE_BACKWARD_BENCH_SM100_SOURCE}"
    "${LIGER_CUTE_FSLCE_BACKWARD_REDUCE_SM90_SOURCE}")

file(GLOB_RECURSE LIGER_CUTE_CORE_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)
list(FILTER LIGER_CUTE_CORE_HEADERS EXCLUDE REGEX "/include/liger_cute/tensor_view\\.h$")

set(LIGER_CUTE_SM90_NONRDC_MOE_CUBIN
    "${CMAKE_CURRENT_BINARY_DIR}/liger_moe_sm90_nonrdc.cubin")
if(LIGER_CUTE_ENABLE_SM90_NONRDC_MOE)
    if(NOT LIGER_CUTE_CUDA_ARCH STREQUAL "90a")
        message(FATAL_ERROR
            "LIGER_CUTE_ENABLE_SM90_NONRDC_MOE requires "
            "LIGER_CUTE_CUDA_ARCH=90a")
    endif()
    if(CUDAToolkit_VERSION VERSION_LESS "12.4")
        message(FATAL_ERROR
            "The SM90 non-RDC MoE path requires CUDA 12.4+ for "
            "cudaFuncGetName and cudaFuncGetParamInfo")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    find_program(LIGER_CUTE_PTXAS ptxas
        HINTS "${CUDAToolkit_BIN_DIR}" REQUIRED)

    set(_nonrdc_moe_source
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/moe_nonrdc_sm90.cu")
    set(_nonrdc_moe_translation_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/moe.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/moe_bwd.cu")
    set(_nonrdc_moe_preinclude
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/nonrdc_nvshmem_module_sm90.cuh")
    set(_nonrdc_moe_ptx
        "${CMAKE_CURRENT_BINARY_DIR}/liger_moe_sm90_nonrdc.ptx")
    set(_nonrdc_moe_visible_ptx
        "${CMAKE_CURRENT_BINARY_DIR}/liger_moe_sm90_nonrdc.visible.ptx")
    set(_nonrdc_moe_export_script
        "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/export_nonrdc_nvshmem_ptx.py")
    set(_nonrdc_moe_include_args
        "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/src/moe"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy"
        "-I${NVSHMEM_INCLUDE_DIR}")
    foreach(_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS CUTLASS_INCLUDE_DIRS)
        list(APPEND _nonrdc_moe_include_args "-I${_include_dir}")
    endforeach()
    file(GLOB_RECURSE _nonrdc_moe_dependencies
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/moe/*.inc")
    file(GLOB_RECURSE _nonrdc_nvshmem_headers
        "${NVSHMEM_INCLUDE_DIR}/*.h"
        "${NVSHMEM_INCLUDE_DIR}/*.cuh")
    list(APPEND _nonrdc_moe_dependencies ${_nonrdc_nvshmem_headers})
    if(LIGER_CUTE_SM90_NONRDC_ALL_CONFIGS)
        set(_nonrdc_moe_config_mode "all-configs")
        set(_nonrdc_moe_config_defines
            -DLIGER_CUTE_SM90_NONRDC_BENCHMARK_ALL_CONFIGS=1)
    else()
        set(_nonrdc_moe_config_mode "target-mixtral")
        set(_nonrdc_moe_config_defines
            -DLIGER_CUTE_SM90_NONRDC_TARGET_MIXTRAL=1)
    endif()
    set(_nonrdc_moe_fingerprint_material
        "sm90-nonrdc-moe-abi-v1:${_nonrdc_moe_config_mode}")
    set(_nonrdc_moe_fingerprint_files
        ${_nonrdc_moe_dependencies}
        "${_nonrdc_moe_source}"
        ${_nonrdc_moe_translation_sources}
        "${_nonrdc_moe_export_script}")
    foreach(_fingerprint_file IN LISTS _nonrdc_moe_fingerprint_files)
        file(SHA256 "${_fingerprint_file}" _fingerprint_file_hash)
        string(APPEND _nonrdc_moe_fingerprint_material
            ":${_fingerprint_file_hash}")
    endforeach()
    string(SHA256 _nonrdc_moe_fingerprint
        "${_nonrdc_moe_fingerprint_material}")
    string(SUBSTRING "${_nonrdc_moe_fingerprint}" 0 16
        _nonrdc_moe_fingerprint_u64)
    set(_nonrdc_moe_fingerprint_literal
        "0x${_nonrdc_moe_fingerprint_u64}ULL")

    add_custom_command(
        OUTPUT "${LIGER_CUTE_SM90_NONRDC_MOE_CUBIN}"
        COMMAND "${CMAKE_CUDA_COMPILER}"
            -std=c++17
            --ptx
            --generate-line-info
            --generate-code=arch=compute_90a,code=compute_90a
            --use_fast_math
            --extra-device-vectorization
            --fmad=true
            --prec-div=false
            --prec-sqrt=false
            -diag-suppress=20036
            -O3
            -DNDEBUG
            -DLIGER_CUTE_BUILDING=1
            -DLIGER_CUTE_DISPATCH_COMPUTE=90
            -DLIGER_CUTE_SM90_NONRDC_SPLIT=1
            -DLIGER_CUTE_SM90_NONRDC_DUAL_TRANSPORT=1
            ${_nonrdc_moe_config_defines}
            "-DLIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT=${_nonrdc_moe_fingerprint_literal}"
            ${_nonrdc_moe_include_args}
            -include "${_nonrdc_moe_preinclude}"
            "${_nonrdc_moe_source}"
            -o "${_nonrdc_moe_ptx}"
        COMMAND "${Python3_EXECUTABLE}"
            "${_nonrdc_moe_export_script}"
            "${_nonrdc_moe_ptx}"
            "${_nonrdc_moe_visible_ptx}"
        COMMAND "${LIGER_CUTE_PTXAS}"
            -O3
            --allow-expensive-optimizations=true
            --generate-line-info
            --warning-as-error
            --gpu-name=sm_90a
            "${_nonrdc_moe_visible_ptx}"
            -o "${LIGER_CUTE_SM90_NONRDC_MOE_CUBIN}"
        DEPENDS
            "${_nonrdc_moe_source}"
            ${_nonrdc_moe_translation_sources}
            "${_nonrdc_moe_preinclude}"
            "${_nonrdc_moe_export_script}"
            ${_nonrdc_moe_dependencies}
        COMMAND_EXPAND_LISTS
        VERBATIM)

    add_custom_target(liger_cute_sm90_nonrdc_moe
        DEPENDS "${LIGER_CUTE_SM90_NONRDC_MOE_CUBIN}")
endif()

if(LIGER_CUTE_CUDA_ARCH MATCHES "^(100|103)")
    set(LIGER_CUTE_FSLCE_FORWARD_TARGET liger_cute_fslce_forward_sm100)
    set(LIGER_CUTE_FSLCE_FORWARD_SOURCE
        "${LIGER_CUTE_FSLCE_FORWARD_SM100_SOURCE}")
else()
    set(LIGER_CUTE_FSLCE_FORWARD_TARGET liger_cute_fslce_forward_sm90)
    set(LIGER_CUTE_FSLCE_FORWARD_SOURCE
        "${LIGER_CUTE_FSLCE_FORWARD_SM90_SOURCE}")
endif()

add_library(${LIGER_CUTE_FSLCE_FORWARD_TARGET} OBJECT
    "${LIGER_CUTE_FSLCE_FORWARD_SOURCE}")
set_target_properties(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
    LIGER_CUTE_BUILDING=1)
if(LIGER_CUTE_CUDA_ARCH MATCHES "^(100|103)")
    # The installed NVSHMEM headers expose their device-state declarations
    # only under __CUDACC_RDC__. Keep SM90 non-RDC, but compile the SM100
    # communication specialization relocatably and resolve it in the final
    # core device link.
    set_target_properties(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
        LIGER_CUTE_FSLCE_SM100_STAGES=${LIGER_CUTE_FSLCE_SM100_STAGES}
        LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES=${LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES}
        LIGER_CUTE_FSLCE_SM100_ENABLE_NVSHMEM=1)
    if(LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_DISABLE_REMOTE)
        target_compile_definitions(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
            LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_DISABLE_REMOTE=1)
    endif()
    if(LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_TIMESTAMPS)
        target_compile_definitions(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
            LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_TIMESTAMPS=1)
    endif()
    if(LIGER_CUTE_FSLCE_SM100_PIPELINED_RING_WAVES)
        target_compile_definitions(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
            LIGER_CUTE_FSLCE_SM100_PIPELINED_RING_WAVES=1)
    endif()
    if(LIGER_CUTE_FSLCE_SM100_USE_WARP_TEAM_COLLECTIVES)
        target_compile_definitions(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
            LIGER_CUTE_FSLCE_SM100_USE_WARP_TEAM_COLLECTIVES=1)
    endif()
endif()
target_include_directories(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
target_link_libraries(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
    CUTLASS::CUTLASS
    CUDA::cudart)
if(LIGER_CUTE_CUDA_ARCH MATCHES "^(100|103)")
    target_link_libraries(${LIGER_CUTE_FSLCE_FORWARD_TARGET} PRIVATE
        NVSHMEM::nvshmem_host
        NVSHMEM::nvshmem_device)
endif()

add_library(liger_cute_fslce_forward_reduce OBJECT
    "${LIGER_CUTE_FSLCE_FORWARD_REDUCE_SOURCE}")
set_target_properties(liger_cute_fslce_forward_reduce PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(liger_cute_fslce_forward_reduce PRIVATE
    LIGER_CUTE_BUILDING=1)
target_compile_options(liger_cute_fslce_forward_reduce PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-UNVSHMEM_ENABLE_ALL_DEVICE_INLINING>)
target_include_directories(liger_cute_fslce_forward_reduce PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
target_link_libraries(liger_cute_fslce_forward_reduce PRIVATE
    CUTLASS::CUTLASS
    NVSHMEM::nvshmem_host
    NVSHMEM::nvshmem_device
    CUDA::cudart
    CUDA::cuda_driver)

add_library(liger_cute_fslce_backward_sm90 OBJECT
    "${LIGER_CUTE_FSLCE_BACKWARD_SM90_SOURCE}")
set_target_properties(liger_cute_fslce_backward_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(liger_cute_fslce_backward_sm90 PRIVATE
    LIGER_CUTE_BUILDING=1)
target_include_directories(liger_cute_fslce_backward_sm90 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
target_link_libraries(liger_cute_fslce_backward_sm90 PRIVATE
    CUTLASS::CUTLASS
    NVSHMEM::nvshmem_host
    CUDA::cudart
    CUDA::cuda_driver)

add_library(liger_cute_fslce_backward_reduce_sm90 OBJECT
    "${LIGER_CUTE_FSLCE_BACKWARD_REDUCE_SM90_SOURCE}")
set_target_properties(liger_cute_fslce_backward_reduce_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(
    liger_cute_fslce_backward_reduce_sm90 PRIVATE
    LIGER_CUTE_BUILDING=1)
target_include_directories(
    liger_cute_fslce_backward_reduce_sm90 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
target_link_libraries(
    liger_cute_fslce_backward_reduce_sm90 PRIVATE
    CUTLASS::CUTLASS
    NVSHMEM::nvshmem_host
    CUDA::cudart
    CUDA::cuda_driver)

# The fused SM100 backward is a single persistent cluster-2 kernel. Like the
# SM100 forward it inlines the warp-scoped NVSHMEM ring, and the installed
# NVSHMEM ABI exposes its device-state symbols only to RDC compilation, so this
# translation unit alone is compiled relocatably and resolved in the final core
# device link. SM90 stays non-RDC.
set(LIGER_CUTE_FSLCE_BACKWARD_SM100_OBJECTS "")
if(LIGER_CUTE_CUDA_ARCH MATCHES "^(100|103)")
    add_library(liger_cute_fslce_backward_sm100 OBJECT
        "${LIGER_CUTE_FSLCE_BACKWARD_SM100_SOURCE}")
    set_target_properties(liger_cute_fslce_backward_sm100 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    target_compile_definitions(liger_cute_fslce_backward_sm100 PRIVATE
        LIGER_CUTE_BUILDING=1
        LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES=${LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES}
        LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS=${LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS}
        LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT=${LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT}
        LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM=1)
    if(LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_CHUNK_PIPELINE)
        target_compile_definitions(liger_cute_fslce_backward_sm100 PRIVATE
            LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_CHUNK_PIPELINE=1)
    endif()
    if(LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS)
        target_compile_definitions(liger_cute_fslce_backward_sm100 PRIVATE
            LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS=1)
    endif()
    target_include_directories(liger_cute_fslce_backward_sm100 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
    target_link_libraries(liger_cute_fslce_backward_sm100 PRIVATE
        CUTLASS::CUTLASS
        NVSHMEM::nvshmem_host
        NVSHMEM::nvshmem_device
        CUDA::cudart
        CUDA::cuda_driver)
    # Benchmark-only per-phase launch paths. Same kernel template, one phase
    # bit each. It reaches the pooled symmetric allocator through
    # buffer_pool.cuh, whose NVSHMEM device-state declarations are only visible
    # under RDC, so this target matches the production one.
    add_library(liger_cute_fslce_backward_bench_sm100 OBJECT
        "${LIGER_CUTE_FSLCE_BACKWARD_BENCH_SM100_SOURCE}")
    set_target_properties(liger_cute_fslce_backward_bench_sm100 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    target_compile_definitions(liger_cute_fslce_backward_bench_sm100 PRIVATE
        LIGER_CUTE_BUILDING=1
        LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES=${LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES}
        LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS=${LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS}
        LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT=${LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT})
    if(LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS)
        target_compile_definitions(liger_cute_fslce_backward_bench_sm100 PRIVATE
            LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS=1)
    endif()
    target_include_directories(liger_cute_fslce_backward_bench_sm100 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
    target_link_libraries(liger_cute_fslce_backward_bench_sm100 PRIVATE
        CUTLASS::CUTLASS
        NVSHMEM::nvshmem_host
        NVSHMEM::nvshmem_device
        CUDA::cudart
        CUDA::cuda_driver)
    set(LIGER_CUTE_FSLCE_BACKWARD_SM100_OBJECTS
        "$<TARGET_OBJECTS:liger_cute_fslce_backward_sm100>"
        "$<TARGET_OBJECTS:liger_cute_fslce_backward_bench_sm100>")
endif()

add_library(liger_cute_kernels SHARED
    ${LIGER_CUTE_CORE_SOURCES}
    ${LIGER_CUTE_CORE_HEADERS}
    $<TARGET_OBJECTS:${LIGER_CUTE_FSLCE_FORWARD_TARGET}>
    $<TARGET_OBJECTS:liger_cute_fslce_forward_reduce>
    $<TARGET_OBJECTS:liger_cute_fslce_backward_sm90>
    $<TARGET_OBJECTS:liger_cute_fslce_backward_reduce_sm90>
    ${LIGER_CUTE_FSLCE_BACKWARD_SM100_OBJECTS}
    "${CMAKE_CURRENT_SOURCE_DIR}/../../liger_cute_kernels/tvm_ffi_bindings.cpp"
)

# Produces libliger_cute_kernels.so (kept the lib prefix for `-l` linking).
set_target_properties(liger_cute_kernels PROPERTIES
    OUTPUT_NAME liger_cute_kernels
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # Hide everything by default; only LIGER_CUTE_API-tagged symbols are exported.
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(liger_cute_kernels PRIVATE LIGER_CUTE_BUILDING=1)
if(LIGER_CUTE_ENABLE_SM90_NONRDC_MOE)
    add_dependencies(liger_cute_kernels liger_cute_sm90_nonrdc_moe)
    target_compile_definitions(liger_cute_kernels PRIVATE
        LIGER_CUTE_HAS_SM90_NONRDC_MOE=1
        LIGER_CUTE_SM90_NONRDC_SPLIT=1
        LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT=${_nonrdc_moe_fingerprint_literal}
        LIGER_CUTE_SM90_NONRDC_MOE_CUBIN_NAME="liger_moe_sm90_nonrdc.cubin")
    if(LIGER_CUTE_SM90_NONRDC_ALL_CONFIGS)
        target_compile_definitions(liger_cute_kernels PRIVATE
            LIGER_CUTE_SM90_NONRDC_BENCHMARK_ALL_CONFIGS=1)
    endif()
endif()
target_compile_options(liger_cute_kernels PRIVATE ${TVM_FFI_CFLAGS})
target_include_directories(liger_cute_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy
)

# Deliberately NO torch here — that is the entire point of the split.
# CUTLASS (header-only) and NVSHMEM bring their include dirs in via their
# imported INTERFACE targets, so no raw ${CUTLASS_HOME}/${NVSHMEM_HOME} paths
# are needed and the build works even when those env vars aren't exported.
target_link_libraries(liger_cute_kernels PUBLIC
    CUTLASS::CUTLASS
    NVSHMEM::nvshmem_host
    NVSHMEM::nvshmem_device
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
    ${TVM_FFI_LDFLAGS}
    tvm_ffi
)

# ── ABI-agnostic hardening (Linux/ELF) ───────────────────────────────────────
option(LIGER_CUTE_STATIC_LIBSTDCXX
    "Statically link libstdc++/libgcc into the core so its internal C++ ABI is \
invisible to consumers. Recommended ON for portable wheels." ON)

if(UNIX AND NOT APPLE)
    # Export only the liger_cute_* C API; localize all C++/CuTe/libstdc++ guts.
    target_link_options(liger_cute_kernels PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/liger_cute.version"
    )
    if(LIGER_CUTE_STATIC_LIBSTDCXX)
        # The FINAL shared-library link is performed by the host compiler (g++)
        # directly, while the device-link step is nvcc. So pass the BARE host
        # driver flags (no -Xcompiler) and scope them to the host link only via
        # $<HOST_LINK:>; without that scoping the flags reach only nvcc's
        # device-link and libstdc++ stays a dynamic dependency.
        target_link_options(liger_cute_kernels PRIVATE
            "$<HOST_LINK:-static-libstdc++>"
            "$<HOST_LINK:-static-libgcc>"
        )
    endif()
endif()
