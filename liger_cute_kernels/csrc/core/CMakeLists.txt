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

# WGMMA/TMA translation units stay non-RDC. Remote NVSHMEM collectives are
# isolated in detail/remote_all_reduce.cu.
set(LIGER_CUTE_FSLCE_FORWARD_SM90_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_forward.cu")
set(LIGER_CUTE_FSLCE_FORWARD_REDUCE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/forward_reduce.cu")
set(LIGER_CUTE_FSLCE_BACKWARD_SM90_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/fused_linear_scaled_cross_entropy_backward.cu")
set(LIGER_CUTE_FSLCE_BACKWARD_FALLBACK_SM90_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy/backward_fallback_sm90.cu")
list(REMOVE_ITEM LIGER_CUTE_CORE_SOURCES
    "${LIGER_CUTE_FSLCE_FORWARD_SM90_SOURCE}"
    "${LIGER_CUTE_FSLCE_FORWARD_REDUCE_SOURCE}"
    "${LIGER_CUTE_FSLCE_BACKWARD_SM90_SOURCE}"
    "${LIGER_CUTE_FSLCE_BACKWARD_FALLBACK_SM90_SOURCE}")

file(GLOB_RECURSE LIGER_CUTE_CORE_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)
list(FILTER LIGER_CUTE_CORE_HEADERS EXCLUDE REGEX "/include/liger_cute/tensor_view\\.h$")

add_library(liger_cute_fslce_forward_sm90 OBJECT
    "${LIGER_CUTE_FSLCE_FORWARD_SM90_SOURCE}")
set_target_properties(liger_cute_fslce_forward_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(liger_cute_fslce_forward_sm90 PRIVATE
    LIGER_CUTE_BUILDING=1)
target_include_directories(liger_cute_fslce_forward_sm90 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy
    ${NCCL_INCLUDE_DIR})
target_link_libraries(liger_cute_fslce_forward_sm90 PRIVATE
    CUTLASS::CUTLASS
    CUDA::cudart
    ${NCCL_LIBRARY})

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

add_library(liger_cute_fslce_backward_fallback_sm90 OBJECT
    "${LIGER_CUTE_FSLCE_BACKWARD_FALLBACK_SM90_SOURCE}")
set_target_properties(liger_cute_fslce_backward_fallback_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(
    liger_cute_fslce_backward_fallback_sm90 PRIVATE
    LIGER_CUTE_BUILDING=1)
target_include_directories(
    liger_cute_fslce_backward_fallback_sm90 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy)
target_link_libraries(
    liger_cute_fslce_backward_fallback_sm90 PRIVATE
    CUTLASS::CUTLASS
    NVSHMEM::nvshmem_host
    CUDA::cudart
    CUDA::cuda_driver)

add_library(liger_cute_kernels SHARED
    ${LIGER_CUTE_CORE_SOURCES}
    ${LIGER_CUTE_CORE_HEADERS}
    $<TARGET_OBJECTS:liger_cute_fslce_forward_sm90>
    $<TARGET_OBJECTS:liger_cute_fslce_forward_reduce>
    $<TARGET_OBJECTS:liger_cute_fslce_backward_sm90>
    $<TARGET_OBJECTS:liger_cute_fslce_backward_fallback_sm90>
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
target_compile_options(liger_cute_kernels PRIVATE ${TVM_FFI_CFLAGS})
target_include_directories(liger_cute_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/moe
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_scaled_linear_cross_entropy
        ${NCCL_INCLUDE_DIR}
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
    ${NCCL_LIBRARY}
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
