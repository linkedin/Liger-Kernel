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
file(GLOB_RECURSE LIGER_CUTE_CORE_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

add_library(liger_cute_kernels SHARED
    ${LIGER_CUTE_CORE_SOURCES}
    ${LIGER_CUTE_CORE_HEADERS}
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

target_include_directories(liger_cute_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
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
