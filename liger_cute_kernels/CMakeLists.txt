# ─────────────────────────────────────────────────────────────────────────────
# LigerCute — native build harness (ported from LigerCommKernels)
#
# Target split:
#
#   1. liger_cute_kernels  (csrc/core)     — CUTLASS + NVSHMEM kernels.
#      *No torch.* Exposes a flat `extern "C"` ABI so the .so is ABI-agnostic
#      and can be linked into a binding built against any torch wheel. This is
#      the slow compile (CuTe templates) and is built ONCE.
#
#   2. TVM FFI exports are compiled into the same core .so.
#      No torch/pybind dependency and no separate shim library.
#
# Build the core alone (no torch needed) with:
#     cmake -S . -B build -DLIGER_CUTE_BUILD_BINDINGS=OFF
#     cmake --build build --target liger_cute_kernels
#
# The fused MoE kernels (csrc/core/src/moe/{moe,moe_bwd,mlp*}.cu) are wired into
# the core; csrc/core/src/moe/tune/ is a separate standalone autotuner project
# (see README "Offline autotuner"), excluded from the core target.
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.24)
project(LigerCute LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(LIGER_CUTE_BUILD_BINDINGS
    "Deprecated compatibility option. The lck wheel now exposes tensor APIs through TVM FFI only." OFF)

# Core-reuse knob: when set to a directory containing a prebuilt
# libliger_cute_kernels.so, the core is NOT compiled from source — it is linked
# as an imported library instead. This is what lets the expensive CUTLASS core
# compile happen ONCE (phase 3.1) and be reused across every torch in the wheel
# matrix. Empty = build the core from source.
set(LIGER_CUTE_CORE_IMPORTED_DIR "" CACHE PATH
    "Dir holding a prebuilt libliger_cute_kernels.so to link instead of building the core")

# Tests-only build: compile just the torch-free C++ unit tests (which include the
# moe device-function headers directly) against CUTLASS + CUDA + gtest. Skips
# NVSHMEM, TVM FFI, the core library, and the bindings entirely — none are needed
# to compile/run the header-only mlp1 kernels. This is how the mlp1 SM100 rewrite
# is built/tested in isolation (e.g. -DLIGER_CUTE_CUDA_ARCH=100a on Blackwell).
option(LIGER_CUTE_TESTS_ONLY
    "Build ONLY the C++ unit tests (CUTLASS+CUDA+gtest); skip NVSHMEM/TVM-FFI/core/bindings" OFF)

# ---------- Common CUDA / device dependencies (needed by core AND bindings) --
find_package(CUDAToolkit REQUIRED)

# The 'a' suffix enables architecture-accelerated features (WGMMA/TMA/multicast
# on sm_90a; tcgen05/UMMA on sm_100a). Override with -DLIGER_CUTE_CUDA_ARCH=100a
# for a Blackwell build. Strip any gencode flags a parent/toolchain may have
# injected, then set the requested arch exclusively.
set(LIGER_CUTE_CUDA_ARCH "90a" CACHE STRING
    "CUDA architecture for -gencode (e.g. 90a for Hopper, 100a for Blackwell)")
set(CMAKE_CUDA_ARCHITECTURES OFF)
string(REGEX REPLACE "-gencode[= ]+arch=compute_[0-9a-z]+,code=sm_[0-9a-z]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${LIGER_CUTE_CUDA_ARCH},code=sm_${LIGER_CUTE_CUDA_ARCH} --use_fast_math --extra-device-vectorization --fmad=true --prec-div=false --prec-sqrt=false --ptxas-options=-O3,--allow-expensive-optimizations=true")
if(LIGER_CUTE_CUDA_ARCH MATCHES "^90")
    add_compile_definitions(LIGER_CUTE_DISPATCH_COMPUTE=90)
elseif(LIGER_CUTE_CUDA_ARCH MATCHES "^100")
    add_compile_definitions(LIGER_CUTE_DISPATCH_COMPUTE=100)
else()
    add_compile_definitions(LIGER_CUTE_DISPATCH_COMPUTE=0)
endif()

# Disable device-side assertions to avoid extern __assert_fail calls that cause
# ptxas to serialize WGMMA instructions.
add_compile_definitions(NDEBUG)
# NVSHMEM_ENABLE_ALL_DEVICE_INLINING is REQUIRED globally — see LigerCommKernels
# root CMakeLists for the full rationale (nvshmem.h device helpers reference
# device-state symbols gated on __CUDACC_RDC__; inlining makes non-RDC host-only
# TUs build cleanly).
add_compile_definitions(NVSHMEM_ENABLE_ALL_DEVICE_INLINING=ON)

# ── NVSHMEM ──────────────────────────────────────────────────────────────────
# Skipped for a tests-only build: the gtest unit tests are NVSHMEM-free.
if(NOT LIGER_CUTE_TESTS_ONLY)
    if(NOT DEFINED NVSHMEM_HOME)
        if(DEFINED ENV{NVSHMEM_HOME})
            set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
        else()
            set(NVSHMEM_HOME "/usr/local/nvshmem")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${NVSHMEM_HOME}")
    find_package(NVSHMEM REQUIRED)
endif()

# ── CUTLASS (header-only) ────────────────────────────────────────────────────
# Needed to COMPILE the core from source AND to build the tests (which include
# the moe device headers directly). A prebuilt/imported core does not need it.
if(NOT LIGER_CUTE_CORE_IMPORTED_DIR)
    if(DEFINED ENV{CUTLASS_HOME})
        set(CUTLASS_HOME "$ENV{CUTLASS_HOME}")
        list(APPEND CMAKE_PREFIX_PATH "${CUTLASS_HOME}")
    endif()
    find_package(CUTLASS REQUIRED)
endif()

# TVM FFI headers/libs used by the in-core TVM FFI exports. Tests don't use it.
if(NOT LIGER_CUTE_TESTS_ONLY)
    find_program(TVM_FFI_CONFIG tvm-ffi-config REQUIRED)
    execute_process(
        COMMAND "${TVM_FFI_CONFIG}" --cflags
        OUTPUT_VARIABLE TVM_FFI_CFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${TVM_FFI_CONFIG}" --ldflags
        OUTPUT_VARIABLE TVM_FFI_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    separate_arguments(TVM_FFI_CFLAGS UNIX_COMMAND "${TVM_FFI_CFLAGS}")
    separate_arguments(TVM_FFI_LDFLAGS UNIX_COMMAND "${TVM_FFI_LDFLAGS}")
endif()

# ---------- Targets ----------
# The core library + bindings are skipped for a tests-only build.
if(NOT LIGER_CUTE_TESTS_ONLY)
if(LIGER_CUTE_CORE_IMPORTED_DIR)
    # Reuse a prebuilt core: define liger_cute_kernels as an IMPORTED target so
    # consumers link the cached .so directly and the core is never recompiled.
    set(_imported_core "${LIGER_CUTE_CORE_IMPORTED_DIR}/libliger_cute_kernels.so")
    if(NOT EXISTS "${_imported_core}")
        message(FATAL_ERROR "LIGER_CUTE_CORE_IMPORTED_DIR set but no core at ${_imported_core}")
    endif()
    add_library(liger_cute_kernels SHARED IMPORTED GLOBAL)
    set_target_properties(liger_cute_kernels PROPERTIES
        IMPORTED_LOCATION "${_imported_core}"
        # Allow dependents to resolve the core's NVSHMEM/CUDA deps.
        INTERFACE_LINK_LIBRARIES "NVSHMEM::nvshmem_host;CUDA::cudart;CUDA::cuda_driver"
    )
    target_include_directories(liger_cute_kernels INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/core/include")
    message(STATUS "Using prebuilt core: ${_imported_core}")
else()
    # Build the torch-free core from source.
    add_subdirectory(csrc/core)
endif()

if(LIGER_CUTE_BUILD_BINDINGS)
    message(FATAL_ERROR "LIGER_CUTE_BUILD_BINDINGS is deprecated. Tensor APIs are exposed through TVM FFI only.")
endif()
endif()  # NOT LIGER_CUTE_TESTS_ONLY

# C++ unit tests are opt-in (require gtest). Torch-free; they include the moe
# device headers directly and need only CUTLASS (found above), CUDA, and gtest —
# not the core library or NVSHMEM. Combine with LIGER_CUTE_TESTS_ONLY for a
# standalone test build (e.g. the mlp1 SM100 rewrite on -DLIGER_CUTE_CUDA_ARCH=100a).
option(LIGER_CUTE_BUILD_TESTS "Build the C++ (gtest) unit tests" OFF)
if(LIGER_CUTE_BUILD_TESTS)
    if(LIGER_CUTE_CORE_IMPORTED_DIR)
        message(FATAL_ERROR
            "LIGER_CUTE_BUILD_TESTS requires CUTLASS headers; unset "
            "LIGER_CUTE_CORE_IMPORTED_DIR.")
    endif()
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
