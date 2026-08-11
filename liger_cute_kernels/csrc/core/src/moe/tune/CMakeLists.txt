# ─────────────────────────────────────────────────────────────────────────────
# tune_moe_fwd_bwd — standalone offline autotuner for the fused MoE fwd/bwd.
#
# NOT part of the wheel. It regenerates the tuned-config tables
#   ../moe_fwd_bwd_tuning_configs_{single,multi}.cuh
# that the runtime auto-dispatcher (find_nearest_tuned / _bwd) searches.
#
# It is its OWN executable (not a wheel artifact) because it must LINK the
# templated launchers liger::moe_fused_fwd_bf16<...> / moe_bwd_fwd_bf16_tuned<...>,
# which the core .so deliberately HIDES (visibility hidden + version script). So
# the tuner compiles the kernel sources itself with DEFAULT visibility. That is a
# full CuTe compile (~45 min for moe.cu + moe_bwd.cu) — build it on demand.
#
# Build (needs CUTLASS_HOME, NVSHMEM_HOME, and an importable torch):
#   cmake -S csrc/core/src/moe/tune -B build-tuner \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-tuner -j
#
# Run (one rank per GPU, PMI bootstrap — e.g. srun --mpi=pmi2 --ntasks=N):
#   LIGER_MOE_FWDBWD_TUNED_OUTPUT=/abs/path/to/moe_fwd_bwd_tuning_configs_multi.cuh \
#     srun --mpi=pmi2 --ntasks=8 ./build-tuner/tune_moe_fwd_bwd
#   (without the env override the .cuh is written to the current directory; cd to
#    csrc/core/src/moe/ first, or pass the absolute path.)
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.24)
project(LigerCuteTuner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Reuse the repo's Find modules (this dir is csrc/core/src/moe/tune).
set(_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
list(APPEND CMAKE_MODULE_PATH "${_REPO_ROOT}/cmake")

# ── Device dependencies (mirror the core build's flags) ──────────────────────
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES OFF)
string(REGEX REPLACE "-gencode[= ]+arch=compute_[0-9a-z]+,code=sm_[0-9a-z]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
set(LIGER_CUTE_CUDA_ARCH "90a" CACHE STRING
    "CUDA architecture for -gencode (e.g. 90a for Hopper, 100a for Blackwell)")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${LIGER_CUTE_CUDA_ARCH},code=sm_${LIGER_CUTE_CUDA_ARCH} --use_fast_math --extra-device-vectorization --fmad=true --prec-div=false --prec-sqrt=false --ptxas-options=-O3,--allow-expensive-optimizations=true")
if(LIGER_CUTE_CUDA_ARCH MATCHES "^90")
    add_compile_definitions(LIGER_CUTE_DISPATCH_COMPUTE=90)
elseif(LIGER_CUTE_CUDA_ARCH MATCHES "^100")
    add_compile_definitions(LIGER_CUTE_DISPATCH_COMPUTE=100)
else()
    add_compile_definitions(LIGER_CUTE_DISPATCH_COMPUTE=0)
endif()

add_compile_definitions(NDEBUG)
add_compile_definitions(NVSHMEM_ENABLE_ALL_DEVICE_INLINING=ON)

# ── NVSHMEM ──────────────────────────────────────────────────────────────────
if(NOT DEFINED NVSHMEM_HOME)
    if(DEFINED ENV{NVSHMEM_HOME})
        set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
    else()
        set(NVSHMEM_HOME "/usr/local/nvshmem")
    endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${NVSHMEM_HOME}")
find_package(NVSHMEM REQUIRED)

# ── CUTLASS (header-only) ────────────────────────────────────────────────────
if(DEFINED ENV{CUTLASS_HOME})
    set(CUTLASS_HOME "$ENV{CUTLASS_HOME}")
    list(APPEND CMAKE_PREFIX_PATH "${CUTLASS_HOME}")
endif()
find_package(CUTLASS REQUIRED)

# ── Torch (libtorch) — auto-detect cmake prefix from the installed wheel ─────
find_program(_python NAMES python3 python REQUIRED)
execute_process(
    COMMAND "${_python}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_torch_prefix)
    list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
endif()
find_package(Torch REQUIRED)

# Match the installed torch's C++ ABI for the WHOLE binary (the tuner TU uses
# torch; everything links into one exe, so all TUs need the same ABI).
execute_process(
    COMMAND "${_python}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
if(TORCH_CXX11_ABI MATCHES "^[01]$")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
endif()

# ── Sources: tuner TU + the kernel/support TUs (default visibility) ──────────
set(_MOE     "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(_CORESRC "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(tune_moe_fwd_bwd
    "${CMAKE_CURRENT_SOURCE_DIR}/tune_moe_fwd_bwd.cu"
    "${_MOE}/moe.cu"
    "${_MOE}/moe_bwd.cu"
    "${_CORESRC}/nvshmem.cu"
    "${_CORESRC}/liger_cute.cu"
)

set_target_properties(tune_moe_fwd_bwd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(tune_moe_fwd_bwd PRIVATE LIGER_CUTE_BUILDING=1)

target_include_directories(tune_moe_fwd_bwd PRIVATE
    "${_REPO_ROOT}/csrc/core/include"   # liger_cute/*.h
    "${_MOE}"                            # moe_launch.h, moe_fwd_bwd_tune_configs.hpp
)

target_link_libraries(tune_moe_fwd_bwd PRIVATE
    "${TORCH_LIBRARIES}"
    CUTLASS::CUTLASS
    NVSHMEM::nvshmem_host
    NVSHMEM::nvshmem_device
    CUDA::cudart
    CUDA::cuda_driver
)
