# ─────────────────────────────────────────────────────────────────────────────
# _C — torch + pybind11 marshalling shim.
#
# Cheap to compile (no CUTLASS); rebuilt once per torch version in the wheel
# matrix. Unwraps torch::Tensor into the flat liger_cute_tensor_t POD and calls
# into the prebuilt, ABI-agnostic liger_cute_kernels.so.
#
# Only configured when LIGER_CUTE_BUILD_BINDINGS=ON (the default). The core build
# (wheel step 3.1) sets it OFF and never needs torch.
# ─────────────────────────────────────────────────────────────────────────────

# ── Python interpreter (used to locate torch + pybind11) ─────────────────────
find_program(_python NAMES python3 python REQUIRED)

# ── PyTorch (libtorch) — auto-detect cmake prefix from the installed wheel ───
execute_process(
    COMMAND "${_python}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _torch_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE  _torch_prefix_err
)
if(_torch_prefix)
    list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
    message(STATUS "Auto-detected Torch cmake prefix: ${_torch_prefix}")
else()
    message(WARNING "Could not detect Torch cmake prefix via Python: ${_torch_prefix_err}")
endif()
find_package(Torch REQUIRED)
message(STATUS "Torch version = ${Torch_VERSION}")

# ── Match the installed torch's C++ ABI for THIS shim only ───────────────────
# The core .so is ABI-agnostic, but the shim links libtorch directly, so it must
# be compiled with the same _GLIBCXX_USE_CXX11_ABI the torch wheel was built
# with or libtorch symbols won't resolve.
execute_process(
    COMMAND "${_python}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Bindings: old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(_liger_cute_abi 0)
elseif(TORCH_CXX11_ABI STREQUAL "1")
    message(STATUS "Bindings: new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(_liger_cute_abi 1)
else()
    message(WARNING "Could not detect torch C++ ABI (got '${TORCH_CXX11_ABI}')")
endif()

# ── pybind11 ─────────────────────────────────────────────────────────────────
execute_process(
    COMMAND "${_python}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_pybind11_cmake_dir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
    message(STATUS "Auto-detected pybind11 cmake dir: ${_pybind11_cmake_dir}")
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# ── The extension module ─────────────────────────────────────────────────────
pybind11_add_module(_C bindings.cpp)

if(DEFINED _liger_cute_abi)
    target_compile_definitions(_C PRIVATE _GLIBCXX_USE_CXX11_ABI=${_liger_cute_abi})
endif()

target_link_libraries(_C PRIVATE
    liger_cute_kernels
    ${TORCH_LIBRARIES}
)

# Find the sibling libliger_cute_kernels.so (and bundled nvshmem) next to _C.so
# at runtime, wherever the wheel unpacks.
set_target_properties(_C PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)
