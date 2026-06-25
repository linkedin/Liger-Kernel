# LigerCute — native MoE kernels (CUTLASS + NVSHMEM)

Native CUDA build for the MoE port (from `LigerCommKernels`). It compiles into
**two** artifacts that are deliberately decoupled:

| Artifact | Sources | Links | Boundary | Built |
|---|---|---|---|---|
| `libliger_cute_kernels.so` (the "core", aka *lck*) | `csrc/core` | CUTLASS + NVSHMEM + CUDA — **no torch** | flat `extern "C"` (`liger_cute.h`) | **once** |
| `tvm_ffi` (the binding) | `liger_cute_kernels/tvm_ffi*.{py,cpp}` | the core + TVM FFI | DLPack/TVM FFI | **source-packaged, JIT-built** |

The core's public ABI is `extern "C"` only (no `std::`/torch types cross it),
symbols are hidden except `liger_cute_*`, and libstdc++/libgcc are linked
statically. That makes the core **ABI-agnostic**: one compiled core links into a
TVM FFI shim loaded from packaged source, so the expensive CUTLASS compile is done
once and the Python binding is not tied to a specific torch wheel.

## Two separate wheels

The top-level **`liger_kernel` wheel is pure Python/Triton** and does **not**
build or contain any of this native code. The native libraries ship as a
**separate, CUDA/torch-version-prefixed `lck` wheel** that installs its own
standalone top-level package **`liger_cute_kernels`** (kept separate from
`liger_kernel` so the native libs don't mix in). `liger_kernel.ops.cute` imports
`liger_cute_kernels.tvm_ffi` at runtime. Intended order:

1. Install the top-level `liger_kernel` wheel (pure Python).
2. *Optionally* install the matching `lck` wheel (package `liger_cute_kernels`)
   for the local CUDA + torch environment.

The lck wheel is built by this module's `setup.py` (see **Building the lck
wheel** below). Selecting/installing the right lck wheel automatically for the
local CUDA + torch environment is a separate follow-up.

## Layout

`liger_cute_kernels/` is a **standalone module at the repo root** holding
everything needed to build the native libraries. Only `__init__.py` (the
in-liger entry point) lives separately, under `src/liger_kernel/ops/cute/`.

```
liger_cute_kernels/             # ← standalone native build module (repo root)
├── README.md
├── setup.py                    # builds the lck wheel (package liger_cute_kernels)
├── pyproject.toml
├── cute_build.py               # build_core() helper + LckBuildExt
├── liger_cute_kernels/         # the lck wheel's package source
│   ├── __init__.py             # (.so are added here at build time)
│   ├── tvm_ffi.py              # Python facade matching the old _C API
│   └── tvm_ffi_bindings.cpp    # TVM FFI C++ shim over the torch-free core
├── test/                       # the lck package's own unit tests
│   └── test_moe_bindings.py
├── CMakeLists.txt              # core (always) + bindings (opt-in)
├── cmake/
│   ├── FindNVSHMEM.cmake
│   └── FindCUTLASS.cmake        # locates main + tools/util/include
└── csrc/
    ├── core/                   # → libliger_cute_kernels.so (torch-free)
    │   ├── include/liger_cute/
    │   │   ├── {liger_cute.h, export.h}     # flat extern "C" ABI (C-parseable)
    │   │   ├── {check.h, tensor_view.h, moe.h}  # C++-only core surface
    │   │   └── detail/symmetric_memory.h    # core-internal (nvshmem+STL); not ABI
    │   ├── src/*.{cu,cpp}
    │   └── liger_cute.version   # exports only liger_cute_*
    └── bindings/               # legacy _C (torch + pybind11)
        ├── bindings.cpp
        └── tensor_view_conversion.h   # torch::Tensor <-> TensorView<N>

src/liger_kernel/ops/cute/
└── __init__.py                 # runtime entry point: liger_kernel.ops.cute
                                #   (loads liger_cute_kernels.tvm_ffi if installed)
```

## Prerequisites

- **CUDA toolkit** with `nvcc` and SM 9.0a (Hopper / `sm_90a`) support.
- **NVSHMEM** install (host `.so`, device `.a`, headers) — point `NVSHMEM_HOME`
  at it (default `/usr/local/nvshmem`). The lck build also auto-detects the
  `nvidia-nvshmem-cu13` pip package layout and creates unversioned compatibility
  symlinks for CMake when needed.
- **CUTLASS** headers (4.x) — point `CUTLASS_HOME` at the repo root (so that
  `$CUTLASS_HOME/include/cutlass/cutlass.h` and
  `$CUTLASS_HOME/tools/util/include` exist). *Not needed when linking a prebuilt
  core.*
- **CMake ≥ 3.24**; **Ninja** recommended.
- For the bindings only: **apache-tvm-ffi** at runtime.

Build commands below are run from the **repository root**; the CMake project is
the `liger_cute_kernels/` module directory.

## Building

### 1. Core only (torch-free)

No torch required. This is the expensive CUTLASS compile, done once.

```bash
cmake -S liger_cute_kernels -B build/core \
      -DLIGER_CUTE_BUILD_BINDINGS=OFF \
      -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build build/core --target liger_cute_kernels -j
# -> build/core/csrc/core/libliger_cute_kernels.so
```

Or from Python (with `liger_cute_kernels/` on `sys.path`):

```python
from cute_build import build_core
build_core("build/core")   # stages libliger_cute_kernels.so + libnvshmem_host.so
```

### 2. Core + bindings from source (single local build)

```bash
cmake -S liger_cute_kernels -B build/all \
      -DLIGER_CUTE_BUILD_BINDINGS=ON \
      -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build build/all --target liger_cute_kernels -j
```

### 3. Bindings against a prebuilt core (reuse across the torch matrix)

Reuses an existing `libliger_cute_kernels.so` instead of recompiling the core.
CUTLASS is not even searched here.

```bash
cmake -S liger_cute_kernels -B build/bind \
      -DLIGER_CUTE_BUILD_BINDINGS=ON \
      -DLIGER_CUTE_CORE_IMPORTED_DIR=/abs/path/to/dir-with-core \
      -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build build/bind --target liger_cute_kernels -j
```

## Building the lck wheel

This module's `setup.py` packages the native libraries into the independent
**lck wheel**, whose package is the standalone top-level **`liger_cute_kernels`**.
It builds the core and ships
`liger_cute_kernels/{libliger_cute_kernels.so, libnvshmem_host.so,
tvm_ffi.py, tvm_ffi_bindings.cpp}`. Build against the **local** CUDA/NVSHMEM
environment (no build isolation), from this module directory:

```bash
cd liger_cute_kernels
pip wheel . --no-deps --no-build-isolation -w dist
# -> dist/liger_cute_kernels-0.1.0+cu130.torch2.9.1-cp312-cp312-linux_x86_64.whl
```

The wheel is tagged with the CUDA + torch version as a PEP 440 local version
(`+cu<ver>.torch<ver>`), so wheels for different environments coexist. To reuse
a core built once across the torch matrix (no core recompile), point at its dir:

```bash
LIGER_CUTE_CORE_DIR=/abs/dir-with-core \
    pip wheel . --no-deps --no-build-isolation -w dist
```

Install order at the consumer side: the `liger_kernel` wheel first, then
optionally the matching lck wheel.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `LIGER_CUTE_BUILD_BINDINGS` | `ON` | Legacy CMake option for the torch/pybind11 `_C` extension. The runtime Python API now uses TVM FFI. |
| `LIGER_CUTE_CORE_IMPORTED_DIR` | *(empty)* | Dir holding a prebuilt `libliger_cute_kernels.so`. When set, the core is linked as an imported library (not compiled) and CUTLASS is not required. |
| `LIGER_CUTE_STATIC_LIBSTDCXX` | `ON` | Statically link libstdc++/libgcc into the core so its internal C++ ABI is invisible to consumers. |
| `NVSHMEM_HOME` | `/usr/local/nvshmem` | NVSHMEM install root (also read from the env var). |
| `CUTLASS_HOME` | *(env)* | CUTLASS repo root (read from the env var; only needed to compile the core). |

Standard CMake flags also apply: `-DCMAKE_BUILD_TYPE=Release`, `-GNinja`,
`-DPython_EXECUTABLE=...`.

CUDA architecture is fixed to `sm_90a` (Hopper, with WGMMA/TMA/multicast) in
`CMakeLists.txt`.

## Environment variables

| Variable | Used by | Effect |
|---|---|---|
| `NVSHMEM_HOME` | CMake / `cute_build` | NVSHMEM install root (default `/usr/local/nvshmem`). |
| `CUTLASS_HOME` | CMake | CUTLASS repo root (core compile only). |
| `LIGER_CUTE_CORE_DIR` | `setup.py` | Dir with a prebuilt core. Set → link it (no core recompile); unset → build the core from source. |
| `LIGER_CUTE_LOCAL_VERSION` | `setup.py` | Override the auto-detected `cu<ver>.torch<ver>` local version tag. |

## Verifying a core build

```bash
SO=build/core/csrc/core/libliger_cute_kernels.so
nm -D --defined-only "$SO" | grep ' T '      # exports: only liger_cute_*
readelf -d "$SO" | grep NEEDED               # no libtorch, no direct libstdc++
```

The core should export only `liger_cute_*` symbols and have no direct
`libstdc++`/`libtorch` `NEEDED` entries (a transitive `libstdc++` via
`libnvshmem_host` is expected and harmless).

## Runtime notes

- `tvm_ffi.py` JIT-builds `tvm_ffi_bindings.cpp` and links its sibling
  `libliger_cute_kernels.so` / `libnvshmem_host.so` via an installed-package
  rpath — no `LD_LIBRARY_PATH` needed once they sit together in the package.
