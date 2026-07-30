# FindNVSHMEM.cmake
#
# Finds the NVSHMEM library and headers.
#
# Search hints (in priority order):
#   1. NVSHMEM_HOME cmake variable  (-DNVSHMEM_HOME=...)
#   2. NVSHMEM_HOME environment variable
#   3. Standard system paths
#
# Imported targets:
#   NVSHMEM::nvshmem_host    - host library (libnvshmem_host)
#   NVSHMEM::nvshmem_device  - device-side static library (libnvshmem_device)
#
# Result variables:
#   NVSHMEM_FOUND
#   NVSHMEM_INCLUDE_DIRS
#   NVSHMEM_HOST_LIBRARY
#   NVSHMEM_DEVICE_LIBRARY

if(DEFINED ENV{NVSHMEM_HOME} AND NOT DEFINED NVSHMEM_HOME)
    set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
endif()

# Keep headers and libraries within the same installation. In particular, a
# PyPI NVSHMEM_HOME must not pick up an unversioned host library from a separate
# system installation.
find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    PATHS "${NVSHMEM_HOME}/include"
    NO_DEFAULT_PATH
)

find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem_host
    PATHS "${NVSHMEM_HOME}/lib" "${NVSHMEM_HOME}/lib64"
    NO_DEFAULT_PATH
)
if(NOT NVSHMEM_HOST_LIBRARY)
    find_file(NVSHMEM_HOST_LIBRARY
        NAMES libnvshmem_host.so.3
        PATHS "${NVSHMEM_HOME}/lib" "${NVSHMEM_HOME}/lib64"
        NO_DEFAULT_PATH
    )
endif()

find_library(NVSHMEM_DEVICE_LIBRARY
    NAMES nvshmem_device
    PATHS "${NVSHMEM_HOME}/lib" "${NVSHMEM_HOME}/lib64"
    NO_DEFAULT_PATH
)

# ── Standard find_package machinery ───────────────────────────────────────────
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
    REQUIRED_VARS NVSHMEM_HOST_LIBRARY NVSHMEM_INCLUDE_DIR
)

if(NVSHMEM_FOUND)
    set(NVSHMEM_INCLUDE_DIRS "${NVSHMEM_INCLUDE_DIR}")

    if(NOT TARGET NVSHMEM::nvshmem_host)
        add_library(NVSHMEM::nvshmem_host SHARED IMPORTED)
        set_target_properties(NVSHMEM::nvshmem_host PROPERTIES
            IMPORTED_LOCATION             "${NVSHMEM_HOST_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
        )
    endif()

    if(NVSHMEM_DEVICE_LIBRARY AND NOT TARGET NVSHMEM::nvshmem_device)
        add_library(NVSHMEM::nvshmem_device STATIC IMPORTED)
        set_target_properties(NVSHMEM::nvshmem_device PROPERTIES
            IMPORTED_LOCATION             "${NVSHMEM_DEVICE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(NVSHMEM_INCLUDE_DIR NVSHMEM_HOST_LIBRARY NVSHMEM_DEVICE_LIBRARY)
