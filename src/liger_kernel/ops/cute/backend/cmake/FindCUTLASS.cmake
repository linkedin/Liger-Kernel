# FindCUTLASS.cmake
# Locates the NVIDIA CUTLASS header-only library.
#
# Sets:
#   CUTLASS_FOUND
#   CUTLASS_INCLUDE_DIRS      - main + util include dirs
#
# Imported target:
#   CUTLASS::CUTLASS          - INTERFACE target carrying both include dirs.
#                              Prefer linking this over raw ${CUTLASS_HOME} paths
#                              so the core builds even when CUTLASS_HOME is not
#                              exported as an env var.

find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    HINTS
        $ENV{CUTLASS_HOME}/include
        $ENV{CUTLASS_DIR}/include
        /usr/local/cutlass/include
)

# The CUTLASS utility headers (cutlass/util/...) live in a separate tree under
# tools/util/include. Locate them relative to the main include dir so we don't
# depend on CUTLASS_HOME being set.
find_path(CUTLASS_UTIL_INCLUDE_DIR
    NAMES cutlass/util/host_tensor.h
    HINTS
        "${CUTLASS_INCLUDE_DIR}/../tools/util/include"
        $ENV{CUTLASS_HOME}/tools/util/include
        $ENV{CUTLASS_DIR}/tools/util/include
        /usr/local/cutlass/tools/util/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTLASS
    REQUIRED_VARS CUTLASS_INCLUDE_DIR
)

set(CUTLASS_INCLUDE_DIRS "${CUTLASS_INCLUDE_DIR}")
if(CUTLASS_UTIL_INCLUDE_DIR)
    list(APPEND CUTLASS_INCLUDE_DIRS "${CUTLASS_UTIL_INCLUDE_DIR}")
else()
    message(WARNING "CUTLASS util headers (cutlass/util/...) not found; "
        "tools/util/include will be missing from CUTLASS::CUTLASS")
endif()

if(CUTLASS_FOUND AND NOT TARGET CUTLASS::CUTLASS)
    add_library(CUTLASS::CUTLASS INTERFACE IMPORTED)
    set_target_properties(CUTLASS::CUTLASS PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CUTLASS_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(CUTLASS_INCLUDE_DIR CUTLASS_UTIL_INCLUDE_DIR)
