// export.h — symbol-visibility control for liger_cute_kernels.
//
// The core is compiled with -fvisibility=hidden so that CUTLASS/CuTe template
// instantiations and any internal C++/libstdc++ symbols stay LOCAL and never
// land on the .so's dynamic symbol table. Only declarations tagged
// LIGER_CUTE_API are exported. Combined with the flat `extern "C"` boundary in
// liger_cute.h (no std:: types crossing the ABI), this keeps the shared library
// ABI-agnostic — one build links into a binding compiled against any torch
// wheel, regardless of its _GLIBCXX_USE_CXX11_ABI.
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef LIGER_CUTE_BUILDING
#    define LIGER_CUTE_API __declspec(dllexport)
#  else
#    define LIGER_CUTE_API __declspec(dllimport)
#  endif
#else
#  define LIGER_CUTE_API __attribute__((visibility("default")))
#endif
