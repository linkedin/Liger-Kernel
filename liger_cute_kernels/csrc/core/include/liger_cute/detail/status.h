// status.h — CORE-INTERNAL error reporting + the exception->status guard.
//
// Shared by every extern "C" core entry point (moe.cu, nvshmem.cu, ...). The
// rule for the flat ABI is "no exception unwinds across the .so boundary" (see
// liger_cute.h / moe.h): each entry point runs its body through guarded(),
// which catches anything escaping, records a human-readable message in a
// thread-local buffer (readable via liger_cute_last_error_string), and returns a
// liger_cute_status_t. NOT part of the ABI — header-only, std:: used freely;
// the binding TU never includes it.
#pragma once

#include <cstdio>
#include <exception>
#include <sstream>
#include <string>

#include "liger_cute/check.h"        // liger_cute::Error, detail::stream_into
#include "liger_cute/liger_cute.h"   // liger_cute_status_t

namespace liger_cute {

// Status-specialized failures. Both subclass liger_cute::Error so a generic
// catch(const Error&) / catch(const std::exception&) still works, but guarded()
// maps them to their dedicated status code (LIGER_CUTE_ERR_NVSHMEM / _CUDA)
// instead of the INVALID_ARGUMENT a bare LIGER_CHECK reports.
class NvshmemError : public Error {
 public:
  using Error::Error;
};

class CudaError : public Error {
 public:
  using Error::Error;
};

namespace detail {

// Per-thread last-error message backing liger_cute_last_error_string(). A fixed
// buffer (not std::string) so recording an error never allocates — the report
// survives even when the exception being handled is std::bad_alloc. snprintf
// truncates gracefully; 1024 comfortably fits a LIGER_CHECK message.
constexpr int kErrorBufLen = 1024;

inline char* tls_error_buf() {
  static thread_local char buf[kErrorBufLen] = {0};
  return buf;
}

inline void set_tls_error(const char* msg) {
  std::snprintf(tls_error_buf(), kErrorBufLen, "%s", msg != nullptr ? msg : "");
}

// Build a free-form message by streaming the args (reuses stream_into from
// check.h). Used by the LIGER_FAIL_* macros below for runtime (non-LIGER_CHECK)
// failures that carry a specific status code.
template <typename... Args>
inline std::string make_message(const Args&... args) {
  std::ostringstream os;
  stream_into(os, args...);
  return os.str();
}

// Run `body`, translating any escaping exception into a status code + a
// last-error message. This is the ONLY place exceptions are allowed to surface;
// every extern "C" entry point routes through it so nothing crosses the ABI.
// Order matters: the status-specialized subclasses must be caught before their
// liger_cute::Error base.
template <typename Body>
liger_cute_status_t guarded(Body&& body) {
  try {
    tls_error_buf()[0] = '\0';
    return body();
  } catch (const NvshmemError& e) {
    set_tls_error(e.what());
    return LIGER_CUTE_ERR_NVSHMEM;
  } catch (const CudaError& e) {
    set_tls_error(e.what());
    return LIGER_CUTE_ERR_CUDA;
  } catch (const liger_cute::Error& e) {
    set_tls_error(e.what());
    return LIGER_CUTE_ERR_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    set_tls_error(e.what());
    return LIGER_CUTE_ERR_INTERNAL;
  } catch (...) {
    set_tls_error("liger_cute: unknown error");
    return LIGER_CUTE_ERR_INTERNAL;
  }
}

}  // namespace detail
}  // namespace liger_cute

// LIGER_FAIL_NVSHMEM(msg...) / LIGER_FAIL_CUDA(msg...) — throw the matching
// status-carrying exception with a streamed message. For runtime failures
// (return codes from NVSHMEM / CUDA APIs) where LIGER_CHECK's INVALID_ARGUMENT
// would mislabel the error. Use LIGER_CHECK for precondition / argument checks.
#define LIGER_FAIL_NVSHMEM(...) \
  throw ::liger_cute::NvshmemError(::liger_cute::detail::make_message(__VA_ARGS__))
#define LIGER_FAIL_CUDA(...) \
  throw ::liger_cute::CudaError(::liger_cute::detail::make_message(__VA_ARGS__))
