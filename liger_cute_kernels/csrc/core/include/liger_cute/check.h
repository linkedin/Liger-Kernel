// check.h — LIGER_CHECK: torch-free precondition/argument checks for the core.
//
// The torch-free analogue of TORCH_CHECK. On a failed condition it formats the
// streamed message and throws liger_cute::Error. That exception is meant to be
// caught at the extern "C" boundary (see moe.h) and converted to a
// liger_cute_status_t — it must NEVER unwind across the .so boundary, which
// would be UB between translation units built against different toolchains.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace liger_cute {

// Thrown by LIGER_CHECK on failure. A std::runtime_error subclass so generic
// catch(const std::exception&) at the boundary still works.
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

inline void stream_into(std::ostream&) {}

template <typename T, typename... Rest>
inline void stream_into(std::ostream& os, const T& head, const Rest&... rest) {
  os << head;
  stream_into(os, rest...);
}

template <typename... Args>
inline std::string format_check_message(const char* cond, const char* file, int line,
                                        const Args&... args) {
  std::ostringstream os;
  os << "LIGER_CHECK(" << cond << ") failed at " << file << ":" << line;
  if (sizeof...(args) > 0) {
    os << ": ";
    stream_into(os, args...);
  }
  return os.str();
}

}  // namespace detail
}  // namespace liger_cute

// LIGER_CHECK(cond, msg...) — if cond is false, throw liger_cute::Error with a
// message built by streaming the (optional) trailing args. Mirrors TORCH_CHECK
// ergonomics: LIGER_CHECK(x == y, "x (", x, ") must equal y (", y, ")").
#define LIGER_CHECK(cond, ...)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      throw ::liger_cute::Error(::liger_cute::detail::format_check_message(    \
          #cond, __FILE__, __LINE__, ##__VA_ARGS__));                          \
    }                                                                          \
  } while (0)
