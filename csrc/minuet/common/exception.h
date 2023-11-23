#pragma once

#include <stdexcept>

#include "minuet/common/macro.h"
#include "minuet/common/stringify.h"

namespace minuet {
class MinuetException : public std::runtime_error {
 public:
  explicit MinuetException(const std::string &message)
      : std::runtime_error("[Minuet] " + message) {}
};

#define MINUET_ERROR(...)                                         \
  do {                                                            \
    std::string message =                                         \
        "[" __FILE__ " (" MINUET_MACRO_STRINGIFY(__LINE__) ")] "; \
    message += Stringify(__VA_ARGS__);                            \
    throw MinuetException(message);                               \
  } while (false)

#define MINUET_CHECK(COND, ...)  \
  do {                           \
    if (!(COND)) {               \
      MINUET_ERROR(__VA_ARGS__); \
    }                            \
  } while (false)

#ifdef __NVCC__
#define MINUET_CHECK_CUDA(stmt)                            \
  do {                                                     \
    auto return_code = (stmt);                             \
    MINUET_CHECK(return_code == cudaSuccess, "[",          \
                 cudaGetErrorName(return_code), "]", ": ", \
                 cudaGetErrorString(return_code));         \
  } while (false)

#define MINUET_CHECK_CUDA_WITH_HINT(stmt, ...)                              \
  do {                                                                      \
    auto return_code = (stmt);                                              \
    MINUET_CHECK(return_code == cudaSuccess, cudaGetErrorName(return_code), \
                 cudaGetErrorString(return_code), " -- ", __VA_ARGS__);     \
  } while (false)
#endif

}  // namespace minuet
