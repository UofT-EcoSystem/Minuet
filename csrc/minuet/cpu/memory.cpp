#include "minuet/cpu/memory.h"

namespace minuet::cpu::detail {

template <>
void *AcquireMemory(std::size_t size) {
  return ::operator new(size);
}

template <>
void ReleaseMemory(void *data) noexcept {
  ::operator delete(data);
}

}  // namespace minuet::cpu::detail
