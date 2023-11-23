#pragma once

#include "minuet/common/exception.h"
#include "minuet/cpu/memory.h"

namespace minuet::cpu {

class Context {
 public:
  Context();

  template <typename T = void>
  Memory<T> NewMemory(std::size_t n) const {
    return Memory<T>(n);
  }
};

}  // namespace minuet::cpu