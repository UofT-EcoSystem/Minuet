#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::device {

// Assume the array is non-decreasing
template <typename IT, typename SourceT>
MINUET_FORCEINLINE MINUET_DEVICE IT
BinarySearchMinimize(IT l, IT r, SourceT source,
                     const std::invoke_result_t<SourceT, IT> &target = 0) {
  while (l < r) {
    auto m = (l + r) >> 1;
    if (source(m) >= target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return l;
}

template <typename IT, typename SourceT>
MINUET_FORCEINLINE MINUET_DEVICE IT
BinarySearchMaximize(IT l, IT r, SourceT source,
                     const std::invoke_result_t<SourceT, IT> &target = 0) {
  while (l < r) {
    auto m = (l + r + 1) >> 1;
    if (target >= source(m)) {
      l = m;
    } else {
      r = m - 1;
    }
  }
  return l;
}

}  // namespace minuet::cuda::device