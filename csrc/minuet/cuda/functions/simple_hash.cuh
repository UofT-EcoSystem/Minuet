#pragma once

#include <cstdint>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

class SimpleHash {
 public:
  template <typename CT, std::size_t T_NDIM>
  void operator()(std::size_t num_sources, const CT *d_sources,
                  std::int64_t *d_targets, bool reverse,
                  const Context &context) const;
};

}  // namespace minuet::cuda
