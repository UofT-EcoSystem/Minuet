#pragma once

#include <cstdint>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

class ArgSortCoordinates {
 public:
  template <std::size_t T_NDIM, typename CT, typename IT>
  void operator()(std::size_t n,            //
                  const CT *d_coordinates,  //
                  IT *d_indices,            //
                  bool enable_flattening,   //
                  const Context &context) const;
};

class MultiArgSortCoordinates {
 public:
  template <std::size_t T_NDIM, typename CT, typename IT>
  void operator()(std::size_t n,            //
                  std::size_t batch_size,   //
                  const CT *d_coordinates,  //
                  const IT *d_batch_dims,   //
                  IT *d_indices,            //
                  bool enable_flattening,   //
                  const Context &context) const;
};

}  // namespace minuet::cuda
