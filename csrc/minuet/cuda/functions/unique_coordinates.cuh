#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

class UniqueCoordinates {
 public:
  template <typename CT, typename IT, std::size_t T_NDIM>
  void operator()(std::size_t num_sources,   //
                  std::size_t &num_uniques,  //
                  const CT *d_sources,       //
                  IT *d_indices,             //
                  const Context &context) const;
};

class MultiUniqueCoordinates {
 public:
  template <typename CT, typename IT, std::size_t T_NDIM>
  void operator()(std::size_t num_batches,   //
                  std::size_t num_sources,   //
                  std::size_t &num_uniques,  //
                  const CT *d_sources,       //
                  const IT *d_batch_dims,    //
                  IT *d_indices,             //
                  const Context &context) const;
};

class FillUniqueCoordinates {
 public:
  template <typename CT, typename IT, std::size_t T_NDIM>
  void operator()(std::size_t num_sources,  //
                  const CT *d_sources,      //
                  const IT *d_indices,      //
                  CT *d_targets,            //
                  const Context &context) const;
};

}  // namespace minuet::cuda