#pragma once

#include <cstdint>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

class QuerySortedIndexWithOffsets {
 public:
  template <typename CT, typename IT, std::size_t T_NDIM>
  void operator()(std::size_t num_sources,  //
                  std::size_t num_targets,  //
                  std::size_t num_offsets,  //
                  const CT *d_sources,      //
                  const CT *d_targets,      //
                  const CT *d_offsets,      //
                  IT *d_indices,            //
                  const Context &context) const;
};

class MultiQuerySortedIndexWithOffsets {
 public:
  template <typename CT, typename IT, std::size_t T_NDIM>
  void operator()(std::size_t num_batches,        //
                  std::size_t num_offsets,        //
                  const IT *d_source_batch_dims,  //
                  const IT *d_target_batch_dims,  //
                  const CT *d_sources,            //
                  const CT *d_targets,            //
                  const CT *d_offsets,            //
                  IT *d_indices,                  //
                  const Context &context) const;
};

}  // namespace minuet::cuda
