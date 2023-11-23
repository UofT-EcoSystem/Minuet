#pragma once

#include <cstdint>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

class MultiExclusiveSum {
 public:
  template <typename IT>
  void operator()(std::size_t batch_size,   //
                  std::size_t num_sources,  //
                  const IT *sources,        //
                  IT *targets,              //
                  const Context &context) const;
};

class MultiInclusiveSum {
 public:
  template <typename IT>
  void operator()(std::size_t batch_size,   //
                  std::size_t num_sources,  //
                  const IT *sources,        //
                  IT *targets,              //
                  const Context &context) const;
};

class MultiTwoSidedSum {
 public:
  template <typename IT>
  void operator()(std::size_t batch_size,   //
                  std::size_t num_sources,  //
                  const IT *sources,        //
                  IT *targets,              //
                  const Context &context) const;
};

class MultiTwoSidedKernelMapRanks {
 public:
  template <typename IT>
  void operator()(std::size_t batch_size,   //
                  std::size_t num_sources,  //
                  const IT *sources,        //
                  IT *targets,              //
                  const Context &context) const;
};

class ComputeKernelMapSizes {
 public:
  template <typename IT>
  void operator()(std::size_t num_offsets,  //
                  std::size_t num_targets,  //
                  const IT *d_kernel_map,   //
                  IT *d_kernel_map_sizes,   //
                  const Context &context) const;
};

class ComputeKernelMapMasks {
 public:
  template <typename IT>
  void operator()(std::size_t num_offsets,       // [O]
                  std::size_t num_sources,       // [S]
                  std::size_t num_targets,       // [T]
                  const IT *d_kernel_map,        // [O, T]
                  const IT *d_kernel_map_sizes,  // [O]
                  IT *d_source_masks,            // [O, S]
                  IT *d_target_masks,            // [O, T]
                  const Context &context) const;
};

}  // namespace minuet::cuda
