#include "minuet/cuda/functions/simple_hash.cuh"
#include "minuet/cuda/kernels/simple_hash.cuh"
#include "minuet/enabled_arguments.h"

namespace minuet::cuda {

template <typename CT, std::size_t T_NDIM>
void SimpleHash::operator()(std::size_t num_sources, const CT *d_sources,
                            std::int64_t *d_targets, bool reverse,
                            const Context &context) const {
  if (reverse) {
    context.Launch(num_sources,                             // size
                   128,                                     // block_size
                   0,                                       // shared_memory
                   kernels::SimpleHashReverse<CT, T_NDIM>,  // kernel
                   num_sources, d_sources, d_targets);
  } else {
    context.Launch(num_sources,                      // size
                   128,                              // block_size
                   0,                                // shared_memory
                   kernels::SimpleHash<CT, T_NDIM>,  // kernel
                   num_sources, d_sources, d_targets);
  }
}

#define MINUET_EXPLICIT_INSTANTIATOR(_, T_NDIM, CT) \
  template void SimpleHash::operator()<CT, T_NDIM>( \
      std::size_t, const CT *, std::int64_t *, bool, const Context &) const

MINUET_FOR_ALL_C_TYPES_AND_DIMS(MINUET_EXPLICIT_INSTANTIATOR);

}  // namespace minuet::cuda
