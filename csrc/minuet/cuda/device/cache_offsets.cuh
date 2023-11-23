#pragma once

#include <cooperative_groups.h>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::device {

template <typename CT, std::size_t T_NDIM, bool T_SYNC = true>
MINUET_FORCEINLINE MINUET_DEVICE void CacheOffsets(std::size_t num_offsets,
                                                   CT *cached_offsets,
                                                   const CT *offsets) {
  auto lid = MINUET_LOCAL_THREAD_ID(x);
  auto lsz = MINUET_N_LOCAL_THREADS(x);
  if (T_SYNC) {
    __syncthreads();
  }
  for (UIter i = lid; i < num_offsets; i += lsz) {
    Iterate<UIter, T_NDIM>([&](auto j) {
      cached_offsets[i * T_NDIM + j] = offsets[i * T_NDIM + j];
    });
  }
  if (T_SYNC) {
    __syncthreads();
  }
}

}  // namespace minuet::cuda::device
