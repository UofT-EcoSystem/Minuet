#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT, std::size_t T_NDIM>
__global__ void FillUniqueCoordinate(std::size_t num_sources, const CT *sources,
                                     const IT *__restrict__ indices,
                                     CT *__restrict__ targets) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_sources; i += gsz) {
    if (indices[i] != indices[i + 1]) {
      auto target = targets + indices[i] * T_NDIM;
      auto source = sources + i * T_NDIM;
      Iterate<UIter, T_NDIM>([&](UIter k) { target[k] = source[k]; });
    }
  }
}

}  // namespace minuet::cuda::kernels
