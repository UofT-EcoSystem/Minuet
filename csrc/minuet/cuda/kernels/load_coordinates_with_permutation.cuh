#pragma once

#include "minuet/cuda/context.cuh"
#include "minuet/cuda/device/binary_search.cuh"

namespace minuet::cuda::kernels {

template <std::size_t T_NDIM, typename CT, typename IT>
__global__ void LoadCoordinatesWithPermutation(
    std::size_t dim, std::size_t n, const CT *__restrict__ sources,
    CT *__restrict__ targets, const IT *__restrict__ permutation) {
  auto tid = MINUET_GLOBAL_THREAD_ID(x);
  auto tsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = tid; i < n; i += tsz) {
    targets[i] = sources[permutation[i] * T_NDIM + dim];
  }
}

}  // namespace minuet::cuda::kernels