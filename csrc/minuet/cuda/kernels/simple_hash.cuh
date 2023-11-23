#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename T, std::size_t T_NDIM>
__global__ void SimpleHash(std::size_t n, const T *__restrict__ sources,
                           std::int64_t *__restrict__ targets) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < n; i += gsz) {
    targets[i] = Hash<T_NDIM>(sources + i * T_NDIM);
  }
}

template <typename T, std::size_t T_NDIM>
__global__ void SimpleHashReverse(std::size_t n, const T *__restrict__ sources,
                                  std::int64_t *__restrict__ targets) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < n; i += gsz) {
    targets[i] = HashReverse<T_NDIM>(sources + i * T_NDIM);
  }
}

}  // namespace minuet::cuda::kernels
