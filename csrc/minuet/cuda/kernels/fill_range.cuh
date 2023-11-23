#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename T>
__global__ void FillRange(std::size_t n, T *targets) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < n; i += gsz) {
    targets[i] = i;
  }
}

}  // namespace minuet::cuda::kernels
