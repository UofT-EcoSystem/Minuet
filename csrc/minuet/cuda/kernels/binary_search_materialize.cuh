#pragma once

#include "minuet/cuda/context.cuh"
#include "minuet/cuda/device/binary_search.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT>
__global__ void BinarySearchMaterialize(std::size_t num_keys,
                                        std::size_t num_values,
                                        const IT *__restrict__ keys,
                                        CT *__restrict__ values) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_values; i += gsz) {
    values[i] = device::BinarySearchMaximize<UIter>(
        0, num_keys - 1, [&](UIter m) { return keys[m]; }, i);
  }
}

}  // namespace minuet::cuda::kernels