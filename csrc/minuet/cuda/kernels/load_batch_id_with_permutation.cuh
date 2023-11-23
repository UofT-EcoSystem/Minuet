#pragma once

#include "minuet/cuda/context.cuh"
#include "minuet/cuda/device/binary_search.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT>
__global__ void LoadBatchIdWithPermutation(std::size_t n,
                                           std::size_t batch_size,
                                           const IT *__restrict__ batch_dims,
                                           const IT *__restrict__ permutation,
                                           CT *__restrict__ indices) {
  auto tid = MINUET_GLOBAL_THREAD_ID(x);
  auto tsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = tid; i < n; i += tsz) {
    // max i s.t. permutation[i] >= batch_dims[i]
    indices[i] = device::BinarySearchMaximize<UIter>(
        0, batch_size, [&](UIter m) { return batch_dims[m]; }, permutation[i]);
  }
}

}  // namespace minuet::cuda::kernels