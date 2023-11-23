#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename IndicesT, typename TargetT>
__global__ void FillBlockIndices(std::size_t num_indices,
                                 const IndicesT *__restrict__ sumblks,
                                 TargetT *__restrict__ blkinds) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_indices; i += gsz) {
    if (sumblks[i] != sumblks[i + 1]) {
      blkinds[sumblks[i]] = i;
    }
  }
}

}  // namespace minuet::cuda::kernels
