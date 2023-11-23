#pragma once

#include <cstddef>

#include "minuet/cuda/device/data_movement.cuh"

namespace minuet::cuda::kernels {

template <typename IT>
__global__ void GenerateMasksFromKernelMap(
    std::size_t num_entries,                                      //
    std::size_t num_sources,                                      //
    std::size_t num_targets,                                      //
    const IT *__restrict__ kernel_map_sizes,                      //
    const std::int64_t *__restrict__ kernel_map_nonzero_indices,  //
    IT *__restrict__ source_masks,                                //
    IT *__restrict__ target_masks) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_entries; i += gsz) {
    auto entry = kernel_map_nonzero_indices[i];  // (value, index)
    auto s = entry % num_sources;
    auto tmp = entry / num_sources;
    auto t = tmp % num_targets;
    auto o = tmp / num_targets;
    auto index = i - kernel_map_sizes[o];
    source_masks[o * num_sources + s] = index;
    target_masks[o * num_targets + t] = index;
  }
}

}  // namespace minuet::cuda::kernels