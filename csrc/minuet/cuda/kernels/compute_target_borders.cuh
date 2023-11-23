#pragma once

#include "minuet/cuda/context.cuh"
#include "minuet/cuda/device/cache_offsets.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT, std::size_t T_NDIM,
          std::size_t T_SOURCE_BLOCK_SIZE>
__global__ void ComputeTargetBorders(
    std::size_t num_source_blocks, std::size_t num_targets,
    std::size_t num_offsets, std::size_t shared_memory_size,
    const CT *__restrict__ sources, const CT *__restrict__ targets,
    const CT *__restrict__ offsets, IT *__restrict__ borders) {
  // shared_memory_size should be at least sizeof(IT) * thread block size
  const auto gid = MINUET_GLOBAL_THREAD_ID(x);
  const auto gsz = MINUET_N_GLOBAL_THREADS(x);

  auto cached_offsets = DynamicSharedMemory<CT>();
  CT cached_sources[T_NDIM];
  auto max_offsets_per_round =
      DivFloor(shared_memory_size, sizeof(CT) * T_NDIM);

  for (UIter h = 0; h < num_offsets; h += max_offsets_per_round) {
    auto num_current_offsets = min(max_offsets_per_round, num_offsets - h);
    device::CacheOffsets<CT, T_NDIM>(num_current_offsets, cached_offsets,
                                     offsets + h * T_NDIM);
    for (UIter i = gid; i < num_source_blocks * num_current_offsets; i += gsz) {
      auto b = i / num_current_offsets;
      auto o = i % num_current_offsets;
      Iterate<UIter, T_NDIM>([&](auto k) {
        cached_sources[k] = sources[(b * T_SOURCE_BLOCK_SIZE) * T_NDIM + k] -
                            cached_offsets[o * T_NDIM + k];
      });
      // minimize i s.t. t >= s[i]
      borders[b * num_offsets + (h + o)] =
          (b > 0) ? device::BinarySearchMinimize<IT>(
                        0, num_targets,
                        [&](auto m) {
                          return CompareCoordinates<CT, T_NDIM>()(
                              [&](auto k) { return targets[m * T_NDIM + k]; },
                              [&](auto k) { return cached_sources[k]; });
                        })
                  : 0;
    }
  }
}

}  // namespace minuet::cuda::kernels