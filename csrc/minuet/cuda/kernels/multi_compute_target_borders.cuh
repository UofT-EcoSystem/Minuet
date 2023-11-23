#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT, std::size_t T_NDIM,
          std::size_t T_SOURCE_BLOCK_SIZE>
__global__ void MultiComputeTargetBorders(
    std::size_t num_batches,                      //
    std::size_t num_offsets,                      //
    std::size_t max_offsets_per_round,            //
    const IT *__restrict__ source_batch_dims,     //
    const IT *__restrict__ target_batch_dims,     //
    const IT *__restrict__ source_cumsum_blocks,  //
    const IT *__restrict__ source_block_indices,  //
    const CT *__restrict__ sources,               //
    const CT *__restrict__ targets,               //
    const CT *__restrict__ offsets,               //
    IT *__restrict__ borders) {
  // shared_memory_size should be at least sizeof(IT) * thread block size
  const auto gid = MINUET_GLOBAL_THREAD_ID(x);
  const auto gsz = MINUET_N_GLOBAL_THREADS(x);
  const auto num_source_blocks = source_cumsum_blocks[num_batches];

  auto shared_memory = DynamicSharedMemory<char>();
  CT *cached_offsets = reinterpret_cast<CT *>(shared_memory);
  CT cached_sources[T_NDIM];

  for (UIter h = 0; h < num_offsets; h += max_offsets_per_round) {
    auto num_current_offsets = min(max_offsets_per_round, num_offsets - h);
    device::CacheOffsets<CT, T_NDIM>(num_current_offsets, cached_offsets,
                                     offsets + h * T_NDIM);
    for (UIter i = gid; i < num_source_blocks * num_current_offsets; i += gsz) {
      auto b = i / num_current_offsets;
      auto o = i % num_current_offsets;
      auto g = source_block_indices[b];
      if (b == source_cumsum_blocks[g]) {
        borders[b * num_offsets + (h + o)] = target_batch_dims[g];
      } else {
        auto source =
            sources + (source_batch_dims[g] +
                       (b - source_cumsum_blocks[g]) * T_SOURCE_BLOCK_SIZE) *
                          T_NDIM;
        auto offset = cached_offsets + o * T_NDIM;
        Iterate<UIter, T_NDIM>(
            [&](auto k) { cached_sources[k] = source[k] - offset[k]; });
        // minimize i s.t. t >= s[i]
        borders[b * num_offsets + (h + o)] =
            (b > 0) ? device::BinarySearchMinimize<IT>(
                          target_batch_dims[g], target_batch_dims[g + 1],
                          [&](auto m) {
                            return CompareCoordinates<CT, T_NDIM>()(
                                [&](auto k) { return targets[m * T_NDIM + k]; },
                                [&](auto k) { return cached_sources[k]; });
                          })
                    : 0;
      }
    }
  }
}

}  // namespace minuet::cuda::kernels