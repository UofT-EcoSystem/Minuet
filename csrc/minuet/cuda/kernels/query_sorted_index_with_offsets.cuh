#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT, std::size_t T_NDIM,
          std::size_t T_SOURCE_BLOCK_SIZE, std::size_t T_TARGET_BLOCK_SIZE>
__global__ void QuerySortedIndexWithOffsets(
    std::size_t num_sources,                      //
    std::size_t num_targets,                      //
    std::size_t num_source_blocks,                //
    std::size_t num_target_blocks,                //
    std::size_t num_offsets,                      //
    const CT *__restrict__ sources,               //
    const CT *__restrict__ targets,               //
    const CT *__restrict__ offsets,               //
    IT *__restrict__ indices,                     //
    const IT *__restrict__ borders,               //
    const IT *__restrict__ target_cumsum_blocks,  //
    const IT *__restrict__ target_block_indices) {
  __shared__ CT cached_sources[T_SOURCE_BLOCK_SIZE * T_NDIM];
  CT cached[T_NDIM];

  const auto lid = MINUET_LOCAL_THREAD_ID(x);
  const auto lsz = MINUET_N_LOCAL_THREADS(x);
  const auto bid = MINUET_THREAD_BLOCK_ID(x);
  const auto bsz = MINUET_N_THREAD_BLOCKS(x);

  for (UIter b = bid; b < num_target_blocks; b += bsz) {
    auto x = target_block_indices[b];
    auto s = x / num_offsets;
    auto o = x % num_offsets;

    __syncthreads();
    Iterate<UIter, T_NDIM>(
        [&](auto k) { cached[k] = offsets[o * T_NDIM + k]; });
    for (UIter i = lid; i < T_SOURCE_BLOCK_SIZE; i += lsz) {
      auto p = s * T_SOURCE_BLOCK_SIZE + i;
      Iterate<UIter, T_NDIM>([&](auto k) {
        cached_sources[i * T_NDIM + k] =
            (p < num_sources) ? (sources[p * T_NDIM + k] - cached[k])
                              : (std::numeric_limits<CT>::max() >> 1);
      });
    }
    __syncthreads();

    UIter begin =
        borders[x] + (b - target_cumsum_blocks[x]) * T_TARGET_BLOCK_SIZE;
    UIter end = (s + 1 < num_source_blocks) ? borders[(s + 1) * num_offsets + o]
                                            : num_targets;
    end = min(end, static_cast<UIter>(begin + T_TARGET_BLOCK_SIZE));
    for (UIter t = begin + lid; t < end; t += lsz) {
      Iterate<UIter, T_NDIM>(
          [&](auto k) { cached[k] = targets[t * T_NDIM + k]; });

      constexpr const auto MAX_ITER = CeilLog2(T_SOURCE_BLOCK_SIZE);
      static_assert(1 << MAX_ITER == T_SOURCE_BLOCK_SIZE);

      IT index = 0;
      Iterate<UIter, MAX_ITER>([&](auto r) {
        auto step = (static_cast<IT>(1) << (MAX_ITER - r - 1));
        auto delta = CompareCoordinates<CT, T_NDIM>()(
            [&](auto j) { return cached_sources[(index + step) * T_NDIM + j]; },
            [&](auto j) { return cached[j]; });
        index += (delta <= 0) * step;
      });
      if (index < num_sources - s * T_SOURCE_BLOCK_SIZE) {
        auto delta = CompareCoordinates<CT, T_NDIM>()(
            [&](auto j) { return cached_sources[index * T_NDIM + j]; },
            [&](auto j) { return cached[j]; });
        indices[o * num_targets + t] =
            (delta == 0) ? (index + s * T_SOURCE_BLOCK_SIZE) : -1;
      } else {
        indices[o * num_targets + t] = -1;
      }
    }
  }
}

}  // namespace minuet::cuda::kernels