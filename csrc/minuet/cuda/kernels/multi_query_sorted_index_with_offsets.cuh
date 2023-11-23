#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename CT, typename IT, std::size_t T_NDIM,
          std::size_t T_SOURCE_BLOCK_SIZE, std::size_t T_TARGET_BLOCK_SIZE>
__global__ void MultiQuerySortedIndexWithOffsets(
    std::size_t num_batches,                      //
    std::size_t num_offsets,                      //
    const CT *__restrict__ sources,               //
    const CT *__restrict__ targets,               //
    const CT *__restrict__ offsets,               //
    IT *__restrict__ indices,                     //
    const IT *__restrict__ borders,               //
    const IT *__restrict__ source_batch_dims,     //
    const IT *__restrict__ source_cumsum_blocks,  //
    const IT *__restrict__ source_block_indices,  //
    const IT *__restrict__ target_batch_dims,     //
    const IT *__restrict__ target_cumsum_blocks,  //
    const IT *__restrict__ target_block_indices) {
  __shared__ CT cached_sources[T_SOURCE_BLOCK_SIZE * T_NDIM];
  CT cached[T_NDIM];

  const auto lid = MINUET_LOCAL_THREAD_ID(x);
  const auto lsz = MINUET_N_LOCAL_THREADS(x);
  const auto bid = MINUET_THREAD_BLOCK_ID(x);
  const auto bsz = MINUET_N_THREAD_BLOCKS(x);

  const auto num_targets = target_batch_dims[num_batches];
  const auto num_source_blocks = source_cumsum_blocks[num_batches];
  const auto num_target_blocks =
      target_cumsum_blocks[num_source_blocks * num_offsets];

  for (UIter b = bid; b < num_target_blocks; b += bsz) {
    auto x = target_block_indices[b];
    auto s = x / num_offsets;
    auto o = x % num_offsets;
    auto batch_index = source_block_indices[s];
    auto source_base =
        source_batch_dims[batch_index] +
        (s - source_cumsum_blocks[batch_index]) * T_SOURCE_BLOCK_SIZE;
    auto source_size = min(source_batch_dims[batch_index + 1] - source_base,
                           T_SOURCE_BLOCK_SIZE);

    __syncthreads();
    Iterate<UIter, T_NDIM>(
        [&](auto k) { cached[k] = offsets[o * T_NDIM + k]; });
    for (UIter i = lid; i < T_SOURCE_BLOCK_SIZE; i += lsz) {
      auto source = sources + (source_base + i) * T_NDIM;
      Iterate<UIter, T_NDIM>([&](auto k) {
        cached_sources[i * T_NDIM + k] =
            (i < source_size) ? (source[k] - cached[k])
                              : (std::numeric_limits<CT>::max() >> 1);
      });
    }
    __syncthreads();

    UIter target_base =
        borders[x] + (b - target_cumsum_blocks[x]) * T_TARGET_BLOCK_SIZE;
    UIter target_end = (s + 1 < num_source_blocks)
                           ? borders[(s + 1) * num_offsets + o]
                           : num_targets;
    target_end =
        min(target_end, static_cast<UIter>(target_base + T_TARGET_BLOCK_SIZE));
    for (UIter t = target_base + lid; t < target_end; t += lsz) {
      Iterate<UIter, T_NDIM>(
          [&](auto k) { cached[k] = targets[t * T_NDIM + k]; });

      constexpr const auto MAX_ITER = CeilLog2(T_SOURCE_BLOCK_SIZE);
      static_assert(1 << MAX_ITER == T_SOURCE_BLOCK_SIZE);

      IT index = 0;
      Iterate<UIter, MAX_ITER>([&](auto r) {
        auto step = (static_cast<IT>(1) << (MAX_ITER - r - 1));
        auto source = cached_sources + (index + step) * T_NDIM;
        auto delta =
            CompareCoordinates<CT, T_NDIM>()([&](auto j) { return source[j]; },
                                             [&](auto j) { return cached[j]; });
        index += (delta <= 0) * step;
      });

      if (index < source_size) {
        auto delta = CompareCoordinates<CT, T_NDIM>()(
            [&](auto j) { return cached_sources[index * T_NDIM + j]; },
            [&](auto j) { return cached[j]; });
        indices[o * num_targets + t] =
            (delta == 0) ? (source_base + index) : -1;
      } else {
        indices[o * num_targets + t] = -1;
      }
    }
  }
}

}  // namespace minuet::cuda::kernels