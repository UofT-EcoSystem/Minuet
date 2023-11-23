#pragma once

#include "minuet/cuda/context.cuh"
#include "minuet/cuda/device/data_movement.cuh"

namespace minuet::cuda::kernels {

template <std::size_t T_NUM_SOURCE_FEATURE_TILES, std::size_t T_TILE_SIZE,
          std::size_t T_BULK_SIZE, std::size_t T_WARP_SIZE, typename IT,
          typename FT>
__global__ void PaddedGatherWarpOptimized(
    std::size_t num_sources,                            //
    std::size_t num_offsets,                            //
    const IT *__restrict__ cumsum_offset_padded_sizes,  //
    const IT *__restrict__ source_masks,                //
    const FT *__restrict__ sources,                     //
    FT *__restrict__ source_buffers) {
  // device::Assign will translate that into vectorized memory instructions
  // according to the bulk size
  // Need to make sure num_source_features % T_TILE_SIZE = 0

  const auto gwid = MINUET_GLOBAL_THREAD_ID(x) / T_WARP_SIZE;
  const auto gwsz = MINUET_N_GLOBAL_THREADS(x) / T_WARP_SIZE;
  const auto wid = MINUET_GLOBAL_THREAD_ID(x) % T_WARP_SIZE;

  static_assert(T_TILE_SIZE % T_BULK_SIZE == 0);
  constexpr const auto NUM_TILE_BULKS = T_TILE_SIZE / T_BULK_SIZE;
  constexpr const auto NUM_WARPS =
      DivCeil<std::size_t>(T_NUM_SOURCE_FEATURE_TILES, T_WARP_SIZE);
  for (UIter i = gwid; i < num_sources * NUM_WARPS; i += gwsz) {
    auto s = i / NUM_WARPS;
    auto w = i % NUM_WARPS;

    FT value[T_TILE_SIZE];
    auto source = sources + s * T_NUM_SOURCE_FEATURE_TILES * T_TILE_SIZE;
    Iterate<UIter, NUM_TILE_BULKS>([&](UIter j) {
      auto x = (w * NUM_TILE_BULKS + j) * T_WARP_SIZE + wid;
      if (x < T_NUM_SOURCE_FEATURE_TILES * NUM_TILE_BULKS) {
        device::Assign<T_BULK_SIZE>(value + j * T_BULK_SIZE,
                                    source + x * T_BULK_SIZE);
      }
    });

    for (UIter o = 0; o < num_offsets; o++) {
      auto d = source_masks[o * num_sources + s];
      if (d == -1) {
        continue;
      }
      d += cumsum_offset_padded_sizes[o];

      Iterate<UIter, NUM_TILE_BULKS>([&](UIter j) {
        auto x = (w * NUM_TILE_BULKS + j) * T_WARP_SIZE + wid;
        if (x < T_NUM_SOURCE_FEATURE_TILES * NUM_TILE_BULKS) {
          // this is where warp diverges
          device::Assign<T_BULK_SIZE>(
              source_buffers + d * T_NUM_SOURCE_FEATURE_TILES * T_TILE_SIZE +
                  x * T_BULK_SIZE,
              value + j * T_BULK_SIZE);
        }
      });
    }
  }
}

}  // namespace minuet::cuda::kernels