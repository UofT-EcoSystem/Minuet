#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename SourceT, typename TargetT, std::size_t T_NDIM,
          typename IndicesT>
__global__ void MultiFlattenCoordinates(
    std::size_t batch_size,               //
    const SourceT *__restrict__ cmin,     //
    const SourceT *__restrict__ cmax,     //
    const SourceT *__restrict__ sources,  //
    TargetT *__restrict__ targets,        //
    const IndicesT *__restrict__ d_batch_dims) {
  static_assert(!std::is_signed_v<TargetT> && T_NDIM > 0);

  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);

  TargetT cached_cpow[T_NDIM];
  SourceT cached_cmin[T_NDIM];

  for (UIter b = 0; b < batch_size; b++) {
    Iterate<UIter, T_NDIM>(
        [&](UIter i) { cached_cmin[i] = cmin[b * T_NDIM + i]; });

    cached_cpow[T_NDIM - 1] = 1;
    Iterate<UIter, T_NDIM - 1>([&](UIter i) {
      UIter r = T_NDIM - (i + 1) - 1;
      // i = 0 => r = T_NDIM - 2
      cached_cpow[r] = cached_cpow[r + 1] *
                       static_cast<TargetT>(cmax[b * T_NDIM + r + 1] -
                                            cmin[b * T_NDIM + r + 1] + 1);
    });

    const auto base = d_batch_dims[b];
    const auto size = d_batch_dims[b + 1] - base;
    for (UIter i = base + gid; i < base + size; i += gsz) {
      auto source = sources + i * T_NDIM;
      TargetT result = 0;
      Iterate<UIter, T_NDIM>([&](auto k) {
        result +=
            static_cast<TargetT>(source[k] - cached_cmin[k]) * cached_cpow[k];
      });
      targets[i] = result;
    }
  }
}

}  // namespace minuet::cuda::kernels
