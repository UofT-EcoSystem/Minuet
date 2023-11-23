#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <typename IT, typename FT>
__global__ void PermuteWeights(std::size_t num_offsets,             //
                               std::size_t num_features,            //
                               std::size_t max_offsets_per_round,   //
                               const IT *__restrict__ permutation,  //
                               const FT *__restrict__ sources,      //
                               FT *__restrict__ targets) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  auto lid = MINUET_LOCAL_THREAD_ID(x);
  auto lsz = MINUET_N_LOCAL_THREADS(x);

  auto shared_memory = DynamicSharedMemory<char>();
  IT *cached_permutation = reinterpret_cast<IT *>(shared_memory);
  for (UIter h = 0; h < num_offsets; h += max_offsets_per_round) {
    auto num_current_offsets = min(max_offsets_per_round, num_offsets - h);

    __syncthreads();
    for (UIter i = lid; i < num_offsets; i += lsz) {
      cached_permutation[i] = permutation[h + i];
    }
    __syncthreads();

    for (UIter i = gid; i < num_current_offsets * num_features; i += gsz) {
      auto o = i / num_features;
      auto f = i % num_features;
      auto x = cached_permutation[o];
      targets[(h + o) * num_features + f] = sources[x * num_features + f];
    }
  }
}

}  // namespace minuet::cuda::kernels