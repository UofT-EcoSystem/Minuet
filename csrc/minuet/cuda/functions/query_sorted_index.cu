#include "minuet/cpu/memory.h"
#include "minuet/cuda/functions/query_sorted_index.cuh"
#include "minuet/cuda/helpers.cuh"
#include "minuet/cuda/kernels/compute_target_borders.cuh"
#include "minuet/cuda/kernels/fill_block_indices.cuh"
#include "minuet/cuda/kernels/multi_compute_target_borders.cuh"
#include "minuet/cuda/kernels/multi_query_sorted_index_with_offsets.cuh"
#include "minuet/cuda/kernels/query_sorted_index_with_offsets.cuh"
#include "minuet/enabled_arguments.h"

namespace minuet::cuda {

template <typename CT, typename IT, std::size_t T_NDIM>
void QuerySortedIndexWithOffsets::operator()(std::size_t num_sources,  //
                                             std::size_t num_targets,  //
                                             std::size_t num_offsets,  //
                                             const CT *d_sources,      //
                                             const CT *d_targets,      //
                                             const CT *d_offsets,      //
                                             IT *d_indices,            //
                                             const Context &context) const {
  constexpr const std::size_t THREAD_BLOCK_SIZE = 128;
  constexpr const std::size_t SOURCE_BLOCK_SIZE = 256;
  constexpr const std::size_t TARGET_BLOCK_SIZE = 512;

  static_assert(TARGET_BLOCK_SIZE >= THREAD_BLOCK_SIZE);
  static_assert(TARGET_BLOCK_SIZE % THREAD_BLOCK_SIZE == 0);

  constexpr const std::size_t MAX_SHARED_MEMORY_SIZE = 8 * 1024;
  constexpr const std::size_t OFFSET_SIZE = T_NDIM * sizeof(CT);
  const auto max_offsets_per_round =
      std::min(DivFloor(MAX_SHARED_MEMORY_SIZE, OFFSET_SIZE), num_offsets);
  const auto shared_memory_size = std::max(max_offsets_per_round * OFFSET_SIZE,
                                           sizeof(IT) * THREAD_BLOCK_SIZE);

  const auto num_source_blocks = DivCeil(num_sources, SOURCE_BLOCK_SIZE);
  auto d_merged =
      context.NewBuffer<IT>(2 * num_source_blocks * num_offsets + 1);
  auto d_borders = d_merged.device_data();
  auto d_target_cumsum_blocks =
      d_merged.device_data() + (num_source_blocks * num_offsets);

  std::size_t d_temp_sum_size = 0;
  MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(
      nullptr,                          // d_temp_storage
      d_temp_sum_size,                  // temp_storage_bytes
      static_cast<IT *>(nullptr),       // d_values_in
      static_cast<IT *>(nullptr),       // d_values_out
      num_source_blocks * num_offsets,  // num_items
      context.stream()));
  auto d_temp_sum = context.NewBuffer(d_temp_sum_size);
  auto NumBlocks = [num_source_blocks, num_targets, num_offsets,
                    borders = d_borders] MINUET_DEVICE(UIter x) {
    auto b = x / num_offsets;
    auto o = x % num_offsets;
    auto n =
        (b + 1 < num_source_blocks) ? borders[x + num_offsets] : num_targets;
    return DivCeil(n - borders[x], TARGET_BLOCK_SIZE);
  };
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NumBlocks, d_in);

  context.Launch(
      num_source_blocks * max_offsets_per_round, THREAD_BLOCK_SIZE,
      shared_memory_size,
      kernels::ComputeTargetBorders<CT, IT, T_NDIM, SOURCE_BLOCK_SIZE>,
      num_source_blocks,   //
      num_targets,         //
      num_offsets,         //
      shared_memory_size,  //
      d_sources,           //
      d_targets,           //
      d_offsets,           //
      d_borders);

  MINUET_CHECK_CUDA(
      cudaMemsetAsync(d_target_cumsum_blocks, 0, sizeof(IT), context.stream()));
  MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(
      d_temp_sum.device_data(),         // d_temp_storage
      d_temp_sum_size,                  // temp_storage_bytes
      d_in,                             // d_values_in
      d_target_cumsum_blocks + 1,       // d_values_out
      num_source_blocks * num_offsets,  // num_items
      context.stream()));
  IT num_target_blocks;
  MINUET_CHECK_CUDA(
      cudaMemcpyAsync(&num_target_blocks,
                      &d_target_cumsum_blocks[num_source_blocks * num_offsets],
                      sizeof(IT), cudaMemcpyDeviceToHost, context.stream()));

  auto d_target_block_indices = context.NewBuffer<IT>(num_target_blocks);
  MINUET_CHECK_CUDA(cudaMemsetAsync(d_target_block_indices.device_data(), 0,
                                    sizeof(IT) * num_target_blocks,
                                    context.stream()));
  context.Launch(num_source_blocks * num_offsets, THREAD_BLOCK_SIZE, 0,
                 kernels::FillBlockIndices<IT, IT>,
                 num_source_blocks * num_offsets,  //
                 d_target_cumsum_blocks,           //
                 d_target_block_indices.device_data());
  Materialize<THREAD_BLOCK_SIZE>(num_source_blocks * num_offsets,
                                 num_target_blocks, d_target_cumsum_blocks,
                                 d_target_block_indices.device_data(), context);
  context.Launch(
      DivCeil<std::size_t>(num_target_blocks, 8) * THREAD_BLOCK_SIZE,
      THREAD_BLOCK_SIZE, 0,
      kernels::QuerySortedIndexWithOffsets<CT, IT, T_NDIM, SOURCE_BLOCK_SIZE,
                                           TARGET_BLOCK_SIZE>,
      num_sources,             //
      num_targets,             //
      num_source_blocks,       //
      num_target_blocks,       //
      num_offsets,             //
      d_sources,               //
      d_targets,               //
      d_offsets,               //
      d_indices,               //
      d_borders,               //
      d_target_cumsum_blocks,  //
      d_target_block_indices.device_data());
}

template <typename CT, typename IT, std::size_t T_NDIM>
void MultiQuerySortedIndexWithOffsets::operator()(
    std::size_t num_batches,        //
    std::size_t num_offsets,        //
    const IT *d_source_batch_dims,  //
    const IT *d_target_batch_dims,  //
    const CT *d_sources,            //
    const CT *d_targets,            //
    const CT *d_offsets,            //
    IT *d_indices,                  //
    const Context &context) const {
  constexpr const std::size_t THREAD_BLOCK_SIZE = 128;
  constexpr const std::size_t SOURCE_BLOCK_SIZE = 256;
  constexpr const std::size_t TARGET_BLOCK_SIZE = 512;
  constexpr const std::size_t MAX_SHARED_MEMORY_SIZE = 8 * 1024;
  constexpr const std::size_t OFFSET_SIZE = T_NDIM * sizeof(CT);

  const auto max_offsets_per_round =
      std::min(DivFloor(MAX_SHARED_MEMORY_SIZE, OFFSET_SIZE), num_offsets);
  const auto shared_memory_size = std::max(max_offsets_per_round * OFFSET_SIZE,
                                           sizeof(IT) * THREAD_BLOCK_SIZE);

  auto [d_source_cumsum_blocks, d_source_block_indices] =
      ComputeMapping<THREAD_BLOCK_SIZE, IT>(
          num_batches,
          [source_batch_dims = d_source_batch_dims] MINUET_DEVICE(UIter x) {
            return DivCeil<IT>(source_batch_dims[x + 1] - source_batch_dims[x],
                               SOURCE_BLOCK_SIZE);
          },
          context);

  auto num_source_blocks = d_source_block_indices.size();
  auto d_borders = context.NewBuffer<IT>(num_source_blocks * num_offsets);
  context.Launch(
      num_source_blocks * num_offsets, THREAD_BLOCK_SIZE, shared_memory_size,
      kernels::MultiComputeTargetBorders<CT, IT, T_NDIM, SOURCE_BLOCK_SIZE>,
      num_batches,                           //
      num_offsets,                           //
      max_offsets_per_round,                 //
      d_source_batch_dims,                   //
      d_target_batch_dims,                   //
      d_source_cumsum_blocks.device_data(),  //
      d_source_block_indices.device_data(),  //
      d_sources,                             //
      d_targets,                             //
      d_offsets,                             //
      d_borders.device_data()                //
  );

  auto [d_target_cumsum_blocks, d_target_block_indices] =
      ComputeMapping<THREAD_BLOCK_SIZE, IT>(
          num_source_blocks * num_offsets,
          [borders = d_borders.device_data(),        //
           target_batch_dims = d_target_batch_dims,  //
           num_batches, num_offsets, num_source_blocks] MINUET_DEVICE(UIter x) {
            auto b = x / num_offsets;
            auto n = (b + 1 < num_source_blocks)
                         ? borders[x + num_offsets]
                         : target_batch_dims[num_batches];
            return DivCeil<IT>(n - borders[x], TARGET_BLOCK_SIZE);
          },
          context);

  context.Launch(
      DivCeil<std::size_t>(d_target_block_indices.size(), num_offsets) *
          THREAD_BLOCK_SIZE,
      THREAD_BLOCK_SIZE, 0,
      kernels::MultiQuerySortedIndexWithOffsets<
          CT, IT, T_NDIM, SOURCE_BLOCK_SIZE, TARGET_BLOCK_SIZE>,
      num_batches,                           //
      num_offsets,                           //
      d_sources,                             //
      d_targets,                             //
      d_offsets,                             //
      d_indices,                             //
      d_borders.device_data(),               //
      d_source_batch_dims,                   //
      d_source_cumsum_blocks.device_data(),  //
      d_source_block_indices.device_data(),  //
      d_target_batch_dims,                   //
      d_target_cumsum_blocks.device_data(),  //
      d_target_block_indices.device_data()   //
  );
}

#define MINUET_EXPLICIT_INSTANTIATOR(_, T_NDIM, CT, IT)                       \
  template void QuerySortedIndexWithOffsets::operator()<CT, IT, T_NDIM>(      \
      std::size_t, std::size_t, std::size_t, const CT *, const CT *,          \
      const CT *, IT *, const Context &) const;                               \
  template void MultiQuerySortedIndexWithOffsets::operator()<CT, IT, T_NDIM>( \
      std::size_t num_batches, std::size_t num_offsets,                       \
      const IT *d_source_batch_dims, const IT *d_target_batch_dims,           \
      const CT *d_sources, const CT *d_targets, const CT *d_offsets,          \
      IT *d_indices, const Context &context) const;

MINUET_FOR_ALL_DIMS_AND_CI_TYPES(MINUET_EXPLICIT_INSTANTIATOR);

}  // namespace minuet::cuda
