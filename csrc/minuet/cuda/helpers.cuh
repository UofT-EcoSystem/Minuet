#pragma once

#include <cub/cub.cuh>

#include "minuet/cuda/kernels/binary_search_materialize.cuh"
#include "minuet/cuda/kernels/fill_block_indices.cuh"
#include "minuet/cuda/kernels/load_coordinates_with_permutation.cuh"

namespace minuet::cuda {

#define CUB_TRANSFORMED_INPUT_ITERATOR(TYPE, FUNC, NAME)         \
  cub::TransformInputIterator<TYPE, decltype(FUNC),              \
                              cub::CountingInputIterator<UIter>> \
  NAME(cub::CountingInputIterator<UIter>(0), FUNC)

enum class MaterializeAlgorithm { kCumSumPlusCumMax = 0, kBinarySearch = 1 };

template <std::size_t THREAD_BLOCK_SIZE, typename IndicesT, typename ValuesT>
inline void Materialize(std::size_t num_indices, std::size_t num_values,
                        const IndicesT *d_indices, ValuesT *d_values,
                        const Context &context,
                        const MaterializeAlgorithm &algorithm =
                            MaterializeAlgorithm::kBinarySearch) {
  if (algorithm == MaterializeAlgorithm::kBinarySearch) {
    context.Launch(num_values, THREAD_BLOCK_SIZE, 0,
                   kernels::BinarySearchMaterialize<ValuesT, IndicesT>,
                   num_indices, num_values, d_indices, d_values);
  } else if (algorithm == MaterializeAlgorithm::kCumSumPlusCumMax) {
    MINUET_CHECK_CUDA(cudaMemsetAsync(d_values, 0, num_values * sizeof(ValuesT),
                                      context.stream()));
    context.Launch(num_indices, THREAD_BLOCK_SIZE, 0,
                   kernels::FillBlockIndices<IndicesT, ValuesT>, num_indices,
                   d_indices, d_values);
    std::size_t d_temp_size = 0;
    MINUET_CHECK_CUDA(
        cub::DeviceScan::InclusiveScan(nullptr,      // d_temp_storage
                                       d_temp_size,  // temp_storage_bytes
                                       d_values,     // d_in
                                       d_values,     // d_out
                                       cub::Max(),   // scan_op
                                       num_values,   // num_items
                                       context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::InclusiveScan(d_temp.device_data(),  // d_temp_storage
                                       d_temp_size,  // temp_storage_bytes
                                       d_values,     // d_in
                                       d_values,     // d_out
                                       cub::Max(),   // scan_op
                                       num_values,   // num_items
                                       context.stream()));
  } else {
    MINUET_ERROR("Found unrecognized materialize algorithm");
  }
}

template <std::size_t THREAD_BLOCK_SIZE, typename IndicesT, typename MappingT>
inline std::size_t ComputeMapping(std::size_t source_space_size,
                                  MappingT mapping, IndicesT *d_cumsums,
                                  IndicesT *d_indices, const Context &context,
                                  const MaterializeAlgorithm &algorithm =
                                      MaterializeAlgorithm::kBinarySearch) {
  // d_cumsums: [num_sources + 1]
  // d_indices: [num_sources]
  std::size_t d_temp_size = 0;
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, mapping, d_in);
  MINUET_CHECK_CUDA(
      cudaMemsetAsync(d_cumsums, 0, sizeof(IndicesT), context.stream()));
  MINUET_CHECK_CUDA(
      cub::DeviceScan::InclusiveSum(nullptr,            // d_temp_storage
                                    d_temp_size,        // temp_storage_bytes
                                    d_in,               // d_in
                                    d_cumsums + 1,      // d_out
                                    source_space_size,  // num_items
                                    context.stream()));
  auto d_temp = context.NewBuffer(d_temp_size);
  MINUET_CHECK_CUDA(
      cub::DeviceScan::InclusiveSum(d_temp.device_data(),  // d_temp_storage
                                    d_temp_size,           // temp_storage_bytes
                                    d_in,                  // d_in
                                    d_cumsums + 1,         // d_out
                                    source_space_size,     // num_items
                                    context.stream()));
  std::size_t target_space_size =
      context.ReadDeviceData(d_cumsums + source_space_size);
  Materialize<THREAD_BLOCK_SIZE>(source_space_size, target_space_size,
                                 d_cumsums, d_indices, context, algorithm);
  return target_space_size;
}

template <std::size_t THREAD_BLOCK_SIZE, typename IndicesT, typename MappingT>
inline auto ComputeMapping(std::size_t source_space_size, MappingT mapping,
                           const Context &context,
                           const MaterializeAlgorithm &algorithm =
                               MaterializeAlgorithm::kBinarySearch) {
  std::size_t d_temp_size = 0;
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, mapping, d_in);
  auto d_cumsums = context.NewBuffer<IndicesT>(source_space_size + 1);
  MINUET_CHECK_CUDA(cudaMemsetAsync(d_cumsums.device_data(), 0,
                                    sizeof(IndicesT), context.stream()));
  {
    MINUET_CHECK_CUDA(
        cub::DeviceScan::InclusiveSum(nullptr,      // d_temp_storage
                                      d_temp_size,  // temp_storage_bytes
                                      d_in,         // d_in
                                      d_cumsums.device_data() + 1,  // d_out
                                      source_space_size,            // num_items
                                      context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::InclusiveSum(d_temp.device_data(),  // d_temp_storage
                                      d_temp_size,  // temp_storage_bytes
                                      d_in,         // d_in
                                      d_cumsums.device_data() + 1,  // d_out
                                      source_space_size,            // num_items
                                      context.stream()));
  }
  std::size_t target_space_size =
      context.ReadDeviceData(d_cumsums.device_data() + source_space_size);
  auto d_indices = context.NewBuffer<IndicesT>(target_space_size);
  Materialize<THREAD_BLOCK_SIZE>(source_space_size, target_space_size,
                                 d_cumsums.device_data(),
                                 d_indices.device_data(), context, algorithm);
  return std::make_pair(std::move(d_cumsums), std::move(d_indices));
}

template <typename FT>
void MatMul(std::size_t m,           //
            std::size_t k,           //
            std::size_t n,           //
            bool is_a_transposed,    //
            bool is_b_transposed,    //
            const FT *d_a,           //
            const FT *d_b,           //
            FT *d_c,                 //
            bool incremental,        //
            const Context &context,  //
            cudaStream_t stream = nullptr);

template <typename FT>
void BatchedMatMul(std::size_t b,           //
                   std::size_t m,           //
                   std::size_t k,           //
                   std::size_t n,           //
                   bool is_a_transposed,    //
                   bool is_b_transposed,    //
                   const FT *d_a,           //
                   const FT *d_b,           //
                   FT *d_c,                 //
                   bool incremental,        //
                   const Context &context,  //
                   cudaStream_t stream = nullptr);

}  // namespace minuet::cuda