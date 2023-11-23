#include "minuet/common/functions.h"
#include "minuet/cuda/device/binary_search.cuh"
#include "minuet/cuda/functions/unique_coordinates.cuh"
#include "minuet/cuda/helpers.cuh"
#include "minuet/cuda/kernels/fill_unique_coordinate.cuh"
#include "minuet/enabled_arguments.h"

namespace minuet::cuda {

template <typename CT, typename IT, std::size_t T_NDIM>
void UniqueCoordinates::operator()(std::size_t num_sources,   //
                                   std::size_t &num_uniques,  //
                                   const CT *d_sources,       //
                                   IT *d_indices,             //
                                   const Context &context) const {
  auto HashCoordinates = [sources = d_sources,
                          num_sources] MINUET_DEVICE(UIter x) {
    if (x + 1 == num_sources) {
      return true;
    }
    auto a = sources + x * T_NDIM;
    auto b = sources + (x + 1) * T_NDIM;
    bool flag = true;
    Iterate<UIter, T_NDIM>([&](UIter k) { flag &= (a[k] == b[k]); });
    return !flag;
  };
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, HashCoordinates, d_in);
  std::size_t d_temp_size = 0;
  MINUET_CHECK_CUDA(
      cudaMemsetAsync(d_indices, 0, sizeof(IT), context.stream()));
  MINUET_CHECK_CUDA(
      cub::DeviceScan::InclusiveSum(nullptr,        // d_temp_storage,
                                    d_temp_size,    // temp_storage_bytes
                                    d_in,           // d_in
                                    d_indices + 1,  // d_out
                                    num_sources,    // num_items
                                    context.stream()));
  auto d_temp = context.NewBuffer(d_temp_size);
  MINUET_CHECK_CUDA(
      cub::DeviceScan::InclusiveSum(d_temp.device_data(),  // d_temp_storage,
                                    d_temp_size,           // temp_storage_bytes
                                    d_in,                  // d_in
                                    d_indices + 1,         // d_out
                                    num_sources,           // num_items
                                    context.stream()));
  num_uniques = context.ReadDeviceData(&d_indices[num_sources]);
}

template <typename CT, typename IT, std::size_t T_NDIM>
void MultiUniqueCoordinates::operator()(std::size_t num_batches,   //
                                        std::size_t num_sources,   //
                                        std::size_t &num_uniques,  //
                                        const CT *d_sources,       //
                                        const IT *d_batch_dims,    //
                                        IT *d_indices,             //
                                        const Context &context) const {
  auto HashCoordinates = [batch_dims = d_batch_dims, sources = d_sources,
                          num_batches] MINUET_DEVICE(UIter x) {
    auto index = device::BinarySearchMinimize<UIter>(
        0, num_batches, [&](UIter m) { return batch_dims[m]; }, x + 1);
    if (batch_dims[index] == x + 1) {
      return true;
    }

    auto a = sources + x * T_NDIM;
    auto b = sources + (x + 1) * T_NDIM;
    bool flag = true;
    Iterate<UIter, T_NDIM>([&](UIter k) { flag &= (a[k] == b[k]); });
    return !flag;
  };
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, HashCoordinates, d_in);
  std::size_t d_temp_size = 0;
  MINUET_CHECK_CUDA(
      cudaMemsetAsync(d_indices, 0, sizeof(IT), context.stream()));
  MINUET_CHECK_CUDA(
      cub::DeviceScan::InclusiveSum(nullptr,        // d_temp_storage,
                                    d_temp_size,    // temp_storage_bytes
                                    d_in,           // d_in
                                    d_indices + 1,  // d_out
                                    num_sources,    // num_items
                                    context.stream()));
  auto d_temp = context.NewBuffer(d_temp_size);
  MINUET_CHECK_CUDA(
      cub::DeviceScan::InclusiveSum(d_temp.device_data(),  // d_temp_storage,
                                    d_temp_size,           // temp_storage_bytes
                                    d_in,                  // d_in
                                    d_indices + 1,         // d_out
                                    num_sources,           // num_items
                                    context.stream()));
  num_uniques = context.ReadDeviceData(&d_indices[num_sources]);
}

template <typename CT, typename IT, std::size_t T_NDIM>
void FillUniqueCoordinates::operator()(std::size_t num_sources,  //
                                       const CT *d_sources,      //
                                       const IT *d_indices,      //
                                       CT *d_targets,            //
                                       const Context &context) const {
  context.Launch(num_sources, 128, 0,
                 kernels::FillUniqueCoordinate<CT, IT, T_NDIM>, num_sources,
                 d_sources, d_indices, d_targets);
}

#define MINUET_EXPLICIT_INSTANTIATOR(_, T_NDIM, CT, IT)                        \
  template void UniqueCoordinates::operator()<CT, IT, T_NDIM>(                 \
      std::size_t num_sources, std::size_t & num_uniques, const CT *d_sources, \
      IT *d_indices, const Context &context) const;                            \
  template void MultiUniqueCoordinates::operator()<CT, IT, T_NDIM>(            \
      std::size_t num_batches, std::size_t num_sources,                        \
      std::size_t & num_uniques, const CT *d_sources, const IT *d_batch_dims,  \
      IT *d_indices, const Context &context) const;                            \
  template void FillUniqueCoordinates::operator()<CT, IT, T_NDIM>(             \
      std::size_t num_sources, const CT *d_sources, const IT *d_indices,       \
      CT *d_targets, const Context &context) const
MINUET_FOR_ALL_DIMS_AND_CI_TYPES(MINUET_EXPLICIT_INSTANTIATOR);

}  // namespace minuet::cuda