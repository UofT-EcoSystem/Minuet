#include "minuet/common/exception.h"
#include "minuet/cuda/functions/unique_coordinates.cuh"
#include "minuet/enabled_arguments.h"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

std::pair<torch::Tensor, torch::Tensor> CUDAUniqueCoordinates(
    const torch::Tensor &coordinates) {
  MINUET_ENSURE_TENSOR_NDIM(coordinates, 2);

  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);

  auto n = coordinates.size(0), ndim = coordinates.size(1);
  auto ctype = coordinates.dtype().toScalarType();
  auto itype = torch::kInt64;
  auto indices =
      torch::empty({n + 1}, torch::TensorOptions(device).dtype(itype));

#define CASE(_, T_NDIM, CT, IT)                                               \
  do {                                                                        \
    if (ctype == torch::CppTypeToScalarType<CT>::value &&                     \
        itype == torch::CppTypeToScalarType<IT>::value && ndim == T_NDIM) {   \
      std::size_t num_uniques;                                                \
      cuda::UniqueCoordinates().operator()<CT, IT, T_NDIM>(                   \
          n, num_uniques, coordinates.data_ptr<CT>(), indices.data_ptr<IT>(), \
          context);                                                           \
      auto targets =                                                          \
          torch::empty({static_cast<std::int64_t>(num_uniques), ndim},        \
                       torch::TensorOptions(device).dtype(ctype));            \
      cuda::FillUniqueCoordinates().operator()<CT, IT, T_NDIM>(               \
          n, coordinates.data_ptr<CT>(), indices.data_ptr<IT>(),              \
          targets.data_ptr<CT>(), context);                                   \
      return {targets, indices};                                              \
    }                                                                         \
  } while (false)
  MINUET_FOR_ALL_DIMS_AND_CI_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for coordinate data type ", torch::toString(ctype),
               " of with ndim=", ndim);
}

std::pair<torch::Tensor, torch::Tensor> CUDAMultiUniqueCoordinates(
    const torch::Tensor &coordinates, const torch::Tensor &batch_dims) {
  MINUET_ENSURE_TENSOR_NDIM(coordinates, 2);
  MINUET_ENSURE_TENSOR_NDIM(batch_dims, 1);

  auto n = coordinates.size(0), ndim = coordinates.size(1);
  auto num_batches = batch_dims.size(0);
  auto ctype = coordinates.dtype().toScalarType();
  auto itype = batch_dims.dtype().toScalarType();

  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto indices =
      torch::empty({n + 1}, torch::TensorOptions(device).dtype(itype));
#define CASE(_, T_NDIM, CT, IT)                                             \
  do {                                                                      \
    if (ctype == torch::CppTypeToScalarType<CT>::value &&                   \
        itype == torch::CppTypeToScalarType<IT>::value && ndim == T_NDIM) { \
      std::size_t num_uniques;                                              \
      cuda::MultiUniqueCoordinates().operator()<CT, IT, T_NDIM>(            \
          num_batches, n, num_uniques, coordinates.data_ptr<CT>(),          \
          batch_dims.data_ptr<IT>(), indices.data_ptr<IT>(), context);      \
      auto targets =                                                        \
          torch::empty({static_cast<std::int64_t>(num_uniques), ndim},      \
                       torch::TensorOptions(device).dtype(ctype));          \
      cuda::FillUniqueCoordinates().operator()<CT, IT, T_NDIM>(             \
          n, coordinates.data_ptr<CT>(), indices.data_ptr<IT>(),            \
          targets.data_ptr<CT>(), context);                                 \
      return {targets, indices};                                            \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_DIMS_AND_CI_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for coordinate data type ", torch::toString(ctype),
               " of with ndim=", ndim);
}

MINUET_TORCH_REGISTER(cuda_unique_coordinates, CUDAUniqueCoordinates);
MINUET_TORCH_REGISTER(cuda_multi_unique_coordinates,
                      CUDAMultiUniqueCoordinates);

}  // namespace minuet
