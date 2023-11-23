#include "minuet/common/exception.h"
#include "minuet/cuda/functions/arg_sort_coodrinates.cuh"
#include "minuet/enabled_arguments.h"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

torch::Tensor CUDAArgSortCoordinates(const torch::Tensor &coordinates,
                                     bool enable_flattening) {
  MINUET_ENSURE_TENSOR_NDIM(coordinates, 2);

  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto n = coordinates.size(0), ndim = coordinates.size(1);
  auto ctype = coordinates.dtype().toScalarType();
  auto itype = n <= std::numeric_limits<std::int32_t>::max() ? torch::kInt32
                                                             : torch::kInt64;
  auto indices = torch::empty({n}, torch::TensorOptions(device).dtype(itype));
#define CASE(_, T_NDIM, CT, IT)                                             \
  do {                                                                      \
    if (ctype == torch::CppTypeToScalarType<CT>::value &&                   \
        itype == torch::CppTypeToScalarType<IT>::value && ndim == T_NDIM) { \
      cuda::ArgSortCoordinates().operator()<T_NDIM, CT, IT>(                \
          n, coordinates.data_ptr<CT>(), indices.data_ptr<IT>(),            \
          enable_flattening, context);                                      \
      return indices;                                                       \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_DIMS_AND_CI_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for coordinate data type ", torch::toString(ctype),
               " of with ndim=", ndim);
}

torch::Tensor CUDAMultiArgSortCoordinates(const torch::Tensor &coordinates,
                                          const torch::Tensor &batch_dims,
                                          bool enable_flattening) {
  MINUET_ENSURE_TENSOR_NDIM(coordinates, 2);
  MINUET_ENSURE_TENSOR_NDIM(batch_dims, 1);

  auto device = GetTorchDeviceFromTensors({coordinates, batch_dims});
  auto n = coordinates.size(0), ndim = coordinates.size(1);
  auto batch_size = batch_dims.size(0) - 1;
  auto ctype = coordinates.dtype().toScalarType();
  auto itype = batch_dims.dtype().toScalarType();
  auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto indices = torch::empty({n}, torch::TensorOptions(device).dtype(itype));
  cuda::Context context(stream.device_index(), stream.stream());
#define CASE(_, T_NDIM, CT, IT)                                             \
  do {                                                                      \
    if (ctype == torch::CppTypeToScalarType<CT>::value &&                   \
        itype == torch::CppTypeToScalarType<IT>::value && ndim == T_NDIM) { \
      cuda::MultiArgSortCoordinates().operator()<T_NDIM, CT, IT>(           \
          n, batch_size, coordinates.data_ptr<CT>(),                        \
          batch_dims.data_ptr<IT>(), indices.data_ptr<IT>(),                \
          enable_flattening, context);                                      \
      return indices;                                                       \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_DIMS_AND_CI_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for coordinate data type ", torch::toString(ctype),
               " of with ndim=", ndim);
}

MINUET_TORCH_REGISTER(cuda_arg_sort_coordinates, CUDAArgSortCoordinates);
MINUET_TORCH_REGISTER(cuda_multi_arg_sort_coordinates,
                      CUDAMultiArgSortCoordinates);

}  // namespace minuet