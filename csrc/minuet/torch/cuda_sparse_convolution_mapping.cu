#include "minuet/common/exception.h"
#include "minuet/cuda/functions/query_sorted_index.cuh"
#include "minuet/cuda/functions/scan.cuh"
#include "minuet/enabled_arguments.h"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

torch::Tensor CUDAQuerySortedIndexWithOffsets(const torch::Tensor &sources,
                                              const torch::Tensor &targets,
                                              const torch::Tensor &offsets) {
  MINUET_ENSURE_TENSOR_NDIM(sources, 2);
  MINUET_ENSURE_TENSOR_NDIM(targets, 2);
  MINUET_ENSURE_TENSOR_NDIM(offsets, 2);

  auto ndim = offsets.size(1);
  MINUET_ENSURE_TENSOR_DIM(sources, 1, ndim);
  MINUET_ENSURE_TENSOR_DIM(targets, 1, ndim);

  auto num_sources = sources.size(0);
  auto num_targets = targets.size(0);
  auto num_offsets = offsets.size(0);

  auto ctype = GetTorchScalarTypeFromTensors({sources, targets, offsets});
  auto itype = ctype;
  auto device = GetTorchDeviceFromTensors({sources, targets});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto indices = torch::empty({num_offsets, num_targets},
                              torch::TensorOptions(device).dtype(itype));

#define CASE(_, T_NDIM, CT, IT)                                             \
  do {                                                                      \
    if (ctype == torch::CppTypeToScalarType<CT>::value &&                   \
        itype == torch::CppTypeToScalarType<IT>::value && ndim == T_NDIM) { \
      cuda::QuerySortedIndexWithOffsets().operator()<CT, IT, T_NDIM>(       \
          num_sources, num_targets, num_offsets, sources.data_ptr<CT>(),    \
          targets.data_ptr<CT>(), offsets.data_ptr<CT>(),                   \
          indices.data_ptr<IT>(), context);                                 \
      return indices;                                                       \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_DIMS_AND_CI_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__, " for data type ",
               torch::toString(ctype), " with ndim=", ndim);
}

torch::Tensor CUDAMultiQuerySortedIndexWithOffsets(
    const torch::Tensor &source_batch_dims,  //
    const torch::Tensor &target_batch_dims,  //
    const torch::Tensor &sources,            //
    const torch::Tensor &targets,            //
    const torch::Tensor &offsets) {
  MINUET_ENSURE_TENSOR_NDIM(sources, 2);
  MINUET_ENSURE_TENSOR_NDIM(targets, 2);
  MINUET_ENSURE_TENSOR_NDIM(offsets, 2);
  MINUET_ENSURE_TENSOR_NDIM(source_batch_dims, 1);
  MINUET_ENSURE_TENSOR_NDIM(target_batch_dims, 1);

  auto ndim = offsets.size(1);
  MINUET_ENSURE_TENSOR_DIM(sources, 1, ndim);
  MINUET_ENSURE_TENSOR_DIM(targets, 1, ndim);

  auto num_offsets = offsets.size(0);
  auto ctype = GetTorchScalarTypeFromTensors({sources, targets, offsets});
  auto itype =
      GetTorchScalarTypeFromTensors({source_batch_dims, target_batch_dims});

  MINUET_CHECK(
      source_batch_dims.size(0) > 1,
      "Tensor source_batch_dims should be at least length 2 but found ",
      source_batch_dims.size(0), " <= ", 1);
  MINUET_CHECK(target_batch_dims.size(0) == source_batch_dims.size(0),
               "Tensor target_batch_dims must have the same length as "
               "source_batch_dims but found ",
               target_batch_dims.size(0), " != ", source_batch_dims.size(0));

  auto device = GetTorchDeviceFromTensors({sources, targets});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto num_batches = source_batch_dims.size(0) - 1;
  auto num_targets = targets.size(0);
  auto indices = torch::empty({num_offsets, num_targets},
                              torch::TensorOptions(device).dtype(itype));
#define CASE(_, T_NDIM, CT, IT)                                             \
  do {                                                                      \
    if (ctype == torch::CppTypeToScalarType<CT>::value &&                   \
        itype == torch::CppTypeToScalarType<IT>::value && ndim == T_NDIM) { \
      cuda::MultiQuerySortedIndexWithOffsets().operator()<CT, IT, T_NDIM>(  \
          num_batches, num_offsets, source_batch_dims.data_ptr<IT>(),       \
          target_batch_dims.data_ptr<IT>(), sources.data_ptr<CT>(),         \
          targets.data_ptr<CT>(), offsets.data_ptr<CT>(),                   \
          indices.data_ptr<IT>(), context);                                 \
      return indices;                                                       \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_DIMS_AND_CI_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__, " for data type ",
               torch::toString(ctype), " with ndim=", ndim);
}

torch::Tensor CUDAComputeKernelMapSizes(const torch::Tensor &kernel_map) {
  MINUET_ENSURE_TENSOR_NDIM(kernel_map, 2);

  auto device = GetTorchDeviceFromTensors({kernel_map});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto num_offsets = kernel_map.size(0);
  auto num_targets = kernel_map.size(1);
  auto itype = kernel_map.dtype().toScalarType();
  auto options = torch::TensorOptions(device).dtype(itype);
  auto results = torch::empty({num_offsets}, options);
#define CASE(_, IT)                                            \
  do {                                                         \
    if (itype == torch::CppTypeToScalarType<IT>::value) {      \
      cuda::ComputeKernelMapSizes().operator()<IT>(            \
          num_offsets, num_targets, kernel_map.data_ptr<IT>(), \
          results.data_ptr<IT>(), context);                    \
      return results;                                          \
    }                                                          \
  } while (false)
  MINUET_FOR_ALL_I_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for tensor data type ", torch::toString(itype));
}

std::pair<torch::Tensor, torch::Tensor> CUDAComputeKernelMapMasks(
    std::int64_t num_sources, const torch::Tensor &kernel_map,
    const torch::Tensor &kernel_map_sizes) {
  MINUET_ENSURE_TENSOR_NDIM(kernel_map, 2);

  auto num_offsets = kernel_map.size(0);
  auto num_targets = kernel_map.size(1);

  auto device = GetTorchDeviceFromTensors({kernel_map});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto itype = kernel_map.dtype().toScalarType();
  auto options = torch::TensorOptions(device).dtype(itype);
  auto source_masks = torch::empty({num_offsets, num_sources}, options);
  auto target_masks = torch::empty({num_offsets, num_targets}, options);
#define CASE(_, IT)                                                         \
  do {                                                                      \
    if (itype == torch::CppTypeToScalarType<IT>::value) {                   \
      cuda::ComputeKernelMapMasks().operator()<IT>(                         \
          num_offsets, num_sources, num_targets, kernel_map.data_ptr<IT>(), \
          kernel_map_sizes.data_ptr<IT>(), source_masks.data_ptr<IT>(),     \
          target_masks.data_ptr<IT>(), context);                            \
      return {source_masks, target_masks};                                  \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_I_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for tensor data type ", torch::toString(itype));
}

MINUET_TORCH_REGISTER(cuda_query_sorted_index_with_offsets,
                      CUDAQuerySortedIndexWithOffsets);
MINUET_TORCH_REGISTER(cuda_multi_query_sorted_index_with_offsets,
                      CUDAMultiQuerySortedIndexWithOffsets);

MINUET_TORCH_REGISTER(cuda_compute_kernel_map_sizes, CUDAComputeKernelMapSizes);
MINUET_TORCH_REGISTER(cuda_compute_kernel_map_masks, CUDAComputeKernelMapMasks);

}  // namespace minuet
