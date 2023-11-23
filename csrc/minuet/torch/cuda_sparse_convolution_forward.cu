#include "minuet/cuda/functions/sparse_convolution.cuh"
#include "minuet/enabled_arguments.h"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

auto CUDATimeGEMM(bool allow_shortcut_matmul,              //
                  std::size_t parallel,                    //
                  const std::optional<double> &threshold,  //
                  const torch::Tensor &weights,            //
                  const torch::Tensor &source_masks,       //
                  const torch::Tensor &target_masks,       //
                  const torch::Tensor &kernel_map_sizes) {
  MINUET_ENSURE_TENSOR_NDIM(weights, 3);
  MINUET_ENSURE_TENSOR_NDIM(source_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(target_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(kernel_map_sizes, 1);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  MINUET_ENSURE_TENSOR_DIM(kernel_map_sizes, 0, num_offsets);
  MINUET_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);

  auto device = GetTorchDeviceFromTensors(
      {weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
  auto itype = GetTorchScalarTypeFromTensors(
      {source_masks, target_masks, kernel_map_sizes});
#define CASE(_, IT, FT)                                                    \
  do {                                                                     \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE &&                        \
        itype == torch::CppTypeToScalarType<IT>::value) {                  \
      return cuda::TimeGEMM().operator()<IT, FT>(                          \
          num_sources, num_targets, num_offsets, threshold, parallel,      \
          allow_shortcut_matmul, num_source_features, num_target_features, \
          kernel_map_sizes.data_ptr<IT>(),                                 \
          TypeConversion<FT>::GetCppPointer(weights), context);            \
    }                                                                      \
  } while (false)
  MINUET_FOR_ALL_IF_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__);
}

auto CUDATimeGather(std::size_t tile_size,                   //
                    bool allow_shortcut_matmul,              //
                    const std::optional<double> &threshold,  //
                    const torch::Tensor &weights,            //
                    const torch::Tensor &source_masks,       //
                    const torch::Tensor &target_masks,       //
                    const torch::Tensor &kernel_map_sizes) {
  MINUET_ENSURE_TENSOR_NDIM(weights, 3);
  MINUET_ENSURE_TENSOR_NDIM(source_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(target_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(kernel_map_sizes, 1);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  MINUET_ENSURE_TENSOR_DIM(kernel_map_sizes, 0, num_offsets);
  MINUET_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);

  auto device = GetTorchDeviceFromTensors(
      {weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
  auto itype = GetTorchScalarTypeFromTensors(
      {source_masks, target_masks, kernel_map_sizes});
#define CASE(_, IT, FT)                                                    \
  do {                                                                     \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE &&                        \
        itype == torch::CppTypeToScalarType<IT>::value) {                  \
      return cuda::TimeGather().operator()<IT, FT>(                        \
          num_sources, num_targets, num_offsets, threshold,                \
          allow_shortcut_matmul, num_source_features, num_target_features, \
          source_masks.data_ptr<IT>(), kernel_map_sizes.data_ptr<IT>(),    \
          tile_size, context);                                             \
    }                                                                      \
  } while (false)
  MINUET_FOR_ALL_IF_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__);
}

auto CUDATimeScatter(std::size_t tile_size,                   //
                     bool allow_shortcut_matmul,              //
                     const std::optional<double> &threshold,  //
                     const torch::Tensor &weights,            //
                     const torch::Tensor &source_masks,       //
                     const torch::Tensor &target_masks,       //
                     const torch::Tensor &kernel_map_sizes) {
  MINUET_ENSURE_TENSOR_NDIM(weights, 3);
  MINUET_ENSURE_TENSOR_NDIM(source_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(target_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(kernel_map_sizes, 1);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  MINUET_ENSURE_TENSOR_DIM(kernel_map_sizes, 0, num_offsets);
  MINUET_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);

  auto device = GetTorchDeviceFromTensors(
      {weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
  auto itype = GetTorchScalarTypeFromTensors(
      {source_masks, target_masks, kernel_map_sizes});
#define CASE(_, IT, FT)                                                    \
  do {                                                                     \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE &&                        \
        itype == torch::CppTypeToScalarType<IT>::value) {                  \
      return cuda::TimeScatter().operator()<IT, FT>(                       \
          num_sources, num_targets, num_offsets, threshold,                \
          allow_shortcut_matmul, num_source_features, num_target_features, \
          target_masks.data_ptr<IT>(), kernel_map_sizes.data_ptr<IT>(),    \
          tile_size, context);                                             \
    }                                                                      \
  } while (false)
  MINUET_FOR_ALL_IF_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__);
}

torch::Tensor CUDASparseConvolutionForward(
    std::size_t gather_tile_size,                          //
    std::size_t scatter_tile_size,                         //
    bool allow_shortcut_matmul,                            //
    std::size_t parallel,                                  //
    const std::optional<double> &threshold,                //
    const torch::Tensor &sources,                          //
    const torch::Tensor &weights,                          //
    const torch::Tensor &source_masks,                     //
    const torch::Tensor &target_masks,                     //
    const std::optional<torch::Tensor> &kernel_map_order,  //
    const torch::Tensor &kernel_map_sizes) {
  MINUET_ENSURE_TENSOR_NDIM(source_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(target_masks, 2);
  MINUET_ENSURE_TENSOR_NDIM(sources, 2);
  MINUET_ENSURE_TENSOR_NDIM(weights, 3);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  if (kernel_map_order.has_value()) {
    MINUET_ENSURE_TENSOR_DIM(kernel_map_order.value(), 0, num_offsets);
  }
  MINUET_ENSURE_TENSOR_DIM(kernel_map_sizes, 0, num_offsets);
  MINUET_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);
  MINUET_ENSURE_TENSOR_DIM(weights, 0, num_offsets);
  MINUET_ENSURE_TENSOR_DIM(sources, 1, num_source_features);

  auto device = GetTorchDeviceFromTensors(
      {sources, weights, source_masks, target_masks,
       kernel_map_order.value_or(sources), kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
  auto itype = GetTorchScalarTypeFromTensors(
      {source_masks, target_masks, kernel_map_order.value_or(source_masks),
       kernel_map_sizes});
  auto targets = torch::empty({num_targets, num_target_features},
                              torch::TensorOptions(device).dtype(ftype));
#define CASE(_, IT, FT)                                                      \
  do {                                                                       \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE &&                          \
        itype == torch::CppTypeToScalarType<IT>::value) {                    \
      if (kernel_map_order.has_value()) {                                    \
        cuda::SparseConvolutionForward().                 \
        operator()<IT, FT>(                                                  \
            num_sources, num_targets, num_offsets, threshold, parallel,      \
            allow_shortcut_matmul, num_source_features, num_target_features, \
            TypeConversion<IT>::GetCppPointer(source_masks),                 \
            TypeConversion<IT>::GetCppPointer(target_masks),                 \
            TypeConversion<IT>::GetCppPointer(kernel_map_order.value()),     \
            TypeConversion<IT>::GetCppPointer(kernel_map_sizes),             \
            TypeConversion<FT>::GetCppPointer(sources),                      \
            TypeConversion<FT>::GetCppPointer(weights),                      \
            TypeConversion<FT>::GetCppPointer(targets), gather_tile_size,    \
            scatter_tile_size, context);                                     \
      } else {                                                               \
        cuda::SparseConvolutionForward().                 \
        operator()<IT, FT>(                                                  \
            num_sources, num_targets, num_offsets, threshold, parallel,      \
            allow_shortcut_matmul, num_source_features, num_target_features, \
            TypeConversion<IT>::GetCppPointer(source_masks),                 \
            TypeConversion<IT>::GetCppPointer(target_masks), nullptr,        \
            TypeConversion<IT>::GetCppPointer(kernel_map_sizes),             \
            TypeConversion<FT>::GetCppPointer(sources),                      \
            TypeConversion<FT>::GetCppPointer(weights),                      \
            TypeConversion<FT>::GetCppPointer(targets), gather_tile_size,    \
            scatter_tile_size, context);                                     \
      }                                                                      \
      return targets;                                                        \
    }                                                                        \
  } while (false)
  MINUET_FOR_ALL_IF_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__);
}

MINUET_TORCH_REGISTER(cuda_sparse_convolution_forward,
                      CUDASparseConvolutionForward);
MINUET_TORCH_REGISTER(cuda_time_gather, CUDATimeGather);
MINUET_TORCH_REGISTER(cuda_time_scatter, CUDATimeScatter);
MINUET_TORCH_REGISTER(cuda_time_gemm, CUDATimeGEMM);

}  // namespace minuet