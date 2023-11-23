#include "minuet/cuda/functions/simple_hash.cuh"
#include "minuet/enabled_arguments.h"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

torch::Tensor CUDASimpleHash(const torch::Tensor &sources, bool reverse) {
  MINUET_ENSURE_TENSOR_NDIM(sources, 2);

  auto device = GetTorchDeviceFromTensors({sources});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ctype = sources.dtype().toScalarType();
  auto num_sources = sources.size(0);
  auto ndim = sources.size(1);
  torch::Tensor targets = torch::empty(
      {num_sources}, torch::TensorOptions(device).dtype(torch::kInt64));

#define CASE(_, T_NDIM, CT)                                                 \
  do {                                                                      \
    if (ctype == torch::CppTypeToScalarType<CT>::value && ndim == T_NDIM) { \
      cuda::SimpleHash().operator()<CT, T_NDIM>(                            \
          num_sources, sources.data_ptr<CT>(),                              \
          targets.data_ptr<std::int64_t>(), reverse, context);              \
      return targets;                                                       \
    }                                                                       \
  } while (false)
  MINUET_FOR_ALL_C_TYPES_AND_DIMS(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__, " for data type ",
               torch::toString(ctype), " with ndim=", ndim);
}

MINUET_TORCH_REGISTER(cuda_simple_hash, CUDASimpleHash);

}  // namespace minuet
