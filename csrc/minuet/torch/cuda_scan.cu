#include "minuet/common/exception.h"
#include "minuet/cuda/functions/scan.cuh"
#include "minuet/enabled_arguments.h"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

enum class CumulativeSumType {
  kInclusive = 0,
  kExclusive = 1,
  kTwoSided = 2,
  kKernelMapRanks = 3
};

torch::Tensor CUDAMultiCumulativeSum(const torch::Tensor &sources,
                                     std::string cumsum_type) {
  MINUET_CHECK(sources.ndimension() >= 1,
               "Tensor sources must have at least 1 dimensions");

  auto device = GetTorchDeviceFromTensors({sources});
  auto context = GetCUDAContextFromTorchCUDADevice(device);

  CumulativeSumType type;
  if (cumsum_type == "exclusive") {
    type = CumulativeSumType::kExclusive;
  } else if (cumsum_type == "inclusive") {
    type = CumulativeSumType::kInclusive;
  } else if (cumsum_type == "two-sided") {
    type = CumulativeSumType::kTwoSided;
  } else if (cumsum_type == "kernel-map-ranks") {
    type = CumulativeSumType::kKernelMapRanks;
  } else {
    MINUET_ERROR("Unknown cumsum type ", cumsum_type);
  }

  auto num_sources = sources.size(-1);
  auto batch_size = sources.numel() / num_sources;
  auto itype = sources.dtype().toScalarType();

  std::vector<std::int64_t> targets_shape(
      sources.sizes().data(), sources.sizes().data() + sources.sizes().size());
  if (type == CumulativeSumType::kTwoSided ||
      type == CumulativeSumType::kKernelMapRanks) {
    targets_shape.back()++;
  }

  auto targets = torch::empty(
      targets_shape,
      torch::TensorOptions(sources.device()).dtype(sources.dtype()));
#define CASE(_, IT)                                          \
  do {                                                       \
    if (itype == torch::CppTypeToScalarType<IT>::value) {    \
      if (type == CumulativeSumType::kExclusive) {           \
        cuda::MultiExclusiveSum().operator()<IT>(            \
            batch_size, num_sources, sources.data_ptr<IT>(), \
            targets.data_ptr<IT>(), context);                \
      } else if (type == CumulativeSumType::kInclusive) {    \
        cuda::MultiInclusiveSum().operator()<IT>(            \
            batch_size, num_sources, sources.data_ptr<IT>(), \
            targets.data_ptr<IT>(), context);                \
      } else if (type == CumulativeSumType::kTwoSided) {     \
        cuda::MultiTwoSidedSum().operator()<IT>(             \
            batch_size, num_sources, sources.data_ptr<IT>(), \
            targets.data_ptr<IT>(), context);                \
      } else {                                               \
        cuda::MultiTwoSidedKernelMapRanks().operator()<IT>(  \
            batch_size, num_sources, sources.data_ptr<IT>(), \
            targets.data_ptr<IT>(), context);                \
      }                                                      \
      return targets;                                        \
    }                                                        \
  } while (false)
  MINUET_FOR_ALL_I_TYPES(CASE);
#undef CASE
  MINUET_ERROR("Cannot find implementation of ", __func__,
               " for tensor data type ", torch::toString(itype));
}

MINUET_TORCH_REGISTER(cuda_multi_cumsum, CUDAMultiCumulativeSum);

}  // namespace minuet
