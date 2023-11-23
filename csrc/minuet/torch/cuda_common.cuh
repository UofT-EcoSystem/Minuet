#pragma once

#include "minuet/cuda/context.cuh"
#include "minuet/torch/common.h"

namespace minuet {

cuda::Context GetCUDAContextFromTorchCUDADevice(const torch::Device &device);

}  // namespace minuet
