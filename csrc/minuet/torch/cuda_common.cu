#include "minuet/torch/cuda_common.cuh"

namespace minuet {

cuda::Context GetCUDAContextFromTorchCUDADevice(const torch::Device &device) {
  MINUET_CHECK(device.is_cuda(),
               "All provided tensors must be one CUDA device");
  auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  cuda::Context context(stream.device_index(), stream.stream());
  context.SetCUBLASHandle(at::cuda::getCurrentCUDABlasHandle());
  return context;
}

}  // namespace minuet
