#include <memory>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

Context::Context(int device, cudaStream_t stream)
    : device_(device), stream_(stream), cublas_handle_() {
  MINUET_CHECK(NumDevices() > 0, "No CUDA device is found");
  MINUET_CHECK(0 <= device && device < NumDevices(),
               "Device ID must be between 0 and ", NumDevices(), " but found ",
               device_);
}

Context::Context(cudaStream_t stream)
    : device_(0), stream_(stream), cublas_handle_() {
  MINUET_CHECK_CUDA(cudaGetDevice(&device_));
}

int Context::NumDevices() {
  static std::unique_ptr<int> num_devices;
  if (num_devices == nullptr) {
    num_devices = std::make_unique<int>();
    MINUET_CHECK_CUDA(cudaGetDeviceCount(num_devices.get()));
  }
  return *num_devices;
}

std::unordered_map<int, cudaDeviceProp> cached_device_properties_;

const cudaDeviceProp &Context::GetDeviceProp() const {
  auto result = cached_device_properties_.try_emplace(device_);
  if (result.second) {
    MINUET_CHECK_CUDA(cudaGetDeviceProperties(&result.first->second, device_));
  }
  return result.first->second;
}

}  // namespace minuet::cuda