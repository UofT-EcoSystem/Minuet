#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <functional>

namespace minuet {

namespace detail {

std::size_t Register(std::function<void(py::module &)> function);

}

#define MINUET_TORCH_REGISTER(NAME, ...)                                 \
  namespace detail {                                                     \
  void TorchRegister##NAME(py::module &m) { m.def(#NAME, __VA_ARGS__); } \
  static std::size_t torch_##NAME = Register(TorchRegister##NAME);       \
  }

template <typename T>
struct TypeConversion {
  static constexpr const auto TORCH_DTYPE =
      torch::CppTypeToScalarType<T>::value;
  static T *GetCppPointer(const torch::Tensor &tensor) {
    return tensor.data_ptr<T>();
  }
};

template <>
struct TypeConversion<half> {
  static constexpr const auto TORCH_DTYPE = torch::kHalf;
  static half *GetCppPointer(const torch::Tensor &tensor) {
    return reinterpret_cast<half *>(tensor.data_ptr<at::Half>());
  }
};

torch::Device GetTorchDeviceFromTensors(
    const std::vector<torch::Tensor> &tensors);
void EnsureTensorNDim(const std::string &name, const torch::Tensor &tensor,
                      std::int64_t ndim);
torch::ScalarType GetTorchScalarTypeFromTensors(
    const std::vector<torch::Tensor> &tensors);
void EnsureTensorDim(const std::string &name, const torch::Tensor &tensor,
                     std::int64_t dim, std::int64_t size);

#define MINUET_ENSURE_TENSOR_NDIM(TENSOR, NDIM) \
  EnsureTensorNDim(#TENSOR, TENSOR, NDIM)
#define MINUET_ENSURE_TENSOR_DIM(TENSOR, DIM, SIZE) \
  EnsureTensorDim(#TENSOR, TENSOR, DIM, SIZE)

}  // namespace minuet
