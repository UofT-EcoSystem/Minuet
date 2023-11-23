#include "minuet/torch/common.h"

#include "minuet/common/exception.h"

namespace minuet {

namespace detail {

std::vector<std::function<void(py::module &)>> &GlobalFunctions() {
  static std::vector<std::function<void(py::module &)>> functions;
  return functions;
}

std::size_t Register(std::function<void(py::module &)> function) {
  GlobalFunctions().push_back(std::move(function));
  return GlobalFunctions().size();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  for (const auto &function : GlobalFunctions()) {
    function(m);
  }
}

}  // namespace detail

torch::Device GetTorchDeviceFromTensors(
    const std::vector<torch::Tensor> &tensors) {
  MINUET_CHECK(!tensors.empty(),
               "Must provide at least one tensor for retrieving device");
  auto device = tensors.begin()->device();
  for (const auto &tensor : tensors) {
    MINUET_CHECK(device == tensor.device(),
                 "Found tensors on two different device (", device.str(),
                 " != ", tensor.device().str(), ")");
  }
  return device;
}

void EnsureTensorNDim(const std::string &name, const torch::Tensor &tensor,
                      std::int64_t ndim) {
  MINUET_CHECK(tensor.ndimension() == ndim, "Tensor ", name,
               " is expected to have ", ndim, " dimension(s) but found ",
               tensor.ndimension(), " dimension(s)");
}

void EnsureTensorDim(const std::string &name, const torch::Tensor &tensor,
                     std::int64_t dim, std::int64_t size) {
  MINUET_CHECK(tensor.size(dim) == size, "The dimension ", dim, " of Tensor ",
               name, " is expected to be ", size, " but found ",
               tensor.size(dim));
}

torch::ScalarType GetTorchScalarTypeFromTensors(
    const std::vector<torch::Tensor> &tensors) {
  MINUET_CHECK(!tensors.empty(),
               "Must provide at least one tensor for retrieving dtype");
  auto dtype = tensors.begin()->dtype().toScalarType();
  for (const auto &tensor : tensors) {
    MINUET_CHECK(dtype == tensor.dtype().toScalarType(),
                 "Found tensors with two different scalar types (",
                 torch::toString(dtype),
                 " != ", torch::toString(tensor.dtype().toScalarType()), ")");
  }
  return dtype;
}

}  // namespace minuet
