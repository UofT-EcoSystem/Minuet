#pragma once

#include "minuet/common/exception.h"

namespace minuet::cuda {

class Stream final {
 public:
  Stream() : stream_(), is_owned_(true) {
    MINUET_CHECK_CUDA(cudaStreamCreate(&stream_));
  }

  explicit Stream(unsigned int flags) : stream_(), is_owned_(true) {
    MINUET_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, flags));
  }

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(Stream &&other) noexcept
      : stream_(other.stream_), is_owned_(other.is_owned_) {
    other.is_owned_ = false;
  }
  Stream &operator=(Stream &&other) noexcept {
    if (this != &other) {
      std::swap(this->stream_, other.stream_);
      std::swap(this->is_owned_, other.is_owned_);
    }
    return *this;
  }

  void Close() {
    if (is_owned_) {
      MINUET_CHECK_CUDA(cudaStreamDestroy(stream_));
      is_owned_ = false;
    }
  }

  void CloseNoExcept() noexcept {
    if (is_owned_) {
      cudaStreamDestroy(stream_);
      is_owned_ = false;
    }
  }

  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  void Synchronize() { MINUET_CHECK_CUDA(cudaStreamSynchronize(stream_)); }

  ~Stream() { CloseNoExcept(); }

 private:
  cudaStream_t stream_;
  bool is_owned_;
};

}  // namespace minuet::cuda