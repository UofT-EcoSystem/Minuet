#pragma once

#include "minuet/common/exception.h"

namespace minuet::cuda {

class Event final {
 public:
  explicit Event() : event_(), closed_(false) {
    MINUET_CHECK_CUDA(cudaEventCreate(&event_));
  }
  explicit Event(unsigned int flags) : event_(), closed_() {
    MINUET_CHECK_CUDA(cudaEventCreate(&event_, flags));
  }

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      std::swap(this->event_, other.event_);
      std::swap(this->closed_, other.closed_);
    }
    return *this;
  }
  Event(Event &&other) noexcept : event_(other.event_), closed_(other.closed_) {
    other.closed_ = true;
  }

  void Record(cudaStream_t stream = cudaStreamDefault);
  void Synchronize();
  [[nodiscard]] float Elapsed(const Event &after) const;

  [[nodiscard]] cudaEvent_t event() const {
    MINUET_CHECK(!closed_, "Event is invalid");
    return event_;
  }

  void Close();

  void CloseNoExcept() noexcept;

  ~Event() { CloseNoExcept(); }

 private:
  bool closed_;
  cudaEvent_t event_;
};

}  // namespace minuet::cuda