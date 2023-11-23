#include "minuet/cuda/event.cuh"

namespace minuet::cuda {

void Event::Record(cudaStream_t stream) {
  MINUET_CHECK(!closed_, "Event is invalid");
  MINUET_CHECK_CUDA(cudaEventRecord(event_, stream));
}

void Event::Synchronize() {
  MINUET_CHECK(!closed_, "Event is invalid");
  MINUET_CHECK_CUDA(cudaEventSynchronize(event_));
}

float Event::Elapsed(const Event &after) const {
  float result;
  MINUET_CHECK(!closed_, "Event is invalid");
  MINUET_CHECK_CUDA(cudaEventElapsedTime(&result, event_, after.event()));
  return result;
}

void Event::Close() {
  if (!closed_) {
    MINUET_CHECK_CUDA(cudaEventDestroy(event_));
    closed_ = true;
  }
}

void Event::CloseNoExcept() noexcept {
  if (!closed_) {
    cudaEventDestroy(event_);
    closed_ = true;
  }
}

}  // namespace minuet::cuda