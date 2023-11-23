#pragma once

#include <cstddef>
#include <memory>

#include "minuet/common/exception.h"
#include "minuet/common/functions.h"
#include "minuet/cpu/memory.h"

namespace minuet::cuda {

template <typename T>
MINUET_DEVICE T *DynamicSharedMemory() {
  // TODO: Do we need this alignment? Probably not
  extern __shared__ __align__(1024) unsigned char memory[];
  return reinterpret_cast<T *>(memory);
}

constexpr const std::size_t CUDA_DEFAULT_ALIGNMENT = 256;

typedef unsigned char OpaqueMemory;

class MemoryFactory {
 public:
  MemoryFactory() = default;
  MemoryFactory(const MemoryFactory &) = delete;
  MemoryFactory &operator=(const MemoryFactory &) = delete;
  MemoryFactory(MemoryFactory &&) noexcept = delete;
  MemoryFactory &operator=(MemoryFactory &&) noexcept = delete;
  virtual ~MemoryFactory() = default;

  [[nodiscard]] virtual void *Acquire(std::size_t size,
                                      std::size_t alignment) = 0;
  virtual void Release(void *data) = 0;
  virtual void SilentRelease(void *data) noexcept = 0;

  template <typename T>
  T *Acquire(std::size_t size, std::size_t alignment) {
    return reinterpret_cast<T *>(Acquire(size * SizeOf<T>, alignment));
  }

  template <typename T>
  T *Acquire(std::size_t size) {
    return Acquire<T>(size, AlignOf<T>);
  }
};

class DirectMemoryFactory : public MemoryFactory {
 public:
  [[nodiscard]] void *Acquire(std::size_t size, std::size_t alignment) override;
  void Release(void *data) override;
  void SilentRelease(void *data) noexcept override;
};

class AsyncDirectMemoryFactory : public MemoryFactory {
 public:
  explicit AsyncDirectMemoryFactory(cudaStream_t stream) : stream_(stream) {}

  [[nodiscard]] void *Acquire(std::size_t size, std::size_t alignment) override;
  void Release(void *data) override;
  void SilentRelease(void *data) noexcept override;

 private:
  cudaStream_t stream_;
};

template <typename T = void>
class Memory {
 public:
  template <typename FactoryT>
  Memory(cudaStream_t stream, std::size_t size, std::size_t alignment,
         FactoryT *factory, bool is_factory_owned)
      : stream_(stream),
        factory_(factory),
        is_factory_owned_(is_factory_owned),
        d_data_(factory_->Acquire<T>(size, alignment)),
        size_(size) {}

  template <typename FactoryT>
  Memory(cudaStream_t stream, std::size_t size, FactoryT *factory,
         bool is_factory_owned)
      : stream_(stream),
        factory_(factory),
        is_factory_owned_(is_factory_owned),
        d_data_(factory_->Acquire<T>(size)),
        size_(size) {}

  template <typename FactoryT>
  Memory(cudaStream_t stream, T *data, std::size_t size, FactoryT *factory,
         bool is_factory_owned)
      : stream_(stream),
        factory_(factory),
        is_factory_owned_(is_factory_owned),
        d_data_(data),
        size_(size) {}

  Memory(const Memory &other) = delete;
  Memory &operator=(const Memory &other) = delete;

  Memory(Memory &&other) noexcept
      : stream_(other.stream_),
        factory_(other.factory_),
        is_factory_owned_(other.is_factory_owned_),
        d_data_(other.d_data_),
        size_(other.size_) {
    other.factory_ = nullptr;
    other.is_factory_owned_ = false;
    other.d_data_ = nullptr;
    other.size_ = 0;
  }

  Memory &operator=(Memory &&other) noexcept {
    if (this != &other) {
      std::swap(this->stream_, other.stream_);
      std::swap(this->factory_, other.factory_);
      std::swap(this->is_factory_owned_, other.is_factory_owned_);
      std::swap(this->size_, other.size_);
      std::swap(this->d_data_, other.d_data_);
    }
    return *this;
  }

  virtual ~Memory() {
    if (d_data_ != nullptr) {
      factory_->SilentRelease(d_data_);
      d_data_ = nullptr;
      size_ = 0;
    }
    if (is_factory_owned_) {
      delete factory_;
      factory_ = nullptr;
    }
  }

  [[nodiscard]] MINUET_HOST_DEVICE const T *device_data() const {
    return d_data_;
  }
  [[nodiscard]] MINUET_HOST_DEVICE T *device_data() { return d_data_; }
  [[nodiscard]] MINUET_HOST_DEVICE std::size_t size() const { return size_; }

  void CopyTo(T *target, std::size_t size, std::size_t base = 0,
              cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) const {
    if (stream_ == cudaStreamDefault) {
      MINUET_CHECK_CUDA(
          cudaMemcpy(target, d_data_ + base, size * SizeOf<T>, kind));
    } else {
      MINUET_CHECK_CUDA(cudaMemcpyAsync(target, d_data_ + base,
                                        size * SizeOf<T>, kind, stream_));
      MINUET_CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
  }
  void CopyTo(T *target,
              cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) const {
    CopyTo(target, size_, 0, kind);
  }
  void AsyncCopyTo(
      T *target, std::size_t size, std::size_t base = 0,
      cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) const {
    MINUET_CHECK_CUDA(cudaMemcpyAsync(target, d_data_ + base, size * SizeOf<T>,
                                      kind, stream_));
  }
  void AsyncCopyTo(T *target, cudaMemcpyKind kind =
                                  cudaMemcpyKind::cudaMemcpyDefault) const {
    AsyncCopyTo(target, size_, 0, kind);
  }

  void CopyFrom(const T *source, std::size_t size, std::size_t base = 0,
                cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) {
    if (stream_ == cudaStreamDefault) {
      MINUET_CHECK_CUDA(
          cudaMemcpy(d_data_ + base, source, size * SizeOf<T>, kind));
    } else {
      MINUET_CHECK_CUDA(cudaMemcpyAsync(d_data_ + base, source,
                                        size * SizeOf<T>, kind, stream_));
      MINUET_CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
  }
  void CopyFrom(const T *source,
                cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) {
    CopyFrom(source, size_, 0, kind);
  }
  void AsyncCopyFrom(const T *source, std::size_t size, std::size_t base = 0,
                     cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) {
    MINUET_CHECK_CUDA(cudaMemcpyAsync(d_data_ + base, source, size * SizeOf<T>,
                                      kind, stream_));
  }
  void AsyncCopyFrom(const T *source,
                     cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDefault) {
    AsyncCopyFrom(source, size_, 0, kind);
  }

  void Memset(int value, std::size_t size, std::size_t base = 0) {
    if (stream_ == cudaStreamDefault) {
      MINUET_CHECK_CUDA(cudaMemset(d_data_ + base, value, SizeOf<T> * size));
    } else {
      MINUET_CHECK_CUDA(
          cudaMemsetAsync(d_data_ + base, value, SizeOf<T> * size, stream_));
      MINUET_CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
  }
  void Memset(int value) { Memset(value, size_, 0); }

  void AsyncMemset(int value, std::size_t size, std::size_t base = 0) {
    MINUET_CHECK_CUDA(
        cudaMemsetAsync(d_data_ + base, value, SizeOf<T> * size, stream_));
  }
  void AsyncMemset(int value) { AsyncMemset(value, size_, 0); }

 private:
  cudaStream_t stream_;
  MemoryFactory *factory_;
  bool is_factory_owned_;
  T *d_data_;
  std::size_t size_;
};

}  // namespace minuet::cuda