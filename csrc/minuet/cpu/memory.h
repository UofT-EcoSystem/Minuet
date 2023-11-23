#pragma once

#include <cstddef>
#include <type_traits>

namespace minuet::cpu {

namespace detail {

template <typename T>
T *AcquireMemory(std::size_t size) {
  return new T[size];
}

template <>
void *AcquireMemory(std::size_t size);

template <typename T>
void ReleaseMemory(T *data) noexcept {
  delete[] (data);
}

template <>
void ReleaseMemory(void *data) noexcept;

}  // namespace detail

template <typename T = void>
class Memory final {
 public:
  explicit Memory(std::size_t size)
      : Memory(detail::AcquireMemory<T>(size), size) {}

  Memory(const Memory &) = delete;
  Memory(Memory &&other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
  }
  Memory &operator=(const Memory &) = delete;
  Memory &operator=(Memory &&other) noexcept {
    if (this != &other) {
      std::swap(this->data_, other.data_);
      std::swap(this->size_, other.size_);
    }
    return *this;
  }

  ~Memory() { SafeRelease(); }

  const T *data() const { return data_; }
  T *data() { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }

  virtual void Reset(T *data) noexcept {
    SafeRelease();
    this->data_ = data;
  }

  template <typename OffsetT>
  T &operator[](OffsetT offset) {
    return data_[offset];
  }

  template <typename OffsetT>
  const T &operator[](OffsetT offset) const {
    return data_[offset];
  }

 protected:
  explicit Memory(T *d_data, std::size_t size) : data_(d_data), size_(size) {}
  virtual void SafeRelease() noexcept {
    // Note that we don't really check the return value here to make
    // the destructor and move constructor noexcept
    detail::ReleaseMemory(data_);
    data_ = nullptr;
  }

  T *data_;
  std::size_t size_;
};

}  // namespace minuet::cpu