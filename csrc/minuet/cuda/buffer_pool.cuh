#pragma once

#include "minuet/cuda/memory.cuh"

namespace minuet::cuda {

class BufferPool {
 public:
  // FIXME: Need to make sure factory is also attached to the same CUDA stream
  explicit BufferPool(cudaStream_t stream, float growth, std::size_t page_size,
                      MemoryFactory *factory, bool is_factory_owned);

  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;
  BufferPool(BufferPool &&) noexcept;
  BufferPool &operator=(BufferPool &&) noexcept;
  virtual ~BufferPool()  noexcept(false);

  static BufferPool &Global(cudaStream_t stream);

  void SetGrowth(float growth);
  void SetPageSize(std::size_t page_size);
  void FreeBuffers();

  template <typename T>
  Memory<T> Acquire(std::size_t size) {
    auto buffer = EnsureBuffer(size * SizeOf<T>, AlignOf<T>);
    auto data =
        reinterpret_cast<T *>(buffer->Acquire(size * SizeOf<T>, AlignOf<T>));
    return {stream_, data, size, buffer, false};
  }
  void EnsureBuffer(std::size_t size);
  [[nodiscard]] std::size_t GetUsedSize() const;
  [[nodiscard]] std::size_t GetTotalSize() const;

 private:
  class Buffer : public MemoryFactory {
   public:
    explicit Buffer(Memory<OpaqueMemory> storage)
        : storage_(std::move(storage)), used_size_(0) {}

    [[nodiscard]] std::size_t used() const { return used_size_; }
    [[nodiscard]] std::size_t idle() const {
      return storage_.size() - used_size_;
    }
    [[nodiscard]] std::size_t num_entries() const { return allocated_.size(); }
    [[nodiscard]] std::size_t total() const { return storage_.size(); }
    [[nodiscard]] std::size_t ComputePaddingSize(std::size_t alignment) const;
    [[nodiscard]] std::size_t ComputeAlignedSize(std::size_t size,
                                                 std::size_t alignment) const;
    [[nodiscard]] void *Acquire(std::size_t size,
                                std::size_t alignment) override;
    void Release(void *data) override;
    void SilentRelease(void *data) noexcept override;

   private:
    Memory<OpaqueMemory> storage_;
    std::size_t used_size_;
    std::vector<std::pair<void *, std::size_t>> allocated_;
  };
  Buffer *EnsureBuffer(std::size_t size, std::size_t alignment);

  cudaStream_t stream_;
  float growth_;
  std::size_t page_size_;
  std::vector<std::unique_ptr<Buffer>> buffers_;
  MemoryFactory *upstream_factory_;
  bool is_factory_owned_;
};

}  // namespace minuet::cuda