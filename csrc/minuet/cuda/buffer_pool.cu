#include <iostream>

#include "minuet/cuda/buffer_pool.cuh"

namespace minuet::cuda {

void BufferPool::EnsureBuffer(std::size_t size) {
  std::ptrdiff_t index = -1;
  for (UIter i = 0; i < buffers_.size(); i++) {
    if (size > buffers_[i]->idle()) {
      continue;
    }
    index = i;
  }
  if (index == -1) {
    FreeBuffers();
    auto buffer_size = page_size_;
    if (!buffers_.empty()) {
      buffer_size = static_cast<std::size_t>(
          std::ceil(static_cast<float>(size) * growth_));
    }
    buffer_size = std::max(size, buffer_size);
#ifdef MINUET_DEBUG
    std::cerr << "[MINUET] Create a new buffer with " << buffer_size
              << " bytes (for ensure " << size << " bytes)" << std::endl;
#endif
    index = static_cast<std::ptrdiff_t>(buffers_.size());
    buffers_.emplace_back(std::make_unique<Buffer>(
        Memory<OpaqueMemory>(stream_, buffer_size, upstream_factory_, false)));
  }
}

BufferPool::Buffer *BufferPool::EnsureBuffer(std::size_t size,
                                             std::size_t alignment) {
  alignment = 256;
  std::ptrdiff_t index = -1;
  for (UIter i = 0; i < buffers_.size(); i++) {
    std::size_t aligned_size = buffers_[i]->ComputeAlignedSize(size, alignment);
    if (aligned_size > buffers_[i]->idle()) {
      continue;
    }
    if (index != -1) {
      auto new_idle_after_allocation = buffers_[i]->idle() - aligned_size;
      auto old_idle_after_allocation =
          buffers_[index]->idle() -
          buffers_[index]->ComputeAlignedSize(size, alignment);
      if (old_idle_after_allocation <= new_idle_after_allocation) {
        continue;
      }
    }
    index = i;
  }
  if (index == -1) {
    FreeBuffers();
    auto buffer_size = page_size_;
    if (!buffers_.empty()) {
      buffer_size = static_cast<std::size_t>(
          std::ceil(static_cast<float>(size) * growth_));
    }
    buffer_size = std::max(size, buffer_size);
#ifdef MINUET_DEBUG
    std::cerr << "[MINUET] Create a new buffer with " << buffer_size
              << " bytes (required " << size << " bytes)" << std::endl;
#endif
    index = static_cast<std::ptrdiff_t>(buffers_.size());
    buffers_.emplace_back(std::make_unique<Buffer>(
        Memory<OpaqueMemory>(stream_, buffer_size, upstream_factory_, false)));
  }
  return buffers_[index].get();
}

BufferPool::~BufferPool() noexcept(false) {
  // Need to make sure all buffers are destructed first
  FreeBuffers();
  if (!buffers_.empty()) {
    std::cerr
        << "[MINUET] Destructing BufferPool with active buffers on stream "
        << stream_ << std::endl;
    for (const auto &buffer : buffers_) {
      std::cerr << "[MINUET] Found Buffer(used=" << buffer->used()
                << ", size=" << buffer->num_entries() << ")" << std::endl;
    }
    MINUET_ERROR("Cannot destruct buffer pool with actively allocated buffers");
  }
  if (is_factory_owned_) {
    delete upstream_factory_;
    upstream_factory_ = nullptr;
  }
}

BufferPool::BufferPool(cudaStream_t stream, float growth, std::size_t page_size,
                       MemoryFactory *factory, bool is_factory_owned)
    : stream_(stream),
      growth_(growth),
      page_size_(page_size),
      upstream_factory_(factory),
      is_factory_owned_(is_factory_owned),
      buffers_() {}

std::unordered_map<cudaStream_t, BufferPool> pool;

BufferPool &BufferPool::Global(cudaStream_t stream) {
  auto it = pool.find(stream);
  if (it != pool.end()) {
    return it->second;
  }
#ifdef MINUET_DEBUG
  std::cerr << "[MINUET] Create a buffer with stream " << stream << std::endl;
#endif
  pool.emplace(stream, BufferPool(stream, 1, 8 * 1024 * 1024,
                                  new AsyncDirectMemoryFactory(stream), true));
  return pool.at(stream);
}

BufferPool::BufferPool(BufferPool &&other) noexcept
    : stream_(other.stream_),
      growth_(other.growth_),
      upstream_factory_(other.upstream_factory_),
      page_size_(other.page_size_),
      is_factory_owned_(other.is_factory_owned_),
      buffers_(std::move(other.buffers_)) {
  other.upstream_factory_ = nullptr;
  other.is_factory_owned_ = false;
}

BufferPool &BufferPool::operator=(BufferPool &&other) noexcept {
  if (this != &other) {
    std::swap(this->stream_, other.stream_);
    std::swap(this->growth_, other.growth_);
    std::swap(this->buffers_, other.buffers_);
    std::swap(this->is_factory_owned_, other.is_factory_owned_);
    std::swap(this->page_size_, other.page_size_);
    std::swap(this->upstream_factory_, other.upstream_factory_);
  }
  return *this;
}
void BufferPool::SetGrowth(float growth) {
  MINUET_CHECK(growth >= 1, "growth has to be larger than 1 but found ",
               growth);
  this->growth_ = growth;
}
void BufferPool::SetPageSize(std::size_t page_size) {
  MINUET_CHECK(page_size > 0, "page_size has to be larger than 0 but found ",
               page_size);
  this->page_size_ = page_size;
}

void BufferPool::FreeBuffers() {
  std::vector<std::unique_ptr<Buffer>> nonempty_buffers;
  for (auto &buffer : buffers_) {
    if (buffer->used()) {
      nonempty_buffers.emplace_back(std::move(buffer));
    }
  }
  buffers_.swap(nonempty_buffers);
}

std::size_t BufferPool::GetUsedSize() const {
  std::size_t result = 0;
  for (const auto &buffer : buffers_) {
    result += buffer->used();
  }
  return result;
}

std::size_t BufferPool::GetTotalSize() const {
  std::size_t result = 0;
  for (const auto &buffer : buffers_) {
    result += buffer->total();
  }
  return result;
}

void *BufferPool::Buffer::Acquire(std::size_t size, std::size_t alignment) {
  alignment = 256;
  MINUET_CHECK(alignment == 0 || CUDA_DEFAULT_ALIGNMENT % alignment == 0,
               "Cannot allocate CUDA memory with alignment ", alignment);
  std::size_t aligned_size = ComputeAlignedSize(size, alignment);
  std::size_t padding_size = ComputePaddingSize(alignment);
  MINUET_CHECK(aligned_size <= idle(), "Acquire ", aligned_size, " bytes (",
               size, " bytes acquired without alignment) where only ", idle(),
               " out of ", total(), " bytes is free at the moment");
  allocated_.emplace_back(storage_.device_data() + used_size_ + padding_size,
                          aligned_size);
  used_size_ += aligned_size;
  return allocated_.back().first;
}

void BufferPool::Buffer::Release(void *data) {
  MINUET_CHECK(!allocated_.empty() && allocated_.back().first == data,
               "Cannot find the pointer to be released in the current buffer");
  used_size_ -= allocated_.back().second;
  allocated_.pop_back();
}

void BufferPool::Buffer::SilentRelease(void *data) noexcept {
  if (!allocated_.empty() && allocated_.back().first == data) {
    used_size_ -= allocated_.back().second;
    allocated_.pop_back();
  } else {
    std::cerr << "[MINUET] Warning: Releasing a buffer that is not on the top "
                 "causes memory leak\n";
  }
}

std::size_t BufferPool::Buffer::ComputePaddingSize(
    std::size_t alignment) const {
  return alignment == 0 ? 0 : (alignment - used_size_ % alignment) % alignment;
}

std::size_t BufferPool::Buffer::ComputeAlignedSize(
    std::size_t size, std::size_t alignment) const {
  std::size_t padding_size = ComputePaddingSize(alignment);
  return size + padding_size;
}

}  // namespace minuet::cuda