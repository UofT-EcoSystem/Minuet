#include "minuet/cuda/buffer_pool.cuh"
#include "minuet/torch/cuda_common.cuh"

namespace minuet {

void CUDAPreallocateBuffer(std::size_t size) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  cuda::BufferPool::Global(stream.stream()).EnsureBuffer(size);
}

void CUDASetBufferGrowth(float growth) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  cuda::BufferPool::Global(stream.stream()).SetGrowth(growth);
}

void CUDASetBufferPageSize(std::size_t page_size) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  cuda::BufferPool::Global(stream.stream()).SetPageSize(page_size);
}

void CUDAFreeBuffers() {
  auto stream = c10::cuda::getCurrentCUDAStream();
  cuda::BufferPool::Global(stream.stream()).FreeBuffers();
}

std::size_t CUDABufferTotalSize() {
  auto stream = c10::cuda::getCurrentCUDAStream();
  return cuda::BufferPool::Global(stream).GetTotalSize();
}

std::size_t CUDABufferUsedSize() {
  auto stream = c10::cuda::getCurrentCUDAStream();
  return cuda::BufferPool::Global(stream).GetUsedSize();
}

void CUDAResetError() { cudaGetLastError(); }

void CUDADeviceReset() { MINUET_CHECK_CUDA(cudaDeviceReset()); }

MINUET_TORCH_REGISTER(cuda_preallocate_buffer, CUDAPreallocateBuffer);
MINUET_TORCH_REGISTER(cuda_free_buffers, CUDAFreeBuffers);
MINUET_TORCH_REGISTER(cuda_set_buffer_growth, CUDASetBufferGrowth);
MINUET_TORCH_REGISTER(cuda_set_buffer_page_size, CUDASetBufferPageSize);
MINUET_TORCH_REGISTER(cuda_buffer_used_size, CUDABufferUsedSize);
MINUET_TORCH_REGISTER(cuda_buffer_total_size, CUDABufferTotalSize);
MINUET_TORCH_REGISTER(cuda_reset_error, CUDAResetError);
MINUET_TORCH_REGISTER(cuda_device_reset, CUDADeviceReset);

}  // namespace minuet
