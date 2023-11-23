#pragma once

#ifndef __NVCC__
#error "minuet/cuda/common.cuh must be compiled with NVCC"
#endif

#include <cooperative_groups.h>
#include <cublas_v2.h>

#include "minuet/common/exception.h"
#include "minuet/common/functions.h"
#include "minuet/common/stringify.h"
#include "minuet/cuda/buffer_pool.cuh"
#include "minuet/cuda/memory.cuh"

#define MINUET_GLOBAL_THREAD_ID(dim) \
  ((blockIdx.dim) * (blockDim.dim) + (threadIdx.dim))
#define MINUET_N_GLOBAL_THREADS(dim) ((gridDim.dim) * (blockDim.dim))
#define MINUET_LOCAL_THREAD_ID(dim) (threadIdx.dim)
#define MINUET_N_LOCAL_THREADS(dim) (blockDim.dim)
#define MINUET_THREAD_BLOCK_ID(dim) (blockIdx.dim)
#define MINUET_N_THREAD_BLOCKS(dim) (gridDim.dim)
#define MINUET_FOR_DIMENSION_AND_INDEX(MACRO, DELIMITER) \
  MACRO(x, 0) DELIMITER MACRO(y, 1) DELIMITER MACRO(z, 2)

namespace minuet::cuda {

namespace cg = ::cooperative_groups;
constexpr const auto WARP_SIZE = 32;

class Context {
 private:
  typedef decltype(std::declval<dim3>().x) CUDADimType;

 public:
  explicit Context(cudaStream_t stream = cudaStreamDefault);
  explicit Context(int device, cudaStream_t stream = cudaStreamDefault);

  template <typename T>
  T ReadDeviceData(const T *data) const {
    T result;
    if (stream_ == cudaStreamDefault) {
      MINUET_CHECK_CUDA(
          cudaMemcpy(&result, data, sizeof(T), cudaMemcpyDeviceToHost));
    } else {
      MINUET_CHECK_CUDA(cudaMemcpyAsync(&result, data, sizeof(T),
                                        cudaMemcpyDeviceToHost, stream_));
      Synchronize();
    }
    return result;
  }

  template <typename T>
  void ReadDeviceData(const T *device_data, T *host_data,
                      std::size_t size) const {
    if (stream_ == cudaStreamDefault) {
      MINUET_CHECK_CUDA(cudaMemcpy(host_data, device_data, sizeof(T) * size,
                                   cudaMemcpyDeviceToHost));
    } else {
      MINUET_CHECK_CUDA(cudaMemcpyAsync(host_data, device_data,
                                        sizeof(T) * size,
                                        cudaMemcpyDeviceToHost, stream_));
    }
  }

  static int NumDevices();
  [[nodiscard]] cudaStream_t stream() const { return stream_; }

  void Synchronize() const {
    MINUET_CHECK_CUDA(cudaStreamSynchronize(stream_));
  }
  void Synchronize(const std::string &hint) const {
    MINUET_CHECK_CUDA_WITH_HINT(cudaStreamSynchronize(stream_), hint);
  }

  static void DeviceSynchronize() {
    MINUET_CHECK_CUDA(cudaDeviceSynchronize());
  }

  static void DeviceSynchronize(const std::string &hint) {
    MINUET_CHECK_CUDA_WITH_HINT(cudaDeviceSynchronize(), hint);
  }

  /*
   * Launch a kernel with given block size. The grid size is automatically
   * calculated based on the problem size.
   */
  template <typename KernelT, typename... ArgTs>
  void Launch(dim3 size, dim3 block_size, std::size_t shared_memory,
              KernelT kernel, ArgTs... args) const {
    static_assert(std::is_invocable_r_v<void, KernelT, ArgTs...>);
    const auto &device_prop = GetDeviceProp();

#define CHECK_BLOCK_SIZE(dim, id)                                              \
  do {                                                                         \
    if (block_size.dim > device_prop.maxThreadsDim[id]) {                      \
      MINUET_ERROR("The dimension" #dim " of the block size (",                \
                   block_size.dim, ") is larger than supported (",             \
                   device_prop.maxThreadsDim[id], ")");                        \
    }                                                                          \
    if (block_size.dim == 0) {                                                 \
      MINUET_ERROR("The dimension " #dim " of the block size cannot be zero"); \
    }                                                                          \
  } while (false)
    MINUET_FOR_DIMENSION_AND_INDEX(CHECK_BLOCK_SIZE, MINUET_SEMICOLON);
#undef CHECK_BLOCK_SIZE

    auto num_threads_per_block = static_cast<std::size_t>(block_size.x) *
                                 static_cast<std::size_t>(block_size.y) *
                                 static_cast<std::size_t>(block_size.z);
    if (num_threads_per_block > device_prop.maxThreadsPerBlock) {
      MINUET_ERROR("The number of threads in each block (",
                   num_threads_per_block,
                   ") for the kernel is larger than supported (",
                   device_prop.maxThreadsPerBlock, ")");
    }
    if (num_threads_per_block % device_prop.warpSize != 0) {
      MINUET_ERROR("The block size (", block_size.x, ", ", block_size.y, ", ",
                   block_size.z, ") must be multiple of warp size (",
                   device_prop.warpSize, ")");
    }

#define GRID_SIZE(dim, id)                                         \
  (std::min(static_cast<CUDADimType>(device_prop.maxGridSize[id]), \
            static_cast<CUDADimType>(DivCeil(size.dim, block_size.dim))))

    dim3 grid_size(MINUET_FOR_DIMENSION_AND_INDEX(GRID_SIZE, MINUET_COMMA));
#undef GRID_SIZE

    kernel<<<grid_size, block_size, shared_memory, stream_>>>(
        std::forward<ArgTs>(args)...);
    MINUET_CHECK_CUDA(cudaGetLastError());
  }

  void SetCUBLASHandle(const cublasHandle_t &handle) {
    this->cublas_handle_ = handle;
  }

  [[nodiscard]] cublasHandle_t GetCUBLASHandle() const {
    return cublas_handle_;
  }

  template <typename T = void>
  [[nodiscard]] Memory<T> NewMemory(std::size_t size) const {
    return Memory<T>(stream_, size, new AsyncDirectMemoryFactory(stream_),
                     true);
  }

  template <typename T = void>
  [[nodiscard]] Memory<T> NewMemoryLike(const cpu::Memory<T> &memory) const {
    return NewMemory<T>(memory.size());
  }

  template <typename T = void>
  [[nodiscard]] Memory<T> NewMemoryFrom(const cpu::Memory<T> &memory) const {
    auto result = NewMemory<T>(memory.size());
    result.CopyFrom(memory.data());
    return result;
  }

  template <typename T = void>
  [[nodiscard]] Memory<T> NewBuffer(std::size_t size) const {
    return BufferPool::Global(stream_).Acquire<T>(size);
  }

  template <typename T = void>
  [[nodiscard]] Memory<T> NewBufferLike(const cpu::Memory<T> &memory) const {
    return NewBuffer<T>(memory.size());
  }

  template <typename T = void>
  [[nodiscard]] Memory<T> NewBufferFrom(const cpu::Memory<T> &memory) const {
    auto result = NewBufferLike(memory);
    result.CopyFrom(memory.data());
    return result;
  }

  [[nodiscard]] const cudaDeviceProp &GetDeviceProp() const;

 private:
  int device_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;
};

}  // namespace minuet::cuda
