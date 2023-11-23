#include <cublas_v2.h>

#include "minuet/cuda/helpers.cuh"

namespace minuet::cuda {

std::string CUBLASErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown Error";
}

#define MINUET_CHECK_CUBLAS(stmt, ...)                              \
  do {                                                              \
    auto return_code = (stmt);                                      \
    MINUET_CHECK(return_code == CUBLAS_STATUS_SUCCESS,              \
                 "CUBLAS Error: ", CUBLASErrorString(return_code)); \
  } while (false)

template <typename T>
struct CUBLASDispatcher;

template <>
struct CUBLASDispatcher<half> {
  static constexpr const auto GEMM = cublasHgemm;
  static constexpr const auto GEMMStridedBatched = cublasHgemmStridedBatched;
};

template <>
struct CUBLASDispatcher<float> {
  static constexpr const auto GEMM = cublasSgemm;
  static constexpr const auto GEMMStridedBatched = cublasSgemmStridedBatched;
};

template <>
struct CUBLASDispatcher<double> {
  static constexpr const auto GEMM = cublasDgemm;
  static constexpr const auto GEMMStridedBatched = cublasDgemmStridedBatched;
};

template <typename FT>
void MatMul(std::size_t m,           //
            std::size_t k,           //
            std::size_t n,           //
            bool is_a_transposed,    //
            bool is_b_transposed,    //
            const FT *d_a,           //
            const FT *d_b,           //
            FT *d_c,                 //
            bool incremental,        //
            const Context &context,  //
            cudaStream_t stream) {
  const FT alpha = 1.0f;
  const FT beta = incremental ? 1.0f : 0.0f;
  cudaStream_t old_stream = nullptr;
  if (stream != nullptr) {
    MINUET_CHECK_CUBLAS(
        cublasGetStream(context.GetCUBLASHandle(), &old_stream));
    MINUET_CHECK_CUBLAS(cublasSetStream(context.GetCUBLASHandle(), stream));
  }
  // We need to swap a and b to get C^T = B^T A^T to receive row-major format
  MINUET_CHECK_CUBLAS(CUBLASDispatcher<FT>::GEMM(
      context.GetCUBLASHandle(),                    // handle
      is_b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,  // transa
      is_a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,  // transb
      n,                                            // m
      m,                                            // n
      k,                                            // k
      &alpha,                                       // alpha
      d_b,                                          // A
      is_b_transposed ? k : n,                      // lda
      d_a,                                          // B
      is_a_transposed ? m : k,                      // ldb
      &beta,                                        // beta
      d_c,                                          // C
      n                                             // ldc
      ));
  if (stream != nullptr) {
    MINUET_CHECK_CUBLAS(cublasSetStream(context.GetCUBLASHandle(), old_stream));
  }
}

template <typename FT>
void BatchedMatMul(std::size_t b,           //
                   std::size_t m,           //
                   std::size_t k,           //
                   std::size_t n,           //
                   bool is_a_transposed,    //
                   bool is_b_transposed,    //
                   const FT *d_a,           //
                   const FT *d_b,           //
                   FT *d_c,                 //
                   bool incremental,        //
                   const Context &context,  //
                   cudaStream_t stream) {
  if (b == 1) {
    MatMul(m,                //
           k,                //
           n,                //
           is_a_transposed,  //
           is_b_transposed,  //
           d_a,              //
           d_b,              //
           d_c,              //
           incremental,      //
           context,          //
           stream);
    return;
  }
  const FT alpha = 1.0f;
  const FT beta = incremental ? 1.0f : 0.0f;
  cudaStream_t old_stream = nullptr;
  if (stream != nullptr) {
    MINUET_CHECK_CUBLAS(
        cublasGetStream(context.GetCUBLASHandle(), &old_stream));
    MINUET_CHECK_CUBLAS(cublasSetStream(context.GetCUBLASHandle(), stream));
  }
  MINUET_CHECK_CUBLAS(CUBLASDispatcher<FT>::GEMMStridedBatched(
      context.GetCUBLASHandle(),                    // handle
      is_b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,  // transa
      is_a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N,  // transb
      n,                                            // m
      m,                                            // n
      k,                                            // k
      &alpha,                                       // alpha
      d_b,                                          // A
      is_b_transposed ? k : n,                      // lda
      k * n,                                        // strideA
      d_a,                                          // B
      is_a_transposed ? m : k,                      // ldb
      m * k,                                        // strideB
      &beta,                                        // beta
      d_c,                                          // C
      n,                                            // ldc
      m * n,                                        // strideC
      b                                             // batchCount
      ));
  if (stream != nullptr) {
    MINUET_CHECK_CUBLAS(cublasSetStream(context.GetCUBLASHandle(), old_stream));
  }
}

#define MINUET_EXPLICIT_INSTANTIATE(FT)                                        \
  template void MatMul<FT>(                                                    \
      std::size_t m, std::size_t k, std::size_t n, bool is_a_transposed,       \
      bool is_b_transposed, const FT *d_a, const FT *d_b, FT *d_c,             \
      bool incremental, const Context &context, cudaStream_t stream);          \
  template void BatchedMatMul<FT>(std::size_t b, std::size_t m, std::size_t k, \
                                  std::size_t n, bool is_a_transposed,         \
                                  bool is_b_transposed, const FT *d_a,         \
                                  const FT *d_b, FT *d_c, bool incremental,    \
                                  const Context &context, cudaStream_t stream)
MINUET_EXPLICIT_INSTANTIATE(half);
MINUET_EXPLICIT_INSTANTIATE(float);
MINUET_EXPLICIT_INSTANTIATE(double);
#undef MINUET_EXPLICIT_INSTANTIATE

}  // namespace minuet::cuda