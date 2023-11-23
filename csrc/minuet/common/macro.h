#pragma once

#ifdef __NVCC__
#define MINUET_HOST_DEVICE __host__ __device__
#define MINUET_FORCEINLINE __forceinline__
#define MINUET_DEVICE __device__
#define MINUET_HOST __host__
#define MINUET_RESTRICT __restrict__
#else
#define MINUET_HOST_DEVICE
#define MINUET_FORCEINLINE inline __attribute__((always_inline))
#define MINUET_RESTIRCT __restrict
#endif

#define MINUET_COMMA ,
#define MINUET_SEMICOLON ;

namespace minuet {

typedef unsigned int UIter;
typedef int Iter;

}  // namespace minuet
