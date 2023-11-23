#pragma once

#include "minuet/common/functions.h"

namespace minuet::cuda::device {

template <typename T, std::size_t T_SIZE>
struct Vectorize;

template <typename T>
struct Devectorize;

#define MINUET_DECLARE_VECTOR_TYPE(T_SOURCE, T_VECTOR, T_SIZE) \
  template <>                                                  \
  struct Vectorize<T_SOURCE, T_SIZE> {                         \
    using VectorType = T_VECTOR;                               \
    using SourceType = T_SOURCE;                               \
    static constexpr const std::size_t size = T_SIZE;          \
  };                                                           \
  template <>                                                  \
  struct Devectorize<T_VECTOR> {                               \
    using SourceType = T_SOURCE;                               \
    using VectorType = T_VECTOR;                               \
    static constexpr const std::size_t size = T_SIZE;          \
  }

#define MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(T_SOURCE, T_VECTOR_HEAD) \
  MINUET_DECLARE_VECTOR_TYPE(T_SOURCE, T_VECTOR_HEAD##1, 1);        \
  MINUET_DECLARE_VECTOR_TYPE(T_SOURCE, T_VECTOR_HEAD##2, 2);        \
  MINUET_DECLARE_VECTOR_TYPE(T_SOURCE, T_VECTOR_HEAD##3, 3);        \
  MINUET_DECLARE_VECTOR_TYPE(T_SOURCE, T_VECTOR_HEAD##4, 4)

MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(int, int);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(long, long);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(unsigned int, uint);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(unsigned long, ulong);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(long long, longlong);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(unsigned long long, ulonglong);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(float, float);
MINUET_DECLARE_VECTOR_TYPE_PLAIN_4(double, double);

#undef MINUET_DECLARE_VECTOR_TYPE_PLAIN_4
#undef MINUET_DECLARE_VECTOR_TYPE

static_assert(SizeOf<char> == 1);

template <std::size_t T_SIZE>
MINUET_FORCEINLINE MINUET_DEVICE void MemoryCopy(
    char *__restrict__ targets, const char *__restrict__ sources) {
  Iterate<UIter, T_SIZE>([&](auto k) { targets[k] = sources[k]; });
}

template <std::size_t T_NUM_BYTES, typename T, typename... Ts>
MINUET_FORCEINLINE MINUET_DEVICE void MemoryCopy(
    char *__restrict__ targets, const char *__restrict__ sources) {
  constexpr auto SIZE = T_NUM_BYTES / SizeOf<T>;
  Iterate<UIter, SIZE>([&](auto k) {
    reinterpret_cast<T *>(targets)[k] = reinterpret_cast<const T *>(sources)[k];
  });
  MemoryCopy<T_NUM_BYTES - SIZE * SizeOf<T>, Ts...>(targets + SIZE * SizeOf<T>,
                                                    sources + SIZE * SizeOf<T>);
}

template <std::size_t T_SIZE, typename T>
MINUET_FORCEINLINE MINUET_DEVICE void Assign(T *__restrict__ targets,
                                             const T *__restrict__ sources) {
  // We do not need type information if we only need to conduct memory copy
  MemoryCopy<T_SIZE * SizeOf<T>, int4, int3, int2, int1, short, char>(
      reinterpret_cast<char *>(targets),
      reinterpret_cast<const char *>(sources));
}

template <std::size_t T_SIZE, typename T>
MINUET_FORCEINLINE MINUET_DEVICE void AssignAdd(T *__restrict__ targets,
                                                const T *__restrict__ sources) {
  // Process with 4 elements as a group
  constexpr const std::size_t BASE4 = 0;
  constexpr const std::size_t SIZE4 = T_SIZE / 4;
  Iterate<UIter, SIZE4>([&](auto k) {
    using VectorType = typename Vectorize<T, 4>::VectorType;
    VectorType source =
        reinterpret_cast<const VectorType *>(sources + BASE4)[k];
    VectorType target = reinterpret_cast<VectorType *>(targets + BASE4)[k];
    Iterate<UIter, 4>([&](auto i) {
      reinterpret_cast<T *>(&target)[i] +=
          reinterpret_cast<const T *>(&source)[i];
    });
  });

  constexpr const std::size_t BASE2 = BASE4 + SIZE4 * 4;
  constexpr const std::size_t SIZE2 = (T_SIZE - BASE2) >> 1;
  Iterate<UIter, SIZE2>([&](auto k) {
    using VectorType = typename Vectorize<T, 2>::VectorType;
    VectorType source =
        reinterpret_cast<const VectorType *>(sources + BASE2)[k];
    VectorType target = reinterpret_cast<VectorType *>(targets + BASE2)[k];
    Iterate<UIter, 2>([&](auto i) {
      reinterpret_cast<T *>(&target)[i] +=
          reinterpret_cast<const T *>(&source)[i];
    });
  });

  constexpr const std::size_t BASE1 = BASE2 + SIZE2 * 2;
  constexpr const std::size_t SIZE1 = T_SIZE - BASE1;
  Iterate<UIter, SIZE1>(
      [&](auto k) { targets[BASE1 + k] = sources[BASE1 + k]; });
}

}  // namespace minuet::cuda::device