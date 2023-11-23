#pragma once

#include <cstdint>

#include "minuet/common/macro.h"

namespace minuet {

template <typename T>
MINUET_FORCEINLINE MINUET_HOST_DEVICE constexpr T DivCeil(const T &a,
                                                          const T &b) {
  return (a + b - 1) / b;
}

template <typename T>
MINUET_FORCEINLINE MINUET_HOST_DEVICE constexpr T DivFloor(const T &a,
                                                           const T &b) {
  return a / b;
}

namespace detail {

template <typename T, T T_BEGIN, T T_END, bool = (T_BEGIN < T_END)>
struct IterateImpl {
  template <typename CallableT>
  MINUET_FORCEINLINE MINUET_HOST_DEVICE static void Evaluate(
      CallableT callable) {
    callable(T_BEGIN);
    IterateImpl<T, T_BEGIN + 1, T_END>::Evaluate(callable);
  }
};

template <typename T, T T_BEGIN, T T_END>
struct IterateImpl<T, T_BEGIN, T_END, false> {
  template <typename CallableT>
  MINUET_FORCEINLINE MINUET_HOST_DEVICE static void Evaluate(CallableT) {}
};

template <typename T>
struct SizeOfImpl : public std::integral_constant<std::size_t, sizeof(T)> {};

template <>
struct SizeOfImpl<void> : public std::integral_constant<std::size_t, 1> {};

template <typename T>
struct AlignOfImpl : public std::integral_constant<std::size_t, alignof(T)> {};

template <>
struct AlignOfImpl<void> : public std::integral_constant<std::size_t, 1> {};

}  // namespace detail

template <typename T, T T_BEGIN, T T_END, typename CallableT>
MINUET_FORCEINLINE MINUET_HOST_DEVICE void Iterate(CallableT callable) {
  detail::IterateImpl<T, T_BEGIN, T_END>::Evaluate(callable);
}

template <typename T, T T_N, typename CallableT>
MINUET_FORCEINLINE MINUET_HOST_DEVICE void Iterate(CallableT callable) {
  detail::IterateImpl<T, 0, T_N>::Evaluate(callable);
}

template <std::size_t T_N, typename T>
MINUET_FORCEINLINE MINUET_HOST_DEVICE std::uint64_t Hash(T data) {
  std::uint64_t result = UINT64_C(14695981039346656037);
  Iterate<UIter, T_N>([&](auto i) {
    result *= UINT64_C(1099511628211);
    result += data(i);
  });
  return result;
}

template <std::size_t T_N, typename T>
MINUET_FORCEINLINE MINUET_HOST_DEVICE std::uint64_t Hash(const T *data) {
  return Hash<T_N>([&](UIter i) { return data[i]; });
}

template <std::size_t T_N, typename T>
MINUET_FORCEINLINE MINUET_HOST_DEVICE std::uint64_t HashReverse(T data) {
  std::uint64_t result = UINT64_C(14695981039346656037);
#pragma unroll
  for (Iter i = T_N - 1; i >= 0; i--) {
    result *= UINT64_C(1099511628211);
    result += data(i);
  };
  return result;
}

template <std::size_t T_N, typename T>
MINUET_FORCEINLINE MINUET_HOST_DEVICE std::uint64_t HashReverse(const T *data) {
  return HashReverse<T_N>([&](UIter i) { return data[i]; });
}

template <typename T, std::size_t T_NDIM>
struct CompareCoordinates {
  template <typename SourceT, typename TargetT>
  MINUET_FORCEINLINE MINUET_HOST_DEVICE T operator()(SourceT source,
                                                     TargetT target) {
    T delta = 0;
    Iterate<UIter, T_NDIM>([&](auto i) {
      delta = (delta == 0) ? (source(i) - target(i)) : delta;
    });
    return delta;
  }
};

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
MINUET_FORCEINLINE MINUET_HOST_DEVICE constexpr const T FloorLog2(T x) {
  return x == 1 ? 0 : 1 + FloorLog2(x >> 1);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
MINUET_FORCEINLINE MINUET_HOST_DEVICE constexpr const T CeilLog2(T x) {
  return x == 1 ? 0 : 1 + FloorLog2(x - 1);
}

template <typename T>
inline constexpr const auto SizeOf = detail::SizeOfImpl<T>::value;

template <typename T>
inline constexpr const auto AlignOf = detail::AlignOfImpl<T>::value;

}  // namespace minuet
