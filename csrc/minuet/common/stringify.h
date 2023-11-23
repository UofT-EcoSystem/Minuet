#pragma once

#include <string>
#include <string_view>

#define MINUET_MACRO_STRINGIFY_IMPL(VALUE) #VALUE
#define MINUET_MACRO_STRINGIFY(VALUE) MINUET_MACRO_STRINGIFY_IMPL(VALUE)

namespace minuet {
namespace detail {

template <int C, class T, class = void>
struct StringifyCase : std::false_type {};

template <class T>
struct StringifyCase<1, T,
                     std::void_t<decltype(std::string(std::declval<T>()))>>
    : std::true_type {
  std::string operator()(T &&arg) { return std::string(std::forward<T>(arg)); };
};

template <class T>
struct StringifyCase<2, T,
                     std::void_t<decltype(std::to_string(std::declval<T>()))>>
    : std::true_type {
  std::string operator()(T &&arg) {
    return std::to_string(std::forward<T>(arg));
  };
};

template <class T>
constexpr int StringifySwitch() {
  if (StringifyCase<1, T>::value) {
    return 1;
  }
  if (StringifyCase<2, T>::value) {
    return 2;
  }
}

template <class T>
std::string StringifyImpl(T &&arg) {
  return detail::StringifyCase<detail::StringifySwitch<T>(), T>()(
      std::forward<T>(arg));
}

}  // namespace detail

inline std::string Stringify() { return {}; }

template <class ArgT>
std::string Stringify(ArgT &&arg) {
  return detail::StringifyImpl(std::forward<ArgT>(arg));
}

template <class ArgT, class... ArgTs>
std::string Stringify(ArgT &&arg, ArgTs &&...args) {
  return Stringify(std::forward<ArgT>(arg)) +
         Stringify<ArgTs...>(std::forward<ArgTs>(args)...);
}

}  // namespace minuet