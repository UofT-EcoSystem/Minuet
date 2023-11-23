#pragma once

#define MINUET_FOR_ALL_C_TYPES(MACRO, ...) \
  MACRO(__VA_ARGS__, std::int64_t);        \
  MACRO(__VA_ARGS__, std::int32_t)

#define MINUET_FOR_ALL_I_TYPES(MACRO, ...) \
  MACRO(__VA_ARGS__, std::int64_t);        \
  MACRO(__VA_ARGS__, std::int32_t)

#define MINUET_FOR_ALL_F_TYPES(MACRO, ...) \
  MACRO(__VA_ARGS__, float)

#define MINUET_FOR_ALL_DIMS(MACRO, ...) \
  MACRO(__VA_ARGS__, 1);                \
  MACRO(__VA_ARGS__, 2);                \
  MACRO(__VA_ARGS__, 3);                \
  MACRO(__VA_ARGS__, 4)

#define MINUET_FOR_ALL_DIMS_AND_CI_TYPES(MACRO, ...)                         \
  MINUET_FOR_ALL_DIMS(MINUET_FOR_ALL_C_TYPES, MINUET_FOR_ALL_I_TYPES, MACRO, \
                      __VA_ARGS__)

#define MINUET_FOR_ALL_IF_TYPES(MACRO, ...) \
  MINUET_FOR_ALL_I_TYPES(MINUET_FOR_ALL_F_TYPES, MACRO, __VA_ARGS__)

#define MINUET_FOR_ALL_C_TYPES_AND_DIMS(MACRO, ...) \
  MINUET_FOR_ALL_DIMS(MINUET_FOR_ALL_C_TYPES, MACRO, __VA_ARGS__)
