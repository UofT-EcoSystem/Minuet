#include <cuda/std/array>

#include "minuet/cuda/context.cuh"
#include "minuet/cuda/functions/arg_sort_coodrinates.cuh"
#include "minuet/cuda/helpers.cuh"
#include "minuet/cuda/kernels/binary_search_materialize.cuh"
#include "minuet/cuda/kernels/fill_range.cuh"
#include "minuet/cuda/kernels/flatten_coordinates.cuh"
#include "minuet/cuda/kernels/load_batch_id_with_permutation.cuh"
#include "minuet/cuda/kernels/load_coordinates_with_permutation.cuh"
#include "minuet/cuda/kernels/multi_flatten_coordinates.cuh"
#include "minuet/cuda/memory.cuh"
#include "minuet/enabled_arguments.h"

namespace minuet::cuda {

template <typename ReduceOpT, std::size_t T_NDIM, typename CT>
struct CoordinateReduceOp {
  using Coordinate = ::cuda::std::array<CT, T_NDIM>;

  MINUET_HOST_DEVICE MINUET_FORCEINLINE Coordinate
  operator()(const Coordinate &a, const Coordinate &b) const {
    Coordinate result;
    Iterate<UIter, T_NDIM>(
        [&](UIter k) { result[k] = ReduceOpT()(a[k], b[k]); });
    return result;
  }
};

template <std::size_t T_NDIM, typename CT>
void GetCoordinatesMinMax(std::size_t n,            //
                          const CT *d_coordinates,  //
                          CT *d_cmin,               //
                          CT *d_cmax,               //
                          const Context &context) {
  using Coordinate = ::cuda::std::array<CT, T_NDIM>;
  std::size_t d_temp_size = 0;
  Coordinate init;
  {
    CoordinateReduceOp<cub::Min, T_NDIM, CT> reduce_op;
    init.fill(std::numeric_limits<CT>::max());
    MINUET_CHECK_CUDA(cub::DeviceReduce::Reduce(
        nullptr,                                              //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmin),               //
        n,                                                    //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceReduce::Reduce(
        d_temp.device_data(),                                 //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmin),               //
        n,                                                    //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
  }
  {
    CoordinateReduceOp<cub::Max, T_NDIM, CT> reduce_op;
    init.fill(std::numeric_limits<CT>::min());
    MINUET_CHECK_CUDA(cub::DeviceReduce::Reduce(
        nullptr,                                              //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmax),               //
        n,                                                    //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceReduce::Reduce(
        d_temp.device_data(),                                 //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmax),               //
        n,                                                    //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
  }
}

template <std::size_t T_NDIM, typename CT, typename IT>
void MultiGetCoordinatesMinMax(std::size_t n,
                               std::size_t batch_size,   //
                               const CT *d_coordinates,  //
                               CT *d_cmin,               //
                               CT *d_cmax,               //
                               const IT *d_batch_dims,   //
                               const Context &context) {
  using Coordinate = ::cuda::std::array<CT, T_NDIM>;
  std::size_t d_temp_size = 0;

#ifdef MINUET_PREFER_SEGMENTED_REDUCE
  Coordinate init;
  {
    CoordinateReduceOp<cub::Min, T_NDIM, CT> reduce_op;
    init.fill(std::numeric_limits<CT>::max());
    MINUET_CHECK_CUDA(cub::DeviceSegmentedReduce::Reduce(
        nullptr,                                              //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmin),               //
        batch_size,                                           //
        d_batch_dims,                                         //
        d_batch_dims + 1,                                     //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceSegmentedReduce::Reduce(
        d_temp.device_data(),                                 //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmin),               //
        batch_size,                                           //
        d_batch_dims,                                         //
        d_batch_dims + 1,                                     //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
  }
  {
    CoordinateReduceOp<cub::Max, T_NDIM, CT> reduce_op;
    init.fill(std::numeric_limits<CT>::min());
    MINUET_CHECK_CUDA(cub::DeviceSegmentedReduce::Reduce(
        nullptr,                                              //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmax),               //
        batch_size,                                           //
        d_batch_dims,                                         //
        d_batch_dims + 1,                                     //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceSegmentedReduce::Reduce(
        d_temp.device_data(),                                 //
        d_temp_size,                                          //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmax),               //
        batch_size,                                           //
        d_batch_dims,                                         //
        d_batch_dims + 1,                                     //
        reduce_op,                                            //
        init,                                                 //
        context.stream()));
  }
#else
  auto GetKey = [batch_size, batch_dims = d_batch_dims] MINUET_DEVICE(UIter x) {
    return device::BinarySearchMaximize<UIter>(
        0, batch_size - 1, [&](UIter m) { return batch_dims[m]; }, x);
  };
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_key_in);
  cub::DiscardOutputIterator d_unique_out;
  cub::DiscardOutputIterator d_num_runs_out;

  {
    CoordinateReduceOp<cub::Min, T_NDIM, CT> reduce_op;
    MINUET_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(
        nullptr,                                              //
        d_temp_size,                                          //
        d_key_in,                                             //
        d_unique_out,                                         //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmin),               //
        d_num_runs_out,                                       //
        reduce_op,                                            //
        n,                                                    //
        context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(
        d_temp.device_data(),                                 //
        d_temp_size,                                          //
        d_key_in,                                             //
        d_unique_out,                                         //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmin),               //
        d_num_runs_out,                                       //
        reduce_op,                                            //
        n,                                                    //
        context.stream()));
  }

  {
    CoordinateReduceOp<cub::Max, T_NDIM, CT> reduce_op;
    MINUET_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(
        nullptr,                                              //
        d_temp_size,                                          //
        d_key_in,                                             //
        d_unique_out,                                         //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmax),               //
        d_num_runs_out,                                       //
        reduce_op,                                            //
        n,                                                    //
        context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(
        d_temp.device_data(),                                 //
        d_temp_size,                                          //
        d_key_in,                                             //
        d_unique_out,                                         //
        reinterpret_cast<const Coordinate *>(d_coordinates),  //
        reinterpret_cast<Coordinate *>(d_cmax),               //
        d_num_runs_out,                                       //
        reduce_op,                                            //
        n,                                                    //
        context.stream()));
  }
#endif
}

template <std::size_t T_NDIM, typename UT, typename CT>
bool IsFlattenable(const CT *h_cmin, const CT *h_cmax) {
  static_assert(!std::is_signed_v<UT>);
  __int128 result = std::numeric_limits<UT>::max();
  for (UIter i = 0; i < T_NDIM; i++) {
    __int128 size =
        static_cast<__int128>(h_cmax[i]) - static_cast<__int128>(h_cmin[i]) + 1;
    if (result < size) {
      return false;
    }
    result /= size;
  }
  return true;
}

template <std::size_t T_NDIM, typename UT, typename CT>
bool MultiIsFlattenable(std::size_t batch_size, const CT *h_cmin,
                        const CT *h_cmax) {
  for (UIter i = 0; i < batch_size; i++) {
    if (!IsFlattenable<T_NDIM, UT, CT>(h_cmin + i * T_NDIM,
                                       h_cmax + i * T_NDIM)) {
      return false;
    }
  }
  return true;
}

template <std::size_t T_NDIM, typename UT, typename CT>
void FlattenCoordinates(std::size_t n,        //
                        const CT *d_sources,  //
                        UT *d_targets,        //
                        const CT *d_cmin,     //
                        const CT *d_cmax,     //
                        const Context &context) {
  context.Launch(n, 128, 0,                                    //
                 kernels::FlattenCoordinates<CT, UT, T_NDIM>,  //
                 n, d_cmin, d_cmax, d_sources, d_targets);
}

template <std::size_t T_NDIM, typename UT, typename CT, typename IT>
void MultiFlattenCoordinates(std::size_t n,           //
                             std::size_t batch_size,  //
                             const CT *d_sources,     //
                             UT *d_targets,           //
                             const CT *d_cmin,        //
                             const CT *d_cmax,        //
                             const IT *d_batch_dims,  //
                             const Context &context) {
  context.Launch(n, 128, 0,                                             //
                 kernels::MultiFlattenCoordinates<CT, UT, T_NDIM, IT>,  //
                 batch_size, d_cmin, d_cmax, d_sources, d_targets,
                 d_batch_dims);
}

template <std::size_t T_NDIM, typename CT, typename IT>
void ArgSortCoordinates::operator()(std::size_t n,            //
                                    const CT *d_coordinates,  //
                                    IT *d_indices,            //
                                    bool enable_flattening,   //
                                    const Context &context) const {
  if (T_NDIM > 1 && enable_flattening) {
    cpu::Memory<CT> h_cmin(T_NDIM);
    cpu::Memory<CT> h_cmax(T_NDIM);
    auto d_cmin = context.NewBufferLike(h_cmin);
    auto d_cmax = context.NewBufferLike(h_cmax);
    GetCoordinatesMinMax<T_NDIM, CT>(n, d_coordinates, d_cmin.device_data(),
                                     d_cmax.device_data(), context);
    d_cmin.CopyTo(h_cmin.data());
    d_cmax.CopyTo(h_cmax.data());
#define TRY(UT)                                                              \
  do {                                                                       \
    if (IsFlattenable<T_NDIM, UT>(h_cmin.data(), h_cmax.data())) {           \
      auto d_buffer = context.NewBuffer<UT>(n);                              \
      FlattenCoordinates<T_NDIM, UT, CT>(                                    \
          n, d_coordinates, d_buffer.device_data(), d_cmin.device_data(),    \
          d_cmax.device_data(), context);                                    \
      ArgSortCoordinates().template operator()<1, UT, IT>(                   \
          n, d_buffer.device_data(), d_indices, enable_flattening, context); \
      return;                                                                \
    }                                                                        \
  } while (false)
    TRY(std::uint8_t);
    TRY(std::uint16_t);
    TRY(std::uint32_t);
    TRY(std::uint64_t);
#undef TRY
  }

  std::size_t d_temp_size;
  MINUET_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
      nullptr,                           // d_temp_storage
      d_temp_size,                       // temp_storage_bytes
      static_cast<const CT *>(nullptr),  // d_keys_in
      static_cast<CT *>(nullptr),        // d_keys_out
      static_cast<const IT *>(nullptr),  // d_values_in
      static_cast<IT *>(nullptr),        // d_values_out,
      static_cast<int>(n),               // num_items
      0,                                 // begin_bit
      sizeof(CT) * CHAR_BIT,             // end_bit
      context.stream()                   // stream
      ));

  auto d_temp = context.NewBuffer(d_temp_size);
  auto d_source_buffer = context.NewBuffer<CT>(n);
  auto d_target_buffer = context.NewBuffer<CT>(n);
  auto d_indices_buffer = context.NewBuffer<IT>(n);
  bool flag = T_NDIM & 1;

  IT *d_indices_swapper[2] = {d_indices, d_indices_buffer.device_data()};
  context.Launch(n,                       // size
                 128,                     // block_size
                 0,                       // shared_memory
                 kernels::FillRange<IT>,  // kernel
                 n, d_indices_swapper[flag]);
  for (std::size_t i = 0; i < T_NDIM; i++) {
    const CT *d_keys_in = d_source_buffer.device_data();
    CT *d_keys_out = d_target_buffer.device_data();
    if (T_NDIM == 1) {
      d_keys_in = d_coordinates;
    } else {
      context.Launch(n,    // size
                     128,  // block_size
                     0,    // shared_memory
                     kernels::LoadCoordinatesWithPermutation<T_NDIM, CT, IT>,
                     T_NDIM - i - 1, n, d_coordinates,
                     d_source_buffer.device_data(), d_indices_swapper[flag]);
    }
    MINUET_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp.device_data(),         // d_temp_storage
        d_temp_size,                  // temp_storage_bytes
        d_keys_in,                    // d_keys_in
        d_keys_out,                   // d_keys_out
        d_indices_swapper[flag],      // d_values_in
        d_indices_swapper[flag ^ 1],  // d_values_out,
        static_cast<int>(n),          // num_items
        0,                            // begin_bit
        sizeof(CT) * CHAR_BIT,        // end_bit
        context.stream()              // stream
        ));
    flag ^= 1;
  }
}

template <std::size_t T_NDIM, typename CT, typename IT>
void MultiArgSortCoordinates::operator()(std::size_t n,            //
                                         std::size_t batch_size,   //
                                         const CT *d_coordinates,  //
                                         const IT *d_batch_dims,   //
                                         IT *d_indices,            //
                                         bool enable_flattening,   //
                                         const Context &context) const {
  if (T_NDIM > 1 && enable_flattening) {
    cpu::Memory<CT> h_cmin(batch_size * T_NDIM);
    cpu::Memory<CT> h_cmax(batch_size * T_NDIM);
    auto d_cmin = context.NewBufferLike(h_cmin);
    auto d_cmax = context.NewBufferLike(h_cmax);
    MultiGetCoordinatesMinMax<T_NDIM, CT, IT>(
        n, batch_size, d_coordinates, d_cmin.device_data(),
        d_cmax.device_data(), d_batch_dims, context);
    d_cmin.CopyTo(h_cmin.data());
    d_cmax.CopyTo(h_cmax.data());

#define TRY(UT)                                                               \
  do {                                                                        \
    if (MultiIsFlattenable<T_NDIM, UT>(batch_size, h_cmin.data(),             \
                                       h_cmax.data())) {                      \
      auto d_buffer = context.NewBuffer<UT>(n);                               \
      MultiFlattenCoordinates<T_NDIM, UT, CT, IT>(                            \
          n, batch_size, d_coordinates, d_buffer.device_data(),               \
          d_cmin.device_data(), d_cmax.device_data(), d_batch_dims, context); \
      MultiArgSortCoordinates().template operator()<1, UT, IT>(               \
          n, batch_size, d_buffer.device_data(), d_batch_dims, d_indices,     \
          enable_flattening, context);                                        \
      return;                                                                 \
    }                                                                         \
  } while (false)
    TRY(std::uint8_t);
    TRY(std::uint16_t);
    TRY(std::uint32_t);
    TRY(std::uint64_t);
#undef TRY
  }

  std::size_t d_temp_size;
  MINUET_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
      nullptr,                           // d_temp_storage
      d_temp_size,                       // temp_storage_bytes
      static_cast<const CT *>(nullptr),  // d_keys_in
      static_cast<CT *>(nullptr),        // d_keys_out
      static_cast<const IT *>(nullptr),  // d_values_in
      static_cast<IT *>(nullptr),        // d_values_out,
      static_cast<int>(n),               // num_items
      0,                                 // begin_bit
      sizeof(CT) * CHAR_BIT,             // end_bit
      context.stream()                   // stream
      ));
  auto d_temp = context.NewBuffer(d_temp_size);
  auto d_source_buffer = context.NewBuffer<CT>(n);
  auto d_target_buffer = context.NewBuffer<CT>(n);
  auto d_indices_buffer = context.NewBuffer<IT>(n);
  bool flag = T_NDIM & 1 ^ 1;

  IT *d_indices_swapper[2] = {d_indices, d_indices_buffer.device_data()};
  context.Launch(n,                       // size
                 128,                     // block_size
                 0,                       // shared_memory
                 kernels::FillRange<IT>,  // kernel
                 n, d_indices_swapper[flag]);
  for (std::size_t i = 0; i < T_NDIM; i++) {
    const CT *d_keys_in = d_source_buffer.device_data();
    CT *d_keys_out = d_target_buffer.device_data();
    if (T_NDIM == 1) {
      d_keys_in = d_coordinates;
    } else {
      context.Launch(n,    // size
                     128,  // block_size
                     0,    // shared_memory
                     kernels::LoadCoordinatesWithPermutation<T_NDIM, CT, IT>,
                     T_NDIM - i - 1, n, d_coordinates,
                     d_source_buffer.device_data(), d_indices_swapper[flag]);
    }
    MINUET_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp.device_data(),         // d_temp_storage
        d_temp_size,                  // temp_storage_bytes
        d_keys_in,                    // d_keys_in
        d_keys_out,                   // d_keys_out
        d_indices_swapper[flag],      // d_values_in
        d_indices_swapper[flag ^ 1],  // d_values_out,
        static_cast<int>(n),          // num_items
        0,                            // begin_bit
        sizeof(CT) * CHAR_BIT,        // end_bit
        context.stream()              // stream
        ));
    flag ^= 1;
  }
  context.Launch(n, 128, 0, kernels::LoadBatchIdWithPermutation<CT, IT>, n,
                 batch_size, d_batch_dims, d_indices_swapper[flag],
                 d_source_buffer.device_data());
  MINUET_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
      d_temp.device_data(),           // d_temp_storage
      d_temp_size,                    // temp_storage_bytes
      d_source_buffer.device_data(),  // d_keys_in
      d_target_buffer.device_data(),  // d_keys_out
      d_indices_swapper[flag],        // d_values_in
      d_indices_swapper[flag ^ 1],    // d_values_out,
      static_cast<int>(n),            // num_items
      0,                              // begin_bit
      sizeof(CT) * CHAR_BIT,          // end_bit
      context.stream()                // stream
      ));
}

#define MINUET_EXPLICIT_INSTANTIATOR(_, T_NDIM, CT, IT)              \
  template void ArgSortCoordinates::operator()<T_NDIM, CT, IT>(      \
      std::size_t, const CT *, IT *, bool, const Context &) const;   \
  template void MultiArgSortCoordinates::operator()<T_NDIM, CT, IT>( \
      std::size_t, std::size_t, const CT *, const IT *, IT *, bool,  \
      const Context &) const
MINUET_FOR_ALL_DIMS_AND_CI_TYPES(MINUET_EXPLICIT_INSTANTIATOR);

}  // namespace minuet::cuda
