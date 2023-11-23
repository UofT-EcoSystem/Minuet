#include "minuet/cuda/functions/scan.cuh"
#include "minuet/cuda/helpers.cuh"
#include "minuet/cuda/kernels/generate_masks_from_kernel_map.cuh"
#include "minuet/enabled_arguments.h"

namespace minuet::cuda {

template <typename IT>
void MultiTwoSidedSum::operator()(std::size_t batch_size,   //
                                  std::size_t num_sources,  //
                                  const IT *d_sources,      //
                                  IT *d_targets,            //
                                  const Context &context) const {
  std::size_t d_temp_size = 0;
  if (batch_size == 1) {
    MINUET_CHECK_CUDA(
        cudaMemsetAsync(d_targets, 0, sizeof(IT), context.stream()));
    MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,        //
                                                    d_temp_size,    //
                                                    d_sources,      //
                                                    d_targets + 1,  //
                                                    num_sources,    //
                                                    context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp.device_data(),  //
                                                    d_temp_size,           //
                                                    d_sources,             //
                                                    d_targets + 1,         //
                                                    num_sources,           //
                                                    context.stream()));
  } else {
    auto GetKey = [num_sources] MINUET_DEVICE(UIter x) {
      return x / (num_sources + 1);
    };
    auto GetValue = [sources = d_sources, num_sources] MINUET_DEVICE(UIter x) {
      auto b = x / (num_sources + 1);
      auto s = x % (num_sources + 1);
      return (s == num_sources) ? 0 : sources[b * num_sources + s];
    };
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetValue, d_values_in);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSumByKey(nullptr,                         //
                                           d_temp_size,                     //
                                           d_keys_in,                       //
                                           d_values_in,                     //
                                           d_targets,                       //
                                           batch_size * (num_sources + 1),  //
                                           cub::Equality(),                 //
                                           context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSumByKey(d_temp.device_data(),            //
                                           d_temp_size,                     //
                                           d_keys_in,                       //
                                           d_values_in,                     //
                                           d_targets,                       //
                                           batch_size * (num_sources + 1),  //
                                           cub::Equality(),                 //
                                           context.stream()));
  }
}

template <typename IT>
void MultiTwoSidedKernelMapRanks::operator()(std::size_t batch_size,   //
                                             std::size_t num_sources,  //
                                             const IT *d_sources,      //
                                             IT *d_targets,            //
                                             const Context &context) const {
  std::size_t d_temp_size = 0;
  if (batch_size == 1) {
    auto GetValue = [sources = d_sources] MINUET_DEVICE(UIter x) {
      return sources[x] != -1;
    };
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetValue, d_in);
    MINUET_CHECK_CUDA(
        cudaMemsetAsync(d_targets, 0, sizeof(IT), context.stream()));
    MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,        //
                                                    d_temp_size,    //
                                                    d_in,           //
                                                    d_targets + 1,  //
                                                    num_sources,    //
                                                    context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp.device_data(),  //
                                                    d_temp_size,           //
                                                    d_in,                  //
                                                    d_targets + 1,         //
                                                    num_sources,           //
                                                    context.stream()));
  } else {
    auto GetKey = [num_sources] MINUET_DEVICE(UIter x) {
      return x / (num_sources + 1);
    };
    auto GetValue = [sources = d_sources, num_sources] MINUET_DEVICE(UIter x) {
      auto b = x / (num_sources + 1);
      auto s = x % (num_sources + 1);
      return (s == num_sources) ? 0 : sources[b * num_sources + s] != -1;
    };
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetValue, d_values_in);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSumByKey(nullptr,                         //
                                           d_temp_size,                     //
                                           d_keys_in,                       //
                                           d_values_in,                     //
                                           d_targets,                       //
                                           batch_size * (num_sources + 1),  //
                                           cub::Equality(),                 //
                                           context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSumByKey(d_temp.device_data(),            //
                                           d_temp_size,                     //
                                           d_keys_in,                       //
                                           d_values_in,                     //
                                           d_targets,                       //
                                           batch_size * (num_sources + 1),  //
                                           cub::Equality(),                 //
                                           context.stream()));
  }
}

template <typename IT>
void MultiExclusiveSum::operator()(std::size_t batch_size,   //
                                   std::size_t num_sources,  //
                                   const IT *d_sources,      //
                                   IT *d_targets,            //
                                   const Context &context) const {
  std::size_t d_temp_size = 0;
  if (batch_size == 1) {
    MINUET_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(nullptr,      //
                                                    d_temp_size,  //
                                                    d_sources,    //
                                                    d_targets,    //
                                                    num_sources,  //
                                                    context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp.device_data(),  //
                                                    d_temp_size,           //
                                                    d_sources,             //
                                                    d_targets,             //
                                                    num_sources,           //
                                                    context.stream()));
  } else {
    auto GetKey = [num_sources] MINUET_DEVICE(UIter x) {
      return x / num_sources;
    };
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSumByKey(nullptr,                   //
                                           d_temp_size,               //
                                           d_keys_in,                 //
                                           d_sources,                 //
                                           d_targets,                 //
                                           batch_size * num_sources,  //
                                           cub::Equality(),           //
                                           context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::ExclusiveSumByKey(d_temp.device_data(),      //
                                           d_temp_size,               //
                                           d_keys_in,                 //
                                           d_sources,                 //
                                           d_targets,                 //
                                           batch_size * num_sources,  //
                                           cub::Equality(),           //
                                           context.stream()));
  }
}

template <typename IT>
void MultiInclusiveSum::operator()(std::size_t batch_size,   //
                                   std::size_t num_sources,  //
                                   const IT *d_sources,      //
                                   IT *d_targets,            //
                                   const Context &context) const {
  std::size_t d_temp_size = 0;
  if (batch_size == 1) {
    MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(nullptr,      //
                                                    d_temp_size,  //
                                                    d_sources,    //
                                                    d_targets,    //
                                                    num_sources,  //
                                                    context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp.device_data(),  //
                                                    d_temp_size,           //
                                                    d_sources,             //
                                                    d_targets,             //
                                                    num_sources,           //
                                                    context.stream()));
  } else {
    auto GetKey = [num_sources] MINUET_DEVICE(UIter x) {
      return x / num_sources;
    };
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::InclusiveSumByKey(nullptr,                   //
                                           d_temp_size,               //
                                           d_keys_in,                 //
                                           d_sources,                 //
                                           d_targets,                 //
                                           batch_size * num_sources,  //
                                           cub::Equality(),           //
                                           context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceScan::InclusiveSumByKey(d_temp.device_data(),      //
                                           d_temp_size,               //
                                           d_keys_in,                 //
                                           d_sources,                 //
                                           d_targets,                 //
                                           batch_size * num_sources,  //
                                           cub::Equality(),           //
                                           context.stream()));
  }
}

template <typename IT>
void ComputeKernelMapSizes::operator()(std::size_t num_offsets,
                                       std::size_t num_targets,
                                       const IT *d_kernel_map,
                                       IT *d_kernel_map_sizes,
                                       const Context &context) const {
  auto IsValidEntry = [kernel_map = d_kernel_map] MINUET_DEVICE(UIter x) {
    return kernel_map[x] != -1;
  };
  std::size_t d_temp_size = 0;
#ifdef MINUET_PREFER_SEGMENTED_REDUCE
  auto GetBeginOffsets = [num_targets] MINUET_DEVICE(UIter x) {
    return x * num_targets;
  };
  auto GetEndOffsets = [num_targets] MINUET_DEVICE(UIter x) {
    return (x + 1) * num_targets;
  };
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, IsValidEntry, d_in);
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetBeginOffsets, d_begin_offsets);
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetEndOffsets, d_end_offsets);

  MINUET_CHECK_CUDA(cub::DeviceSegmentedReduce::Sum(nullptr,             //
                                                    d_temp_size,         //
                                                    d_in,                //
                                                    d_kernel_map_sizes,  //
                                                    num_offsets,         //
                                                    d_begin_offsets,     //
                                                    d_end_offsets,       //
                                                    context.stream()));
  auto d_temp = context.NewBuffer(d_temp_size);
  MINUET_CHECK_CUDA(cub::DeviceSegmentedReduce::Sum(d_temp.device_data(),  //
                                                    d_temp_size,           //
                                                    d_in,                  //
                                                    d_kernel_map_sizes,    //
                                                    num_offsets,           //
                                                    d_begin_offsets,       //
                                                    d_end_offsets,         //
                                                    context.stream()));
#else
  auto GetKey = [num_targets] MINUET_DEVICE(UIter x) {
    return x / num_targets;
  };
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
  cub::DiscardOutputIterator d_unique_out;
  CUB_TRANSFORMED_INPUT_ITERATOR(UIter, IsValidEntry, d_values_in);
  cub::DiscardOutputIterator d_num_runs_out;
  MINUET_CHECK_CUDA(
      cub::DeviceReduce::ReduceByKey(nullptr,                    //
                                     d_temp_size,                //
                                     d_keys_in,                  //
                                     d_unique_out,               //
                                     d_values_in,                //
                                     d_kernel_map_sizes,         //
                                     d_num_runs_out,             //
                                     cub::Sum(),                 //
                                     num_offsets * num_targets,  //
                                     context.stream()));
  auto d_temp = context.NewBuffer(d_temp_size);
  MINUET_CHECK_CUDA(
      cub::DeviceReduce::ReduceByKey(d_temp.device_data(),       //
                                     d_temp_size,                //
                                     d_keys_in,                  //
                                     d_unique_out,               //
                                     d_values_in,                //
                                     d_kernel_map_sizes,         //
                                     d_num_runs_out,             //
                                     cub::Sum(),                 //
                                     num_offsets * num_targets,  //
                                     context.stream()));
#endif
}

template <typename IT>
void ComputeKernelMapMasks::operator()(std::size_t num_offsets,       // [O]
                                       std::size_t num_sources,       // [S]
                                       std::size_t num_targets,       // [T]
                                       const IT *d_kernel_map,        // [O, T]
                                       const IT *d_kernel_map_sizes,  // [O]
                                       IT *d_source_masks,            // [O, S]
                                       IT *d_target_masks,            // [O, T]
                                       const Context &context) const {
  cpu::Memory<IT> h_cumsum_kernel_sizes(num_offsets + 1);
  context.ReadDeviceData(d_kernel_map_sizes, h_cumsum_kernel_sizes.data() + 1,
                         num_offsets);

  h_cumsum_kernel_sizes[0] = 0;
  for (UIter i = 1; i <= num_offsets; i++) {
    h_cumsum_kernel_sizes[i] += h_cumsum_kernel_sizes[i - 1];
  }

  auto num_entries = h_cumsum_kernel_sizes[num_offsets];
  auto d_cumsum_kernel_sizes = context.NewBufferFrom(h_cumsum_kernel_sizes);

  using EntryType = std::int64_t;
  auto d_kernel_map_nonzero_indices = context.NewBuffer<EntryType>(num_entries);
  {
    auto IsValidEntry = [] MINUET_DEVICE(const EntryType &entry) {
      return entry != -1;
    };
    auto GetEntryPair = [num_sources,
                         kernel_map = d_kernel_map] MINUET_DEVICE(auto index) {
      auto value = kernel_map[index];
      return (value != -1) ? static_cast<EntryType>(index) * num_sources + value
                           : -1;
    };
    CUB_TRANSFORMED_INPUT_ITERATOR(EntryType, GetEntryPair, d_in);
    std::size_t d_temp_size = 0;
    cub::DiscardOutputIterator d_num_selected_out;
    MINUET_CHECK_CUDA(
        cub::DeviceSelect::If(nullptr,                                     //
                              d_temp_size,                                 //
                              d_in,                                        //
                              d_kernel_map_nonzero_indices.device_data(),  //
                              d_num_selected_out,                          //
                              num_offsets * num_targets,                   //
                              IsValidEntry,                                //
                              context.stream()));
    auto d_temp = context.NewBuffer(d_temp_size);
    MINUET_CHECK_CUDA(
        cub::DeviceSelect::If(d_temp.device_data(),                        //
                              d_temp_size,                                 //
                              d_in,                                        //
                              d_kernel_map_nonzero_indices.device_data(),  //
                              d_num_selected_out,                          //
                              num_offsets * num_targets,                   //
                              IsValidEntry,                                //
                              context.stream()));
  }
  MINUET_CHECK_CUDA(cudaMemsetAsync(d_source_masks, -1,
                                    sizeof(IT) * num_offsets * num_sources,
                                    context.stream()));
  MINUET_CHECK_CUDA(cudaMemsetAsync(d_target_masks, -1,
                                    sizeof(IT) * num_offsets * num_targets,
                                    context.stream()));
  if (num_entries > 0) {
    context.Launch(DivCeil<UIter>(num_entries, 4), 128, 0,      //
                   kernels::GenerateMasksFromKernelMap<IT>,     //
                   num_entries,                                 //
                   num_sources,                                 //
                   num_targets,                                 //
                   d_cumsum_kernel_sizes.device_data(),         //
                   d_kernel_map_nonzero_indices.device_data(),  //
                   d_source_masks,                              //
                   d_target_masks);
  }
}

#define MINUET_EXPLICIT_INSTANTIATOR(_, IT)                                   \
  template void MultiInclusiveSum::operator()<IT>(                            \
      std::size_t batch_size, std::size_t num_sources, const IT *d_sources,   \
      IT *d_targets, const Context &context) const;                           \
  template void MultiExclusiveSum::operator()<IT>(                            \
      std::size_t batch_size, std::size_t num_sources, const IT *d_sources,   \
      IT *d_targets, const Context &context) const;                           \
  template void MultiTwoSidedSum::operator()<IT>(                             \
      std::size_t batch_size, std::size_t num_sources, const IT *d_sources,   \
      IT *d_targets, const Context &context) const;                           \
  template void MultiTwoSidedKernelMapRanks::operator()<IT>(                  \
      std::size_t batch_size, std::size_t num_sources, const IT *d_sources,   \
      IT *d_targets, const Context &context) const;                           \
  template void ComputeKernelMapSizes::operator()<IT>(                        \
      std::size_t num_offsets, std::size_t num_targets,                       \
      const IT *d_kernel_map, IT *d_kernel_map_sizes, const Context &context) \
      const;                                                                  \
  template void ComputeKernelMapMasks::operator()<IT>(                        \
      std::size_t num_offsets, std::size_t num_sources,                       \
      std::size_t num_targets, const IT *d_kernel_map,                        \
      const IT *d_kernel_map_sizes, IT *d_source_masks, IT *d_target_masks,   \
      const Context &context) const

MINUET_FOR_ALL_I_TYPES(MINUET_EXPLICIT_INSTANTIATOR);
#undef MINUET_EXPLICIT_INSTANTIATOR

}  // namespace minuet::cuda