#include <iomanip>

#include "minuet/cuda/event.cuh"
#include "minuet/cuda/functions/sparse_convolution.cuh"
#include "minuet/cuda/helpers.cuh"
#include "minuet/cuda/kernels/padded_gather_warp_optimized.cuh"
#include "minuet/cuda/kernels/padded_scatter_warp_optimized.cuh"
#include "minuet/cuda/kernels/permute_weights.cuh"
#include "minuet/cuda/stream.cuh"
#include "minuet/enabled_arguments.h"

namespace minuet::cuda {

std::vector<Stream> stream_pool_;

template <typename IT>
std::vector<IT> GeneratePaddingBuckets(std::size_t num_offsets,
                                       std::size_t num_kernel_features,
                                       const std::optional<double> &threshold,
                                       const IT *h_kernel_map_sizes) {
  std::vector<IT> h_buckets;
  h_buckets.push_back(0);
  for (UIter i = 0; i < num_offsets; i++) {
    auto j = i;
    std::size_t actual_size = h_kernel_map_sizes[i];
    std::size_t summit_size = h_kernel_map_sizes[i];
    while (i + 1 < num_offsets) {
      actual_size += h_kernel_map_sizes[i + 1];
      summit_size = std::max(
          summit_size, static_cast<std::size_t>(h_kernel_map_sizes[i + 1]));
      auto padded_size = summit_size * ((i + 1) - j + 1);
      auto redundancy = static_cast<double>(padded_size - actual_size) /
                        static_cast<double>(actual_size);
      if (threshold.has_value() && (threshold < 0 || redundancy > threshold)) {
        break;
      }
      i++;
    }
    h_buckets.push_back(i + 1);
  }
  MINUET_CHECK(h_buckets.back() == num_offsets);
  return h_buckets;
}

#define MINUET_GENERATE_TILE_CASES(MACRO) \
  case 512: {                             \
    MACRO(512, 512)                       \
    MACRO(512, 256)                       \
    MACRO(512, 128)                       \
    MACRO(512, 64)                        \
    MACRO(512, 32)                        \
    MACRO(512, 16)                        \
    MACRO(512, 8)                         \
    MACRO(512, 4)                         \
    MACRO(512, 2)                         \
    MACRO(512, 1)                         \
    break;                                \
  }                                       \
  case 384: {                             \
    MACRO(384, 384)                       \
    MACRO(384, 192)                       \
    MACRO(384, 128)                       \
    MACRO(384, 96)                        \
    MACRO(384, 64)                        \
    MACRO(384, 48)                        \
    MACRO(384, 32)                        \
    MACRO(384, 24)                        \
    MACRO(384, 16)                        \
    MACRO(384, 12)                        \
    MACRO(384, 8)                         \
    MACRO(384, 6)                         \
    MACRO(384, 4)                         \
    MACRO(384, 3)                         \
    MACRO(384, 2)                         \
    MACRO(384, 1)                         \
    break;                                \
  }                                       \
  case 256: {                             \
    MACRO(256, 256)                       \
    MACRO(256, 128)                       \
    MACRO(256, 64)                        \
    MACRO(256, 32)                        \
    MACRO(256, 16)                        \
    MACRO(256, 8)                         \
    MACRO(256, 4)                         \
    MACRO(256, 2)                         \
    MACRO(256, 1)                         \
    break;                                \
  }                                       \
  case 192: {                             \
    MACRO(192, 192)                       \
    MACRO(192, 96)                        \
    MACRO(192, 64)                        \
    MACRO(192, 48)                        \
    MACRO(192, 32)                        \
    MACRO(192, 24)                        \
    MACRO(192, 16)                        \
    MACRO(192, 12)                        \
    MACRO(192, 8)                         \
    MACRO(192, 6)                         \
    MACRO(192, 4)                         \
    MACRO(192, 3)                         \
    MACRO(192, 2)                         \
    MACRO(192, 1)                         \
    break;                                \
  }                                       \
  case 128: {                             \
    MACRO(128, 128)                       \
    MACRO(128, 64)                        \
    MACRO(128, 32)                        \
    MACRO(128, 16)                        \
    MACRO(128, 8)                         \
    MACRO(128, 4)                         \
    MACRO(128, 2)                         \
    MACRO(128, 1)                         \
    break;                                \
  }                                       \
  case 96: {                              \
    MACRO(96, 96)                         \
    MACRO(96, 48)                         \
    MACRO(96, 32)                         \
    MACRO(96, 24)                         \
    MACRO(96, 16)                         \
    MACRO(96, 12)                         \
    MACRO(96, 8)                          \
    MACRO(96, 6)                          \
    MACRO(96, 4)                          \
    MACRO(96, 3)                          \
    MACRO(96, 2)                          \
    MACRO(96, 1)                          \
    break;                                \
  }                                       \
  case 64: {                              \
    MACRO(64, 64)                         \
    MACRO(64, 32)                         \
    MACRO(64, 16)                         \
    MACRO(64, 8)                          \
    MACRO(64, 4)                          \
    MACRO(64, 2)                          \
    MACRO(64, 1)                          \
    break;                                \
  }                                       \
  case 32: {                              \
    MACRO(32, 32)                         \
    MACRO(32, 16)                         \
    MACRO(32, 8)                          \
    MACRO(32, 4)                          \
    MACRO(32, 2)                          \
    MACRO(32, 1)                          \
    break;                                \
  }                                       \
  case 16: {                              \
    MACRO(16, 16)                         \
    MACRO(16, 8)                          \
    MACRO(16, 4)                          \
    MACRO(16, 2)                          \
    MACRO(16, 1)                          \
    break;                                \
  }                                       \
  case 8: {                               \
    MACRO(8, 8)                           \
    MACRO(8, 4)                           \
    MACRO(8, 2)                           \
    MACRO(8, 1)                           \
    break;                                \
  }                                       \
  case 4: {                               \
    MACRO(4, 4)                           \
    MACRO(4, 2)                           \
    MACRO(4, 1)                           \
    break;                                \
  }                                       \
  case 2: {                               \
    MACRO(2, 2)                           \
    MACRO(2, 1)                           \
    break;                                \
  }                                       \
  case 1: {                               \
    MACRO(1, 1)                           \
    break;                                \
  }

template <typename IT, typename FT>
void Gather(std::size_t num_sources,                 //
            std::size_t num_offsets,                 //
            std::size_t num_source_features,         //
            const IT *d_cumsum_kernel_padded_sizes,  //
            const IT *d_source_masks,                //
            const FT *d_sources,                     //
            FT *d_source_buffers,                    //
            std::size_t tile_size,                   //
            const Context &context) {
  constexpr std::size_t THREAD_BLOCK_SIZE = 128;
  switch (num_source_features) {
#define MINUET_GATHER_CASE(NUM_FEATURES, TILE_SIZE)                         \
  if (tile_size == TILE_SIZE) {                                             \
    constexpr const auto BULK_SIZE =                                        \
        std::min(static_cast<std::size_t>(TILE_SIZE),                       \
                 (TILE_SIZE % (SizeOf<int4> / SizeOf<FT>) == 0)             \
                     ? SizeOf<int4> / SizeOf<FT>                            \
                     : SizeOf<int3> / SizeOf<FT>);                          \
    static_assert(NUM_FEATURES % TILE_SIZE == 0);                           \
    constexpr const auto NUM_TILES = NUM_FEATURES / TILE_SIZE;              \
    constexpr const auto FAKE_WARP_SIZE =                                   \
        std::min(static_cast<std::size_t>(NUM_TILES),                       \
                 static_cast<std::size_t>(WARP_SIZE));                      \
    constexpr const auto NUM_WARPS =                                        \
        DivCeil<std::size_t>(NUM_TILES, FAKE_WARP_SIZE);                    \
    context.Launch(                                                         \
        num_sources *NUM_WARPS *FAKE_WARP_SIZE, THREAD_BLOCK_SIZE, 0,       \
        kernels::PaddedGatherWarpOptimized<NUM_TILES, TILE_SIZE, BULK_SIZE, \
                                           FAKE_WARP_SIZE, IT, FT>,         \
        num_sources, num_offsets, d_cumsum_kernel_padded_sizes,             \
        d_source_masks, d_sources, d_source_buffers);                       \
    return;                                                                 \
  }
    MINUET_GENERATE_TILE_CASES(MINUET_GATHER_CASE)
#undef MINUET_GATHER_CASE
    default: {
      break;
    }
  }
  MINUET_ERROR("Unsupported num_source_features = ", num_source_features,
               " gather_tile_size = ", tile_size);
}

template <typename IT, typename FT>
void Scatter(std::size_t num_targets,                 //
             std::size_t num_offsets,                 //
             std::size_t num_target_features,         //
             const IT *d_cumsum_kernel_padded_sizes,  //
             const IT *d_target_masks,                //
             const FT *d_target_buffers,              //
             FT *d_targets,                           //
             std::size_t tile_size,                   //
             const Context &context) {
  constexpr std::size_t THREAD_BLOCK_SIZE = 128;
  switch (num_target_features) {
#define MINUET_SCATTER_CASE(NUM_FEATURES, TILE_SIZE)                         \
  if (tile_size == TILE_SIZE) {                                              \
    constexpr const auto BULK_SIZE =                                         \
        std::min(static_cast<std::size_t>(TILE_SIZE),                        \
                 (TILE_SIZE % (SizeOf<int4> / SizeOf<FT>) == 0)              \
                     ? SizeOf<int4> / SizeOf<FT>                             \
                     : SizeOf<int3> / SizeOf<FT>);                           \
    static_assert(NUM_FEATURES % TILE_SIZE == 0);                            \
    constexpr const auto NUM_TILES = NUM_FEATURES / TILE_SIZE;               \
    constexpr const auto FAKE_WARP_SIZE =                                    \
        std::min(static_cast<std::size_t>(1) << CeilLog2(NUM_TILES),         \
                 static_cast<std::size_t>(WARP_SIZE));                       \
    constexpr const auto NUM_WARPS =                                         \
        DivCeil<std::size_t>(NUM_TILES, FAKE_WARP_SIZE);                     \
    context.Launch(                                                          \
        num_targets *NUM_WARPS *FAKE_WARP_SIZE, THREAD_BLOCK_SIZE, 0,        \
        kernels::PaddedScatterWarpOptimized<NUM_TILES, TILE_SIZE, BULK_SIZE, \
                                            FAKE_WARP_SIZE, IT, FT>,         \
        num_targets, num_offsets, d_cumsum_kernel_padded_sizes,              \
        d_target_masks, d_target_buffers, d_targets);                        \
                                                                             \
    return;                                                                  \
  }
    MINUET_GENERATE_TILE_CASES(MINUET_SCATTER_CASE)
#undef MINUET_SCATTER_CASE
    default: {
      break;
    }
  }
  MINUET_ERROR("Unsupported num_target_features = ", num_target_features,
               "scatter_tile_size = ", tile_size);
}

template <typename IT>
cpu::Memory<IT> ComputeCumsumKernelPaddedSizes(
    std::size_t num_offsets,           //
    const std::vector<IT> &h_buckets,  //
    const cpu::Memory<IT> &h_kernel_map_sizes) {
  // Compute padding strategy & padding indices
  cpu::Memory<IT> h_cumsum_kernel_padded_sizes(num_offsets + 1);
  h_cumsum_kernel_padded_sizes[0] = 0;
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    std::size_t summit_size = 0;
    for (UIter j = h_buckets[i]; j < h_buckets[i + 1]; j++) {
      summit_size = std::max(summit_size,
                             static_cast<std::size_t>(h_kernel_map_sizes[j]));
    }
    for (UIter j = h_buckets[i]; j < h_buckets[i + 1]; j++) {
      h_cumsum_kernel_padded_sizes[j + 1] = h_cumsum_kernel_padded_sizes[j];
      h_cumsum_kernel_padded_sizes[j + 1] += summit_size;
    }
  }
  return h_cumsum_kernel_padded_sizes;
}

template <typename IT, typename FT>
float TimeGather::operator()(
    std::size_t num_sources,                 // S
    std::size_t num_targets,                 // T
    std::size_t num_offsets,                 // O
    const std::optional<double> &threshold,  //
    bool allow_shortcut_matmul,       // True if kernel stride & dilation is 1
    std::size_t num_source_features,  // C_in
    std::size_t num_target_features,  // C_out
    const IT *d_source_masks,         // [S, O]
    const IT *d_kernel_map_sizes,     // [O]
    std::size_t gather_tile_size,     //
    const Context &context) const {
  const auto num_kernel_features = num_source_features * num_target_features;

  cpu::Memory<IT> h_kernel_map_sizes(num_offsets);
  context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(),
                         num_offsets);
  const bool shortcut_last_offset =
      allow_shortcut_matmul &&
      h_kernel_map_sizes[num_offsets - 1] == num_targets &&
      h_kernel_map_sizes[num_offsets - 1] == num_sources;
  if (shortcut_last_offset) {
    num_offsets--;
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  //
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_sources = context.NewBuffer<FT>(num_sources * num_source_features);

  Event prior, after;

  prior.Record(context.stream());

  Gather(num_sources, num_offsets, num_source_features,
         d_cumsum_kernel_padded_sizes.device_data(), d_source_masks,
         d_sources.device_data(), d_source_buffers.device_data(),
         gather_tile_size, context);
  after.Record(context.stream());
  after.Synchronize();
  return prior.Elapsed(after);
}

template <typename IT, typename FT>
float TimeScatter::operator()(
    std::size_t num_sources,                 // S
    std::size_t num_targets,                 // T
    std::size_t num_offsets,                 // O
    const std::optional<double> &threshold,  //
    bool allow_shortcut_matmul,       // True if kernel stride & dilation is 1
    std::size_t num_source_features,  // C_in
    std::size_t num_target_features,  // C_out
    const IT *d_target_masks,         // [T, O]
    const IT *d_kernel_map_sizes,     // [O]
    std::size_t tile_size,            //
    const Context &context) const {
  const auto num_kernel_features = num_source_features * num_target_features;

  cpu::Memory<IT> h_kernel_map_sizes(num_offsets);
  context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(),
                         num_offsets);
  const bool shortcut_last_offset =
      allow_shortcut_matmul &&
      h_kernel_map_sizes[num_offsets - 1] == num_targets &&
      h_kernel_map_sizes[num_offsets - 1] == num_sources;
  if (shortcut_last_offset) {
    num_offsets--;
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  //
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);
  auto d_targets = context.NewBuffer<FT>(num_targets * num_target_features);

  Event prior, after;
  prior.Record(context.stream());
  Scatter(num_targets, num_offsets, num_target_features,
          d_cumsum_kernel_padded_sizes.device_data(), d_target_masks,
          d_target_buffers.device_data(), d_targets.device_data(), tile_size,
          context);
  after.Record(context.stream());
  after.Synchronize();
  return prior.Elapsed(after);
}

template <typename IT, typename FT>
float TimeGEMM::operator()(
    std::size_t num_sources,                 // S
    std::size_t num_targets,                 // T
    std::size_t num_offsets,                 // O
    const std::optional<double> &threshold,  //
    std::size_t parallel,
    bool allow_shortcut_matmul,       // True if kernel stride & dilation is 1
    std::size_t num_source_features,  // C_in
    std::size_t num_target_features,  // C_out
    const IT *d_kernel_map_sizes,     // [O]
    const FT *d_weights,              // [O, C_in, C_out]
    const Context &context) const {
  const auto num_kernel_features = num_source_features * num_target_features;
  cpu::Memory<IT> h_kernel_map_sizes(num_offsets);
  context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(),
                         num_offsets);
  const bool shortcut_last_offset =
      allow_shortcut_matmul &&
      h_kernel_map_sizes[num_offsets - 1] == num_targets &&
      h_kernel_map_sizes[num_offsets - 1] == num_sources;
  if (shortcut_last_offset) {
    num_offsets--;
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  // Create buffer for sources and targets
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);

  parallel = std::min(parallel, h_buckets.size() - 1);
  if (parallel > 1) {
    while (stream_pool_.size() < parallel) {
      stream_pool_.emplace_back(cudaStreamNonBlocking);
    }
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(context.stream());
      MINUET_CHECK_CUDA(
          cudaStreamWaitEvent(stream_pool_[i].stream(), event.event()));
    }
  }

  Event prior, after;
  prior.Record(context.stream());
  // Perform GEMMs
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    auto padded_base = h_cumsum_kernel_padded_sizes[h_buckets[i]];

    auto m = *std::max_element(h_kernel_map_sizes.data() + h_buckets[i],
                               h_kernel_map_sizes.data() + h_buckets[i + 1]);
    //    auto m = h_kernel_map_sizes[h_buckets[i + 1] - 1];
    if (m == 0) {
      continue;
    }
    auto b = h_buckets[i + 1] - h_buckets[i];

    auto d_a = d_source_buffers.device_data();
    d_a += padded_base * num_source_features;

    auto d_b = d_weights;
    d_b += h_buckets[i] * num_kernel_features;

    auto d_c = d_target_buffers.device_data();
    d_c += padded_base * num_target_features;

    auto stream =
        (parallel > 1) ? stream_pool_[i % parallel].stream() : context.stream();

    BatchedMatMul(b,                    // b
                  m,                    // m
                  num_source_features,  // k
                  num_target_features,  // n
                  false,                // is_a_transposed
                  false,                // is_b_transposed
                  d_a,                  // d_a
                  d_b,                  // d_b
                  d_c,                  // d_c
                  false,                // incremental
                  context,              // context
                  stream);
  }
  if (parallel > 1) {
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(stream_pool_[i].stream());
      MINUET_CHECK_CUDA(cudaStreamWaitEvent(context.stream(), event.event()));
    }
  }
  // Note that we don't include the last offset (if there is) since it's
  // always the same regardless of `threshold`
  after.Record(context.stream());
  after.Synchronize();
  return prior.Elapsed(after);
}

template <typename IT, typename FT>
void SparseConvolutionForward::operator()(
    std::size_t num_sources,                 // S
    std::size_t num_targets,                 // T
    std::size_t num_offsets,                 // O
    const std::optional<double> &threshold,  //
    std::size_t parallel,                    //
    bool allow_shortcut_matmul,       // True if kernel stride & dilation is 1
    std::size_t num_source_features,  // C_in
    std::size_t num_target_features,  // C_out
    const IT *d_source_masks,         // [S, O]
    const IT *d_target_masks,         // [T, O]
    const IT *d_kernel_map_order,     // [O]
    const IT *d_kernel_map_sizes,     // [O]
    const FT *d_sources,              // [S, C_in]
    const FT *d_weights,              // [O, C_in, C_out]
    FT *d_targets,                    // [T, C_out]
    std::size_t gather_tile_size,     //
    std::size_t scatter_tile_size,    //
    const Context &context) const {
  const auto num_kernel_features = num_source_features * num_target_features;

  const FT *d_ordered_weights;
  std::optional<Memory<FT>> d_ordered_weights_buffer;
  if (d_kernel_map_order != nullptr) {
    d_ordered_weights_buffer =
        context.NewBuffer<FT>(num_offsets * num_kernel_features);
    d_ordered_weights = d_ordered_weights_buffer.value().device_data();

    constexpr std::size_t THREAD_BLOCK_SIZE = 128;
    constexpr std::size_t MAX_SHARED_MEMORY_SIZE = 8 * 1024;
    constexpr const std::size_t OFFSET_SIZE = sizeof(IT);
    const auto max_offsets_per_round =
        std::min(DivFloor(MAX_SHARED_MEMORY_SIZE, OFFSET_SIZE), num_offsets);
    const auto shared_memory_size = std::max(
        max_offsets_per_round * OFFSET_SIZE, sizeof(IT) * THREAD_BLOCK_SIZE);
    context.Launch(num_offsets * num_kernel_features,              //
                   THREAD_BLOCK_SIZE,                              //
                   shared_memory_size,                             //
                   kernels::PermuteWeights<IT, FT>,                //
                   num_offsets,                                    //
                   num_kernel_features,                            //
                   max_offsets_per_round,                          //
                   d_kernel_map_order,                             //
                   d_weights,                                      //
                   d_ordered_weights_buffer.value().device_data()  //
    );
  } else {
    d_ordered_weights = d_weights;
  }

  cpu::Memory<IT> h_kernel_map_sizes(num_offsets);
  context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(),
                         num_offsets);
  const bool shortcut_last_offset =
      allow_shortcut_matmul &&
      h_kernel_map_sizes[num_offsets - 1] == num_targets &&
      h_kernel_map_sizes[num_offsets - 1] == num_sources;
  if (shortcut_last_offset) {
    num_offsets--;
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  // Create buffer for sources and targets
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);

  // Gather from all sources
  Gather(num_sources, num_offsets, num_source_features,
         d_cumsum_kernel_padded_sizes.device_data(), d_source_masks, d_sources,
         d_source_buffers.device_data(), gather_tile_size, context);

  parallel = std::min(parallel, h_buckets.size() - 1);
  if (parallel > 1) {
    while (stream_pool_.size() < parallel) {
      stream_pool_.emplace_back(cudaStreamNonBlocking);
    }
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(context.stream());
      MINUET_CHECK_CUDA(
          cudaStreamWaitEvent(stream_pool_[i].stream(), event.event()));
    }
  }

  // Perform GEMMs
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    auto padded_base = h_cumsum_kernel_padded_sizes[h_buckets[i]];

    auto m = *std::max_element(h_kernel_map_sizes.data() + h_buckets[i],
                               h_kernel_map_sizes.data() + h_buckets[i + 1]);
    if (m == 0) {
      continue;
    }
    auto b = h_buckets[i + 1] - h_buckets[i];

    auto d_a = d_source_buffers.device_data();
    d_a += padded_base * num_source_features;

    auto d_b = d_ordered_weights;
    d_b += h_buckets[i] * num_kernel_features;

    auto d_c = d_target_buffers.device_data();
    d_c += padded_base * num_target_features;

    auto stream =
        (parallel > 1) ? stream_pool_[i % parallel].stream() : context.stream();

    BatchedMatMul(b,                    // b
                  m,                    // m
                  num_source_features,  // k
                  num_target_features,  // n
                  false,                // is_a_transposed
                  false,                // is_b_transposed
                  d_a,                  // d_a
                  d_b,                  // d_b
                  d_c,                  // d_c
                  false,                // incremental
                  context,              //
                  stream);
  }
  if (parallel > 1) {
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(stream_pool_[i].stream());
      MINUET_CHECK_CUDA(cudaStreamWaitEvent(context.stream(), event.event()));
    }
  }
  Scatter(num_targets, num_offsets, num_target_features,
          d_cumsum_kernel_padded_sizes.device_data(), d_target_masks,
          d_target_buffers.device_data(), d_targets, scatter_tile_size,
          context);

  if (shortcut_last_offset) {
    auto d_b = d_ordered_weights;
    d_b += num_offsets * num_kernel_features;
    MatMul(num_targets,          // m
           num_source_features,  // k
           num_target_features,  // n
           false,                // is_a_transposed
           false,                // is_b_transposed
           d_sources,            // d_a
           d_b,                  // d_b
           d_targets,            // d_c
           true,                 // incremental
           context);
  }
}  // namespace minuet::cuda

#define MINUET_EXPLICIT_INSTANTIATION(_, IT, FT)                        \
  template void SparseConvolutionForward::operator()<IT, FT>(           \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, const std::optional<double> &threshold,  \
      std::size_t parallel, bool allow_shortcut_matmul,                 \
      std::size_t num_source_features, std::size_t num_target_features, \
      const IT *d_source_masks, const IT *d_target_masks,               \
      const IT *d_kernel_map_order, const IT *d_kernel_map_sizes,       \
      const FT *d_sources, const FT *d_weights, FT *d_targets,          \
      std::size_t gather_tile_size, std::size_t scatter_tile_size,      \
      const Context &context) const;                                    \
  template float TimeGather::operator()<IT, FT>(                        \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, const std::optional<double> &threshold,  \
      bool allow_shortcut_matmul, std::size_t num_source_features,      \
      std::size_t num_target_features, const IT *d_source_masks,        \
      const IT *d_kernel_map_sizes, std::size_t gather_tile_size,       \
      const Context &context) const;                                    \
  template float TimeScatter::operator()<IT, FT>(                       \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, const std::optional<double> &threshold,  \
      bool allow_shortcut_matmul, std::size_t num_source_features,      \
      std::size_t num_target_features, const IT *d_target_masks,        \
      const IT *d_kernel_map_sizes, std::size_t tile_size,              \
      const minuet::cuda::Context &context) const;                      \
  template float TimeGEMM::operator()<IT, FT>(                          \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, const std::optional<double> &threshold,  \
      std::size_t parallel, bool allow_shortcut_matmul,                 \
      std::size_t num_source_features, std::size_t num_target_features, \
      const IT *d_kernel_map_sizes, const FT *d_weights,                \
      const minuet::cuda::Context &context) const

MINUET_FOR_ALL_IF_TYPES(MINUET_EXPLICIT_INSTANTIATION);

}  // namespace minuet::cuda