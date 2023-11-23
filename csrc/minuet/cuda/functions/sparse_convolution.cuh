#pragma once

#include <optional>

#include "minuet/cuda/context.cuh"

namespace minuet::cuda {

struct SparseConvolutionForward {
  template <typename IT, typename FT>
  void operator()(std::size_t num_sources,                 // S
                  std::size_t num_targets,                 // T
                  std::size_t num_offsets,                 // O
                  const std::optional<double> &threshold,  //
                  std::size_t parallel,                    //
                  bool allow_shortcut_matmul,              //
                  std::size_t num_source_features,         // C_in
                  std::size_t num_target_features,         // C_out
                  const IT *d_source_masks,                // [S, O]
                  const IT *d_target_masks,                // [T, O]
                  const IT *d_kernel_map_order,            // [O]
                  const IT *d_kernel_map_sizes,            // [O]
                  const FT *d_sources,                     // [S, C_in]
                  const FT *d_weights,                     // [O, C_in, C_out]
                  FT *d_targets,                           // [T, C_out]
                  std::size_t gather_tile_size,            //
                  std::size_t scatter_tile_size,           //
                  const Context &context) const;
};

struct TimeGEMM {
  template <typename IT, typename FT>
  float operator()(
      std::size_t num_sources,                 // S
      std::size_t num_targets,                 // T
      std::size_t num_offsets,                 // O
      const std::optional<double> &threshold,  //
      std::size_t parallel,                    //
      bool allow_shortcut_matmul,       // True if kernel stride & dilation is 1
      std::size_t num_source_features,  // C_in
      std::size_t num_target_features,  // C_out
      const IT *d_kernel_map_sizes,     // [O]
      const FT *d_weights,              // [O, C_in, C_out]
      const Context &context) const;
};

struct TimeGather {
  template <typename IT, typename FT>
  float operator()(
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
      const Context &context) const;
};

struct TimeScatter {
  template <typename IT, typename FT>
  float operator()(
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
      const Context &context) const;
};

}  // namespace minuet::cuda