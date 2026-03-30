// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vortex {

enum class TmemWindowTarget : uint8_t {
  A = 0,
  B,
  C,
  D,
  Meta,
};

struct TensorShape2D {
  uint16_t rows = 0;
  uint16_t cols = 0;

  bool empty() const {
    return rows == 0 || cols == 0;
  }
};

struct TmemWindowPlannerInput {
  TensorShape2D a_shape = {};
  TensorShape2D b_shape = {};
  TensorShape2D c_shape = {};
  TensorShape2D d_shape = {};
  TensorShape2D meta_shape = {};
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint32_t sparse_mode = 0;
  uint32_t allocation_col_span = 0;
};

struct TmemWindowPlan {
  uint32_t window_id = 0;
  TmemWindowTarget target = TmemWindowTarget::A;
  TensorShape2D elem_shape = {};
  uint32_t fmt = 0;
  uint32_t sparse_mode = 0;
  uint32_t logical_col_base = 0;
  uint32_t logical_line_base = 0;
  uint32_t logical_col_span = 0;
  uint32_t logical_line_span = 0;
  uint32_t logical_tile_col_span = 0;
  uint32_t logical_tile_line_span = 0;
  uint32_t logical_packet_col_span = 0;
  uint32_t logical_packet_line_span = 0;
  uint32_t packet_base = 0;
  uint32_t tile_rows = 0;
  uint32_t tile_cols = 0;
  uint32_t tile_count = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
  uint32_t packets_per_tile = 0;
};

struct TmemLayoutPlan {
  bool valid = false;
  uint32_t epoch = 0;
  uint32_t required_col_span = 0;
  uint32_t required_logical_col_span = 0;
  uint32_t required_logical_line_span = 0;
  std::vector<TmemWindowPlan> windows;
};

class TmemWindowPlanner {
public:
  static constexpr uint32_t kMinAllocationCols = 16;
  static constexpr uint32_t kLogicalCols = 64;
  static constexpr uint32_t kLogicalLines = 128;
  static constexpr uint32_t kTileRows = 16;
  static constexpr uint32_t kTileCols = 16;
  static constexpr uint32_t kPacketBytes = 64;

  static bool build_dense_plan(const TmemWindowPlannerInput& input,
                               TmemLayoutPlan* out,
                               std::string* reason = nullptr);
  static bool build_single_dense_window(TmemWindowTarget target,
                                        const TensorShape2D& shape,
                                        uint32_t fmt,
                                        uint32_t sparse_mode,
                                        uint32_t window_id,
                                        TmemWindowPlan* out,
                                        std::string* reason = nullptr);

private:
  static bool append_dense_window(TmemLayoutPlan* plan,
                                  uint32_t* next_window_id,
                                  uint32_t* required_alloc_col_span,
                                  uint32_t* next_line_base,
                                  TmemWindowTarget target,
                                  const TensorShape2D& shape,
                                  uint32_t fmt,
                                  uint32_t sparse_mode,
                                  std::string* reason);
};

} // namespace vortex
