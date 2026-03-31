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

#include <algorithm>
#include <sstream>
#include "tensor_cfg.h"
#include "tmem_window_planner.h"

using namespace vortex;

namespace {

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

uint32_t align_up(uint32_t value, uint32_t align) {
  return ceil_div(value, align) * align;
}

uint32_t fmt_bytes(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
    return 1;
  case vortex::tensor::fp16::id:
  case vortex::tensor::bf16::id:
    return 2;
  case vortex::tensor::fp32::id:
    return 4;
  default:
    return 0;
  }
}

uint32_t cmem_packet_count(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
    return 4;
  case vortex::tensor::fp16::id:
    return 8;
  case vortex::tensor::fp32::id:
    return 16;
  default:
    return 0;
  }
}

uint32_t cmem_packets_per_subtile(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
    return 1;
  case vortex::tensor::fp16::id:
    return 2;
  case vortex::tensor::fp32::id:
    return 4;
  default:
    return 0;
  }
}

uint32_t ab_packet_count(TmemWindowTarget target, uint32_t fmt, uint32_t sparse_mode) {
  (void)target;
  (void)sparse_mode;
  switch (fmt) {
  case vortex::tensor::fp8::id:
    return 4;
  case vortex::tensor::fp16::id:
    return 8;
  default:
    return 0;
  }
}

void set_reason(std::string* reason, const std::string& text) {
  if (reason) {
    *reason = text;
  }
}

bool packet_math_region(const TmemWindowPlan& window,
                        uint32_t packet_idx,
                        uint32_t* math_row_base,
                        uint32_t* math_col_base,
                        uint32_t* packet_rows,
                        uint32_t* packet_cols) {
  auto elem_bytes = fmt_bytes(window.fmt);
  if (0 == elem_bytes || window.tile_cols == 0 || window.packets_per_tile == 0) {
    return false;
  }

  auto tile_idx = packet_idx / window.packets_per_tile;
  auto local_packet_idx = packet_idx % window.packets_per_tile;
  auto tile_row = tile_idx / window.tile_cols;
  auto tile_col = tile_idx % window.tile_cols;

  switch (window.layout_kind) {
  case TmemWindowLayoutKind::ALineNative: {
    auto packets_per_line = ab_packet_count(TmemWindowTarget::A, window.fmt, window.sparse_mode) / 4;
    if (0 == packets_per_line) {
      return false;
    }
    auto line_id = local_packet_idx / packets_per_line;
    auto packet_in_line = local_packet_idx % packets_per_line;
    auto line_row = line_id / 2;
    auto line_col = line_id % 2;
    if (math_row_base) {
      *math_row_base = tile_row * TmemWindowPlanner::kTileRows
                     + line_row * 8
                     + packet_in_line * window.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * TmemWindowPlanner::kTileCols
                     + line_col * 8;
    }
    break;
  }
  case TmemWindowLayoutKind::BLineNative: {
    auto packets_per_line = ab_packet_count(TmemWindowTarget::B, window.fmt, window.sparse_mode) / 4;
    if (0 == packets_per_line) {
      return false;
    }
    auto line_id = local_packet_idx / packets_per_line;
    auto packet_in_line = local_packet_idx % packets_per_line;
    auto line_col = line_id / 2;
    auto line_row = line_id % 2;
    if (math_row_base) {
      *math_row_base = tile_row * TmemWindowPlanner::kTileRows
                     + line_row * 8
                     + packet_in_line * window.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * TmemWindowPlanner::kTileCols
                     + line_col * 8;
    }
    break;
  }
  case TmemWindowLayoutKind::MathRowMajor: {
    auto packet_cols_per_tile = ceil_div(TmemWindowPlanner::kTileCols, window.packet_cols);
    auto packet_row_group = local_packet_idx / packet_cols_per_tile;
    auto packet_col_group = local_packet_idx % packet_cols_per_tile;
    if (math_row_base) {
      *math_row_base = tile_row * TmemWindowPlanner::kTileRows + packet_row_group * window.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * TmemWindowPlanner::kTileCols + packet_col_group * window.packet_cols;
    }
    break;
  }
  case TmemWindowLayoutKind::CSubtileNative: {
    auto packets_per_subtile = cmem_packets_per_subtile(window.fmt);
    if (0 == packets_per_subtile) {
      return false;
    }
    auto subtile_id = local_packet_idx / packets_per_subtile;
    auto packet_in_subtile = local_packet_idx % packets_per_subtile;
    auto subtile_row = subtile_id / 2;
    auto subtile_col = subtile_id % 2;
    if (math_row_base) {
      *math_row_base = tile_row * TmemWindowPlanner::kTileRows
                     + subtile_row * 8
                     + packet_in_subtile * window.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * TmemWindowPlanner::kTileCols
                     + subtile_col * 8;
    }
    break;
  }
  case TmemWindowLayoutKind::LinearPacketStream:
  default:
    return false;
  }

  if (packet_rows) {
    *packet_rows = window.packet_rows;
  }
  if (packet_cols) {
    *packet_cols = window.packet_cols;
  }
  return true;
}

} // namespace

bool TmemWindowPlanner::append_dense_window(TmemLayoutPlan* plan,
                                            uint32_t* next_window_id,
                                            uint32_t* required_alloc_col_span,
                                            uint32_t* next_line_base,
                                            TmemWindowTarget target,
                                            const TensorShape2D& shape,
                                            uint32_t fmt,
                                            uint32_t sparse_mode,
                                            std::string* reason) {
  if (nullptr == plan || nullptr == next_window_id || nullptr == required_alloc_col_span || nullptr == next_line_base) {
    return false;
  }
  if (shape.empty()) {
    return true;
  }

  TmemWindowPlan window{};
  if (!build_single_dense_window(target, shape, fmt, sparse_mode, (*next_window_id)++, &window, reason)) {
    return false;
  }
  window.logical_col_base = 0;
  window.logical_line_base = *next_line_base;
  uint32_t packet_base = 0;
  for (const auto& existing : plan->windows) {
    packet_base += existing.tile_count * existing.packets_per_tile;
  }
  window.packet_base = packet_base;

  auto next_logical_line = *next_line_base + window.logical_line_span;
  if (next_logical_line > kLogicalLines) {
    std::ostringstream oss;
    oss << "window " << window.window_id << " exceeds TMEM logical line capacity";
    set_reason(reason, oss.str());
    return false;
  }

  if (window.logical_col_span > kLogicalCols) {
    std::ostringstream oss;
    oss << "window " << window.window_id << " exceeds TMEM logical column capacity";
    set_reason(reason, oss.str());
    return false;
  }

  *next_line_base = next_logical_line;
  *required_alloc_col_span = std::max<uint32_t>(*required_alloc_col_span, window.logical_col_span);
  plan->required_col_span = std::max<uint32_t>(plan->required_col_span, *required_alloc_col_span);
  plan->required_logical_col_span = std::max<uint32_t>(plan->required_logical_col_span, window.logical_col_span);
  plan->required_logical_line_span = std::max<uint32_t>(plan->required_logical_line_span, next_logical_line);
  plan->windows.push_back(window);
  return true;
}

bool TmemWindowPlanner::build_single_dense_window(TmemWindowTarget target,
                                                  const TensorShape2D& shape,
                                                  uint32_t fmt,
                                                  uint32_t sparse_mode,
                                                  uint32_t window_id,
                                                  TmemWindowPlan* out,
                                                  std::string* reason) {
  if (nullptr == out) {
    return false;
  }
  if (shape.empty()) {
    set_reason(reason, "empty shape cannot form a TMEM window");
    return false;
  }

  auto elem_bytes = fmt_bytes(fmt);
  if (0 == elem_bytes) {
    std::ostringstream oss;
    oss << "unsupported format " << fmt << " for TMEM window target " << static_cast<uint32_t>(target);
    set_reason(reason, oss.str());
    return false;
  }

  TmemWindowPlan window{};
  window.window_id = window_id;
  window.target = target;
  window.layout_kind = TmemWindowLayoutKind::LinearPacketStream;
  window.elem_shape = shape;
  window.fmt = fmt;
  window.sparse_mode = sparse_mode;
  window.logical_line_base = 0;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, shape.rows), kTileRows);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, shape.cols), kTileCols);
  window.tile_count = window.tile_rows * window.tile_cols;
  if (target == TmemWindowTarget::A || target == TmemWindowTarget::B
   || target == TmemWindowTarget::C || target == TmemWindowTarget::D) {
    auto packets_per_tile = (target == TmemWindowTarget::C || target == TmemWindowTarget::D)
                          ? cmem_packet_count(fmt)
                          : ab_packet_count(target, fmt, sparse_mode);
    if (0 == packets_per_tile) {
      std::ostringstream oss;
      oss << "unsupported window format " << fmt
          << " for target " << static_cast<uint32_t>(target);
      set_reason(reason, oss.str());
      return false;
    }
    // Explicit A/B windows keep their local packet stream. C/D windows are
    // stored internally in subtile-native packet order while external TMA
    // still sees mathematical row-major packets through the adapter helpers.
    if (target == TmemWindowTarget::A || target == TmemWindowTarget::B) {
      window.layout_kind = (target == TmemWindowTarget::A)
                         ? TmemWindowLayoutKind::ALineNative
                         : TmemWindowLayoutKind::BLineNative;
      window.packet_cols = 8;
      window.packet_rows = std::max<uint32_t>(1, kPacketBytes / (window.packet_cols * elem_bytes));
    } else if (target == TmemWindowTarget::C || target == TmemWindowTarget::D) {
      window.layout_kind = TmemWindowLayoutKind::CSubtileNative;
      window.packet_cols = 8;
      window.packet_rows = std::max<uint32_t>(1, kPacketBytes / (window.packet_cols * elem_bytes));
    }
    window.packets_per_tile = packets_per_tile;
    window.logical_col_span = kPacketBytes;
    window.logical_line_span = window.tile_count * window.packets_per_tile;
    window.logical_tile_col_span = kPacketBytes;
    window.logical_tile_line_span = window.packets_per_tile;
    window.logical_packet_col_span = kPacketBytes;
    window.logical_packet_line_span = 1;
  } else {
    window.layout_kind = TmemWindowLayoutKind::MathRowMajor;
    window.packet_cols = 8;
    window.packet_rows = std::max<uint32_t>(1, kPacketBytes / (window.packet_cols * elem_bytes));
    window.packets_per_tile = ceil_div(kTileRows, window.packet_rows) * ceil_div(kTileCols, window.packet_cols);
    // Dense A/B windows use a packet-major logical layout:
    // each math packet is packed row-major into one 64B logical line.
    window.logical_col_span = kPacketBytes;
    window.logical_line_span = window.tile_count * window.packets_per_tile;
    window.logical_tile_col_span = kPacketBytes;
    window.logical_tile_line_span = window.packets_per_tile;
    window.logical_packet_col_span = kPacketBytes;
    window.logical_packet_line_span = 1;
  }
  *out = window;
  return true;
}

bool TmemWindowPlanner::build_dense_plan(const TmemWindowPlannerInput& input,
                                         TmemLayoutPlan* out,
                                         std::string* reason) {
  if (nullptr == out) {
    return false;
  }

  TmemLayoutPlan plan{};
  uint32_t next_window_id = 0;
  uint32_t required_alloc_col_span = 0;
  uint32_t next_line_base = 0;

  if (!append_dense_window(&plan, &next_window_id, &required_alloc_col_span, &next_line_base,
                           TmemWindowTarget::A, input.a_shape, input.fmt_a, input.sparse_mode, reason)
   || !append_dense_window(&plan, &next_window_id, &required_alloc_col_span, &next_line_base,
                           TmemWindowTarget::B, input.b_shape, input.fmt_b, input.sparse_mode, reason)
   || !append_dense_window(&plan, &next_window_id, &required_alloc_col_span, &next_line_base,
                           TmemWindowTarget::C, input.c_shape, input.fmt_c, input.sparse_mode, reason)
   || !append_dense_window(&plan, &next_window_id, &required_alloc_col_span, &next_line_base,
                           TmemWindowTarget::D, input.d_shape, input.fmt_d, input.sparse_mode, reason)
   || !append_dense_window(&plan, &next_window_id, &required_alloc_col_span, &next_line_base,
                           TmemWindowTarget::Meta, input.meta_shape, vortex::tensor::uint8::id, input.sparse_mode, reason)) {
    return false;
  }

  plan.required_col_span = std::max<uint32_t>(kMinAllocationCols, align_up(plan.required_col_span, kMinAllocationCols));
  if (input.allocation_col_span != 0 && plan.required_col_span > input.allocation_col_span) {
    std::ostringstream oss;
    oss << "planned windows require " << plan.required_col_span
        << " logical columns but allocation only provides " << input.allocation_col_span;
    set_reason(reason, oss.str());
    return false;
  }
  if (plan.required_col_span > kLogicalCols) {
    set_reason(reason, "planned windows exceed TMEM logical column capacity");
    return false;
  }
  if (plan.required_logical_col_span > kLogicalCols) {
    set_reason(reason, "planned logical TMEM window footprint exceeds TMEM logical column capacity");
    return false;
  }
  if (plan.required_logical_line_span > kLogicalLines) {
    set_reason(reason, "planned logical TMEM window footprint exceeds TMEM logical line capacity");
    return false;
  }

  plan.valid = true;
  *out = plan;
  return true;
}

bool TmemWindowPlanner::uses_math_packet_adapter(const TmemWindowPlan& window) {
  return !window.elem_shape.empty()
      && window.packets_per_tile != 0
      && window.tile_cols != 0
      && (window.layout_kind == TmemWindowLayoutKind::MathRowMajor
       || window.layout_kind == TmemWindowLayoutKind::ALineNative
       || window.layout_kind == TmemWindowLayoutKind::BLineNative
       || window.layout_kind == TmemWindowLayoutKind::CSubtileNative);
}

bool TmemWindowPlanner::pack_math_packet(const TmemWindowPlan& window,
                                         const std::vector<uint8_t>& payload,
                                         uint32_t packet_idx,
                                         std::array<uint8_t, kPacketBytes>* out) {
  if (nullptr == out || !uses_math_packet_adapter(window)) {
    return false;
  }
  out->fill(0);
  auto elem_bytes = fmt_bytes(window.fmt);
  auto rows = window.elem_shape.rows;
  auto cols = window.elem_shape.cols;
  auto row_bytes = cols * elem_bytes;
  uint32_t math_row_base = 0;
  uint32_t math_col_base = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
  if (!packet_math_region(window, packet_idx, &math_row_base, &math_col_base, &packet_rows, &packet_cols)) {
    return false;
  }

  uint32_t offset = 0;
  for (uint32_t pr = 0; pr < packet_rows; ++pr) {
    auto math_row = math_row_base + pr;
    for (uint32_t pc = 0; pc < packet_cols; ++pc) {
      auto math_col = math_col_base + pc;
      for (uint32_t b = 0; b < elem_bytes; ++b) {
        if (math_row < rows && math_col < cols) {
          auto payload_off = math_row * row_bytes + math_col * elem_bytes + b;
          if (payload_off < payload.size()) {
            out->at(offset) = payload.at(payload_off);
          }
        }
        ++offset;
      }
    }
  }
  return true;
}

bool TmemWindowPlanner::unpack_math_packet(const TmemWindowPlan& window,
                                           uint32_t packet_idx,
                                           const std::array<uint8_t, kPacketBytes>& packet,
                                           std::vector<uint8_t>* payload) {
  if (nullptr == payload || !uses_math_packet_adapter(window)) {
    return false;
  }
  auto elem_bytes = fmt_bytes(window.fmt);
  auto rows = window.elem_shape.rows;
  auto cols = window.elem_shape.cols;
  auto row_bytes = cols * elem_bytes;
  uint32_t math_row_base = 0;
  uint32_t math_col_base = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
  if (!packet_math_region(window, packet_idx, &math_row_base, &math_col_base, &packet_rows, &packet_cols)) {
    return false;
  }

  uint32_t offset = 0;
  for (uint32_t pr = 0; pr < packet_rows; ++pr) {
    auto math_row = math_row_base + pr;
    for (uint32_t pc = 0; pc < packet_cols; ++pc) {
      auto math_col = math_col_base + pc;
      for (uint32_t b = 0; b < elem_bytes; ++b) {
        if (math_row < rows && math_col < cols) {
          auto payload_off = math_row * row_bytes + math_col * elem_bytes + b;
          if (payload_off < payload->size()) {
            payload->at(payload_off) = packet.at(offset);
          }
        }
        ++offset;
      }
    }
  }
  return true;
}
