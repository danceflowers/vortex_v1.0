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
#include "tensor_mem_layout_utils.h"
#include "tensor_cfg.h"

using namespace vortex;

namespace {

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

bool uses_generic_math_window(const TmaDescriptor& descriptor) {
  return descriptor.tile_role == 0
      && static_cast<TcuPayloadKind>(descriptor.payload_kind) != TcuPayloadKind::SparseMeta;
}

bool build_generic_math_window(uint32_t window_id,
                               const TmaDescriptor& descriptor,
                               TmemWindowPlan* out) {
  if (nullptr == out || descriptor.rows == 0 || descriptor.cols == 0) {
    return false;
  }
  auto fmt = infer_window_fmt(descriptor);
  auto elem_bytes = fmt_elem_bytes(fmt);
  if (0 == elem_bytes) {
    return false;
  }
  TmemWindowPlan window{};
  window.window_id = window_id;
  window.target = map_window_target(descriptor);
  window.layout_kind = TmemWindowLayoutKind::MathRowMajor;
  window.elem_shape = {descriptor.rows, descriptor.cols};
  window.fmt = fmt;
  window.logical_col_base = 0;
  window.logical_line_base = 0;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, descriptor.rows), TmemWindowPlanner::kTileRows);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, descriptor.cols), TmemWindowPlanner::kTileCols);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.packet_cols = 8;
  window.packet_rows = std::max<uint32_t>(1, TmemWindowPlanner::kPacketBytes / (window.packet_cols * elem_bytes));
  window.packets_per_tile = ceil_div(TmemWindowPlanner::kTileRows, window.packet_rows)
                          * ceil_div(TmemWindowPlanner::kTileCols, window.packet_cols);
  window.logical_col_span = TmemWindowPlanner::kPacketBytes;
  window.logical_line_span = window.tile_count * window.packets_per_tile;
  window.logical_tile_col_span = TmemWindowPlanner::kPacketBytes;
  window.logical_tile_line_span = window.packets_per_tile;
  window.logical_packet_col_span = TmemWindowPlanner::kPacketBytes;
  window.logical_packet_line_span = 1;
  *out = window;
  return true;
}

uint32_t next_available_window_line(const TmemAllocation& allocation) {
  uint32_t next_line = 0;
  for (const auto& window : allocation.windows) {
    next_line = std::max<uint32_t>(next_line, window.logical_line_base + window.logical_line_span);
  }
  return next_line;
}

} // namespace

TmemWindowTarget vortex::map_window_target(const TmaDescriptor& descriptor) {
  if (static_cast<TcuPayloadKind>(descriptor.payload_kind) == TcuPayloadKind::SparseMeta) {
    return TmemWindowTarget::Meta;
  }
  switch (descriptor.tile_role) {
  case 1: return TmemWindowTarget::A;
  case 2: return TmemWindowTarget::B;
  case 3: return TmemWindowTarget::C;
  case 4: return TmemWindowTarget::D;
  default: return TmemWindowTarget::A;
  }
}

uint32_t vortex::infer_window_fmt(const TmaDescriptor& descriptor) {
  switch (descriptor.elem_bytes) {
  case 1: return vortex::tensor::fp8::id;
  case 2: return vortex::tensor::fp16::id;
  case 4: return vortex::tensor::fp32::id;
  default: return 0;
  }
}

uint32_t vortex::fmt_elem_bytes(uint32_t fmt) {
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

uint32_t vortex::meta_shadow_window_id(uint32_t window_id) {
  return window_id | 0x80000000u;
}

bool vortex::build_sparse_meta_window_plan(const TmaDescriptor& descriptor,
                                           uint32_t window_id,
                                           TmemWindowPlan* out) {
  if (nullptr == out
   || descriptor.meta_addr == 0
   || descriptor.meta_size_bytes == 0
   || descriptor.meta_col_span == 0
   || descriptor.tile_role != 1) {
    return false;
  }
  TmemWindowPlan window{};
  window.window_id = meta_shadow_window_id(window_id);
  window.target = TmemWindowTarget::Meta;
  window.layout_kind = TmemWindowLayoutKind::MathRowMajor;
  window.elem_shape = {static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, descriptor.rows), 16u) * 4u),
                       static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, descriptor.cols), 16u) * 16u)};
  window.fmt = vortex::tensor::uint8::id;
  window.logical_line_base = 0;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, descriptor.rows), 16u);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, descriptor.cols), 16u);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.logical_col_span = Tmem::kPacketBytes;
  window.logical_line_span = window.tile_count;
  window.logical_tile_col_span = Tmem::kPacketBytes;
  window.logical_tile_line_span = 1;
  window.packet_cols = 16;
  window.packet_rows = 4;
  window.logical_packet_col_span = Tmem::kPacketBytes;
  window.logical_packet_line_span = 1;
  window.packets_per_tile = 1;
  *out = window;
  return true;
}

TmemWindowPlan vortex::build_legacy_window_plan(uint32_t window_id,
                                                const TmaDescriptor& descriptor,
                                                uint32_t col_base,
                                                uint32_t col_span) {
  (void)col_base;
  TmemWindowPlan window{};
  auto shape = TensorShape2D{descriptor.rows, descriptor.cols};
  auto fmt = infer_window_fmt(descriptor);
  bool built_from_shape = false;
  if (!shape.empty() && fmt_elem_bytes(fmt) != 0) {
    if (uses_generic_math_window(descriptor)) {
      (void)build_generic_math_window(window_id, descriptor, &window);
    } else {
      (void)TmemWindowPlanner::build_single_dense_window(map_window_target(descriptor),
                                                         shape,
                                                         fmt,
                                                         0,
                                                         0, // col_span: auto
                                                         window_id,
                                                         &window,
                                                         nullptr);
    }
    built_from_shape = true;
  } else {
    window.window_id = window_id;
    window.target = map_window_target(descriptor);
  }
  if (!built_from_shape) {
    auto packet_line_bytes = std::min<uint32_t>(Tmem::kPacketBytes, std::max<uint32_t>(1, col_span));
    window.logical_col_span = packet_line_bytes;
    window.logical_line_span = std::max<uint32_t>(1, ceil_div(std::max<uint32_t>(1, descriptor.size_bytes), packet_line_bytes));
    window.logical_tile_col_span = packet_line_bytes;
    window.logical_tile_line_span = window.logical_line_span;
    window.packet_cols = packet_line_bytes;
    window.packet_rows = 1;
    window.logical_packet_col_span = packet_line_bytes;
    window.logical_packet_line_span = 1;
    window.tile_rows = 1;
    window.tile_cols = 1;
    window.tile_count = 1;
    window.packets_per_tile = ceil_div(std::max<uint32_t>(1, descriptor.size_bytes), packet_line_bytes);
  }
  window.logical_col_base = 0;
  window.logical_line_base = 0;
  return window;
}

void vortex::adjust_legacy_subwindow_from_existing(const TmaDescriptor& descriptor,
                                                   const TmemWindowPlan* existing_window,
                                                   TmemWindowPlan* window) {
  if (nullptr == existing_window || nullptr == window) {
    return;
  }
  if ((descriptor.rows != 0 && descriptor.cols != 0) || descriptor.elem_bytes != 0) {
    return;
  }

  auto target = static_cast<TcuTarget>(descriptor.tile_role);
  switch (target) {
  case TcuTarget::C:
    if (existing_window->logical_col_span == window->logical_col_span
     && existing_window->logical_line_span >= window->logical_line_span) {
      window->logical_col_base = existing_window->logical_col_base;
      window->logical_line_base = existing_window->logical_line_base
                                + (existing_window->logical_line_span - window->logical_line_span);
    }
    break;
  case TcuTarget::A:
    window->logical_col_base = existing_window->logical_col_base;
    window->logical_line_base = existing_window->logical_line_base;
    break;
  default:
    break;
  }
}

bool vortex::preserve_existing_math_window(const TmemWindowPlan* existing_window,
                                           const TmaDescriptor& descriptor) {
  if (nullptr == existing_window || !TmemWindowPlanner::uses_math_packet_adapter(*existing_window)) {
    return false;
  }
  if (!uses_generic_math_window(descriptor) && existing_window->target != map_window_target(descriptor)) {
    return false;
  }
  auto inferred_fmt = infer_window_fmt(descriptor);
  if (fmt_elem_bytes(inferred_fmt) != 0 && existing_window->fmt != inferred_fmt) {
    return false;
  }
  if (descriptor.rows != 0 && existing_window->elem_shape.rows != descriptor.rows) {
    return false;
  }
  if (descriptor.cols != 0 && existing_window->elem_shape.cols != descriptor.cols) {
    return false;
  }
  return true;
}

bool vortex::place_window_after_existing(const TmemAllocation& allocation,
                                         TmemWindowPlan* window) {
  if (nullptr == window) {
    return false;
  }
  window->logical_col_base = 0;
  window->logical_line_base = next_available_window_line(allocation);
  return window->logical_col_span <= allocation.col_span
      && (window->logical_line_base + window->logical_line_span) <= TmemWindowPlanner::kLogicalLines;
}

void vortex::encode_math_window_packet(const TmemWindowPlan& window,
                                       const std::vector<uint8_t>& math_window_bytes,
                                       uint32_t packet_idx,
                                       TmemPacket* packet) {
  if (nullptr == packet) {
    return;
  }
  if (!TmemWindowPlanner::pack_math_packet(window, math_window_bytes, packet_idx, &packet->bytes)) {
    packet->bytes.fill(0);
  }
}

void vortex::decode_math_window_packet(const TmemWindowPlan& window,
                                       uint32_t packet_idx,
                                       const TmemPacket& packet,
                                       std::vector<uint8_t>* math_bytes) {
  if (nullptr == math_bytes) {
    return;
  }
  (void)TmemWindowPlanner::unpack_math_packet(window, packet_idx, packet.bytes, math_bytes);
}
