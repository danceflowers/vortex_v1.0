// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "tmem_math_packet.h"

#include "tensor_cfg.h"

namespace vortex {
namespace tmem_math_packet {
namespace {

constexpr uint32_t kTileRows = 16;
constexpr uint32_t kTileCols = 16;

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

uint32_t fmt_bytes(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
  case vortex::tensor::uint8::id:
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

uint32_t ab_packet_count(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
    return 4;
  case vortex::tensor::fp16::id:
    return 8;
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

// Map packet_idx to a math-tile rectangle for the selected native layout.
bool packet_math_region_impl(const TmemMathPacketLayout& layout,
                             uint32_t packet_idx,
                             uint32_t* math_row_base,
                             uint32_t* math_col_base,
                             uint32_t* packet_rows,
                             uint32_t* packet_cols) {
  auto elem_bytes = fmt_bytes(layout.fmt);
  if (0 == elem_bytes || layout.tile_cols == 0 || layout.packets_per_tile == 0) {
    return false;
  }

  auto tile_idx = packet_idx / layout.packets_per_tile;
  auto local_packet_idx = packet_idx % layout.packets_per_tile;
  auto tile_row = tile_idx / layout.tile_cols;
  auto tile_col = tile_idx % layout.tile_cols;

  switch (layout.layout_kind) {
  case TmemMathPacketLayoutKind::ALineNative: {
    auto packets_per_line = ab_packet_count(layout.fmt) / 2;
    if (0 == packets_per_line) {
      return false;
    }

    // For A-line native, consecutive packets in the same line cover disjoint math rows of the same math columns;
    // the next line starts after all math rows of the previous line are covered.
    auto line_id = local_packet_idx / packets_per_line;
    auto packet_in_line = local_packet_idx % packets_per_line;
    if (math_row_base) {
      *math_row_base = tile_row * kTileRows
                     + line_id * 8
                     + packet_in_line * layout.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * kTileCols;
    }
    break;
  }
  case TmemMathPacketLayoutKind::BLineNative: {
    auto packets_per_line = ab_packet_count(layout.fmt) / 2;
    if (0 == packets_per_line) {
      return false;
    }
    auto line_id = local_packet_idx / packets_per_line;
    auto packet_in_line = local_packet_idx % packets_per_line;
    if (math_row_base) {
      *math_row_base = tile_row * kTileRows
                     + packet_in_line * layout.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * kTileCols + line_id * 8;
    }
    break;
  }
  case TmemMathPacketLayoutKind::MathRowMajor: {
    auto packet_cols_per_tile = ceil_div(kTileCols, layout.packet_cols);
    auto packet_row_group = local_packet_idx / packet_cols_per_tile;
    auto packet_col_group = local_packet_idx % packet_cols_per_tile;
    if (math_row_base) {
      *math_row_base = tile_row * kTileRows + packet_row_group * layout.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * kTileCols + packet_col_group * layout.packet_cols;
    }
    break;
  }
  case TmemMathPacketLayoutKind::CSubtileNative: {
    auto packets_per_subtile = cmem_packets_per_subtile(layout.fmt);
    if (0 == packets_per_subtile) {
      return false;
    }
    auto subtile_id = local_packet_idx / packets_per_subtile;
    auto packet_in_subtile = local_packet_idx % packets_per_subtile;
    auto subtile_row = subtile_id / 2;
    auto subtile_col = subtile_id % 2;
    if (math_row_base) {
      *math_row_base = tile_row * kTileRows
                     + subtile_row * 8
                     + packet_in_subtile * layout.packet_rows;
    }
    if (math_col_base) {
      *math_col_base = tile_col * kTileCols + subtile_col * 8;
    }
    break;
  }
  case TmemMathPacketLayoutKind::LinearPacketStream:
  default:
    return false;
  }

  if (packet_rows) {
    *packet_rows = layout.packet_rows;
  }
  if (packet_cols) {
    *packet_cols = layout.packet_cols;
  }
  return true;
}

} // namespace

// Decide whether the caller must use the math-packet adapter for this layout.
bool uses_math_packet_adapter(const TmemMathPacketLayout& layout) {
  return !layout.elem_shape.empty()
      && layout.packets_per_tile != 0
      && layout.tile_cols != 0
      && (layout.layout_kind == TmemMathPacketLayoutKind::MathRowMajor
       || layout.layout_kind == TmemMathPacketLayoutKind::ALineNative
       || layout.layout_kind == TmemMathPacketLayoutKind::BLineNative
       || layout.layout_kind == TmemMathPacketLayoutKind::CSubtileNative);
}

// Public wrapper that exposes the packet's logical math-coordinate span.
bool packet_math_region(const TmemMathPacketLayout& layout,
                        uint32_t packet_idx,
                        TmemMathPacketRegion* out) {
  if (nullptr == out) {
    return false;
  }
  return packet_math_region_impl(layout,
                                 packet_idx,
                                 &out->math_row_base,
                                 &out->math_col_base,
                                 &out->packet_rows,
                                 &out->packet_cols);
}

// Gather bytes from the logical payload into the selected packet shape.
bool pack_math_packet(const TmemMathPacketLayout& layout,
                      const std::vector<uint8_t>& payload,
                      uint32_t packet_idx,
                      std::array<uint8_t, kPacketBytes>* out,
                      bool transpose) {
  if (nullptr == out || !uses_math_packet_adapter(layout)) {
    return false;
  }
  out->fill(0);
  auto elem_bytes = fmt_bytes(layout.fmt);
  auto rows = layout.elem_shape.rows;
  auto cols = layout.elem_shape.cols;
  auto row_bytes = cols * elem_bytes;
  uint32_t math_row_base = 0;
  uint32_t math_col_base = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
  if (!packet_math_region_impl(layout, packet_idx, &math_row_base, &math_col_base, &packet_rows, &packet_cols)) {
    return false;
  }

  auto src_row_bytes = transpose ? (rows * elem_bytes) : row_bytes;
  uint32_t offset = 0;
  for (uint32_t pr = 0; pr < packet_rows; ++pr) {
    auto math_row = math_row_base + pr;
    for (uint32_t pc = 0; pc < packet_cols; ++pc) {
      auto math_col = math_col_base + pc;
      for (uint32_t b = 0; b < elem_bytes; ++b) {
        if (math_row < rows && math_col < cols) {
          uint32_t payload_off = transpose
                               ? math_col * src_row_bytes + math_row * elem_bytes + b
                               : math_row * row_bytes + math_col * elem_bytes + b;
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

// Scatter packet bytes back into their logical math-tile payload offsets.
bool unpack_math_packet(const TmemMathPacketLayout& layout,
                        uint32_t packet_idx,
                        const std::array<uint8_t, kPacketBytes>& packet,
                        std::vector<uint8_t>* payload) {
  if (nullptr == payload || !uses_math_packet_adapter(layout)) {
    return false;
  }
  auto elem_bytes = fmt_bytes(layout.fmt);
  auto rows = layout.elem_shape.rows;
  auto cols = layout.elem_shape.cols;
  auto row_bytes = cols * elem_bytes;
  uint32_t math_row_base = 0;
  uint32_t math_col_base = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
  if (!packet_math_region_impl(layout, packet_idx, &math_row_base, &math_col_base, &packet_rows, &packet_cols)) {
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

} // namespace tmem_math_packet
} // namespace vortex
