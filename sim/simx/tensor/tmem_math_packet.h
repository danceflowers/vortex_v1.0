// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace vortex {
namespace tmem_math_packet {

static constexpr uint32_t kPacketBytes = 64;

enum class TmemMathPacketLayoutKind : uint8_t {
  LinearPacketStream = 0,
  MathRowMajor,
  ALineNative,
  BLineNative,
  CSubtileNative,
};

struct TmemMathPacketShape {
  uint16_t rows = 0;
  uint16_t cols = 0;

  bool empty() const {
    return rows == 0 || cols == 0;
  }
};

struct TmemMathPacketLayout {
  TmemMathPacketLayoutKind layout_kind = TmemMathPacketLayoutKind::LinearPacketStream;
  TmemMathPacketShape elem_shape = {};
  uint32_t fmt = 0;
  uint16_t tile_cols = 0;
  uint16_t packets_per_tile = 0;
  uint16_t packet_rows = 0;
  uint16_t packet_cols = 0;
  uint16_t tile_count = 0;
};

struct TmemMathPacketRegion {
  uint32_t math_row_base = 0;
  uint32_t math_col_base = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
};

bool uses_math_packet_adapter(const TmemMathPacketLayout& layout);

bool packet_math_region(const TmemMathPacketLayout& layout,
                        uint32_t packet_idx,
                        TmemMathPacketRegion* out);

bool pack_math_packet(const TmemMathPacketLayout& layout,
                      const std::vector<uint8_t>& payload,
                      uint32_t packet_idx,
                      std::array<uint8_t, kPacketBytes>* out,
                      bool transpose = false);

bool unpack_math_packet(const TmemMathPacketLayout& layout,
                        uint32_t packet_idx,
                        const std::array<uint8_t, kPacketBytes>& packet,
                        std::vector<uint8_t>* payload);

} // namespace tmem_math_packet
} // namespace vortex
