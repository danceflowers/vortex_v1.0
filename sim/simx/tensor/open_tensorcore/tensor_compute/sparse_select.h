#pragma once

#include <cstdint>
#include <cstdlib>

// Sparse metadata helpers for OpenRVGPU structured sparse MMA.
// The metadata line is 16 B. Each row owns 16 bits:
//   2:4: four 4-wide groups, each group has two 2-bit indices.
//   1:4: eight 4-wide groups, each group has one 2-bit index.
// These helpers deliberately decode metadata per row. This is important because
// different A rows may use different sparse patterns.

inline uint16_t sparse_row_meta(const uint8_t meta_line[16], uint32_t row) {
  if (row >= 8) {
    std::abort();
  }
  return static_cast<uint16_t>(meta_line[row * 2 + 0])
       | static_cast<uint16_t>(meta_line[row * 2 + 1] << 8);
}

inline uint32_t sparse_2_4_lane0(uint16_t row_meta, uint32_t group) {
  if (group >= 4) {
    std::abort();
  }
  return (row_meta >> (group * 4 + 0)) & 0x3;
}

inline uint32_t sparse_2_4_lane1(uint16_t row_meta, uint32_t group) {
  if (group >= 4) {
    std::abort();
  }
  return (row_meta >> (group * 4 + 2)) & 0x3;
}

inline uint32_t sparse_1_4_lane(uint16_t row_meta, uint32_t group) {
  if (group >= 8) {
    std::abort();
  }
  return (row_meta >> (group * 2)) & 0x3;
}

// Legacy compatibility helpers. They are kept for callers that only need a
// selected B matrix. Note that these helpers are only correct when all rows in
// the primitive share the same metadata pattern. The row-dependent sparse path
// in tensor_unit.cpp does not use these functions for final compute.
inline void sparse_select_2_4(const uint8_t meta_line[16],
                              const uint16_t b_source[16][8],
                              uint16_t b_selected[8][8]) {
  uint16_t row_meta = sparse_row_meta(meta_line, 0);
  for (uint32_t block = 0; block < 4; ++block) {
    uint32_t lane0_sel = sparse_2_4_lane0(row_meta, block);
    uint32_t lane1_sel = sparse_2_4_lane1(row_meta, block);
    uint32_t k0 = block * 4 + lane0_sel;
    uint32_t k1 = block * 4 + lane1_sel;
    for (uint32_t col = 0; col < 8; ++col) {
      b_selected[block * 2 + 0][col] = b_source[k0][col];
      b_selected[block * 2 + 1][col] = b_source[k1][col];
    }
  }
}

inline void sparse_select_1_4(const uint8_t meta_line[16],
                              const uint16_t b_source[32][8],
                              uint16_t b_selected[8][8]) {
  uint16_t row_meta = sparse_row_meta(meta_line, 0);
  for (uint32_t block = 0; block < 8; ++block) {
    uint32_t lane_sel = sparse_1_4_lane(row_meta, block);
    uint32_t k = block * 4 + lane_sel;
    for (uint32_t col = 0; col < 8; ++col) {
      b_selected[block][col] = b_source[k][col];
    }
  }
}
