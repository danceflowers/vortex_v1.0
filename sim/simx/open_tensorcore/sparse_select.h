#pragma once

#include <cstdint>

inline void sparse_select_2_4(const uint8_t meta_line[16],
                              const uint16_t b_source[16][8],
                              uint16_t b_selected[8][8]) {
  for (uint32_t row = 0; row < 8; ++row) {
    uint16_t row_meta = static_cast<uint16_t>(meta_line[row * 2 + 0])
                      | (static_cast<uint16_t>(meta_line[row * 2 + 1]) << 8);
    for (uint32_t block = 0; block < 4; ++block) {
      uint32_t lane0_sel = (row_meta >> (block * 4 + 0)) & 0x3;
      uint32_t lane1_sel = (row_meta >> (block * 4 + 2)) & 0x3;
      uint32_t k0 = block * 4 + lane0_sel;
      uint32_t k1 = block * 4 + lane1_sel;
      for (uint32_t col = 0; col < 8; ++col) {
        b_selected[block * 2 + 0][col] = b_source[k0][col];
        b_selected[block * 2 + 1][col] = b_source[k1][col];
      }
    }
  }
}

inline void sparse_select_1_4(const uint8_t meta_line[16],
                              const uint16_t b_source[32][8],
                              uint16_t b_selected[8][8]) {
  for (uint32_t row = 0; row < 8; ++row) {
    uint16_t row_meta = static_cast<uint16_t>(meta_line[row * 2 + 0])
                      | (static_cast<uint16_t>(meta_line[row * 2 + 1]) << 8);
    for (uint32_t block = 0; block < 8; ++block) {
      uint32_t lane_sel = (row_meta >> (block * 2)) & 0x3;
      uint32_t k = block * 4 + lane_sel;
      for (uint32_t col = 0; col < 8; ++col) {
        b_selected[block][col] = b_source[k][col];
      }
    }
  }
}
