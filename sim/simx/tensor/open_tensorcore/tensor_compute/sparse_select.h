#pragma once

#include <cstdlib>
#include <cstdint>

namespace vortex::sparse {

// Clear an 8x8 primitive tile before sparse routing fills the active lanes.
inline void zero_primitive_u16(uint16_t out[8][8]) {
  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      out[i][j] = 0;
    }
  }
}

// Each sparse metadata row is stored as two little-endian bytes.
inline uint16_t row_meta_bits(const uint8_t meta_line[16], uint32_t row) {
  return static_cast<uint16_t>(meta_line[row * 2 + 0])
       | (static_cast<uint16_t>(meta_line[row * 2 + 1]) << 8);
}

// Select the four lane-local products used by the native 2:4 sparse datapath.
inline void select_sparse_2_4_lane(uint16_t row_meta,
                                   const uint32_t a_compact[8],
                                   const uint32_t b_dense[8],
                                   uint32_t a4[4],
                                   uint32_t b4[4]) {
  for (uint32_t group = 0; group < 2; ++group) {
    uint32_t base = group * 4;
    uint32_t idx0 = (row_meta >> (group * 4 + 0)) & 0x3;
    uint32_t idx1 = (row_meta >> (group * 4 + 2)) & 0x3;
    if (idx0 == idx1) {
      std::abort();
    }
    uint32_t out = group * 2;
    a4[out + 0] = a_compact[out + 0];
    b4[out + 0] = b_dense[base + idx0];
    a4[out + 1] = a_compact[out + 1];
    b4[out + 1] = b_dense[base + idx1];
  }
}

// Select the two lane-local products used by 1:4 sparse mode and zero-fill the
// inactive multiplier inputs.
inline void select_sparse_1_4_lane(uint16_t row_meta,
                                   const uint32_t a_compact[8],
                                   const uint32_t b_dense[8],
                                   uint32_t a4[4],
                                   uint32_t b4[4]) {
  for (uint32_t lane = 0; lane < 4; ++lane) {
    a4[lane] = 0;
    b4[lane] = 0;
  }
  for (uint32_t group = 0; group < 2; ++group) {
    uint32_t base = group * 4;
    uint32_t idx = (row_meta >> (group * 2)) & 0x3;
    a4[group] = a_compact[group];
    b4[group] = b_dense[base + idx];
  }
}

// Expand a 2:4-compressed A primitive and mask B to the selected K lanes.
inline void route_sparse_2_4_primitive(const uint8_t meta_line[16],
                                       const uint16_t a_compact[8][8],
                                       const uint16_t b_dense[8][8],
                                       uint16_t a_routed[8][8],
                                       uint16_t b_routed[8][8]) {
  zero_primitive_u16(a_routed);
  zero_primitive_u16(b_routed);

  for (uint32_t row = 0; row < 8; ++row) {
    auto row_meta = row_meta_bits(meta_line, row);
    uint32_t payload_cursor = 0;
    for (uint32_t group = 0; group < 2; ++group) {
      uint32_t base = group * 4;
      uint32_t idx0 = (row_meta >> (group * 4 + 0)) & 0x3;
      uint32_t idx1 = (row_meta >> (group * 4 + 2)) & 0x3;
      uint32_t k0 = base + idx0;
      uint32_t k1 = base + idx1;
      a_routed[row][k0] = a_compact[row][payload_cursor++];
      a_routed[row][k1] = a_compact[row][payload_cursor++];
      for (uint32_t col = 0; col < 8; ++col) {
        b_routed[k0][col] = b_dense[k0][col];
        b_routed[k1][col] = b_dense[k1][col];
      }
    }
  }
}

// Expand a 1:4-compressed A primitive and mask B to the selected K lane.
inline void route_sparse_1_4_primitive(const uint8_t meta_line[16],
                                       const uint16_t a_compact[8][8],
                                       const uint16_t b_dense[8][8],
                                       uint16_t a_routed[8][8],
                                       uint16_t b_routed[8][8]) {
  zero_primitive_u16(a_routed);
  zero_primitive_u16(b_routed);

  for (uint32_t row = 0; row < 8; ++row) {
    auto row_meta = row_meta_bits(meta_line, row);
    uint32_t payload_cursor = 0;
    for (uint32_t group = 0; group < 2; ++group) {
      uint32_t base = group * 4;
      uint32_t idx = (row_meta >> (group * 2)) & 0x3;
      uint32_t k = base + idx;
      a_routed[row][k] = a_compact[row][payload_cursor++];
      for (uint32_t col = 0; col < 8; ++col) {
        b_routed[k][col] = b_dense[k][col];
      }
    }
  }
}

// Dispatch sparse primitive routing by decoded tcgen05 sparse mode.
inline bool route_sparse_primitive(uint32_t sparse_mode,
                                   const uint8_t meta_line[16],
                                   const uint16_t a_compact[8][8],
                                   const uint16_t b_dense[8][8],
                                   uint16_t a_routed[8][8],
                                   uint16_t b_routed[8][8]) {
  switch (sparse_mode) {
  case 1:
    route_sparse_2_4_primitive(meta_line, a_compact, b_dense, a_routed, b_routed);
    return true;
  case 2:
    route_sparse_1_4_primitive(meta_line, a_compact, b_dense, a_routed, b_routed);
    return true;
  default:
    return false;
  }
}

} // namespace vortex::sparse

// Legacy helper: select B rows for 2:4 sparse metadata without routing A.
inline void sparse_select_2_4(const uint8_t meta_line[16],
                              const uint16_t b_source[16][8],
                              uint16_t b_selected[8][8]) {
  vortex::sparse::zero_primitive_u16(b_selected);
  for (uint32_t row = 0; row < 8; ++row) {
    auto row_meta = vortex::sparse::row_meta_bits(meta_line, row);
    for (uint32_t group = 0; group < 2; ++group) {
      uint32_t base = group * 4;
      uint32_t idx0 = (row_meta >> (group * 4 + 0)) & 0x3;
      uint32_t idx1 = (row_meta >> (group * 4 + 2)) & 0x3;
      uint32_t k0 = base + idx0;
      uint32_t k1 = base + idx1;
      for (uint32_t col = 0; col < 8; ++col) {
        b_selected[base + idx0][col] = b_source[k0][col];
        b_selected[base + idx1][col] = b_source[k1][col];
      }
    }
  }
}

// Legacy helper: select B rows for 1:4 sparse metadata without routing A.
inline void sparse_select_1_4(const uint8_t meta_line[16],
                              const uint16_t b_source[32][8],
                              uint16_t b_selected[8][8]) {
  vortex::sparse::zero_primitive_u16(b_selected);
  for (uint32_t row = 0; row < 8; ++row) {
    auto row_meta = vortex::sparse::row_meta_bits(meta_line, row);
    for (uint32_t group = 0; group < 2; ++group) {
      uint32_t base = group * 4;
      uint32_t idx = (row_meta >> (group * 2)) & 0x3;
      uint32_t k = base + idx;
      for (uint32_t col = 0; col < 8; ++col) {
        b_selected[k][col] = b_source[k][col];
      }
    }
  }
}
