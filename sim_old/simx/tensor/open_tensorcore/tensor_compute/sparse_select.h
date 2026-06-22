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

namespace vortex {
namespace sparse {

// Functional helper used by tensor_helper/test/main.cpp.
//
// This helper converts a compact sparse A primitive back into a dense-routed
// A primitive with zeros inserted at pruned positions, while keeping B dense.
// It is mainly used by OTC standalone tests, where the dense TensorCoreTop
// primitive is reused as the functional checker.
//
// sparse_mode:
//   1 = 2:4
//   2 = 1:4
inline bool route_sparse_primitive(uint32_t sparse_mode,
                                   const uint8_t meta_line[16],
                                   const uint16_t a_compact[8][8],
                                   const uint16_t b_dense[8][8],
                                   uint16_t a_routed[8][8],
                                   uint16_t b_routed[8][8]) {
  if (meta_line == nullptr || a_compact == nullptr || b_dense == nullptr
      || a_routed == nullptr || b_routed == nullptr) {
    return false;
  }

  // Clear routed A, because pruned sparse positions should become explicit 0.
  for (uint32_t r = 0; r < 8; ++r) {
    for (uint32_t k = 0; k < 8; ++k) {
      a_routed[r][k] = 0;
    }
  }

  // B stays dense in the current sparse design.
  for (uint32_t k = 0; k < 8; ++k) {
    for (uint32_t c = 0; c < 8; ++c) {
      b_routed[k][c] = b_dense[k][c];
    }
  }

  if (sparse_mode == 1) {  // 2:4
    for (uint32_t row = 0; row < 8; ++row) {
      const uint16_t row_meta = sparse_row_meta(meta_line, row);
      uint32_t cursor = 0;

      for (uint32_t group = 0; group < 2; ++group) {
        const uint32_t idx0 = sparse_2_4_lane0(row_meta, group);
        const uint32_t idx1 = sparse_2_4_lane1(row_meta, group);

        // In legal 2:4 metadata, two selected positions must be different.
        if (idx0 == idx1) {
          return false;
        }

        a_routed[row][group * 4 + idx0] = a_compact[row][cursor++];
        a_routed[row][group * 4 + idx1] = a_compact[row][cursor++];
      }
    }
    return true;
  }

  if (sparse_mode == 2) {  // 1:4
    for (uint32_t row = 0; row < 8; ++row) {
      const uint16_t row_meta = sparse_row_meta(meta_line, row);
      uint32_t cursor = 0;

      for (uint32_t group = 0; group < 2; ++group) {
        const uint32_t idx = sparse_1_4_lane(row_meta, group);
        a_routed[row][group * 4 + idx] = a_compact[row][cursor++];
      }
    }
    return true;
  }

  return false;
}

} // namespace sparse
} // namespace vortex
