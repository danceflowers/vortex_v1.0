#pragma once

// ============================================================================
// AMem -- single-instance A operand SRAM.
// ============================================================================
//
// Storage layout:
//   - depth = 4 lines, one 8x8 primitive block per line.
//   - m16n16k16 mapping: lines 0/1 are K phase 0 M blocks, and lines 2/3
//     are K phase 1 M blocks.
//
// Fill path:
//   convert_fill_packets() and write_fill_line() unpack TMEM packets and
//   convert external fp8/fp16 payloads into the internal fp9 operand format.
//
// Compute path:
//   read_primitive() returns one 8x8 block selected by
//   k_phase * 2 + storage_m for TensorCore issue.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tensor_cfg.h"
#include "fp_types.h"

class AMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kDepth = 4;
  static constexpr uint32_t kLineElems = 64;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = kLineElems / kBankCount;

  /// Number of AMem lines needed to fill one complete tile.
  static constexpr uint32_t fill_lines() {
    return kDepth;
  }

  AMem() {
    reset();
  }

  void reset() {
    for (auto& line : lines_) {
      for (auto& bank : line) {
        bank.fill(0);
      }
    }
    row_valid_.fill(false);
  }

  void clear() {
    for (auto& line : lines_) {
      for (auto& bank : line) {
        bank.fill(0);
      }
    }
    row_valid_.fill(false);
  }

  /// Number of TMEM packets required for one m16n16k16 A tile.
  static uint32_t packet_count(uint32_t fmt_a) {
    return (fmt_a == vortex::tensor::fp16::id) ? 8 : 4;
  }

  static constexpr uint32_t all_bank_mask() {
    return (1u << kBankCount) - 1u;
  }

  static uint32_t line_bank_mask(uint32_t line_idx) {
    if (line_idx >= kDepth) {
      std::abort();
    }
    uint32_t mask = 0;
    for (uint32_t elem = 0; elem < kLineElems; ++elem) {
      mask |= (1u << (elem / kBankElems));
    }
    return mask;
  }

  /// Bank mask touched when TensorCore reads one primitive block.
  static uint32_t primitive_bank_mask(uint32_t line_idx,
                                      bool transpose = false) {
    (void)transpose;
    return line_bank_mask(line_idx);
  }

  /// Number of TMEM packets required to fill one AMem line.
  static uint32_t packets_per_fill_line(uint32_t fmt_a) {
    return (fmt_a == vortex::tensor::fp16::id) ? 2 : 1;
  }

  /// Convert external fp8/fp16 fill packets into internal fp9 elements.
  static bool convert_fill_packets(uint32_t fmt_a,
                                   const std::vector<packet_t>& packets,
                                   uint16_t out[8][8]) {
    auto needed = packets_per_fill_line(fmt_a);
    if (packets.size() < needed) {
      return false;
    }

    if (fmt_a == vortex::tensor::fp16::id) {
      const auto& p0 = packets.at(0);
      const auto& p1 = packets.at(1);
      for (uint32_t elem = 0; elem < kLineElems; ++elem) {
        auto byte_idx = elem * 2;
        uint16_t fp16 = 0;
        if (byte_idx < 64) {
          fp16 = static_cast<uint16_t>(p0.at(byte_idx + 0) | (p0.at(byte_idx + 1) << 8));
        } else {
          auto local = byte_idx - 64;
          fp16 = static_cast<uint16_t>(p1.at(local + 0) | (p1.at(local + 1) << 8));
        }
        out[elem / 8][elem % 8] = convert_to_fp9(fp16, PREC_FP16);
      }
      return true;
    }

    if (fmt_a == vortex::tensor::fp8::id) {
      const auto& p = packets.at(0);
      for (uint32_t elem = 0; elem < kLineElems; ++elem) {
        out[elem / 8][elem % 8] = convert_to_fp9(p.at(elem), PREC_FP8_E4M3);
      }
      return true;
    }

    return false;
  }

  /// Convert the incoming packet group and write one AMem line.
  bool write_fill_line(uint32_t fmt_a,
                       uint32_t line_idx,
                       const std::vector<packet_t>& packets) {
    auto needed = packets_per_fill_line(fmt_a);
    if (line_idx >= kDepth || packets.size() < needed) {
      return false;
    }

    auto& dst = lines_.at(line_idx);
    for (auto& bank : dst) {
      bank.fill(0);
    }
    uint16_t converted[8][8] = {};
    if (!convert_fill_packets(fmt_a, packets, converted)) {
      return false;
    }
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        store_elem(dst, i * 8 + j, converted[i][j]);
      }
    }
    row_valid_.at(line_idx) = true;
    return true;
  }

  /// Timing path helper: write already-converted fp9 data directly.
  bool write_converted_line(uint32_t line_idx,
                            const uint16_t in[8][8]) {
    if (line_idx >= kDepth) {
      return false;
    }
    auto& dst = lines_.at(line_idx);
    for (auto& bank : dst) {
      bank.fill(0);
    }
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        store_elem(dst, i * 8 + j, in[i][j]);
      }
    }
    row_valid_.at(line_idx) = true;
    return true;
  }

  /// True when all four AMem lines contain a valid tile.
  bool valid() const {
    for (uint32_t line = 0; line < kDepth; ++line) {
      if (!row_valid_.at(line)) {
        return false;
      }
    }
    return true;
  }

  /// Read the selected 8x8 primitive block for TensorCore issue.
  void read_primitive(uint32_t line_idx,
                      uint16_t out[8][8],
                      bool transpose = false) const {
    if (line_idx >= kDepth || !row_valid_.at(line_idx)) {
      std::abort();
    }
    const auto& line = lines_.at(line_idx);
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        auto elem_idx = transpose ? (j * 8 + i) : (i * 8 + j);
        out[i][j] = load_elem(line, elem_idx);
      }
    }
  }

private:
  using row_t = std::array<std::array<uint16_t, kBankElems>, kBankCount>;

  static void store_elem(row_t& row, uint32_t elem_idx, uint16_t value) {
    row.at(elem_idx / kBankElems).at(elem_idx % kBankElems) = value;
  }

  static uint16_t load_elem(const row_t& row, uint32_t elem_idx) {
    return row.at(elem_idx / kBankElems).at(elem_idx % kBankElems);
  }

  std::array<row_t, kDepth> lines_;
  std::array<bool, kDepth> row_valid_;
};
