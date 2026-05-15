#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tensor_cfg.h"
#include "fp_types.h"

class BMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kNumSlots = 2;
  static constexpr uint32_t kRowsPerSlot = 4;
  static constexpr uint32_t kDepth = kNumSlots * kRowsPerSlot;
  static constexpr uint32_t kTileLines = kRowsPerSlot;
  static constexpr uint32_t kDenseLines = kRowsPerSlot;
  static constexpr uint32_t kLineElems = 64;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = kLineElems / kBankCount;

  BMem() {
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

  void clear_slot(uint32_t slot_id) {
    auto base = slot_base(slot_id);
    for (uint32_t line = 0; line < kRowsPerSlot; ++line) {
      auto& row = lines_.at(base + line);
      for (auto& bank : row) {
        bank.fill(0);
      }
      row_valid_.at(base + line) = false;
    }
  }

  static uint32_t packet_count(uint32_t fmt_ab, uint32_t sparse_mode = vortex::tensor::sparse_none) {
    if (sparse_mode != vortex::tensor::sparse_none) {
      return (fmt_ab == vortex::tensor::fp16::id) ? 8 : 4;
    }
    return (fmt_ab == vortex::tensor::fp16::id) ? 8 : 4;
  }

  static uint32_t fill_beats(uint32_t sparse_mode = vortex::tensor::sparse_none) {
    if (sparse_mode != vortex::tensor::sparse_none) {
      return kRowsPerSlot;
    }
    return kRowsPerSlot;
  }

  static uint32_t packets_per_fill_beat(uint32_t fmt_ab, uint32_t sparse_mode = vortex::tensor::sparse_none) {
    if (sparse_mode != vortex::tensor::sparse_none) {
      return (fmt_ab == vortex::tensor::fp16::id) ? 2 : 1;
    }
    return (fmt_ab == vortex::tensor::fp16::id) ? 2 : 1;
  }

  bool write_fill_beat(uint32_t slot_id,
                       uint32_t fmt_ab,
                       uint32_t beat_idx,
                       const std::vector<packet_t>& packets,
                       uint32_t sparse_mode = vortex::tensor::sparse_none) {
    auto needed = packets_per_fill_beat(fmt_ab, sparse_mode);
    if (beat_idx >= kRowsPerSlot || packets.size() < needed) {
      return false;
    }

    auto& dst = lines_.at(slot_base(slot_id) + beat_idx);
    for (auto& bank : dst) {
      bank.fill(0);
    }
    if (fmt_ab == vortex::tensor::fp16::id) {
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
        store_elem(dst, elem, convert_to_fp9(fp16, PREC_FP16));
      }
    } else {
      const auto& p = packets.at(0);
      for (uint32_t elem = 0; elem < kLineElems; ++elem) {
        store_elem(dst, elem, convert_to_fp9(p.at(elem), PREC_FP8_E4M3));
      }
    }
    row_valid_.at(slot_base(slot_id) + beat_idx) = true;
    return true;
  }

  bool fill_tile(uint32_t slot_id,
                 uint32_t fmt_ab,
                 const std::vector<packet_t>& packets,
                 uint32_t sparse_mode = vortex::tensor::sparse_none) {
    auto expected_packets = packet_count(fmt_ab, sparse_mode);
    if (packets.size() < expected_packets) {
      return false;
    }

    clear_slot(slot_id);
    auto packets_per_beat = packets_per_fill_beat(fmt_ab, sparse_mode);
    for (uint32_t beat = 0; beat < kRowsPerSlot; ++beat) {
      std::vector<packet_t> beat_packets;
      for (uint32_t packet = 0; packet < packets_per_beat; ++packet) {
        beat_packets.push_back(packets.at(beat * packets_per_beat + packet));
      }
      if (!write_fill_beat(slot_id, fmt_ab, beat, beat_packets, sparse_mode)) {
        return false;
      }
    }
    return true;
  }

  bool fill_tile(uint32_t fmt_ab,
                 const std::vector<packet_t>& packets,
                 uint32_t sparse_mode = vortex::tensor::sparse_none) {
    return fill_tile(0, fmt_ab, packets, sparse_mode);
  }

  bool valid(uint32_t slot_id) const {
    auto base = slot_base(slot_id);
    for (uint32_t line = 0; line < kRowsPerSlot; ++line) {
      if (!row_valid_.at(base + line)) {
        return false;
      }
    }
    return true;
  }

  void read_primitive(uint32_t slot_id,
                      uint32_t step_k,
                      uint32_t step_n,
                      uint16_t out[8][8],
                      bool transpose = false) const {
    auto line_idx = slot_base(slot_id) + ((transpose ? step_k : step_n) * 2) + (transpose ? step_n : step_k);
    if (!row_valid_.at(line_idx)) {
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

  void read_primitive(uint32_t step_k, uint32_t step_n, uint16_t out[8][8], bool transpose = false) const {
    read_primitive(0, step_k, step_n, out, transpose);
  }

  // Read dense B source rows for 2:4 sparse A.
  // One sparse 8x8 A payload represents 16 logical K positions.
  // For each output N half, the source is two consecutive dense B lines.
  void read_sparse_2_4_source(uint32_t slot_id, uint32_t storage_n, uint16_t out[16][8]) const {
    if (storage_n >= 2) {
      std::abort();
    }
    const uint32_t base = slot_base(slot_id) + storage_n * 2;
    for (uint32_t k_block = 0; k_block < 2; ++k_block) {
      const uint32_t line_idx = base + k_block;
      if (line_idx >= lines_.size() || !row_valid_.at(line_idx)) {
        std::abort();
      }
      const auto& line = lines_.at(line_idx);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          out[k_block * 8 + i][j] = load_elem(line, i * 8 + j);
        }
      }
    }
  }

  // Read dense B source rows for 1:4 sparse A.
  // One sparse 8x8 A payload represents 32 logical K positions.
  // Current Cmodel storage can hold 32x8 B source rows per slot. For N=16,
  // software should load the second N half into the next B slot and issue/use
  // that slot accordingly. If slot_id + storage_n is available and valid, use
  // it; otherwise fall back to slot_id for backward-compatible N=8 tests.
  void read_sparse_1_4_source(uint32_t slot_id, uint32_t storage_n, uint16_t out[32][8]) const {
    uint32_t source_slot = slot_id;
    if (storage_n != 0 && (slot_id + storage_n) < kNumSlots) {
      source_slot = slot_id + storage_n;
    }
    const uint32_t base = slot_base(source_slot);
    for (uint32_t k_block = 0; k_block < 4; ++k_block) {
      const uint32_t line_idx = base + k_block;
      if (line_idx >= lines_.size() || !row_valid_.at(line_idx)) {
        std::abort();
      }
      const auto& line = lines_.at(line_idx);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          out[k_block * 8 + i][j] = load_elem(line, i * 8 + j);
        }
      }
    }
  }

private:
  using row_t = std::array<std::array<uint16_t, kBankElems>, kBankCount>;

  static uint32_t slot_base(uint32_t slot_id) {
    if (slot_id >= kNumSlots) {
      std::abort();
    }
    return slot_id * kRowsPerSlot;
  }

  static void store_elem(row_t& row, uint32_t elem_idx, uint16_t value) {
    row.at(elem_idx / kBankElems).at(elem_idx % kBankElems) = value;
  }

  static uint16_t load_elem(const row_t& row, uint32_t elem_idx) {
    return row.at(elem_idx / kBankElems).at(elem_idx % kBankElems);
  }

  std::array<row_t, kDepth> lines_;
  std::array<bool, kDepth> row_valid_;
};
