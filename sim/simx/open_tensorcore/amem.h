#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tensor_cfg.h"
#include "fp_types.h"

class AMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kNumSlots = 2;
  static constexpr uint32_t kRowsPerSlot = 4;
  static constexpr uint32_t kDepth = kNumSlots * kRowsPerSlot;
  static constexpr uint32_t kTileLines = kRowsPerSlot;
  static constexpr uint32_t kLineElems = 64;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = kLineElems / kBankCount;

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

  static uint32_t packet_count(uint32_t fmt_ab) {
    return (fmt_ab == vortex::tensor::fp16::id) ? 8 : 4;
  }

  static constexpr uint32_t fill_beats() {
    return kRowsPerSlot;
  }

  static uint32_t packets_per_fill_beat(uint32_t fmt_ab) {
    return (fmt_ab == vortex::tensor::fp16::id) ? 2 : 1;
  }

  bool write_fill_beat(uint32_t slot_id,
                       uint32_t fmt_ab,
                       uint32_t beat_idx,
                       const std::vector<packet_t>& packets) {
    auto needed = packets_per_fill_beat(fmt_ab);
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

  bool fill_tile(uint32_t slot_id, uint32_t fmt_ab, const std::vector<packet_t>& packets) {
    auto expected_packets = packet_count(fmt_ab);
    if (packets.size() < expected_packets) {
      return false;
    }

    clear_slot(slot_id);
    auto packets_per_beat = packets_per_fill_beat(fmt_ab);
    for (uint32_t beat = 0; beat < kRowsPerSlot; ++beat) {
      std::vector<packet_t> beat_packets;
      for (uint32_t packet = 0; packet < packets_per_beat; ++packet) {
        beat_packets.push_back(packets.at(beat * packets_per_beat + packet));
      }
      if (!write_fill_beat(slot_id, fmt_ab, beat, beat_packets)) {
        return false;
      }
    }
    return true;
  }

  bool fill_tile(uint32_t fmt_ab, const std::vector<packet_t>& packets) {
    return fill_tile(0, fmt_ab, packets);
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

  void read_primitive(uint32_t slot_id, uint32_t step_m, uint32_t step_k, uint16_t out[8][8]) const {
    auto line_idx = slot_base(slot_id) + step_m * 2 + step_k;
    if (!row_valid_.at(line_idx)) {
      std::abort();
    }
    const auto& line = lines_.at(line_idx);
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        out[i][j] = load_elem(line, i * 8 + j);
      }
    }
  }

  void read_primitive(uint32_t step_m, uint32_t step_k, uint16_t out[8][8]) const {
    read_primitive(0, step_m, step_k, out);
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
