#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tensor_cfg.h"
#include "config_register.h"
#include "fadd_s1.h"
#include "fadd_s2.h"
#include "fp22_to_fp16.h"
#include "fp_types.h"

// CMem models the TensorCore accumulator SRAM. Each slot owns four 8x8
// subtiles stored in fp22. Load/store paths operate in subtile granularity,
// while packet counts depend on the exposed C/D precision.
class CMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kNumSlots = 2;
  static constexpr uint32_t kRowsPerSlot = 4;
  static constexpr uint32_t kDepth = kNumSlots * kRowsPerSlot;
  static constexpr uint32_t kDim = 16;
  static constexpr uint32_t kPrimitiveDim = 8;
  static constexpr uint32_t kSubtileElems = kPrimitiveDim * kPrimitiveDim;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = kSubtileElems / kBankCount;

  CMem() {
    reset();
  }

  void reset() {
    for (auto& row : rows_) {
      for (auto& bank : row) {
        bank.fill(0);
      }
    }
    row_valid_.fill(false);
  }

  void clear_slot(uint32_t slot_id) {
    auto base = slot_base(slot_id);
    for (uint32_t row = 0; row < kRowsPerSlot; ++row) {
      auto& dst = rows_.at(base + row);
      for (auto& bank : dst) {
        bank.fill(0);
      }
      row_valid_.at(base + row) = false;
    }
  }

  static uint32_t packet_count(uint32_t fmt) {
    switch (fmt) {
    case vortex::tensor::fp8::id:
      return 4;
    case vortex::tensor::fp16::id:
      return 8;
    case vortex::tensor::fp32::id:
      return 16;
    default:
      return 0;
    }
  }

  static constexpr uint32_t fill_subtiles(uint32_t) {
    return kRowsPerSlot;
  }

  static constexpr uint32_t dump_subtiles(uint32_t) {
    return kRowsPerSlot;
  }

  static uint32_t packets_per_subtile(uint32_t fmt_c) {
    switch (fmt_c) {
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

  // One local write action fills one 8x8 C subtile.
  bool write_fill_subtile(uint32_t slot_id,
                          uint32_t fmt_c,
                          uint32_t subtile_idx,
                       const std::vector<packet_t>& packets) {
    auto needed = packets_per_subtile(fmt_c);
    if (subtile_idx >= kRowsPerSlot || packets.size() < needed) {
      return false;
    }

    auto row_idx = slot_base(slot_id) + subtile_idx;
    auto& row = rows_.at(row_idx);
    for (auto& bank : row) {
      bank.fill(0);
    }

    if (fmt_c == vortex::tensor::fp32::id) {
      for (uint32_t packet_idx = 0; packet_idx < packets.size(); ++packet_idx) {
        const auto& p = packets.at(packet_idx);
        for (uint32_t row_pair = 0; row_pair < 2; ++row_pair) {
          auto row_in_subtile = packet_idx * 2 + row_pair;
          for (uint32_t col_in_subtile = 0; col_in_subtile < kPrimitiveDim; ++col_in_subtile) {
            auto off = (row_pair * kPrimitiveDim + col_in_subtile) * 4;
            uint32_t bits = static_cast<uint32_t>(p.at(off + 0))
                          | (static_cast<uint32_t>(p.at(off + 1)) << 8)
                          | (static_cast<uint32_t>(p.at(off + 2)) << 16)
                          | (static_cast<uint32_t>(p.at(off + 3)) << 24);
            store_elem(row, row_in_subtile * kPrimitiveDim + col_in_subtile, convert_c_to_fp22(bits, PREC_FP32));
          }
        }
      }
    } else if (fmt_c == vortex::tensor::fp16::id) {
      for (uint32_t packet_idx = 0; packet_idx < packets.size(); ++packet_idx) {
        const auto& p = packets.at(packet_idx);
        for (uint32_t row_pair = 0; row_pair < 4; ++row_pair) {
          auto row_in_subtile = packet_idx * 4 + row_pair;
          for (uint32_t col_in_subtile = 0; col_in_subtile < kPrimitiveDim; ++col_in_subtile) {
            auto off = (row_pair * kPrimitiveDim + col_in_subtile) * 2;
            uint16_t bits = static_cast<uint16_t>(p.at(off + 0) | (p.at(off + 1) << 8));
            store_elem(row,
                       row_in_subtile * kPrimitiveDim + col_in_subtile,
                       convert_c_to_fp22(bits, PREC_FP16));
          }
        }
      }
    } else if (fmt_c == vortex::tensor::fp8::id) {
      const auto& p = packets.front();
      for (uint32_t row_in_subtile = 0; row_in_subtile < kPrimitiveDim; ++row_in_subtile) {
        for (uint32_t col_in_subtile = 0; col_in_subtile < kPrimitiveDim; ++col_in_subtile) {
          auto off = row_in_subtile * kPrimitiveDim + col_in_subtile;
          store_elem(row,
                     row_in_subtile * kPrimitiveDim + col_in_subtile,
                     convert_c_to_fp22(p.at(off), PREC_FP8_E4M3));
        }
      }
    } else {
      return false;
    }

    row_valid_.at(row_idx) = true;
    return true;
  }

  bool fill_tile(uint32_t slot_id, uint32_t fmt_c, const std::vector<packet_t>& packets) {
    auto expected_packets = packet_count(fmt_c);
    if (packets.size() < expected_packets) {
      return false;
    }

    clear_slot(slot_id);
    auto packets_per_subtile_group = packets_per_subtile(fmt_c);
    for (uint32_t subtile = 0; subtile < kRowsPerSlot; ++subtile) {
      std::vector<packet_t> subtile_packets;
      auto group_base = subtile * packets_per_subtile_group;
      for (uint32_t packet = 0; packet < packets_per_subtile_group; ++packet) {
        subtile_packets.push_back(packets.at(group_base + packet));
      }
      if (!write_fill_subtile(slot_id, fmt_c, subtile, subtile_packets)) {
        return false;
      }
    }
    return true;
  }

  bool fill_tile(uint32_t fmt_c, const std::vector<packet_t>& packets) {
    return fill_tile(0, fmt_c, packets);
  }

  bool dump_subtile_packets(uint32_t slot_id,
                            uint32_t fmt_c,
                            uint32_t subtile_id,
                            std::vector<packet_t>* packets) const {
    if (nullptr == packets || subtile_id >= kRowsPerSlot || !valid(slot_id)) {
      return false;
    }

    auto packets_per_group = packets_per_subtile(fmt_c);
    packets->assign(packets_per_group, packet_t{});
    if (fmt_c == vortex::tensor::fp32::id) {
      for (uint32_t packet_idx = 0; packet_idx < packets_per_group; ++packet_idx) {
        auto& p = packets->at(packet_idx);
        for (uint32_t row_pair = 0; row_pair < 2; ++row_pair) {
          auto row = packet_idx * 2 + row_pair;
          for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
            auto bits = fp22_to_fp32(load_subtile_elem(slot_id, subtile_id, row, col));
            auto off = (row_pair * kPrimitiveDim + col) * 4;
            p.at(off + 0) = bits & 0xff;
            p.at(off + 1) = (bits >> 8) & 0xff;
            p.at(off + 2) = (bits >> 16) & 0xff;
            p.at(off + 3) = (bits >> 24) & 0xff;
          }
        }
      }
      return true;
    }

    if (fmt_c == vortex::tensor::fp16::id) {
      for (uint32_t packet_idx = 0; packet_idx < packets_per_group; ++packet_idx) {
        auto& p = packets->at(packet_idx);
        for (uint32_t row_quad = 0; row_quad < 4; ++row_quad) {
          auto row = packet_idx * 4 + row_quad;
          for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
            auto bits = fp22_to_fp16(load_subtile_elem(slot_id, subtile_id, row, col));
            auto off = (row_quad * kPrimitiveDim + col) * 2;
            p.at(off + 0) = bits & 0xff;
            p.at(off + 1) = (bits >> 8) & 0xff;
          }
        }
      }
      return true;
    }

    if (fmt_c == vortex::tensor::fp8::id) {
      auto& p = packets->front();
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
          p.at(row * kPrimitiveDim + col) = fp22_to_fp8_e4m3(load_subtile_elem(slot_id, subtile_id, row, col));
        }
      }
      return true;
    }

    return false;
  }

  bool dump_tile(uint32_t slot_id, uint32_t fmt_c, std::vector<packet_t>* packets) const {
    if (nullptr == packets || !valid(slot_id)) {
      return false;
    }

    packets->assign(packet_count(fmt_c), packet_t{});
    auto packets_per_group = packets_per_subtile(fmt_c);
    for (uint32_t subtile = 0; subtile < kRowsPerSlot; ++subtile) {
      auto base = subtile * packets_per_group;
      std::vector<packet_t> subtile_packets;
      if (!dump_subtile_packets(slot_id, fmt_c, subtile, &subtile_packets)) {
        return false;
      }
      for (uint32_t packet_idx = 0; packet_idx < packets_per_group; ++packet_idx) {
        packets->at(base + packet_idx) = subtile_packets.at(packet_idx);
      }
    }
    return true;
  }

  bool dump_tile(uint32_t fmt_c, std::vector<packet_t>* packets) const {
    return dump_tile(0, fmt_c, packets);
  }

  bool valid(uint32_t slot_id) const {
    auto base = slot_base(slot_id);
    for (uint32_t row = 0; row < kRowsPerSlot; ++row) {
      if (!row_valid_.at(base + row)) {
        return false;
      }
    }
    return true;
  }

  void read_subtile_fp22(uint32_t slot_id, uint32_t subtile_id, uint32_t out[kPrimitiveDim][kPrimitiveDim]) const {
    auto row_idx = slot_base(slot_id) + subtile_id;
    if (row_idx >= rows_.size() || !row_valid_.at(row_idx)) {
      std::abort();
    }
    const auto& row = rows_.at(row_idx);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        out[i][j] = load_elem(row, i * kPrimitiveDim + j);
      }
    }
  }

  void accumulate_subtile(uint32_t slot_id, uint32_t subtile_id, const uint32_t in[kPrimitiveDim][kPrimitiveDim]) {
    auto row_idx = slot_base(slot_id) + subtile_id;
    if (!row_valid_.at(row_idx)) {
      std::abort();
    }
    auto& row = rows_.at(row_idx);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        auto elem_idx = i * kPrimitiveDim + j;
        auto current = load_elem(row, elem_idx);
        store_elem(row, elem_idx, add_fp22_raw(current, in[i][j]));
      }
    }
  }

  void load_block_fp32(uint32_t storage_m, uint32_t storage_n, float out[kPrimitiveDim][kPrimitiveDim]) const {
    uint32_t raw[kPrimitiveDim][kPrimitiveDim] = {};
    read_subtile_fp22(0, storage_m * 2 + storage_n, raw);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        out[i][j] = vortex::bit_cast<float>(fp22_to_fp32(raw[i][j]));
      }
    }
  }

  void load_block_fp16(uint32_t storage_m, uint32_t storage_n, uint16_t out[kPrimitiveDim][kPrimitiveDim]) const {
    uint32_t raw[kPrimitiveDim][kPrimitiveDim] = {};
    read_subtile_fp22(0, storage_m * 2 + storage_n, raw);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        out[i][j] = fp22_to_fp16(raw[i][j]);
      }
    }
  }

  void store_block_fp32(uint32_t storage_m, uint32_t storage_n, const float in[kPrimitiveDim][kPrimitiveDim]) {
    auto& row = rows_.at(slot_base(0) + storage_m * 2 + storage_n);
    row_valid_.at(slot_base(0) + storage_m * 2 + storage_n) = true;
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        store_elem(row, i * kPrimitiveDim + j,
                   convert_c_to_fp22(vortex::bit_cast<uint32_t>(in[i][j]), PREC_FP32));
      }
    }
  }

  void store_block_fp16(uint32_t storage_m, uint32_t storage_n, const uint16_t in[kPrimitiveDim][kPrimitiveDim]) {
    auto& row = rows_.at(slot_base(0) + storage_m * 2 + storage_n);
    row_valid_.at(slot_base(0) + storage_m * 2 + storage_n) = true;
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        store_elem(row, i * kPrimitiveDim + j, convert_c_to_fp22(in[i][j], PREC_FP16));
      }
    }
  }

private:
  using row_t = std::array<std::array<uint32_t, kBankElems>, kBankCount>;

  static uint32_t slot_base(uint32_t slot_id) {
    if (slot_id >= kNumSlots) {
      std::abort();
    }
    return slot_id * kRowsPerSlot;
  }

  static void store_elem(row_t& row, uint32_t elem_idx, uint32_t value) {
    row.at(elem_idx / kBankElems).at(elem_idx % kBankElems) = value;
  }

  static uint32_t load_elem(const row_t& row, uint32_t elem_idx) {
    return row.at(elem_idx / kBankElems).at(elem_idx % kBankElems);
  }

  static uint32_t add_fp22_raw(uint32_t a, uint32_t b) {
    auto s1 = fadd_s1(a, b, 8, 14, 14, g_cfg.rm);
    return fadd_s2(s1, 8, 14);
  }

  void store_matrix_elem(uint32_t slot_id, uint32_t row, uint32_t col, uint32_t value) {
    auto subtile = (row / kPrimitiveDim) * 2 + (col / kPrimitiveDim);
    auto elem_idx = (row % kPrimitiveDim) * kPrimitiveDim + (col % kPrimitiveDim);
    store_elem(rows_.at(slot_base(slot_id) + subtile), elem_idx, value);
  }

  uint32_t load_matrix_elem(uint32_t slot_id, uint32_t row, uint32_t col) const {
    auto subtile = (row / kPrimitiveDim) * 2 + (col / kPrimitiveDim);
    auto elem_idx = (row % kPrimitiveDim) * kPrimitiveDim + (col % kPrimitiveDim);
    return load_elem(rows_.at(slot_base(slot_id) + subtile), elem_idx);
  }

  uint32_t load_subtile_elem(uint32_t slot_id, uint32_t subtile, uint32_t row, uint32_t col) const {
    return load_elem(rows_.at(slot_base(slot_id) + subtile), row * kPrimitiveDim + col);
  }

  std::array<row_t, kDepth> rows_;
  std::array<bool, kDepth> row_valid_;
};
