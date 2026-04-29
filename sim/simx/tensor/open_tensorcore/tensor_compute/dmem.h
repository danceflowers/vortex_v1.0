#pragma once

// ============================================================================
// DMem -- 输出结果缓冲 (单实例版本)
// ============================================================================
//
// 存储组织:
//   - 单实例，深度 = 4 subtile（与 CMem 对称）
//   - 每个 subtile = 8×8 fp22
//
// 数据流:
//   1. WMMA 计算结果写入:
//      - first/single 模式：A*B + CMem_bypass 结果直接写 DMem[subtile]
//      - middle/tail 模式：A*B + DMem[subtile]_prev 结果原位写回 DMem[subtile]
//   2. MMA_STORE 读取:
//      - 从 DMem 读出 fp22 subtile，精度转换为 fp8/fp16/fp32 后写回 TMEM
// ============================================================================

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tensor_cfg.h"
#include "fp22_to_fp16.h"
#include "fp_types.h"

class DMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kDepth = 4;
  static constexpr uint32_t kPrimitiveDim = 8;
  static constexpr uint32_t kSubtileElems = kPrimitiveDim * kPrimitiveDim;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = kSubtileElems / kBankCount;

  DMem() {
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

  void clear() {
    reset();
  }

  bool valid() const {
    for (uint32_t row = 0; row < kDepth; ++row) {
      if (!row_valid_.at(row)) {
        return false;
      }
    }
    return true;
  }

  bool subtile_valid(uint32_t subtile_id) const {
    if (subtile_id >= kDepth) {
      std::abort();
    }
    return row_valid_.at(subtile_id);
  }

  void write_subtile_fp22(uint32_t subtile_id,
                          const uint32_t in[kPrimitiveDim][kPrimitiveDim]) {
    if (subtile_id >= kDepth) {
      std::abort();
    }
    auto& row = rows_.at(subtile_id);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        store_elem(row, i * kPrimitiveDim + j, in[i][j]);
      }
    }
    row_valid_.at(subtile_id) = true;
  }

  void read_subtile_fp22(uint32_t subtile_id,
                         uint32_t out[kPrimitiveDim][kPrimitiveDim]) const {
    if (subtile_id >= kDepth || !row_valid_.at(subtile_id)) {
      std::abort();
    }
    const auto& row = rows_.at(subtile_id);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        out[i][j] = load_elem(row, i * kPrimitiveDim + j);
      }
    }
  }

  static constexpr uint32_t dump_subtiles(uint32_t) {
    return kDepth;
  }

  static constexpr uint32_t all_bank_mask() {
    return (1u << kBankCount) - 1u;
  }

  static uint32_t subtile_bank_mask(uint32_t subtile_id) {
    if (subtile_id >= kDepth) {
      std::abort();
    }
    uint32_t mask = 0;
    for (uint32_t elem = 0; elem < kSubtileElems; ++elem) {
      mask |= (1u << (elem / kBankElems));
    }
    return mask;
  }

  static uint32_t packets_per_subtile(uint32_t fmt_d) {
    switch (fmt_d) {
    case vortex::tensor::fp8::id:  return 1;
    case vortex::tensor::fp16::id: return 2;
    case vortex::tensor::fp32::id: return 4;
    default:                        return 0;
    }
  }

  static bool build_dump_packet(uint32_t fmt_d,
                                const uint32_t subtile[kPrimitiveDim][kPrimitiveDim],
                                uint32_t segment_idx,
                                packet_t* out) {
    if (nullptr == out) {
      return false;
    }
    packet_t packet{};
    if (fmt_d == vortex::tensor::fp32::id) {
      for (uint32_t row_pair = 0; row_pair < 2; ++row_pair) {
        auto row = segment_idx * 2 + row_pair;
        for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
          auto bits = fp22_to_fp32(subtile[row][col]);
          auto off = (row_pair * kPrimitiveDim + col) * 4;
          packet.at(off + 0) = bits & 0xff;
          packet.at(off + 1) = (bits >> 8) & 0xff;
          packet.at(off + 2) = (bits >> 16) & 0xff;
          packet.at(off + 3) = (bits >> 24) & 0xff;
        }
      }
      *out = packet;
      return true;
    }

    if (fmt_d == vortex::tensor::fp16::id) {
      for (uint32_t row_quad = 0; row_quad < 4; ++row_quad) {
        auto row = segment_idx * 4 + row_quad;
        for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
          auto bits = fp22_to_fp16(subtile[row][col]);
          auto off = (row_quad * kPrimitiveDim + col) * 2;
          packet.at(off + 0) = bits & 0xff;
          packet.at(off + 1) = (bits >> 8) & 0xff;
        }
      }
      *out = packet;
      return true;
    }

    if (fmt_d == vortex::tensor::fp8::id) {
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
          packet.at(row * kPrimitiveDim + col) = fp22_to_fp8_e4m3(subtile[row][col]);
        }
      }
      *out = packet;
      return true;
    }

    return false;
  }

  bool dump_subtile_packets(uint32_t fmt_d,
                            uint32_t subtile_id,
                            std::vector<packet_t>* packets) const {
    if (nullptr == packets || subtile_id >= kDepth || !valid()) {
      return false;
    }

    auto packets_per_group = packets_per_subtile(fmt_d);
    packets->assign(packets_per_group, packet_t{});
    uint32_t subtile[kPrimitiveDim][kPrimitiveDim] = {};
    read_subtile_fp22(subtile_id, subtile);
    for (uint32_t packet_idx = 0; packet_idx < packets_per_group; ++packet_idx) {
      if (!build_dump_packet(fmt_d, subtile, packet_idx, &packets->at(packet_idx))) {
        return false;
      }
    }
    return true;
  }

private:
  using row_t = std::array<std::array<uint32_t, kBankElems>, kBankCount>;

  static void store_elem(row_t& row, uint32_t elem_idx, uint32_t value) {
    row.at(elem_idx / kBankElems).at(elem_idx % kBankElems) = value;
  }

  static uint32_t load_elem(const row_t& row, uint32_t elem_idx) {
    return row.at(elem_idx / kBankElems).at(elem_idx % kBankElems);
  }

  std::array<row_t, kDepth> rows_;
  std::array<bool, kDepth> row_valid_;
};
