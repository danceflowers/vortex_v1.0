#pragma once

// ============================================================================
// CMem -- C 累加器输入 SRAM (单实例版本)
// ============================================================================
//
// 存储组织:
//   - 单实例, 深度 = 4 subtile (kDepth = 4)
//   - 一个 16×16 tile = 2×2 的 4 个 8×8 subtile:
//       subtile 0 = 左上, subtile 1 = 右上,
//       subtile 2 = 左下, subtile 3 = 右下
//   - 8 个 bank，每个 bank 持有 kBankElems = 8 个元素
//
// 内部精度: fp22 (8-bit 指数 + 14-bit 尾数)
//
// 支持的操作:
//   1. Fill (TMEM → CMem): write_fill_subtile / convert_fill_packets
//   2. WMMA 读取: read_subtile_fp22(subtile_id, out) 作为累加序列 first 模式的
//      c_bypass 输入
//
// 累加序列简化版:
//   - first（OR=0 或 OR=1 且 prev_or=0）: c_bypass 从 CMem 读出，透传进阵列
//   - middle/tail（OR=1 && prev_or=1）: c_bypass 从 DMem 读出，实现原位累加
// ============================================================================

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

class CMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kDepth = 4;
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

  void clear() {
    reset();
  }

  /// fill 一个完整 tile 所需 TMEM 包数: fp8=4, fp16=8, fp32=16
  static uint32_t packet_count(uint32_t fmt) {
    switch (fmt) {
    case vortex::tensor::fp8::id:  return 4;
    case vortex::tensor::fp16::id: return 8;
    case vortex::tensor::fp32::id: return 16;
    default:                        return 0;
    }
  }

  static constexpr uint32_t fill_subtiles(uint32_t) {
    return kDepth;
  }

  static constexpr uint32_t dump_subtiles(uint32_t) {
    return kDepth;
  }

  static constexpr uint32_t all_bank_mask() {
    return (1u << kBankCount) - 1u;
  }

  static uint32_t subtile_bank_mask(uint32_t subtile_idx) {
    if (subtile_idx >= kDepth) {
      std::abort();
    }
    uint32_t mask = 0;
    for (uint32_t elem = 0; elem < kSubtileElems; ++elem) {
      mask |= (1u << (elem / kBankElems));
    }
    return mask;
  }

  static uint32_t packets_per_subtile(uint32_t fmt_c) {
    switch (fmt_c) {
    case vortex::tensor::fp8::id:  return 1;
    case vortex::tensor::fp16::id: return 2;
    case vortex::tensor::fp32::id: return 4;
    default:                        return 0;
    }
  }

  /// Fill 精度转换: 外部格式 → fp22
  static bool convert_fill_packets(uint32_t fmt_c,
                                   const std::vector<packet_t>& packets,
                                   uint32_t out[kPrimitiveDim][kPrimitiveDim]) {
    auto needed = packets_per_subtile(fmt_c);
    if (packets.size() < needed) {
      return false;
    }

    if (fmt_c == vortex::tensor::fp32::id) {
      for (uint32_t packet_idx = 0; packet_idx < packets.size(); ++packet_idx) {
        const auto& p = packets.at(packet_idx);
        for (uint32_t row_pair = 0; row_pair < 2; ++row_pair) {
          auto row = packet_idx * 2 + row_pair;
          for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
            auto off = (row_pair * kPrimitiveDim + col) * 4;
            uint32_t bits = static_cast<uint32_t>(p.at(off + 0))
                          | (static_cast<uint32_t>(p.at(off + 1)) << 8)
                          | (static_cast<uint32_t>(p.at(off + 2)) << 16)
                          | (static_cast<uint32_t>(p.at(off + 3)) << 24);
            out[row][col] = convert_c_to_fp22(bits, PREC_FP32);
          }
        }
      }
      return true;
    }

    if (fmt_c == vortex::tensor::fp16::id) {
      for (uint32_t packet_idx = 0; packet_idx < packets.size(); ++packet_idx) {
        const auto& p = packets.at(packet_idx);
        for (uint32_t row_quad = 0; row_quad < 4; ++row_quad) {
          auto row = packet_idx * 4 + row_quad;
          for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
            auto off = (row_quad * kPrimitiveDim + col) * 2;
            uint16_t bits = static_cast<uint16_t>(p.at(off + 0) | (p.at(off + 1) << 8));
            out[row][col] = convert_c_to_fp22(bits, PREC_FP16);
          }
        }
      }
      return true;
    }

    if (fmt_c == vortex::tensor::fp8::id) {
      const auto& p = packets.front();
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
          out[row][col] = convert_c_to_fp22(p.at(row * kPrimitiveDim + col), PREC_FP8_E4M3);
        }
      }
      return true;
    }

    return false;
  }

  /// Dump 精度转换: fp22 → 外部格式 (按 segment 构建 64B 包)
  static bool build_dump_packet(uint32_t fmt_c,
                                const uint32_t subtile[kPrimitiveDim][kPrimitiveDim],
                                uint32_t segment_idx,
                                packet_t* out) {
    if (nullptr == out) {
      return false;
    }
    packet_t packet{};
    if (fmt_c == vortex::tensor::fp32::id) {
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

    if (fmt_c == vortex::tensor::fp16::id) {
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

    if (fmt_c == vortex::tensor::fp8::id) {
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

  /// 便捷路径: 转换 + 写入
  bool write_fill_subtile(uint32_t fmt_c,
                          uint32_t subtile_idx,
                          const std::vector<packet_t>& packets) {
    auto needed = packets_per_subtile(fmt_c);
    if (subtile_idx >= kDepth || packets.size() < needed) {
      return false;
    }

    auto& row = rows_.at(subtile_idx);
    for (auto& bank : row) {
      bank.fill(0);
    }
    uint32_t converted[kPrimitiveDim][kPrimitiveDim] = {};
    if (!convert_fill_packets(fmt_c, packets, converted)) {
      return false;
    }
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        store_elem(row, i * kPrimitiveDim + j, converted[i][j]);
      }
    }
    row_valid_.at(subtile_idx) = true;
    return true;
  }

  /// 时序路径: 已转换好的 fp22 直接写入
  bool write_converted_subtile(uint32_t subtile_idx,
                               const uint32_t in[kPrimitiveDim][kPrimitiveDim]) {
    if (subtile_idx >= kDepth) {
      return false;
    }
    write_subtile_fp22(subtile_idx, in);
    return true;
  }

  bool dump_subtile_packets(uint32_t fmt_c,
                            uint32_t subtile_id,
                            std::vector<packet_t>* packets) const {
    if (nullptr == packets || subtile_id >= kDepth || !valid()) {
      return false;
    }

    auto packets_per_group = packets_per_subtile(fmt_c);
    packets->assign(packets_per_group, packet_t{});
    uint32_t subtile[kPrimitiveDim][kPrimitiveDim] = {};
    read_subtile_fp22(subtile_id, subtile);
    for (uint32_t packet_idx = 0; packet_idx < packets_per_group; ++packet_idx) {
      if (!build_dump_packet(fmt_c, subtile, packet_idx, &packets->at(packet_idx))) {
        return false;
      }
    }
    return true;
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

  /// 便捷路径 (参考验证用)
  void accumulate_subtile(uint32_t subtile_id,
                          const uint32_t in[kPrimitiveDim][kPrimitiveDim]) {
    if (subtile_id >= kDepth || !row_valid_.at(subtile_id)) {
      std::abort();
    }
    auto& row = rows_.at(subtile_id);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        auto elem_idx = i * kPrimitiveDim + j;
        auto current = load_elem(row, elem_idx);
        store_elem(row, elem_idx, add_fp22_raw(current, in[i][j]));
      }
    }
  }

  // 参考验证辅助接口
  void load_block_fp32(uint32_t storage_m, uint32_t storage_n, float out[kPrimitiveDim][kPrimitiveDim]) const {
    uint32_t raw[kPrimitiveDim][kPrimitiveDim] = {};
    read_subtile_fp22(storage_m * 2 + storage_n, raw);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        out[i][j] = vortex::bit_cast<float>(fp22_to_fp32(raw[i][j]));
      }
    }
  }

  void load_block_fp16(uint32_t storage_m, uint32_t storage_n, uint16_t out[kPrimitiveDim][kPrimitiveDim]) const {
    uint32_t raw[kPrimitiveDim][kPrimitiveDim] = {};
    read_subtile_fp22(storage_m * 2 + storage_n, raw);
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        out[i][j] = fp22_to_fp16(raw[i][j]);
      }
    }
  }

  void store_block_fp32(uint32_t storage_m, uint32_t storage_n, const float in[kPrimitiveDim][kPrimitiveDim]) {
    auto idx = storage_m * 2 + storage_n;
    auto& row = rows_.at(idx);
    row_valid_.at(idx) = true;
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        store_elem(row, i * kPrimitiveDim + j,
                   convert_c_to_fp22(vortex::bit_cast<uint32_t>(in[i][j]), PREC_FP32));
      }
    }
  }

  void store_block_fp16(uint32_t storage_m, uint32_t storage_n, const uint16_t in[kPrimitiveDim][kPrimitiveDim]) {
    auto idx = storage_m * 2 + storage_n;
    auto& row = rows_.at(idx);
    row_valid_.at(idx) = true;
    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        store_elem(row, i * kPrimitiveDim + j, convert_c_to_fp22(in[i][j], PREC_FP16));
      }
    }
  }

private:
  using row_t = std::array<std::array<uint32_t, kBankElems>, kBankCount>;

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

  std::array<row_t, kDepth> rows_;
  std::array<bool, kDepth> row_valid_;
};
