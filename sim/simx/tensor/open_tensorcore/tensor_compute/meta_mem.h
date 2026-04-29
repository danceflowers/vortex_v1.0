#pragma once

// ============================================================================
// MetaMem -- 稀疏元数据存储 (单实例版本)
// ============================================================================
//
// 存储组织:
//   - 单实例, 保存 1 个 64B packet
//   - 64B 被逻辑上划分为 4 条 16B line (kTileLines = 4)
//
// Fill 路径:
//   FillA 在传输 A 数据包之后，额外传输 1 个元数据包。
//
// Read 路径:
//   read_line(step_m, step_k, out) 读出 16B 元数据。
// ============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

class MetaMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kTileLines = 4;
  static constexpr uint32_t kLineBytes = 16;

  MetaMem() {
    reset();
  }

  void reset() {
    packet_.fill(0);
    valid_ = false;
  }

  void clear() {
    packet_.fill(0);
    valid_ = false;
  }

  static constexpr uint32_t packet_count() {
    return 1;
  }

  static constexpr uint32_t fill_packets() {
    return 1;
  }

  bool write_fill_packet(const packet_t& packet) {
    packet_ = packet;
    valid_ = true;
    return true;
  }

  bool valid() const {
    return valid_;
  }

  void read_line(uint32_t step_m, uint32_t step_k, uint8_t out[kLineBytes]) const {
    if (!valid_) {
      std::abort();
    }
    uint32_t line_index = step_m * 2 + step_k;
    uint32_t byte_offset = line_index * kLineBytes;
    std::copy_n(packet_.begin() + byte_offset, kLineBytes, out);
  }

private:
  packet_t packet_;
  bool valid_;
};
