#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "tensor_cfg.h"

class CMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kDim = 16;
  static constexpr uint32_t kDepth = 16;

  CMem() {
    reset();
  }

  void reset() {
    fmt_c_ = 0;
    valid_ = false;
    for (auto& row : fp16_tile_) {
      row.fill(0);
    }
    for (auto& row : fp32_tile_) {
      row.fill(0.0f);
    }
  }

  static uint32_t packet_count(uint32_t fmt_c) {
    return (fmt_c == vortex::tensor::fp32::id) ? 16 : 8;
  }

  bool fill_tile(uint32_t fmt_c, const std::vector<packet_t>& packets) {
    auto expected_packets = packet_count(fmt_c);
    if (packets.size() < expected_packets) {
      return false;
    }
    fmt_c_ = fmt_c;
    if (fmt_c == vortex::tensor::fp32::id) {
      for (uint32_t row = 0; row < kDim; ++row) {
        const auto& p = packets.at(row);
        for (uint32_t col = 0; col < kDim; ++col) {
          uint32_t off = col * 4;
          uint32_t bits = static_cast<uint32_t>(p.at(off + 0))
                        | (static_cast<uint32_t>(p.at(off + 1)) << 8)
                        | (static_cast<uint32_t>(p.at(off + 2)) << 16)
                        | (static_cast<uint32_t>(p.at(off + 3)) << 24);
          union {
            uint32_t u;
            float f;
          } cvt = {bits};
          fp32_tile_.at(row).at(col) = cvt.f;
        }
      }
    } else {
      for (uint32_t packet = 0; packet < expected_packets; ++packet) {
        const auto& p = packets.at(packet);
        for (uint32_t elem = 0; elem < 32; ++elem) {
          uint32_t row = packet * 2 + (elem / 16);
          uint32_t col = elem % 16;
          uint32_t off = elem * 2;
          fp16_tile_.at(row).at(col) = static_cast<uint16_t>(p.at(off + 0) | (p.at(off + 1) << 8));
        }
      }
    }
    valid_ = true;
    return true;
  }

  bool dump_tile(uint32_t fmt_c, std::vector<packet_t>* packets) const {
    if (nullptr == packets) {
      return false;
    }
    packets->assign(packet_count(fmt_c), packet_t{});
    if (fmt_c == vortex::tensor::fp32::id) {
      for (uint32_t row = 0; row < kDim; ++row) {
        auto& p = packets->at(row);
        for (uint32_t col = 0; col < kDim; ++col) {
          union {
            uint32_t u;
            float f;
          } cvt = {0};
          cvt.f = fp32_tile_.at(row).at(col);
          uint32_t off = col * 4;
          p.at(off + 0) = cvt.u & 0xff;
          p.at(off + 1) = (cvt.u >> 8) & 0xff;
          p.at(off + 2) = (cvt.u >> 16) & 0xff;
          p.at(off + 3) = (cvt.u >> 24) & 0xff;
        }
      }
    } else {
      for (uint32_t packet = 0; packet < packets->size(); ++packet) {
        auto& p = packets->at(packet);
        for (uint32_t elem = 0; elem < 32; ++elem) {
          uint32_t row = packet * 2 + (elem / 16);
          uint32_t col = elem % 16;
          uint32_t off = elem * 2;
          auto value = fp16_tile_.at(row).at(col);
          p.at(off + 0) = value & 0xff;
          p.at(off + 1) = (value >> 8) & 0xff;
        }
      }
    }
    return true;
  }

  bool valid() const {
    return valid_;
  }

  uint32_t fmt_c() const {
    return fmt_c_;
  }

  void load_block_fp16(uint32_t step_m, uint32_t step_n, uint16_t out[8][8]) const {
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        out[i][j] = fp16_tile_.at(step_m * 8 + i).at(step_n * 8 + j);
      }
    }
  }

  void store_block_fp16(uint32_t step_m, uint32_t step_n, const uint16_t in[8][8]) {
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        fp16_tile_.at(step_m * 8 + i).at(step_n * 8 + j) = in[i][j];
      }
    }
    valid_ = true;
  }

  void load_block_fp32(uint32_t step_m, uint32_t step_n, float out[8][8]) const {
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        out[i][j] = fp32_tile_.at(step_m * 8 + i).at(step_n * 8 + j);
      }
    }
  }

  void store_block_fp32(uint32_t step_m, uint32_t step_n, const float in[8][8]) {
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        fp32_tile_.at(step_m * 8 + i).at(step_n * 8 + j) = in[i][j];
      }
    }
    valid_ = true;
  }

private:
  uint32_t fmt_c_;
  bool valid_;
  std::array<std::array<uint16_t, kDim>, kDim> fp16_tile_;
  std::array<std::array<float, kDim>, kDim> fp32_tile_;
};
