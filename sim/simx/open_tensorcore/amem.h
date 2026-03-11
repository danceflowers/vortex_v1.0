#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "tensor_cfg.h"
#include "fp_types.h"

class AMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kDepth = 8;
  static constexpr uint32_t kTileLines = 4;
  static constexpr uint32_t kLineElems = 64;

  AMem() {
    reset();
  }

  void reset() {
    for (auto& line : lines_) {
      line.fill(0);
    }
    active_base_ = 0;
    stage_base_ = kTileLines;
    valid_ = false;
  }

  static uint32_t packet_count(uint32_t fmt_ab) {
    return (fmt_ab == vortex::tensor::fp16::id) ? 8 : 4;
  }

  bool fill_tile(uint32_t fmt_ab, const std::vector<packet_t>& packets) {
    auto expected_packets = packet_count(fmt_ab);
    if (packets.size() < expected_packets) {
      return false;
    }
    for (uint32_t line = 0; line < kTileLines; ++line) {
      auto& dst = lines_.at(stage_base_ + line);
      if (fmt_ab == vortex::tensor::fp16::id) {
        const auto& p0 = packets.at(line * 2 + 0);
        const auto& p1 = packets.at(line * 2 + 1);
        for (uint32_t elem = 0; elem < kLineElems; ++elem) {
          uint32_t byte_idx = elem * 2;
          uint16_t fp16 = 0;
          if (byte_idx < 64) {
            fp16 = static_cast<uint16_t>(p0.at(byte_idx + 0) | (p0.at(byte_idx + 1) << 8));
          } else {
            auto local = byte_idx - 64;
            fp16 = static_cast<uint16_t>(p1.at(local + 0) | (p1.at(local + 1) << 8));
          }
          dst.at(elem) = convert_to_fp9(fp16, PREC_FP16);
        }
      } else {
        const auto& p = packets.at(line);
        for (uint32_t elem = 0; elem < kLineElems; ++elem) {
          dst.at(elem) = convert_to_fp9(p.at(elem), PREC_FP8_E4M3);
        }
      }
    }
    active_base_ = stage_base_;
    stage_base_ = (stage_base_ == 0) ? kTileLines : 0;
    valid_ = true;
    return true;
  }

  bool valid() const {
    return valid_;
  }

  void read_primitive(uint32_t step_m, uint32_t step_k, uint16_t out[8][8]) const {
    const auto& line = lines_.at(active_base_ + step_m * 2 + step_k);
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        out[i][j] = line.at(i * 8 + j);
      }
    }
  }

private:
  std::array<std::array<uint16_t, kLineElems>, kDepth> lines_;
  uint32_t active_base_;
  uint32_t stage_base_;
  bool valid_;
};
