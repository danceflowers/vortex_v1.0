#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

class MetaMem {
public:
  using packet_t = std::array<uint8_t, 64>;

  static constexpr uint32_t kNumSlots = 2;
  static constexpr uint32_t kTileLines = 4;
  static constexpr uint32_t kLineBytes = 16;

  MetaMem() {
    reset();
  }

  void reset() {
    for (auto& packet : packets_) {
      packet.fill(0);
    }
    valid_.fill(false);
  }

  void clear_slot(uint32_t slot_id) {
    if (slot_id >= kNumSlots) {
      std::abort();
    }
    packets_.at(slot_id).fill(0);
    valid_.at(slot_id) = false;
  }

  static constexpr uint32_t packet_count() {
    return 1;
  }

  static constexpr uint32_t fill_beats() {
    return 1;
  }

  bool write_fill_beat(uint32_t slot_id, const packet_t& packet) {
    if (slot_id >= kNumSlots) {
      return false;
    }
    packets_.at(slot_id) = packet;
    valid_.at(slot_id) = true;
    return true;
  }

  bool fill_tile(uint32_t slot_id, const std::vector<packet_t>& packets) {
    if (slot_id >= kNumSlots || packets.size() < packet_count()) {
      return false;
    }
    return write_fill_beat(slot_id, packets.front());
  }

  bool fill_tile(const std::vector<packet_t>& packets) {
    return fill_tile(0, packets);
  }

  bool valid(uint32_t slot_id) const {
    if (slot_id >= kNumSlots) {
      std::abort();
    }
    return valid_.at(slot_id);
  }

  void read_line(uint32_t slot_id, uint32_t step_m, uint32_t step_k, uint8_t out[kLineBytes]) const {
    if (slot_id >= kNumSlots || !valid_.at(slot_id)) {
      std::abort();
    }
    uint32_t line_index = step_m * 2 + step_k;
    uint32_t byte_offset = line_index * kLineBytes;
    std::copy_n(packets_.at(slot_id).begin() + byte_offset, kLineBytes, out);
  }

  void read_line(uint32_t step_m, uint32_t step_k, uint8_t out[kLineBytes]) const {
    read_line(0, step_m, step_k, out);
  }

private:
  std::array<packet_t, kNumSlots> packets_;
  std::array<bool, kNumSlots> valid_;
};
