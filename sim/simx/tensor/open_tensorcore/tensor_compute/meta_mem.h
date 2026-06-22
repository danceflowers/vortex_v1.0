#pragma once

// ============================================================================
// MetaMem -- single-packet sparse metadata store.
// ============================================================================
//
// A sparse MMA carries one 64B metadata packet next to the A payload. MetaMem
// splits it into four 16B lines indexed by (step_m, step_k); each primitive
// uses one line to expand the compressed A payload and mask B.
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

  static constexpr uint32_t fill_beats() {
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

  /// Read the 16B sparse metadata line for one primitive issue.
  void read_line(uint32_t step_m, uint32_t step_k, uint8_t out[kLineBytes]) const {
    if (!valid_) {
      std::abort();
    }
    uint32_t line_index = step_m * 2 + step_k;
    uint32_t byte_offset = line_index * kLineBytes;
    std::copy_n(packet_.begin() + byte_offset, kLineBytes, out);
  }

  // Slot-aware compatibility wrappers used by the TensorUnit timing path.
  // Current MetaMem is single-instance; slot_id is accepted for interface
  // compatibility and ignored.
  void clear_slot(uint32_t) {
    clear();
  }

  bool write_fill_beat(uint32_t, const packet_t& packet) {
    return write_fill_packet(packet);
  }

  void read_line(uint32_t,
                 uint32_t step_m,
                 uint32_t step_k,
                 uint8_t out[kLineBytes]) const {
    read_line(step_m, step_k, out);
  }

private:
  packet_t packet_;
  bool valid_;
};
