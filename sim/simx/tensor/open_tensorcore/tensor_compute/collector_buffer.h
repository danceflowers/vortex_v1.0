#pragma once

// ============================================================================
// collector_buffer.h — Collector buffer classes for OpenTensorCore pipeline
// ============================================================================
//
// ABuf, MBuf, and BBuf mirror the storage dimensions of AMem, MetaMem, and
// BMem respectively, but add warp-level ownership tracking via CollectorState.
// These buffers sit between the operand-fetch stage and the compute stage,
// holding converted fp9 operands and metadata for a single warp's tile.
//
// CollectorState lifecycle:
//   INVALID → FILL (valid=true, owner=wid, compute_inflight=true)
//          → USE   (compute_inflight stays true while computing)
//          → LASTUSE / DISCARD → INVALID
// ============================================================================

#include <array>
#include <cstdint>
#include <cstdlib>

namespace vortex {

// ============================================================================
// CollectorState — shared ownership tracking struct for all collector buffers
// ============================================================================
struct CollectorState {
  bool     valid            = false;
  uint32_t owner_wid        = 0;
  bool     compute_inflight = false;

  void reset() {
    valid = false;
    owner_wid = 0;
    compute_inflight = false;
  }
};

// ============================================================================
// ABuf — A-matrix collector buffer (mirrors AMem: 4 lines x 8x8 FP9, 8 banks)
// ============================================================================
class ABuf {
public:
  static constexpr uint32_t kDepth     = 4;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = 8;
  using row_t = std::array<std::array<uint16_t, kBankElems>, kBankCount>;

  ABuf() { clear(); }

  void clear() {
    for (auto& line : lines_) { for (auto& bank : line) bank.fill(0); }
    line_valid_.fill(false);
    state_.reset();
  }

  bool write_line(uint32_t line_idx, const uint16_t data[8][8]) {
    if (line_idx >= kDepth) return false;
    auto& dst = lines_.at(line_idx);
    for (auto& bank : dst) bank.fill(0);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        store_elem(dst, i * 8 + j, data[i][j]);
    line_valid_.at(line_idx) = true;
    return true;
  }

  void read_primitive(uint32_t line_idx, uint16_t out[8][8]) const {
    if (line_idx >= kDepth || !line_valid_.at(line_idx)) std::abort();
    const auto& line = lines_.at(line_idx);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        out[i][j] = load_elem(line, i * 8 + j);
  }

  bool all_lines_valid() const {
    for (uint32_t l = 0; l < kDepth; ++l) if (!line_valid_.at(l)) return false;
    return true;
  }

  // Collector lifecycle
  void fill(uint32_t wid) { state_.valid = true; state_.owner_wid = wid; state_.compute_inflight = true; }
  void mark_compute_started() { state_.compute_inflight = true; }
  void mark_compute_done()   { state_.compute_inflight = false; }
  void invalidate()          { state_.valid = false; state_.compute_inflight = false; }

  bool is_ready_for(uint32_t wid, uint8_t collector_buffer) const {
    switch (collector_buffer) {
    case 0x0: return !state_.valid || !state_.compute_inflight;  // FILL: ok if INVALID or compute done
    case 0x1: case 0x2: return state_.valid && state_.owner_wid == wid;  // USE/LASTUSE: must match owner
    case 0x3: return !(state_.valid && state_.owner_wid == wid && state_.compute_inflight);  // DISCARD: stall if own compute inflight
    default: return true;
    }
  }

  bool valid() const { return state_.valid; }
  uint32_t owner() const { return state_.owner_wid; }
  bool compute_inflight() const { return state_.compute_inflight; }

private:
  static void store_elem(row_t& row, uint32_t idx, uint16_t v) { row.at(idx / kBankElems).at(idx % kBankElems) = v; }
  static uint16_t load_elem(const row_t& row, uint32_t idx) { return row.at(idx / kBankElems).at(idx % kBankElems); }

  std::array<row_t, kDepth> lines_;
  std::array<bool, kDepth>  line_valid_;
  CollectorState            state_;
};

// ============================================================================
// MBuf — sparse metadata collector buffer (mirrors MetaMem: 1 packet x 64B)
// ============================================================================
class MBuf {
public:
  static constexpr uint32_t kLineBytes = 16;

  MBuf() { clear(); }
  void clear() { data_.fill(0); state_.reset(); }
  void write(const std::array<uint8_t, 64>& data) { data_ = data; }
  void read(uint8_t out[kLineBytes], uint32_t step_m, uint32_t step_k) const {
    uint32_t line_index = step_m * 2 + step_k;
    uint32_t byte_offset = line_index * kLineBytes;
    for (uint32_t i = 0; i < kLineBytes; ++i) out[i] = data_[byte_offset + i];
  }
  void fill(uint32_t wid) { state_.valid = true; state_.owner_wid = wid; state_.compute_inflight = true; }
  void mark_compute_started() { state_.compute_inflight = true; }
  void mark_compute_done()    { state_.compute_inflight = false; }
  void invalidate()           { state_.valid = false; state_.compute_inflight = false; }
  bool valid() const { return state_.valid; }
private:
  std::array<uint8_t, 64> data_;
  CollectorState           state_;
};

// ============================================================================
// BBuf — B-matrix collector buffer (4 instances, mirrors BMem: 4 lines x 8x8 FP9, 8 banks)
// ============================================================================
class BBuf {
public:
  static constexpr uint32_t kDepth     = 4;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = 8;
  using row_t = std::array<std::array<uint16_t, kBankElems>, kBankCount>;

  explicit BBuf(uint8_t index = 0) : index_(index) { clear(); }

  void clear() {
    for (auto& line : lines_) { for (auto& bank : line) bank.fill(0); }
    line_valid_.fill(false);
    state_.reset();
  }

  bool write_line(uint32_t line_idx, const uint16_t data[8][8]) {
    if (line_idx >= kDepth) return false;
    auto& dst = lines_.at(line_idx);
    for (auto& bank : dst) bank.fill(0);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        store_elem(dst, i * 8 + j, data[i][j]);
    line_valid_.at(line_idx) = true;
    return true;
  }

  void read_primitive(uint32_t line_idx, uint16_t out[8][8]) const {
    if (line_idx >= kDepth || !line_valid_.at(line_idx)) std::abort();
    const auto& line = lines_.at(line_idx);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        out[i][j] = load_elem(line, i * 8 + j);
  }

  bool all_lines_valid() const {
    for (uint32_t l = 0; l < kDepth; ++l) if (!line_valid_.at(l)) return false;
    return true;
  }

  // Collector lifecycle (identical to ABuf)
  void fill(uint32_t wid) { state_.valid = true; state_.owner_wid = wid; state_.compute_inflight = true; }
  void mark_compute_started() { state_.compute_inflight = true; }
  void mark_compute_done()    { state_.compute_inflight = false; }
  void invalidate()           { state_.valid = false; state_.compute_inflight = false; }

  bool is_ready_for(uint32_t wid, uint8_t collector_buffer) const {
    switch (collector_buffer) {
    case 0x0: return !state_.valid || !state_.compute_inflight;
    case 0x1: case 0x2: return state_.valid && state_.owner_wid == wid;
    case 0x3: return !(state_.valid && state_.owner_wid == wid && state_.compute_inflight);
    default: return true;
    }
  }

  bool valid() const { return state_.valid; }
  uint32_t owner() const { return state_.owner_wid; }
  uint8_t index() const { return index_; }

private:
  static void store_elem(row_t& row, uint32_t idx, uint16_t v) { row.at(idx / kBankElems).at(idx % kBankElems) = v; }
  static uint16_t load_elem(const row_t& row, uint32_t idx) { return row.at(idx / kBankElems).at(idx % kBankElems); }

  std::array<row_t, kDepth> lines_;
  std::array<bool, kDepth>  line_valid_;
  CollectorState            state_;
  uint8_t                   index_;
};

}  // namespace vortex
