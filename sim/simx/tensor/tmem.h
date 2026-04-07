// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tmem_window_planner.h"

namespace vortex {

struct TmemPacket {
  std::array<uint8_t, 64> bytes;

  TmemPacket() {
    bytes.fill(0);
  }
};

enum class TmemHandleBlockReason : uint8_t {
  None = 0,
  Invalid,
  BusyTmaLoad,
  BusyTmaStoreOrShift,
  PayloadNotReady,
  MetaNotReady,
};

// One software-visible TMEM allocation handle.
//
// The allocation reserves a slab of logical TMEM columns, tracks readiness for
// payload/meta traffic, and owns the planned windows that map mathematical
// matrices onto the TMEM logical view.
struct TmemAllocation {
  bool valid = false;
  uint32_t payload_col_base = 0;
  uint32_t col_span = 0;
  uint32_t row_bytes = 0;
  uint32_t meta_col_base = 0;
  uint32_t meta_col_span = 0;
  bool layout_valid = false;
  uint32_t layout_epoch = 0;
  uint32_t visible_layout_epoch = 0;
  std::vector<TmemWindowPlan> windows;
  bool payload_ready = true;
  bool meta_ready = true;
  bool visible_payload_ready = true;
  bool visible_meta_ready = true;
  // TMEM_ALLOC 绑定的 MMA descriptor ID 和预构建 layout plan
  uint32_t mma_desc_id = 0;
  TmemLayoutPlan prebuilt_layout;  // alloc 时由 build_dense_plan 生成
};

// TMEM models the tensor scratchpad between mathematical matrix windows and
// the physical banked SRAM array.
//
// The interface is intentionally split into three views:
// - mathematical windows, tiles and packets owned by TMA / TensorUnit
// - logical TMEM columns and logical lines used by the window planner
// - physical banks / rows / byte lanes used by the banked SRAM model
//
// Clients operate on allocations, windows, packets and logical lines; TMEM
// resolves those accesses onto banks, rows and per-cycle read/write
// arbitration.
class Tmem {
public:
  enum class PortRequestKind : uint8_t {
    RegionRead = 0,
    RegionWrite,
    WindowRead,
    WindowWrite,
    WindowLinearRead,
    WindowLinearWrite,
    WindowLineRead,
    WindowLineWrite,
  };

  struct PortRequestDesc {
    PortRequestKind kind = PortRequestKind::RegionRead;
    uint64_t age = 0;
    uint32_t handle = 0;
    uint32_t window_id = 0;
    uint32_t packet_idx = 0;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
    uint32_t line_idx = 0;
    uint32_t chunk_idx = 0;
  };

  static constexpr uint32_t kPacketBytes = 64;
  static constexpr uint32_t kPhysicalBankBytes = 8;
  static constexpr uint32_t kPacketLanes = kPacketBytes / kPhysicalBankBytes;
  static constexpr uint32_t kLogicalLines = 128;
  static constexpr uint32_t kPayloadCols = 64;
  static constexpr uint32_t kMetaCols = 0;
  static constexpr uint32_t kMetaColBase = kPayloadCols;
  static constexpr uint32_t kNumCols = kPayloadCols + kMetaCols;
  static constexpr uint32_t kColBytes = kLogicalLines;
  static constexpr uint32_t kDefaultPhysicalBanks = 16;
  static constexpr uint32_t kPacketsPerCol = kColBytes / kPacketBytes;
  static constexpr uint32_t kReadPortsPerBank = 1;
  static constexpr uint32_t kWritePortsPerBank = 1;
  static constexpr uint32_t kReadPacketsPerCycle = 1;
  static constexpr uint32_t kWritePacketsPerCycle = 1;
  // Legacy aliases retained while the rest of simx is migrated to col-span naming.
  static constexpr uint32_t kPayloadBanks = kPayloadCols;
  static constexpr uint32_t kMetaBanks = kMetaCols;
  static constexpr uint32_t kMetaBankBase = kMetaColBase;
  static constexpr uint32_t kNumBanks = kNumCols;
  static constexpr uint32_t kBankSize = kColBytes;

  Tmem();

  uint32_t physical_banks() const {
    return num_physical_banks_;
  }

  uint32_t bank_slice_bytes() const {
    return bank_slice_bytes_;
  }

  void reset();

  uint32_t alloc(uint32_t col_span);
  bool free(uint32_t handle);
  void seal_allocator();
  bool allocator_sealed() const;

  bool lookup_allocation(uint32_t handle, TmemAllocation** allocation);
  bool lookup_allocation(uint32_t handle, const TmemAllocation** allocation) const;

  bool region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const;
  bool query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const;

  bool region_copy_in(uint32_t col_base, uint32_t col_span, const uint8_t* data, uint32_t size_bytes);
  bool region_copy_out(uint32_t col_base, uint32_t col_span, uint8_t* data, uint32_t size_bytes) const;
  bool region_read_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, TmemPacket* out) const;
  bool region_write_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, const TmemPacket& in);
  bool copy_in(uint32_t handle, const uint8_t* data, uint32_t size_bytes);
  bool copy_out(uint32_t handle, uint8_t* data, uint32_t size_bytes) const;
  bool region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes);
  bool read_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const;
  bool write_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in);
  bool read_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const;
  bool write_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in);
  bool read_window_line_chunk(uint32_t handle, uint32_t window_id, uint32_t line_idx, uint32_t chunk_idx, TmemPacket* out) const;
  bool window_packet_count(uint32_t handle, uint32_t window_id, uint32_t* count) const;
  bool write_window_line_chunk(uint32_t handle, uint32_t window_id, uint32_t line_idx, uint32_t chunk_idx, const TmemPacket& in);
  bool window_line_chunk_count(uint32_t handle, uint32_t window_id, uint32_t* count) const;
  bool read_window_math_row(uint32_t handle,
                            uint32_t window_id,
                            uint32_t math_row,
                            uint8_t* out_row,
                            uint32_t row_bytes) const;
  bool write_window_math_row(uint32_t handle,
                             uint32_t window_id,
                             uint32_t math_row,
                             const uint8_t* row_bytes,
                             uint32_t row_size_bytes);
  bool shift_window_logical_lines_down(uint32_t handle, uint32_t window_id);
  bool shift_window_math_rows_down(uint32_t handle,
                                   uint32_t window_id,
                                   const uint8_t* refill_math_row,
                                   uint32_t refill_math_row_bytes);

  void set_payload_ready(uint32_t handle, bool ready);
  void set_meta_ready(uint32_t handle, bool ready);
  void set_meta_region(uint32_t handle, uint32_t meta_col_base, uint32_t meta_col_span);
  bool bind_layout(uint32_t handle, const TmemLayoutPlan& layout);
  bool upsert_window(uint32_t handle, const TmemWindowPlan& window);
  bool lookup_window(uint32_t handle, uint32_t window_id, const TmemWindowPlan** out) const;
  bool lookup_window(uint32_t handle, uint32_t window_id, TmemWindowPlan** out);
  bool window_epoch(uint32_t handle, uint32_t* epoch) const;
  bool bump_window_epoch(uint32_t handle);
  void publish_visible_state();
  bool set_row_bytes(uint32_t handle, uint32_t row_bytes);
  bool row_bytes(uint32_t handle, uint32_t* row_bytes) const;

  void reset_port_budgets(uint64_t cycle);
  void ensure_port_budgets(uint64_t cycle);
  bool try_acquire_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);
  bool try_acquire_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);
  bool try_acquire_window_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  bool try_acquire_window_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  bool try_acquire_window_linear_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  bool try_acquire_window_linear_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  bool try_acquire_window_line_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t line_idx, uint32_t chunk_idx);
  uint64_t enqueue_port_request(uint64_t cycle, const PortRequestDesc& desc);
  void arbitrate_requests(uint64_t cycle);
  bool request_granted(uint64_t tag) const;
  void consume_request_grant(uint64_t tag);
  void refund_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);
  void refund_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);
  void refund_window_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  void refund_window_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  void refund_window_linear_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  void refund_window_linear_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx);
  void refund_window_line_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t line_idx, uint32_t chunk_idx);

  bool validate(std::string* reason) const;

private:
  void assert_valid() const;
  void resize_storage();
  static uint32_t ceil_div(uint32_t value, uint32_t divisor);
  uint8_t read_region_byte(uint32_t col_base, uint32_t byte_offset) const;
  void write_region_byte(uint32_t col_base, uint32_t byte_offset, uint8_t value);
  uint8_t read_logical_byte(uint32_t logical_col, uint32_t logical_line) const;
  void write_logical_byte(uint32_t logical_col, uint32_t logical_line, uint8_t value);
  void clear_region(uint32_t col_base, uint32_t col_span);
  bool try_acquire_region_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx, bool write);
  bool try_acquire_window_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx, bool write);
  bool try_acquire_window_linear_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx, bool write);
  bool try_acquire_window_line_chunk(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t line_idx, uint32_t chunk_idx, bool write);
  void refund_region_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx, bool write);
  void refund_window_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx, bool write);
  void refund_window_linear_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx, bool write);
  void refund_window_line_chunk(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t line_idx, uint32_t chunk_idx, bool write);
  static bool region_packet_location(uint32_t col_base,
                                     uint32_t col_span,
                                     uint32_t packet_idx,
                                     uint32_t* logical_col);
  bool window_packet_location(uint32_t handle,
                              uint32_t window_id,
                              uint32_t packet_idx,
                              uint32_t* logical_col_base,
                              uint32_t* logical_line_base,
                              uint32_t* logical_col_span,
                              uint32_t* logical_line_span) const;
  bool resolve_window_line_chunk_region(uint32_t handle,
                                        uint32_t window_id,
                                        uint32_t line_idx,
                                        uint32_t chunk_idx,
                                        uint32_t* logical_col_base,
                                        uint32_t* logical_line_base,
                                        uint32_t* logical_col_span) const;
  bool resolve_window_linear_packet_region(uint32_t handle,
                                           uint32_t window_id,
                                           uint32_t packet_idx,
                                           uint32_t* logical_col_base,
                                           uint32_t* logical_line_base,
                                           uint32_t* logical_col_span,
                                           uint32_t* byte_offset,
                                           uint32_t* valid_bytes) const;
  uint32_t line_chunk_bank(uint32_t logical_line,
                           uint32_t chunk_idx) const;
  uint32_t packet_lane_bank(uint32_t logical_col,
                            uint32_t packet_in_col,
                            uint32_t lane) const;
  void logical_byte_to_physical(uint32_t logical_col,
                                uint32_t logical_line,
                                uint32_t* bank,
                                uint32_t* row,
                                uint32_t* bank_byte) const;
  uint32_t resolve_window_col_base(const TmemAllocation& allocation,
                                   const TmemWindowPlan& window) const;
  struct PortRequest {
    uint64_t tag = 0;
    uint64_t age = 0;
    uint64_t submit_cycle = 0;
    PortRequestKind kind = PortRequestKind::RegionRead;
    uint32_t handle = 0;
    uint32_t window_id = 0;
    uint32_t packet_idx = 0;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
    uint32_t line_idx = 0;
    uint32_t chunk_idx = 0;
  };
  bool port_request_is_write(PortRequestKind kind) const;
  bool try_grant_request(const PortRequest& request, uint64_t cycle);
  void insert_pending_request(uint64_t tag, bool write);

  using PhysicalRow = std::vector<uint8_t>;
  static constexpr uint32_t kPhysicalRows = 64;

  uint32_t num_physical_banks_;
  uint32_t bank_slice_bytes_;
  uint32_t bank_swizzle_base_stride_;
  uint32_t bank_swizzle_lane_stride_;
  std::vector<std::vector<PhysicalRow>> banks_;
  std::array<bool, kPayloadCols> payload_col_allocs_;
  std::unordered_map<uint32_t, TmemAllocation> allocations_;
  bool allocator_sealed_;
  uint32_t next_handle_;
  uint64_t port_cycle_;
  uint32_t read_packet_budget_;
  uint32_t write_packet_budget_;
  std::vector<uint32_t> read_bank_budgets_;
  std::vector<uint32_t> write_bank_budgets_;
  uint64_t next_request_tag_;
  uint64_t arbitration_cycle_;
  std::unordered_map<uint64_t, PortRequest> pending_requests_;
  std::unordered_map<uint64_t, PortRequest> granted_requests_;
  std::deque<uint64_t> pending_read_requests_;
  std::deque<uint64_t> pending_write_requests_;
};

} // namespace vortex
