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

namespace vortex {

struct TmemPacket {
  std::array<uint8_t, 64> bytes;

  TmemPacket() {
    bytes.fill(0);
  }
};

enum class TmemTaddrBlockReason : uint8_t {
  None = 0,
  Invalid,
  BusyTmemShift,
  PayloadNotReady,
  MetaNotReady,
};

// One software-visible PTX TMEM allocation.
//
// The allocation reserves a slab of logical TMEM lanes. The returned taddr is
// the PTX TADDR lane base, so code can round-trip TADDRs without a descriptor
// table or window binding.
struct TmemAllocation {
  bool valid = false;
  uint32_t payload_col_base = 0;
  uint32_t col_span = 0;
  uint32_t row_bytes = 0;
  uint32_t meta_col_base = 0;
  uint32_t meta_col_span = 0;
  bool payload_ready = true;
  bool meta_ready = true;
  bool visible_payload_ready = true;
  bool visible_meta_ready = true;
};

// TMEM models the PTX tensor scratchpad and its physical banked SRAM array.
//
// Clients operate on PTX TADDR allocations and 64 B packet/byte ranges. TMEM
// resolves those accesses onto banks, rows and per-cycle read/write
// arbitration. Descriptors live in LMEM/DRAM, and TMEM is addressed by PTX
// TADDRs.
class Tmem {
public:
  enum class PortRequestKind : uint8_t {
    RegionRead = 0,
    RegionWrite,
  };

  struct PortRequestDesc {
    PortRequestKind kind = PortRequestKind::RegionRead;
    uint64_t age = 0;
    uint32_t packet_idx = 0;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
  };

  // Phase-3.4 Stage 0: PTX-strict TADDR sentinel. tcgen05.alloc returns a
  // TADDR `[15:0]=lane`,`[31:16]=col_byte`. col_byte 0..511 is valid, so the
  // theoretical max valid TADDR is 0x01FF_007F (col_byte=511, lane=127). Any
  // TADDR >= 0x80000000 is treated as invalid; we use 0xFFFFFFFFu as the
  // canonical failure value (cannot collide with a successful alloc).
  static constexpr uint32_t kInvalidTaddr = 0xFFFFFFFFu;

  static constexpr uint32_t kPacketBytes = 64;
  static constexpr uint32_t kPhysicalBankBytes = 8;
  static constexpr uint32_t kPacketLanes = kPacketBytes / kPhysicalBankBytes;
  // Phase-3.3.3: expand logical view to 128 lanes × 512 cols (1-byte cell).
  // Total = 128 × 512 = 64 KB. Banks / ports / swizzle unchanged.
  static constexpr uint32_t kLogicalLines = 128;   // was 128 (cols per lane)
  static constexpr uint32_t kPayloadCols = 512;    // was 64  (lanes)
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
  Tmem();

  uint32_t physical_banks() const {
    return num_physical_banks_;
  }

  uint32_t bank_slice_bytes() const {
    return bank_slice_bytes_;
  }

  void reset();

  // Allocate a PTX-visible TMEM TADDR region and return its lane-base TADDR.
  uint32_t alloc(uint32_t col_span);

  // Free a previously allocated TADDR region and clear its payload storage.
  bool free(uint32_t taddr);

  // Seal the allocator after tcgen05.alloc permit is released.
  void seal_allocator();
  bool allocator_sealed() const;

  // Look up allocation metadata by the kernel-visible TADDR value.
  bool lookup_allocation(uint32_t taddr, TmemAllocation** allocation);
  bool lookup_allocation(uint32_t taddr, const TmemAllocation** allocation) const;

  // Phase-3.4 Stage 0: PTX TADDR lookup helper. Per-thread TCU_LD/TCU_ST
  // computes actual_lane = taddr.lane + thread_id; this scans allocations_
  // for one whose [col_base, col_base + col_span) covers actual_lane.
  // Returns the containing allocation's col_base on success.
  bool find_allocation_by_lane(uint32_t lane, uint32_t* col_base) const;

  bool region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const;
  bool query(uint32_t taddr, uint32_t* col_span, uint32_t* size_bytes) const;

  bool region_copy_in(uint32_t col_base, uint32_t col_span, const uint8_t* data, uint32_t size_bytes);
  bool region_copy_out(uint32_t col_base, uint32_t col_span, uint8_t* data, uint32_t size_bytes) const;

  // Byte-range R/W via taddr (Phase-3.3.1 GAP-4 path for tcu_ld/tcu_st).
  // byte_offset is relative to the allocation's payload region (0..col_span*kColBytes-1).
  bool taddr_read_bytes(uint32_t taddr, uint32_t byte_offset, uint8_t* dst, uint32_t bytes) const;
  bool taddr_write_bytes(uint32_t taddr, uint32_t byte_offset, const uint8_t* src, uint32_t bytes);
  bool region_read_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, TmemPacket* out) const;
  bool region_write_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, const TmemPacket& in);
  bool copy_in(uint32_t taddr, const uint8_t* data, uint32_t size_bytes);
  bool copy_out(uint32_t taddr, uint8_t* data, uint32_t size_bytes) const;

  // Shift a region down by one logical row; TmemSystem models the async timing.
  bool region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes);

  // Update producer/consumer readiness that becomes visible on publish_visible_state().
  void set_payload_ready(uint32_t taddr, bool ready);
  void set_meta_ready(uint32_t taddr, bool ready);
  void set_meta_region(uint32_t taddr, uint32_t meta_col_base, uint32_t meta_col_span);
  void publish_visible_state();
  bool set_row_bytes(uint32_t taddr, uint32_t row_bytes);
  bool row_bytes(uint32_t taddr, uint32_t* row_bytes) const;

  void reset_port_budgets(uint64_t cycle);
  void ensure_port_budgets(uint64_t cycle);

  // Try to reserve packet/bank ports directly; used by tests and the port arbiter.
  bool try_acquire_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);
  bool try_acquire_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);

  // Enqueue packet traffic from TMEM clients and grant it through shared budgets.
  uint64_t enqueue_port_request(uint64_t cycle, const PortRequestDesc& desc);
  void arbitrate_requests(uint64_t cycle);
  bool request_granted(uint64_t tag) const;
  void consume_request_grant(uint64_t tag);
  void refund_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);
  void refund_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx);

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
  void refund_region_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx, bool write);
  static bool region_packet_location(uint32_t col_base,
                                     uint32_t col_span,
                                     uint32_t packet_idx,
                                     uint32_t* logical_col);
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
  struct PortRequest {
    uint64_t tag = 0;
    uint64_t age = 0;
    uint64_t submit_cycle = 0;
    PortRequestKind kind = PortRequestKind::RegionRead;
    uint32_t packet_idx = 0;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
  };
  bool port_request_is_write(PortRequestKind kind) const;
  bool try_grant_request(const PortRequest& request, uint64_t cycle);
  void insert_pending_request(uint64_t tag, bool write);

  using PhysicalRow = std::vector<uint8_t>;
  // Phase-3.3.3: kPhysicalRows tracks kLogicalLines (2 logical lines pack into
  // one physical row). With kLogicalLines=512, kPhysicalRows=256.
  static constexpr uint32_t kPhysicalRows = kLogicalLines / 2;

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
