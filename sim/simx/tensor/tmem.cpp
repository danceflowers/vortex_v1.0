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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <vector>
#include "tensor_cfg.h"
#include "tmem.h"
#include "tmem_utils.h"

using namespace vortex;

namespace {

// Tmem.cpp implements the back half of the PTX tensor-memory path:
// - logical TMEM allocations addressed by TADDR lane base
// - logical (column, line) to physical (bank, row, byte) mapping
// - per-cycle ingress/egress arbitration
// - logical-line shift used by tcgen05.shift

} // namespace

Tmem::Tmem()
  : num_physical_banks_(tmem_functional::configured_physical_bank_count(kDefaultPhysicalBanks, kPacketLanes))
  , bank_slice_bytes_(kPhysicalBankBytes)
  , bank_swizzle_base_stride_(tmem_functional::choose_coprime_stride(num_physical_banks_, (num_physical_banks_ / 4) + 1))
  , bank_swizzle_lane_stride_(tmem_functional::choose_coprime_stride(num_physical_banks_, (num_physical_banks_ / 2) - 1))
  , allocator_sealed_(false)
  , next_handle_(1)
  , port_cycle_(std::numeric_limits<uint64_t>::max())
  , read_packet_budget_(0)
  , write_packet_budget_(0)
  , next_request_tag_(1)
  , arbitration_cycle_(std::numeric_limits<uint64_t>::max()) {
  this->resize_storage();
  this->reset();
}

uint32_t Tmem::ceil_div(uint32_t value, uint32_t divisor) {
  return tmem_functional::ceil_div(value, divisor);
}

void Tmem::resize_storage() {
  tmem_functional::resize_storage(&banks_, num_physical_banks_, kPhysicalRows, bank_slice_bytes_);
  read_bank_budgets_.assign(num_physical_banks_, 0);
  write_bank_budgets_.assign(num_physical_banks_, 0);
}

void Tmem::reset() {
  tmem_functional::reset_storage(&banks_);
  payload_col_allocs_.fill(false);
  allocations_.clear();
  allocator_sealed_ = false;
  next_handle_ = 1;
  port_cycle_ = std::numeric_limits<uint64_t>::max();
  read_packet_budget_ = 0;
  write_packet_budget_ = 0;
  next_request_tag_ = 1;
  arbitration_cycle_ = std::numeric_limits<uint64_t>::max();
  std::fill(read_bank_budgets_.begin(), read_bank_budgets_.end(), 0);
  std::fill(write_bank_budgets_.begin(), write_bank_budgets_.end(), 0);
  pending_requests_.clear();
  granted_requests_.clear();
  pending_read_requests_.clear();
  pending_write_requests_.clear();
  assert_valid();
}

bool Tmem::port_request_is_write(PortRequestKind kind) const {
  switch (kind) {
  case PortRequestKind::RegionWrite:
    return true;
  case PortRequestKind::RegionRead:
    return false;
  default:
    std::abort();
  }
}

void Tmem::insert_pending_request(uint64_t tag, bool write) {
  auto& fifo = write ? pending_write_requests_ : pending_read_requests_;
  auto request_it = pending_requests_.find(tag);
  if (request_it == pending_requests_.end()) {
    return;
  }
  auto req_age = request_it->second.age;
  auto insert_it = fifo.begin();
  for (; insert_it != fifo.end(); ++insert_it) {
    auto other_it = pending_requests_.find(*insert_it);
    if (other_it == pending_requests_.end()) {
      break;
    }
    auto other_age = other_it->second.age;
    if (req_age < other_age || (req_age == other_age && tag < *insert_it)) {
      break;
    }
  }
  fifo.insert(insert_it, tag);
}

uint64_t Tmem::enqueue_port_request(uint64_t cycle, const PortRequestDesc& desc) {
  PortRequest request{};
  request.tag = next_request_tag_++;
  request.age = desc.age;
  request.submit_cycle = cycle;
  request.kind = desc.kind;
  request.packet_idx = desc.packet_idx;
  request.col_base = desc.col_base;
  request.col_span = desc.col_span;
  auto tag = request.tag;
  pending_requests_.emplace(tag, request);
  insert_pending_request(tag, port_request_is_write(desc.kind));
  return tag;
}

bool Tmem::request_granted(uint64_t tag) const {
  return granted_requests_.count(tag) != 0;
}

void Tmem::consume_request_grant(uint64_t tag) {
  granted_requests_.erase(tag);
}

bool Tmem::try_grant_request(const PortRequest& request, uint64_t cycle) {
  switch (request.kind) {
  case PortRequestKind::RegionRead:
    return try_acquire_region_packet(cycle, request.col_base, request.col_span, request.packet_idx, false);
  case PortRequestKind::RegionWrite:
    return try_acquire_region_packet(cycle, request.col_base, request.col_span, request.packet_idx, true);
  default:
    std::abort();
  }
}

void Tmem::arbitrate_requests(uint64_t cycle) {
  if (arbitration_cycle_ == cycle) {
    return;
  }
  arbitration_cycle_ = cycle;
  reset_port_budgets(cycle);

  auto arbitrate_fifo = [this, cycle](std::deque<uint64_t>& fifo) {
    while (!fifo.empty()) {
      auto tag = fifo.front();
      auto it = pending_requests_.find(tag);
      if (it == pending_requests_.end()) {
        fifo.pop_front();
        continue;
      }
      if (!try_grant_request(it->second, cycle)) {
        break;
      }
      granted_requests_.emplace(tag, it->second);
      pending_requests_.erase(it);
      fifo.pop_front();
      break;
    }
  };

  arbitrate_fifo(pending_read_requests_);
  arbitrate_fifo(pending_write_requests_);
}

bool Tmem::lookup_allocation(uint32_t handle, TmemAllocation** allocation) {
  auto it = allocations_.find(handle);
  if (it == allocations_.end() || !it->second.valid) {
    return false;
  }
  if (allocation) {
    *allocation = &it->second;
  }
  return true;
}

bool Tmem::lookup_allocation(uint32_t handle, const TmemAllocation** allocation) const {
  auto it = allocations_.find(handle);
  if (it == allocations_.end() || !it->second.valid) {
    return false;
  }
  if (allocation) {
    *allocation = &it->second;
  }
  return true;
}

bool Tmem::find_allocation_by_lane(uint32_t lane, uint32_t* col_base) const {
  if (lane >= kPayloadCols) {
    return false;
  }
  for (auto& kv : allocations_) {
    auto& a = kv.second;
    if (!a.valid) continue;
    if (lane >= a.payload_col_base
     && lane < (a.payload_col_base + a.col_span)) {
      if (col_base) *col_base = a.payload_col_base;
      return true;
    }
  }
  return false;
}

bool Tmem::region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const {
  return tmem_functional::region_query(col_base, col_span, kNumCols, kColBytes, size_bytes);
}

bool Tmem::query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  if (col_span) {
    *col_span = allocation->col_span;
  }
  return region_query(allocation->payload_col_base, allocation->col_span, size_bytes);
}

uint32_t Tmem::line_chunk_bank(uint32_t logical_line, uint32_t chunk_idx) const {
  return tmem_functional::line_chunk_bank(logical_line,
                                          chunk_idx,
                                          num_physical_banks_,
                                          bank_swizzle_base_stride_,
                                          kPacketLanes);
}

uint32_t Tmem::packet_lane_bank(uint32_t logical_col,
                                uint32_t packet_in_col,
                                uint32_t lane) const {
  return tmem_functional::packet_lane_bank(logical_col,
                                           packet_in_col,
                                           lane,
                                           num_physical_banks_,
                                           kPhysicalBankBytes,
                                           kLogicalLines,
                                           bank_swizzle_base_stride_,
                                           kPacketLanes);
}

void Tmem::logical_byte_to_physical(uint32_t logical_col,
                                    uint32_t logical_line,
                                    uint32_t* bank,
                                    uint32_t* row,
                                    uint32_t* bank_byte) const {
  tmem_functional::logical_byte_to_physical(logical_col,
                                            logical_line,
                                            num_physical_banks_,
                                            kPhysicalBankBytes,
                                            bank_swizzle_base_stride_,
                                            kPacketLanes,
                                            bank,
                                            row,
                                            bank_byte);
}

uint8_t Tmem::read_logical_byte(uint32_t logical_col, uint32_t logical_line) const {
  if (logical_col >= kNumCols || logical_line >= kLogicalLines) {
    return 0;
  }
  uint32_t bank = 0;
  uint32_t row = 0;
  uint32_t bank_byte = 0;
  logical_byte_to_physical(logical_col, logical_line, &bank, &row, &bank_byte);
  return banks_.at(bank).at(row).at(bank_byte);
}

void Tmem::write_logical_byte(uint32_t logical_col, uint32_t logical_line, uint8_t value) {
  if (logical_col >= kNumCols || logical_line >= kLogicalLines) {
    return;
  }
  uint32_t bank = 0;
  uint32_t row = 0;
  uint32_t bank_byte = 0;
  logical_byte_to_physical(logical_col, logical_line, &bank, &row, &bank_byte);
  banks_.at(bank).at(row).at(bank_byte) = value;
}

uint8_t Tmem::read_region_byte(uint32_t col_base, uint32_t byte_offset) const {
  auto logical_col = col_base + (byte_offset / kColBytes);
  auto logical_line = byte_offset % kColBytes;
  return read_logical_byte(logical_col, logical_line);
}

void Tmem::write_region_byte(uint32_t col_base, uint32_t byte_offset, uint8_t value) {
  auto logical_col = col_base + (byte_offset / kColBytes);
  auto logical_line = byte_offset % kColBytes;
  write_logical_byte(logical_col, logical_line, value);
}

bool Tmem::region_copy_in(uint32_t col_base, uint32_t col_span, const uint8_t* data, uint32_t size_bytes) {
  uint32_t capacity = 0;
  if (!region_query(col_base, col_span, &capacity) || (size_bytes != 0 && nullptr == data) || size_bytes > capacity) {
    return false;
  }

  for (uint32_t offset = 0; offset < size_bytes; ++offset) {
    write_region_byte(col_base, offset, data[offset]);
  }
  for (uint32_t offset = size_bytes; offset < capacity; ++offset) {
    write_region_byte(col_base, offset, 0);
  }

  assert_valid();
  return true;
}

bool Tmem::region_copy_out(uint32_t col_base, uint32_t col_span, uint8_t* data, uint32_t size_bytes) const {
  uint32_t capacity = 0;
  if (!region_query(col_base, col_span, &capacity) || (size_bytes != 0 && nullptr == data) || size_bytes > capacity) {
    return false;
  }

  for (uint32_t offset = 0; offset < size_bytes; ++offset) {
    data[offset] = read_region_byte(col_base, offset);
  }
  return true;
}

// Phase-3.3.1 GAP-4: handle-based byte-range read/write for tcgen05.ld/st.
// Resolves handle to (col_base, col_span), then walks bytes through the
// existing private byte API. byte_offset is within the allocation's payload
// region [0, col_span * kColBytes).
bool Tmem::handle_region_read_bytes(uint32_t handle, uint32_t byte_offset, uint8_t* dst, uint32_t bytes) const {
  const TmemAllocation* alloc = nullptr;
  if (!lookup_allocation(handle, &alloc) || nullptr == alloc || nullptr == dst) {
    return false;
  }
  uint32_t capacity = alloc->col_span * kColBytes;
  if (byte_offset + bytes > capacity) {
    return false;
  }
  for (uint32_t i = 0; i < bytes; ++i) {
    dst[i] = read_region_byte(alloc->payload_col_base, byte_offset + i);
  }
  return true;
}

bool Tmem::handle_region_write_bytes(uint32_t handle, uint32_t byte_offset, const uint8_t* src, uint32_t bytes) {
  TmemAllocation* alloc = nullptr;
  if (!lookup_allocation(handle, &alloc) || nullptr == alloc || nullptr == src) {
    return false;
  }
  uint32_t capacity = alloc->col_span * kColBytes;
  if (byte_offset + bytes > capacity) {
    return false;
  }
  for (uint32_t i = 0; i < bytes; ++i) {
    write_region_byte(alloc->payload_col_base, byte_offset + i, src[i]);
  }
  assert_valid();
  return true;
}

bool Tmem::region_read_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, TmemPacket* out) const {
  if (nullptr == out) {
    return false;
  }
  uint32_t size_bytes = 0;
  if (!region_query(col_base, col_span, &size_bytes)) {
    return false;
  }
  uint32_t offset = packet_idx * kPacketBytes;
  if (offset + kPacketBytes > size_bytes) {
    return false;
  }
  for (uint32_t i = 0; i < kPacketBytes; ++i) {
    out->bytes[i] = read_region_byte(col_base, offset + i);
  }
  return true;
}

bool Tmem::region_packet_location(uint32_t col_base,
                                  uint32_t col_span,
                                  uint32_t packet_idx,
                                  uint32_t* logical_col) {
  return tmem_functional::region_packet_location(col_base,
                                                 col_span,
                                                 packet_idx,
                                                 kNumCols,
                                                 kColBytes,
                                                 kPacketBytes,
                                                 logical_col);
}

bool Tmem::region_write_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, const TmemPacket& in) {
  uint32_t size_bytes = 0;
  if (!region_query(col_base, col_span, &size_bytes)) {
    return false;
  }
  uint32_t offset = packet_idx * kPacketBytes;
  if (offset + kPacketBytes > size_bytes) {
    return false;
  }
  for (uint32_t i = 0; i < kPacketBytes; ++i) {
    write_region_byte(col_base, offset + i, in.bytes[i]);
  }
  assert_valid();
  return true;
}

bool Tmem::copy_in(uint32_t handle, const uint8_t* data, uint32_t size_bytes) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  auto success = region_copy_in(allocation->payload_col_base, allocation->col_span, data, size_bytes);
  if (success) {
    allocation->payload_ready = true;
  }
  return success;
}

bool Tmem::copy_out(uint32_t handle, uint8_t* data, uint32_t size_bytes) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  return region_copy_out(allocation->payload_col_base, allocation->col_span, data, size_bytes);
}

bool Tmem::region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes) {
  (void)row_bytes;
  if (col_span == 0 || col_base >= kNumCols || (col_base + col_span) > kNumCols) {
    return false;
  }
  for (uint32_t line = kLogicalLines - 1; line > 0; --line) {
    for (uint32_t col = 0; col < col_span; ++col) {
      auto value = read_logical_byte(col_base + col, line - 1);
      write_logical_byte(col_base + col, line, value);
    }
  }
  for (uint32_t col = 0; col < col_span; ++col) {
    write_logical_byte(col_base + col, 0, 0);
  }
  assert_valid();
  return true;
}

void Tmem::clear_region(uint32_t col_base, uint32_t col_span) {
  uint32_t size_bytes = 0;
  if (!region_query(col_base, col_span, &size_bytes)) {
    return;
  }
  for (uint32_t offset = 0; offset < size_bytes; ++offset) {
    write_region_byte(col_base, offset, 0);
  }
}

uint32_t Tmem::alloc(uint32_t col_span) {
  // Phase-3.4 Stage 0: PTX-strict taddr encoding.
  // tcgen05.alloc returns a TADDR (PTX §9.7.16.1): bits[15:0]=lane base,
  // bits[31:16]=col_byte offset (=0 for fresh alloc). Vortex maps PTX lane
  // axis to its internal logical_col axis, so the returned taddr is just
  // the chosen col_base (lane base) directly. allocations_ is now keyed by
  // col_base instead of an opaque handle counter.
  // Failure sentinel: kInvalidTaddr (0xFFFFFFFFu) — col_base 0 is a valid taddr.
  constexpr uint32_t granularity = 16;
  if (col_span == 0) {
    return kInvalidTaddr;
  }
  col_span = ceil_div(col_span, granularity) * granularity;
  if (allocator_sealed_ || col_span > kPayloadCols) {
    return kInvalidTaddr;
  }

  for (uint32_t start_col = 0; start_col + col_span <= kPayloadCols; ++start_col) {
    bool available = true;
    for (uint32_t i = 0; i < col_span; ++i) {
      if (payload_col_allocs_.at(start_col + i)) {
        available = false;
        break;
      }
    }
    if (!available) {
      continue;
    }

    for (uint32_t i = 0; i < col_span; ++i) {
      payload_col_allocs_.at(start_col + i) = true;
    }
    clear_region(start_col, col_span);

    // Key allocations_ by col_base so the kernel-visible taddr round-trips
    // back to its own metadata via lookup_allocation(taddr).
    uint32_t taddr = start_col;
    TmemAllocation allocation{};
    allocation.valid = true;
    allocation.payload_col_base = start_col;
    allocation.col_span = col_span;
    allocation.row_bytes = col_span;
    allocation.visible_payload_ready = allocation.payload_ready;
    allocation.visible_meta_ready = allocation.meta_ready;
    allocations_[taddr] = allocation;
    assert_valid();
    return taddr;
  }

  return kInvalidTaddr;
}

bool Tmem::free(uint32_t handle) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }

  clear_region(allocation->payload_col_base, allocation->col_span);
  for (uint32_t i = 0; i < allocation->col_span; ++i) {
    payload_col_allocs_.at(allocation->payload_col_base + i) = false;
  }
  allocations_.erase(handle);
  assert_valid();
  return true;
}

void Tmem::seal_allocator() {
  allocator_sealed_ = true;
}

bool Tmem::allocator_sealed() const {
  return allocator_sealed_;
}

void Tmem::set_payload_ready(uint32_t handle, bool ready) {
  if (TmemAllocation* allocation = nullptr; lookup_allocation(handle, &allocation)) {
    allocation->payload_ready = ready;
  }
}

void Tmem::set_meta_ready(uint32_t handle, bool ready) {
  if (TmemAllocation* allocation = nullptr; lookup_allocation(handle, &allocation)) {
    allocation->meta_ready = ready;
  }
}

void Tmem::publish_visible_state() {
  for (auto& entry : allocations_) {
    auto& allocation = entry.second;
    allocation.visible_payload_ready = allocation.payload_ready;
    allocation.visible_meta_ready = allocation.meta_ready;
  }
}

void Tmem::set_meta_region(uint32_t handle, uint32_t meta_col_base, uint32_t meta_col_span) {
  if (TmemAllocation* allocation = nullptr; lookup_allocation(handle, &allocation)) {
    allocation->meta_col_base = meta_col_base;
    allocation->meta_col_span = meta_col_span;
  }
  assert_valid();
}

bool Tmem::set_row_bytes(uint32_t handle, uint32_t row_bytes) {
  if (TmemAllocation* allocation = nullptr; lookup_allocation(handle, &allocation)) {
    auto clamped = std::min<uint32_t>(row_bytes, allocation->col_span * kColBytes);
    if (0 == clamped) {
      return false;
    }
    allocation->row_bytes = clamped;
    assert_valid();
    return true;
  }
  return false;
}

bool Tmem::row_bytes(uint32_t handle, uint32_t* row_bytes) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || nullptr == row_bytes) {
    return false;
  }
  *row_bytes = allocation->row_bytes;
  return true;
}

void Tmem::reset_port_budgets(uint64_t cycle) {
  port_cycle_ = cycle;
  tmem_timing::reset_port_budgets(kReadPacketsPerCycle,
                                  kWritePacketsPerCycle,
                                  kReadPortsPerBank,
                                  kWritePortsPerBank,
                                  &read_packet_budget_,
                                  &write_packet_budget_,
                                  &read_bank_budgets_,
                                  &write_bank_budgets_);
}

void Tmem::ensure_port_budgets(uint64_t cycle) {
  if (port_cycle_ != cycle) {
    reset_port_budgets(cycle);
  }
}

bool Tmem::try_acquire_region_packet(uint64_t cycle,
                                     uint32_t col_base,
                                     uint32_t col_span,
                                     uint32_t packet_idx,
                                     bool write) {
  ensure_port_budgets(cycle);
  uint32_t logical_col = 0;
  if (!region_packet_location(col_base, col_span, packet_idx, &logical_col)) {
    return false;
  }

  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t line = 0; line < kLogicalLines; ++line) {
    uint32_t bank = 0;
    logical_byte_to_physical(logical_col, line, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  return tmem_timing::consume_packet_ports(touched,
                                           write,
                                           &read_packet_budget_,
                                           &write_packet_budget_,
                                           &read_bank_budgets_,
                                           &write_bank_budgets_);
}

void Tmem::refund_region_packet(uint64_t cycle,
                                uint32_t col_base,
                                uint32_t col_span,
                                uint32_t packet_idx,
                                bool write) {
  ensure_port_budgets(cycle);
  uint32_t logical_col = 0;
  if (!region_packet_location(col_base, col_span, packet_idx, &logical_col)) {
    return;
  }
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t line = 0; line < kLogicalLines; ++line) {
    uint32_t bank = 0;
    logical_byte_to_physical(logical_col, line, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  tmem_timing::refund_packet_ports(touched,
                                   write,
                                   kReadPacketsPerCycle,
                                   kWritePacketsPerCycle,
                                   kReadPortsPerBank,
                                   kWritePortsPerBank,
                                   &read_packet_budget_,
                                   &write_packet_budget_,
                                   &read_bank_budgets_,
                                   &write_bank_budgets_);
}

bool Tmem::try_acquire_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  return try_acquire_region_packet(cycle, col_base, col_span, packet_idx, false);
}

bool Tmem::try_acquire_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  return try_acquire_region_packet(cycle, col_base, col_span, packet_idx, true);
}

void Tmem::refund_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  refund_region_packet(cycle, col_base, col_span, packet_idx, false);
}

void Tmem::refund_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  refund_region_packet(cycle, col_base, col_span, packet_idx, true);
}

bool Tmem::validate(std::string* reason) const {
  std::array<bool, kPayloadCols> seen_payload{};
  seen_payload.fill(false);

  for (const auto& entry : allocations_) {
    const auto& allocation = entry.second;
    if (!allocation.valid) {
      if (reason) {
        *reason = "encountered invalid TMEM allocation entry";
      }
      return false;
    }

    if (0 == entry.first) {
      if (reason) {
        *reason = "TMEM handle 0 is reserved but present in the allocation table";
      }
      return false;
    }

    if (allocation.col_span == 0
     || allocation.payload_col_base >= kPayloadCols
     || (allocation.payload_col_base + allocation.col_span) > kPayloadCols) {
      if (reason) {
        *reason = "TMEM payload column allocation is outside the payload window";
      }
      return false;
    }

    if (allocation.row_bytes == 0 || allocation.row_bytes > allocation.col_span * kColBytes) {
      if (reason) {
        *reason = "TMEM allocation row_bytes is outside the allocated payload capacity";
      }
      return false;
    }

    for (uint32_t i = 0; i < allocation.col_span; ++i) {
      auto col = allocation.payload_col_base + i;
      if (seen_payload.at(col)) {
        if (reason) {
          *reason = "TMEM payload column overlap detected";
        }
        return false;
      }
      seen_payload.at(col) = true;
    }

    if (allocation.meta_col_span != 0) {
      if ((allocation.meta_col_base + allocation.meta_col_span) > kNumCols) {
        if (reason) {
          *reason = "TMEM meta column range is outside the TMEM logical view";
        }
        return false;
      }
    }
  }

  for (uint32_t col = 0; col < kPayloadCols; ++col) {
    if (payload_col_allocs_.at(col) != seen_payload.at(col)) {
      if (reason) {
        std::ostringstream oss;
        oss << "TMEM allocator bookkeeping mismatch on payload column " << col;
        *reason = oss.str();
      }
      return false;
    }
  }

  return true;
}

void Tmem::assert_valid() const {
#ifndef NDEBUG
  std::string reason;
  auto valid = this->validate(&reason);
  assert(valid && "TMEM state validation failed");
  (void)reason;
#endif
}
