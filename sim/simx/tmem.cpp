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
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <vector>
#include "tensor_cfg.h"
#include "tmem.h"

using namespace vortex;

namespace {

uint32_t configured_tmem_bank_count() {
  constexpr uint32_t kDefault = Tmem::kDefaultPhysicalBanks;
  auto value = std::getenv("VORTEX_SIMX_TMEM_BANKS");
  if (nullptr == value || '\0' == value[0]) {
    return kDefault;
  }
  char* end = nullptr;
  auto parsed = std::strtoul(value, &end, 0);
  if (end == value
   || *end != '\0'
   || parsed < Tmem::kPacketLanes
   || parsed > 256) {
    return kDefault;
  }
  return static_cast<uint32_t>(parsed);
}

uint32_t choose_coprime_stride(uint32_t count, uint32_t preferred) {
  if (count <= 1) {
    return 1;
  }

  preferred = std::min<uint32_t>(std::max<uint32_t>(1, preferred), count - 1);

  for (uint32_t candidate = preferred; candidate >= 1; --candidate) {
    if (std::gcd(candidate, count) == 1) {
      return candidate;
    }
    if (candidate == 1) {
      break;
    }
  }

  for (uint32_t candidate = preferred + 1; candidate < count; ++candidate) {
    if (std::gcd(candidate, count) == 1) {
      return candidate;
    }
  }

  return 1;
}

uint32_t fmt_bytes(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
  case vortex::tensor::uint8::id:
    return 1;
  case vortex::tensor::fp16::id:
  case vortex::tensor::bf16::id:
    return 2;
  case vortex::tensor::fp32::id:
    return 4;
  default:
    return 0;
  }
}

bool uses_single_legacy_linear_window(const TmemAllocation& allocation) {
  if (!allocation.layout_valid || allocation.windows.size() != 1) {
    return false;
  }
  const auto& window = allocation.windows.front();
  return window.window_id == 0
      && window.logical_col_span != 0
      && window.logical_line_span != 0;
}

} // namespace

Tmem::Tmem()
  : num_physical_banks_(configured_tmem_bank_count())
  , bank_slice_bytes_(kPhysicalBankBytes)
  , bank_swizzle_base_stride_(choose_coprime_stride(num_physical_banks_, (num_physical_banks_ / 4) + 1))
  , bank_swizzle_lane_stride_(choose_coprime_stride(num_physical_banks_, (num_physical_banks_ / 2) - 1))
  , allocator_sealed_(false)
  , next_handle_(1)
  , port_cycle_(std::numeric_limits<uint64_t>::max())
  , read_packet_budget_(0)
  , write_packet_budget_(0) {
  this->resize_storage();
  this->reset();
}

uint32_t Tmem::ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

void Tmem::resize_storage() {
  banks_.assign(num_physical_banks_, std::vector<PhysicalRow>(kPhysicalRows));
  for (auto& bank : banks_) {
    for (auto& row : bank) {
      row.assign(bank_slice_bytes_, 0);
    }
  }
  read_bank_budgets_.assign(num_physical_banks_, 0);
  write_bank_budgets_.assign(num_physical_banks_, 0);
}

void Tmem::reset() {
  for (auto& bank : banks_) {
    for (auto& row : bank) {
      std::fill(row.begin(), row.end(), 0);
    }
  }
  payload_col_allocs_.fill(false);
  allocations_.clear();
  allocator_sealed_ = false;
  next_handle_ = 1;
  port_cycle_ = std::numeric_limits<uint64_t>::max();
  read_packet_budget_ = 0;
  write_packet_budget_ = 0;
  std::fill(read_bank_budgets_.begin(), read_bank_budgets_.end(), 0);
  std::fill(write_bank_budgets_.begin(), write_bank_budgets_.end(), 0);
  assert_valid();
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

bool Tmem::region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const {
  if (col_span == 0 || col_base >= kNumCols || (col_base + col_span) > kNumCols) {
    return false;
  }
  if (size_bytes) {
    *size_bytes = col_span * kColBytes;
  }
  return true;
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
  auto physical_row = logical_line / 2;
  auto line_slot = logical_line % 2;
  auto base = (physical_row * bank_swizzle_base_stride_) % num_physical_banks_;
  return (base + line_slot * kPacketLanes + chunk_idx) % num_physical_banks_;
}

uint32_t Tmem::packet_lane_bank(uint32_t logical_col,
                                uint32_t packet_in_col,
                                uint32_t lane) const {
  auto logical_line = packet_in_col * kPacketLanes + lane;
  return line_chunk_bank(logical_line % kLogicalLines, logical_col / kPhysicalBankBytes);
}

void Tmem::logical_byte_to_physical(uint32_t logical_col,
                                    uint32_t logical_line,
                                    uint32_t* bank,
                                    uint32_t* row,
                                    uint32_t* bank_byte) const {
  auto chunk_idx = logical_col / kPhysicalBankBytes;
  if (bank) {
    *bank = line_chunk_bank(logical_line, chunk_idx);
  }
  if (row) {
    *row = logical_line / 2;
  }
  if (bank_byte) {
    *bank_byte = logical_col % kPhysicalBankBytes;
  }
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
  if (col_span == 0 || col_base >= kNumCols || (col_base + col_span) > kNumCols) {
    return false;
  }
  uint32_t size_bytes = col_span * kColBytes;
  uint32_t offset = packet_idx * kPacketBytes;
  if (offset + kPacketBytes > size_bytes) {
    return false;
  }
  if (logical_col) {
    *logical_col = col_base + (offset / kColBytes);
  }
  return true;
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
  auto granularity = TmemWindowPlanner::kMinAllocationCols;
  if (col_span == 0) {
    return 0;
  }
  col_span = ceil_div(col_span, granularity) * granularity;
  if (allocator_sealed_ || col_span > kPayloadCols) {
    return 0;
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

    auto handle = next_handle_++;
    TmemAllocation allocation{};
    allocation.valid = true;
    allocation.payload_col_base = start_col;
    allocation.col_span = col_span;
    allocation.row_bytes = col_span;
    allocations_[handle] = allocation;
    assert_valid();
    return handle;
  }

  return 0;
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

bool Tmem::read_packet(uint32_t handle, uint32_t packet_idx, TmemPacket* out) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  if (uses_single_legacy_linear_window(*allocation)) {
    return read_window_linear_packet(handle, 0, packet_idx, out);
  }
  return region_read_packet(allocation->payload_col_base, allocation->col_span, packet_idx, out);
}

bool Tmem::read_meta_packet(uint32_t handle, uint32_t packet_idx, TmemPacket* out) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || allocation->meta_col_span == 0) {
    return false;
  }
  return region_read_packet(allocation->meta_col_base, allocation->meta_col_span, packet_idx, out);
}

uint32_t Tmem::resolve_window_col_base(const TmemAllocation& allocation,
                                       const TmemWindowPlan& window) const {
  if ((window.logical_col_base + window.logical_col_span) <= allocation.col_span) {
    return allocation.payload_col_base + window.logical_col_base;
  }
  return window.logical_col_base;
}

bool Tmem::window_packet_location(uint32_t handle,
                                  uint32_t window_id,
                                  uint32_t packet_idx,
                                  uint32_t* logical_col_base,
                                  uint32_t* logical_line_base,
                                  uint32_t* logical_col_span,
                                  uint32_t* logical_line_span) const {
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation)
   || !lookup_window(handle, window_id, &window)
   || window->logical_packet_col_span == 0
   || window->logical_packet_line_span == 0) {
    return false;
  }

  auto packets_per_row = ceil_div(window->logical_col_span, window->logical_packet_col_span);
  auto packets_per_col = ceil_div(window->logical_line_span, window->logical_packet_line_span);
  auto packet_count = packets_per_row * packets_per_col;
  if (packet_idx >= packet_count) {
    return false;
  }

  auto packet_row = packet_idx / packets_per_row;
  auto packet_col = packet_idx % packets_per_row;
  auto line_offset = packet_row * window->logical_packet_line_span;
  auto col_offset = packet_col * window->logical_packet_col_span;
  auto abs_col_base = resolve_window_col_base(*allocation, *window);
  auto abs_line_base = window->logical_line_base + line_offset;
  auto abs_col_span = std::min(window->logical_packet_col_span, window->logical_col_span - col_offset);
  auto abs_line_span = std::min(window->logical_packet_line_span, window->logical_line_span - line_offset);
  if ((abs_col_base + col_offset + abs_col_span) > kNumCols
   || (abs_line_base + abs_line_span) > kLogicalLines) {
    return false;
  }

  if (logical_col_base) {
    *logical_col_base = abs_col_base + col_offset;
  }
  if (logical_line_base) {
    *logical_line_base = abs_line_base;
  }
  if (logical_col_span) {
    *logical_col_span = abs_col_span;
  }
  if (logical_line_span) {
    *logical_line_span = abs_line_span;
  }
  return true;
}

bool Tmem::window_line_chunk_location(uint32_t handle,
                                      uint32_t window_id,
                                      uint32_t line_idx,
                                      uint32_t chunk_idx,
                                      uint32_t* logical_col_base,
                                      uint32_t* logical_line_base,
                                      uint32_t* logical_col_span) const {
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation)
   || !lookup_window(handle, window_id, &window)
   || window->logical_col_span == 0
   || window->logical_line_span == 0
   || line_idx >= window->logical_line_span) {
    return false;
  }

  auto chunks_per_line = ceil_div(window->logical_col_span, kPacketBytes);
  if (chunk_idx >= chunks_per_line) {
    return false;
  }

  auto col_offset = chunk_idx * kPacketBytes;
  auto abs_col_base = resolve_window_col_base(*allocation, *window);
  auto abs_line_base = window->logical_line_base + line_idx;
  auto abs_col_span = std::min(kPacketBytes, window->logical_col_span - col_offset);
  if ((abs_col_base + col_offset + abs_col_span) > kNumCols
   || abs_line_base >= kLogicalLines) {
    return false;
  }

  if (logical_col_base) {
    *logical_col_base = abs_col_base + col_offset;
  }
  if (logical_line_base) {
    *logical_line_base = abs_line_base;
  }
  if (logical_col_span) {
    *logical_col_span = abs_col_span;
  }
  return true;
}

bool Tmem::window_linear_packet_info(uint32_t handle,
                                     uint32_t window_id,
                                     uint32_t packet_idx,
                                     uint32_t* logical_col_base,
                                     uint32_t* logical_line_base,
                                     uint32_t* logical_col_span,
                                     uint32_t* byte_offset,
                                     uint32_t* valid_bytes) const {
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation)
   || !lookup_window(handle, window_id, &window)
   || window->logical_col_span == 0
   || window->logical_line_span == 0) {
    return false;
  }

  uint32_t total_bytes = window->logical_col_span * window->logical_line_span;
  uint32_t offset = packet_idx * kPacketBytes;
  if (offset >= total_bytes) {
    return false;
  }

  if (logical_col_base) {
    *logical_col_base = resolve_window_col_base(*allocation, *window);
  }
  if (logical_line_base) {
    *logical_line_base = window->logical_line_base;
  }
  if (logical_col_span) {
    *logical_col_span = window->logical_col_span;
  }
  if (byte_offset) {
    *byte_offset = offset;
  }
  if (valid_bytes) {
    *valid_bytes = std::min<uint32_t>(kPacketBytes, total_bytes - offset);
  }
  return true;
}

bool Tmem::read_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const {
  if (nullptr == out) {
    return false;
  }
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t line_span = 0;
  if (!window_packet_location(handle, window_id, packet_idx, &col_base, &line_base, &col_span, &line_span)) {
    return false;
  }
  out->bytes.fill(0);
  uint32_t offset = 0;
  for (uint32_t line = 0; line < line_span; ++line) {
    for (uint32_t col = 0; col < col_span; ++col) {
      out->bytes.at(offset++) = read_logical_byte(col_base + col, line_base + line);
    }
  }
  return true;
}

bool Tmem::write_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in) {
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t line_span = 0;
  if (!window_packet_location(handle, window_id, packet_idx, &col_base, &line_base, &col_span, &line_span)) {
    return false;
  }
  uint32_t offset = 0;
  for (uint32_t line = 0; line < line_span; ++line) {
    for (uint32_t col = 0; col < col_span; ++col) {
      write_logical_byte(col_base + col, line_base + line, in.bytes.at(offset++));
    }
  }
  assert_valid();
  return true;
}

bool Tmem::read_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const {
  if (nullptr == out) {
    return false;
  }
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t byte_offset = 0;
  uint32_t valid_bytes = 0;
  if (!window_linear_packet_info(handle, window_id, packet_idx,
                                 &col_base, &line_base, &col_span,
                                 &byte_offset, &valid_bytes)) {
    return false;
  }
  out->bytes.fill(0);
  for (uint32_t i = 0; i < valid_bytes; ++i) {
    auto local_offset = byte_offset + i;
    auto logical_line = line_base + (local_offset / col_span);
    auto logical_col = col_base + (local_offset % col_span);
    out->bytes.at(i) = read_logical_byte(logical_col, logical_line);
  }
  return true;
}

bool Tmem::write_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in) {
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t byte_offset = 0;
  uint32_t valid_bytes = 0;
  if (!window_linear_packet_info(handle, window_id, packet_idx,
                                 &col_base, &line_base, &col_span,
                                 &byte_offset, &valid_bytes)) {
    return false;
  }
  for (uint32_t i = 0; i < valid_bytes; ++i) {
    auto local_offset = byte_offset + i;
    auto logical_line = line_base + (local_offset / col_span);
    auto logical_col = col_base + (local_offset % col_span);
    write_logical_byte(logical_col, logical_line, in.bytes.at(i));
  }
  assert_valid();
  return true;
}

bool Tmem::window_packet_count(uint32_t handle, uint32_t window_id, uint32_t* count) const {
  if (nullptr == count) {
    return false;
  }
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation)
   || !lookup_window(handle, window_id, &window)
   || window->logical_packet_col_span == 0
   || window->logical_packet_line_span == 0) {
    return false;
  }
  auto packets_per_row = ceil_div(window->logical_col_span, window->logical_packet_col_span);
  auto packets_per_col = ceil_div(window->logical_line_span, window->logical_packet_line_span);
  *count = packets_per_row * packets_per_col;
  return true;
}

bool Tmem::window_line_chunk_count(uint32_t handle, uint32_t window_id, uint32_t* count) const {
  if (nullptr == count) {
    return false;
  }
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation)
   || !lookup_window(handle, window_id, &window)
   || window->logical_col_span == 0) {
    return false;
  }
  *count = ceil_div(window->logical_col_span, kPacketBytes);
  return true;
}

bool Tmem::write_window_line_chunk(uint32_t handle,
                                   uint32_t window_id,
                                   uint32_t line_idx,
                                   uint32_t chunk_idx,
                                   const TmemPacket& in) {
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  if (!window_line_chunk_location(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
    return false;
  }
  for (uint32_t col = 0; col < col_span; ++col) {
    write_logical_byte(col_base + col, line_base, in.bytes.at(col));
  }
  assert_valid();
  return true;
}

bool Tmem::shift_window_down(uint32_t handle, uint32_t window_id) {
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation) || !lookup_window(handle, window_id, &window)) {
    return false;
  }
  auto col_base = resolve_window_col_base(*allocation, *window);
  auto line_base = window->logical_line_base;
  auto col_span = window->logical_col_span;
  auto line_span = window->logical_line_span;
  if (col_span == 0 || line_span == 0
   || (col_base + col_span) > kNumCols
   || (line_base + line_span) > kLogicalLines) {
    return false;
  }
  for (uint32_t line = line_base + line_span - 1; line > line_base; --line) {
    for (uint32_t col = 0; col < col_span; ++col) {
      auto value = read_logical_byte(col_base + col, line - 1);
      write_logical_byte(col_base + col, line, value);
    }
  }
  for (uint32_t col = 0; col < col_span; ++col) {
    write_logical_byte(col_base + col, line_base, 0);
  }
  assert_valid();
  return true;
}

bool Tmem::shift_window_math_row_down(uint32_t handle,
                                      uint32_t window_id,
                                      const uint8_t* refill_row,
                                      uint32_t refill_size_bytes) {
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation) || !lookup_window(handle, window_id, &window)) {
    return false;
  }

  auto elem_bytes = fmt_bytes(window->fmt);
  if (0 == elem_bytes || window->elem_shape.empty() || !TmemWindowPlanner::uses_math_packet_adapter(*window)) {
    if (!shift_window_down(handle, window_id)) {
      return false;
    }
    if (refill_row != nullptr && refill_size_bytes != 0) {
      auto col_base = resolve_window_col_base(*allocation, *window);
      auto top_line = window->logical_line_base;
      auto copy_bytes = std::min<uint32_t>(window->logical_col_span, refill_size_bytes);
      for (uint32_t i = 0; i < copy_bytes; ++i) {
        write_logical_byte(col_base + i, top_line, refill_row[i]);
      }
    }
    assert_valid();
    return true;
  }

  auto rows = std::max<uint32_t>(1, window->elem_shape.rows);
  auto cols = std::max<uint32_t>(1, window->elem_shape.cols);
  auto row_bytes = cols * elem_bytes;
  std::vector<uint8_t> math_bytes(rows * row_bytes, 0);
  for (uint32_t tile_idx = 0; tile_idx < window->tile_count; ++tile_idx) {
    for (uint32_t local_packet_idx = 0; local_packet_idx < window->packets_per_tile; ++local_packet_idx) {
      auto packet_idx = tile_idx * window->packets_per_tile + local_packet_idx;
      TmemPacket packet;
      if (!read_window_packet(handle, window_id, packet_idx, &packet)) {
        return false;
      }
      if (!TmemWindowPlanner::unpack_math_packet(*window, packet_idx, packet.bytes, &math_bytes)) {
        return false;
      }
    }
  }

  if (rows > 1) {
    std::memmove(math_bytes.data() + row_bytes, math_bytes.data(), (rows - 1) * row_bytes);
  }
  std::fill_n(math_bytes.data(), row_bytes, 0);
  if (refill_row != nullptr && refill_size_bytes != 0) {
    std::copy_n(refill_row, std::min<uint32_t>(row_bytes, refill_size_bytes), math_bytes.data());
  }

  for (uint32_t tile_idx = 0; tile_idx < window->tile_count; ++tile_idx) {
    for (uint32_t local_packet_idx = 0; local_packet_idx < window->packets_per_tile; ++local_packet_idx) {
      auto packet_idx = tile_idx * window->packets_per_tile + local_packet_idx;
      TmemPacket packet;
      if (!TmemWindowPlanner::pack_math_packet(*window, math_bytes, packet_idx, &packet.bytes)) {
        return false;
      }
      if (!write_window_packet(handle, window_id, packet_idx, packet)) {
        return false;
      }
    }
  }

  assert_valid();
  return true;
}

bool Tmem::write_packet(uint32_t handle, uint32_t packet_idx, const TmemPacket& in) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  if (uses_single_legacy_linear_window(*allocation)) {
    if (!write_window_linear_packet(handle, 0, packet_idx, in)) {
      return false;
    }
    allocation->payload_ready = true;
    return true;
  }
  if (!region_write_packet(allocation->payload_col_base, allocation->col_span, packet_idx, in)) {
    return false;
  }
  allocation->payload_ready = true;
  return true;
}

bool Tmem::bind_layout(uint32_t handle, const TmemLayoutPlan& layout) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || !layout.valid || layout.required_col_span > allocation->col_span) {
    return false;
  }
  allocation->layout_valid = true;
  allocation->layout_epoch = layout.epoch;
  allocation->windows = layout.windows;
  return true;
}

bool Tmem::upsert_window(uint32_t handle, const TmemWindowPlan& window) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  auto it = std::find_if(allocation->windows.begin(), allocation->windows.end(),
                         [&window](const TmemWindowPlan& existing) {
                           return existing.window_id == window.window_id;
                         });
  if (it == allocation->windows.end()) {
    allocation->windows.push_back(window);
  } else {
    *it = window;
  }
  allocation->layout_valid = !allocation->windows.empty();
  return true;
}

bool Tmem::lookup_window(uint32_t handle, uint32_t window_id, const TmemWindowPlan** out) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || !allocation->layout_valid) {
    return false;
  }
  auto it = std::find_if(allocation->windows.begin(), allocation->windows.end(),
                         [window_id](const TmemWindowPlan& window) {
                           return window.window_id == window_id;
                         });
  if (it == allocation->windows.end()) {
    return false;
  }
  if (out) {
    *out = &(*it);
  }
  return true;
}

bool Tmem::lookup_window(uint32_t handle, uint32_t window_id, TmemWindowPlan** out) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || !allocation->layout_valid) {
    return false;
  }
  auto it = std::find_if(allocation->windows.begin(), allocation->windows.end(),
                         [window_id](const TmemWindowPlan& window) {
                           return window.window_id == window_id;
                         });
  if (it == allocation->windows.end()) {
    return false;
  }
  if (out) {
    *out = &(*it);
  }
  return true;
}

bool Tmem::window_epoch(uint32_t handle, uint32_t* epoch) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || !allocation->layout_valid || nullptr == epoch) {
    return false;
  }
  *epoch = allocation->layout_epoch;
  return true;
}

bool Tmem::bump_window_epoch(uint32_t handle) {
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  ++allocation->layout_epoch;
  allocation->layout_valid = !allocation->windows.empty();
  return true;
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
  read_packet_budget_ = kReadPacketsPerCycle;
  write_packet_budget_ = kWritePacketsPerCycle;
  std::fill(read_bank_budgets_.begin(), read_bank_budgets_.end(), kReadPortsPerBank);
  std::fill(write_bank_budgets_.begin(), write_bank_budgets_.end(), kWritePortsPerBank);
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

  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  if (0 == packet_budget) {
    return false;
  }

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t line = 0; line < kLogicalLines; ++line) {
    uint32_t bank = 0;
    logical_byte_to_physical(logical_col, line, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank) && 0 == budgets.at(bank)) {
      return false;
    }
  }
  --packet_budget;
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      --budgets.at(bank);
    }
  }
  return true;
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
  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  auto max_packet_budget = write ? kWritePacketsPerCycle : kReadPacketsPerCycle;
  packet_budget = std::min<uint32_t>(packet_budget + 1, max_packet_budget);

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  auto max_budget = write ? kWritePortsPerBank : kReadPortsPerBank;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t line = 0; line < kLogicalLines; ++line) {
    uint32_t bank = 0;
    logical_byte_to_physical(logical_col, line, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      budgets.at(bank) = std::min<uint32_t>(budgets.at(bank) + 1, max_budget);
    }
  }
}

bool Tmem::try_acquire_window_packet(uint64_t cycle,
                                     uint32_t handle,
                                     uint32_t window_id,
                                     uint32_t packet_idx,
                                     bool write) {
  ensure_port_budgets(cycle);
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t line_span = 0;
  if (!window_packet_location(handle, window_id, packet_idx, &col_base, &line_base, &col_span, &line_span)) {
    return false;
  }

  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  if (0 == packet_budget) {
    return false;
  }

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t line = 0; line < line_span; ++line) {
    for (uint32_t col = 0; col < col_span; ++col) {
      uint32_t bank = 0;
      logical_byte_to_physical(col_base + col, line_base + line, &bank, nullptr, nullptr);
      touched.at(bank) = true;
    }
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank) && 0 == budgets.at(bank)) {
      return false;
    }
  }
  --packet_budget;
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      --budgets.at(bank);
    }
  }
  return true;
}

bool Tmem::try_acquire_window_linear_packet(uint64_t cycle,
                                            uint32_t handle,
                                            uint32_t window_id,
                                            uint32_t packet_idx,
                                            bool write) {
  ensure_port_budgets(cycle);
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t byte_offset = 0;
  uint32_t valid_bytes = 0;
  if (!window_linear_packet_info(handle, window_id, packet_idx,
                                 &col_base, &line_base, &col_span,
                                 &byte_offset, &valid_bytes)) {
    return false;
  }

  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  if (0 == packet_budget) {
    return false;
  }

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t i = 0; i < valid_bytes; ++i) {
    auto local_offset = byte_offset + i;
    auto logical_line = line_base + (local_offset / col_span);
    auto logical_col = col_base + (local_offset % col_span);
    uint32_t bank = 0;
    logical_byte_to_physical(logical_col, logical_line, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank) && 0 == budgets.at(bank)) {
      return false;
    }
  }
  --packet_budget;
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      --budgets.at(bank);
    }
  }
  return true;
}

bool Tmem::try_acquire_window_line_chunk(uint64_t cycle,
                                         uint32_t handle,
                                         uint32_t window_id,
                                         uint32_t line_idx,
                                         uint32_t chunk_idx,
                                         bool write) {
  ensure_port_budgets(cycle);
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  if (!window_line_chunk_location(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
    return false;
  }

  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  if (0 == packet_budget) {
    return false;
  }

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t col = 0; col < col_span; ++col) {
    uint32_t bank = 0;
    logical_byte_to_physical(col_base + col, line_base, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank) && 0 == budgets.at(bank)) {
      return false;
    }
  }
  --packet_budget;
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      --budgets.at(bank);
    }
  }
  return true;
}

void Tmem::refund_window_packet(uint64_t cycle,
                                uint32_t handle,
                                uint32_t window_id,
                                uint32_t packet_idx,
                                bool write) {
  ensure_port_budgets(cycle);
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t line_span = 0;
  if (!window_packet_location(handle, window_id, packet_idx, &col_base, &line_base, &col_span, &line_span)) {
    return;
  }
  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  auto max_packet_budget = write ? kWritePacketsPerCycle : kReadPacketsPerCycle;
  packet_budget = std::min<uint32_t>(packet_budget + 1, max_packet_budget);

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  auto max_budget = write ? kWritePortsPerBank : kReadPortsPerBank;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t line = 0; line < line_span; ++line) {
    for (uint32_t col = 0; col < col_span; ++col) {
      uint32_t bank = 0;
      logical_byte_to_physical(col_base + col, line_base + line, &bank, nullptr, nullptr);
      touched.at(bank) = true;
    }
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      budgets.at(bank) = std::min<uint32_t>(budgets.at(bank) + 1, max_budget);
    }
  }
}

void Tmem::refund_window_linear_packet(uint64_t cycle,
                                       uint32_t handle,
                                       uint32_t window_id,
                                       uint32_t packet_idx,
                                       bool write) {
  ensure_port_budgets(cycle);
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  uint32_t byte_offset = 0;
  uint32_t valid_bytes = 0;
  if (!window_linear_packet_info(handle, window_id, packet_idx,
                                 &col_base, &line_base, &col_span,
                                 &byte_offset, &valid_bytes)) {
    return;
  }
  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  auto max_packet_budget = write ? kWritePacketsPerCycle : kReadPacketsPerCycle;
  packet_budget = std::min<uint32_t>(packet_budget + 1, max_packet_budget);

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  auto max_budget = write ? kWritePortsPerBank : kReadPortsPerBank;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t i = 0; i < valid_bytes; ++i) {
    auto local_offset = byte_offset + i;
    auto logical_line = line_base + (local_offset / col_span);
    auto logical_col = col_base + (local_offset % col_span);
    uint32_t bank = 0;
    logical_byte_to_physical(logical_col, logical_line, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      budgets.at(bank) = std::min<uint32_t>(budgets.at(bank) + 1, max_budget);
    }
  }
}

void Tmem::refund_window_line_chunk(uint64_t cycle,
                                    uint32_t handle,
                                    uint32_t window_id,
                                    uint32_t line_idx,
                                    uint32_t chunk_idx,
                                    bool write) {
  ensure_port_budgets(cycle);
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  if (!window_line_chunk_location(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
    return;
  }
  auto& packet_budget = write ? write_packet_budget_ : read_packet_budget_;
  auto max_packet_budget = write ? kWritePacketsPerCycle : kReadPacketsPerCycle;
  packet_budget = std::min<uint32_t>(packet_budget + 1, max_packet_budget);

  auto& budgets = write ? write_bank_budgets_ : read_bank_budgets_;
  auto max_budget = write ? kWritePortsPerBank : kReadPortsPerBank;
  std::vector<bool> touched(num_physical_banks_, false);
  for (uint32_t col = 0; col < col_span; ++col) {
    uint32_t bank = 0;
    logical_byte_to_physical(col_base + col, line_base, &bank, nullptr, nullptr);
    touched.at(bank) = true;
  }
  for (uint32_t bank = 0; bank < touched.size(); ++bank) {
    if (touched.at(bank)) {
      budgets.at(bank) = std::min<uint32_t>(budgets.at(bank) + 1, max_budget);
    }
  }
}

bool Tmem::try_acquire_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  return try_acquire_region_packet(cycle, col_base, col_span, packet_idx, false);
}

bool Tmem::try_acquire_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  return try_acquire_region_packet(cycle, col_base, col_span, packet_idx, true);
}

bool Tmem::try_acquire_window_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  return try_acquire_window_packet(cycle, handle, window_id, packet_idx, false);
}

bool Tmem::try_acquire_window_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  return try_acquire_window_packet(cycle, handle, window_id, packet_idx, true);
}

bool Tmem::try_acquire_window_linear_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  return try_acquire_window_linear_packet(cycle, handle, window_id, packet_idx, false);
}

bool Tmem::try_acquire_window_linear_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  return try_acquire_window_linear_packet(cycle, handle, window_id, packet_idx, true);
}

bool Tmem::try_acquire_window_line_write_packet(uint64_t cycle,
                                                uint32_t handle,
                                                uint32_t window_id,
                                                uint32_t line_idx,
                                                uint32_t chunk_idx) {
  return try_acquire_window_line_chunk(cycle, handle, window_id, line_idx, chunk_idx, true);
}

bool Tmem::try_acquire_read_packet(uint64_t cycle, uint32_t handle, uint32_t packet_idx) {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  if (uses_single_legacy_linear_window(*allocation)) {
    return try_acquire_window_linear_read_packet(cycle, handle, 0, packet_idx);
  }
  return try_acquire_region_read_packet(cycle, allocation->payload_col_base, allocation->col_span, packet_idx);
}

bool Tmem::try_acquire_meta_read_packet(uint64_t cycle, uint32_t handle, uint32_t packet_idx) {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation) || allocation->meta_col_span == 0) {
    return false;
  }
  return try_acquire_region_read_packet(cycle, allocation->meta_col_base, allocation->meta_col_span, packet_idx);
}

bool Tmem::try_acquire_write_packet(uint64_t cycle, uint32_t handle, uint32_t packet_idx) {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(handle, &allocation)) {
    return false;
  }
  if (uses_single_legacy_linear_window(*allocation)) {
    return try_acquire_window_linear_write_packet(cycle, handle, 0, packet_idx);
  }
  return try_acquire_region_write_packet(cycle, allocation->payload_col_base, allocation->col_span, packet_idx);
}

void Tmem::refund_region_read_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  refund_region_packet(cycle, col_base, col_span, packet_idx, false);
}

void Tmem::refund_region_write_packet(uint64_t cycle, uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  refund_region_packet(cycle, col_base, col_span, packet_idx, true);
}

void Tmem::refund_window_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  refund_window_packet(cycle, handle, window_id, packet_idx, false);
}

void Tmem::refund_window_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  refund_window_packet(cycle, handle, window_id, packet_idx, true);
}

void Tmem::refund_window_linear_read_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  refund_window_linear_packet(cycle, handle, window_id, packet_idx, false);
}

void Tmem::refund_window_linear_write_packet(uint64_t cycle, uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  refund_window_linear_packet(cycle, handle, window_id, packet_idx, true);
}

void Tmem::refund_window_line_write_packet(uint64_t cycle,
                                           uint32_t handle,
                                           uint32_t window_id,
                                           uint32_t line_idx,
                                           uint32_t chunk_idx) {
  refund_window_line_chunk(cycle, handle, window_id, line_idx, chunk_idx, true);
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
