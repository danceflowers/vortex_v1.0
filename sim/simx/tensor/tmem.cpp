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

// Tmem.cpp implements the back half of the tensor-memory path:
// - logical TMEM allocations and planned windows
// - logical (column, line) to physical (bank, row, byte) mapping
// - per-cycle ingress/egress arbitration
// - logical-line and math-row shift helpers used by TMEM_SHIFT

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
  , write_packet_budget_(0)
  , next_request_tag_(1)
  , arbitration_cycle_(std::numeric_limits<uint64_t>::max()) {
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
  case PortRequestKind::WindowWrite:
  case PortRequestKind::WindowLinearWrite:
  case PortRequestKind::WindowLineWrite:
    return true;
  case PortRequestKind::RegionRead:
  case PortRequestKind::WindowRead:
  case PortRequestKind::WindowLinearRead:
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
  request.handle = desc.handle;
  request.window_id = desc.window_id;
  request.packet_idx = desc.packet_idx;
  request.col_base = desc.col_base;
  request.col_span = desc.col_span;
  request.line_idx = desc.line_idx;
  request.chunk_idx = desc.chunk_idx;
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
  case PortRequestKind::WindowRead:
    return try_acquire_window_packet(cycle, request.handle, request.window_id, request.packet_idx, false);
  case PortRequestKind::WindowWrite:
    return try_acquire_window_packet(cycle, request.handle, request.window_id, request.packet_idx, true);
  case PortRequestKind::WindowLinearRead:
    return try_acquire_window_linear_packet(cycle, request.handle, request.window_id, request.packet_idx, false);
  case PortRequestKind::WindowLinearWrite:
    return try_acquire_window_linear_packet(cycle, request.handle, request.window_id, request.packet_idx, true);
  case PortRequestKind::WindowLineRead:
    return try_acquire_window_line_chunk(cycle, request.handle, request.window_id, request.line_idx, request.chunk_idx, false);
  case PortRequestKind::WindowLineWrite:
    return try_acquire_window_line_chunk(cycle, request.handle, request.window_id, request.line_idx, request.chunk_idx, true);
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
  // Phase-3.4 Stage 0: PTX-strict taddr encoding.
  // tcgen05.alloc returns a TADDR (PTX §9.7.16.1): bits[15:0]=lane base,
  // bits[31:16]=col_byte offset (=0 for fresh alloc). Vortex maps PTX lane
  // axis to its internal logical_col axis, so the returned taddr is just
  // the chosen col_base (lane base) directly. allocations_ is now keyed by
  // col_base instead of an opaque handle counter.
  // Failure sentinel: kInvalidTaddr (0xFFFFFFFFu) — col_base 0 is a valid taddr.
  auto granularity = TmemWindowPlanner::kMinAllocationCols;
  if (col_span == 0) {
    fprintf(stderr, "[TMEM_ALLOC_DBG] col_span==0, returning invalid\n");
    return kInvalidTaddr;
  }
  col_span = ceil_div(col_span, granularity) * granularity;
  if (allocator_sealed_ || col_span > kPayloadCols) {
    fprintf(stderr, "[TMEM_ALLOC_DBG] sealed=%d col_span=%u kPayloadCols=%u, invalid\n",
            (int)allocator_sealed_, col_span, kPayloadCols);
    return kInvalidTaddr;
  }
  fprintf(stderr, "[TMEM_ALLOC_DBG] entering scan loop col_span=%u\n", col_span);

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
    allocation.visible_layout_epoch = allocation.layout_epoch;
    allocation.visible_payload_ready = allocation.payload_ready;
    allocation.visible_meta_ready = allocation.meta_ready;
    allocations_[taddr] = allocation;
    fprintf(stderr, "[TMEM_ALLOC_DBG] success: taddr=%u col_span=%u (allocations_.size=%zu)\n",
            taddr, col_span, allocations_.size());
    assert_valid();
    return taddr;
  }

  fprintf(stderr, "[TMEM_ALLOC_DBG] no free range found for col_span=%u\n", col_span);
  for (uint32_t i = 0; i < kPayloadCols; ++i) {
    fprintf(stderr, "%c", payload_col_allocs_[i] ? '1' : '0');
  }
  fprintf(stderr, "\n");
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

// Resolve one logical line chunk within a planned window into an absolute
// logical TMEM region. This is used by top-line refill traffic, which is
// expressed as line chunks rather than generic tile packets.
bool Tmem::resolve_window_line_chunk_region(uint32_t handle,
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

// Resolve one packet in the window-linear view used by legacy traffic and
// line-oriented refill helpers. The returned byte_offset/valid_bytes describe
// where this packet falls inside the window's logical rectangle.
bool Tmem::resolve_window_linear_packet_region(uint32_t handle,
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
  if (!resolve_window_linear_packet_region(handle, window_id, packet_idx,
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
  if (!resolve_window_linear_packet_region(handle, window_id, packet_idx,
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

bool Tmem::read_window_line_chunk(uint32_t handle,
                                  uint32_t window_id,
                                  uint32_t line_idx,
                                  uint32_t chunk_idx,
                                  TmemPacket* out) const {
  if (nullptr == out) {
    return false;
  }
  uint32_t col_base = 0;
  uint32_t line_base = 0;
  uint32_t col_span = 0;
  if (!resolve_window_line_chunk_region(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
    return false;
  }
  out->bytes.fill(0);
  for (uint32_t col = 0; col < col_span; ++col) {
    out->bytes.at(col) = read_logical_byte(col_base + col, line_base);
  }
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
  if (!resolve_window_line_chunk_region(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
    return false;
  }
  for (uint32_t col = 0; col < col_span; ++col) {
    write_logical_byte(col_base + col, line_base, in.bytes.at(col));
  }
  assert_valid();
  return true;
}

bool Tmem::read_window_math_row(uint32_t handle,
                                uint32_t window_id,
                                uint32_t math_row,
                                uint8_t* out_row,
                                uint32_t row_bytes) const {
  if (nullptr == out_row || row_bytes == 0) {
    return false;
  }
  const TmemWindowPlan* window = nullptr;
  if (!lookup_window(handle, window_id, &window) || !TmemWindowPlanner::uses_math_packet_adapter(*window)) {
    return false;
  }
  auto elem_bytes = fmt_bytes(window->fmt);
  auto cols = window->elem_shape.cols;
  if (row_bytes != cols * elem_bytes) {
    return false;
  }
  std::fill_n(out_row, row_bytes, 0);
  auto packet_count = window->tile_count * window->packets_per_tile;
  for (uint32_t packet_idx = 0; packet_idx < packet_count; ++packet_idx) {
    TmemMathPacketRegion region{};
    if (!TmemWindowPlanner::packet_math_region(*window, packet_idx, &region)) {
      return false;
    }
    if (math_row < region.math_row_base || math_row >= (region.math_row_base + region.packet_rows)) {
      continue;
    }
    TmemPacket packet{};
    if (!read_window_packet(handle, window_id, packet_idx, &packet)) {
      return false;
    }
    auto packet_row = math_row - region.math_row_base;
    auto packet_row_byte_offset = packet_row * region.packet_cols * elem_bytes;
    auto valid_cols = (region.math_col_base < cols)
                    ? std::min<uint32_t>(region.packet_cols, cols - region.math_col_base)
                    : 0;
    auto valid_bytes = valid_cols * elem_bytes;
    if (valid_bytes != 0) {
      std::copy_n(packet.bytes.data() + packet_row_byte_offset,
                  valid_bytes,
                  out_row + region.math_col_base * elem_bytes);
    }
  }
  return true;
}

bool Tmem::write_window_math_row(uint32_t handle,
                                 uint32_t window_id,
                                 uint32_t math_row,
                                 const uint8_t* row_bytes,
                                 uint32_t row_size_bytes) {
  if (nullptr == row_bytes || row_size_bytes == 0) {
    return false;
  }
  const TmemWindowPlan* window = nullptr;
  if (!lookup_window(handle, window_id, &window) || !TmemWindowPlanner::uses_math_packet_adapter(*window)) {
    return false;
  }
  auto elem_bytes = fmt_bytes(window->fmt);
  auto cols = window->elem_shape.cols;
  if (row_size_bytes != cols * elem_bytes) {
    return false;
  }
  auto packet_count = window->tile_count * window->packets_per_tile;
  for (uint32_t packet_idx = 0; packet_idx < packet_count; ++packet_idx) {
    TmemMathPacketRegion region{};
    if (!TmemWindowPlanner::packet_math_region(*window, packet_idx, &region)) {
      return false;
    }
    if (math_row < region.math_row_base || math_row >= (region.math_row_base + region.packet_rows)) {
      continue;
    }
    TmemPacket packet{};
    if (!read_window_packet(handle, window_id, packet_idx, &packet)) {
      return false;
    }
    auto packet_row = math_row - region.math_row_base;
    auto packet_row_byte_offset = packet_row * region.packet_cols * elem_bytes;
    auto valid_cols = (region.math_col_base < cols)
                    ? std::min<uint32_t>(region.packet_cols, cols - region.math_col_base)
                    : 0;
    auto valid_bytes = valid_cols * elem_bytes;
    if (valid_bytes != 0) {
      std::copy_n(row_bytes + region.math_col_base * elem_bytes,
                  valid_bytes,
                  packet.bytes.data() + packet_row_byte_offset);
    }
    if (!write_window_packet(handle, window_id, packet_idx, packet)) {
      return false;
    }
  }
  return true;
}

// Shift the raw logical TMEM lines inside one window down by one line. This
// helper is only correct for windows whose external semantics already match
// logical TMEM lines.
bool Tmem::shift_window_logical_lines_down(uint32_t handle, uint32_t window_id) {
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

// Shift one window down by one mathematical matrix row. When the window uses a
// math-packet adapter, TMEM reconstructs the mathematical image, performs the
// row shift, optionally writes a refill top row, then repacks packets back into
// the TMEM window layout.
bool Tmem::shift_window_math_rows_down(uint32_t handle,
                                       uint32_t window_id,
                                       const uint8_t* refill_math_row,
                                       uint32_t refill_math_row_bytes) {
  const TmemAllocation* allocation = nullptr;
  const TmemWindowPlan* window = nullptr;
  if (!lookup_allocation(handle, &allocation) || !lookup_window(handle, window_id, &window)) {
    return false;
  }

  auto elem_bytes = fmt_bytes(window->fmt);
  if (0 == elem_bytes || window->elem_shape.empty() || !TmemWindowPlanner::uses_math_packet_adapter(*window)) {
    if (!shift_window_logical_lines_down(handle, window_id)) {
      return false;
    }
    if (refill_math_row != nullptr && refill_math_row_bytes != 0) {
      auto col_base = resolve_window_col_base(*allocation, *window);
      auto top_line = window->logical_line_base;
      auto copy_bytes = std::min<uint32_t>(window->logical_col_span, refill_math_row_bytes);
      for (uint32_t i = 0; i < copy_bytes; ++i) {
        write_logical_byte(col_base + i, top_line, refill_math_row[i]);
      }
    }
    assert_valid();
    return true;
  }

  auto rows = std::max<uint32_t>(1, window->elem_shape.rows);
  auto cols = std::max<uint32_t>(1, window->elem_shape.cols);
  auto row_bytes = cols * elem_bytes;
  std::vector<uint8_t> math_row_buffer(row_bytes, 0);
  for (uint32_t math_row = rows - 1; math_row > 0; --math_row) {
    if (!read_window_math_row(handle, window_id, math_row - 1, math_row_buffer.data(), row_bytes)) {
      return false;
    }
    if (!write_window_math_row(handle, window_id, math_row, math_row_buffer.data(), row_bytes)) {
      return false;
    }
  }
  std::fill(math_row_buffer.begin(), math_row_buffer.end(), 0);
  if (refill_math_row != nullptr && refill_math_row_bytes != 0) {
    std::copy_n(refill_math_row,
                std::min<uint32_t>(row_bytes, refill_math_row_bytes),
                math_row_buffer.data());
  }
  if (!write_window_math_row(handle, window_id, 0, math_row_buffer.data(), row_bytes)) {
    return false;
  }

  assert_valid();
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
  *epoch = allocation->visible_layout_epoch;
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

void Tmem::publish_visible_state() {
  for (auto& entry : allocations_) {
    auto& allocation = entry.second;
    allocation.visible_layout_epoch = allocation.layout_epoch;
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
  if (!resolve_window_linear_packet_region(handle, window_id, packet_idx,
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
  if (!resolve_window_line_chunk_region(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
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
  if (!resolve_window_linear_packet_region(handle, window_id, packet_idx,
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
  if (!resolve_window_line_chunk_region(handle, window_id, line_idx, chunk_idx, &col_base, &line_base, &col_span)) {
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
