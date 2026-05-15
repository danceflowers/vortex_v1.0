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

#include "tmem_system.h"
#include "core.h"

using namespace vortex;

TmemSystem::TmemSystem(const SimContext& ctx, const char* name, Core* core)
  : SimObject(ctx, name)
  , TensorExecuteReqIn(this)
  , TensorExecuteRspOut(this)
  , CoreTransferReqIn(this)
  , CoreTransferRspOut(this)
  , AsyncOpCompletionOut(this)
  , core_(core)
{
  (void)name;
}

void TmemSystem::reset() {
  tmem_.reset();
  pending_tensor_execute_requests_.clear();
  pending_core_transfer_requests_.clear();
  shift_transactions_.clear();
  shift_transaction_fifo_.clear();
  live_shift_busy_taddrs_.clear();
  visible_shift_busy_taddrs_.clear();
}

// Publish producer-ready bits into the cross-module visible snapshot.
void TmemSystem::publish_visible_state() {
  visible_shift_busy_taddrs_ = live_shift_busy_taddrs_;
  tmem_.publish_visible_state();
}

// Keep the FIFO head valid so the service loop does not revisit completed shifts.
void TmemSystem::compact_shift_transaction_fifo() {
  std::deque<uint32_t> compacted_fifo;
  for (auto async_id : shift_transaction_fifo_) {
    auto it = shift_transactions_.find(async_id);
    if (it == shift_transactions_.end() || it->second.completed) {
      continue;
    }
    compacted_fifo.push_back(async_id);
  }
  shift_transaction_fifo_.swap(compacted_fifo);
}

void TmemSystem::drain_one_inbound_request(SimPort<TensorMemPortReq>& request_port,
                                           std::unordered_map<uint64_t, PendingPortRequest>& pending_requests) {
  if (request_port.empty()) {
    return;
  }

  auto request = request_port.front();
  auto port_request = request.port_request;
  port_request.age = request.arbitration_age;
  PendingPortRequest pending{};
  pending.port_request = request;
  pending.tmem_request_tag =
      tmem_.enqueue_port_request(SimPlatform::instance().cycles(),
                                 port_request);
  pending_requests[request.request_id] = pending;
  request_port.pop();
}

void TmemSystem::complete_granted_port_requests(std::unordered_map<uint64_t, PendingPortRequest>& pending_requests,
                                                SimPort<TensorMemPortRsp>& response_port) {
  std::vector<uint64_t> completed_request_ids;
  for (auto& entry : pending_requests) {
    auto request_id = entry.first;
    auto& pending = entry.second;
    if (!tmem_.request_granted(pending.tmem_request_tag)) {
      continue;
    }

    tmem_.consume_request_grant(pending.tmem_request_tag);
    TensorMemPortRsp response{};
    response.request_id = request_id;
    response.access_type = pending.port_request.access_type;

    const auto& request = pending.port_request.port_request;
    bool success = false;
    switch (request.kind) {
    case Tmem::PortRequestKind::RegionRead:
      success = tmem_.region_read_packet(request.col_base,
                                         request.col_span,
                                         request.packet_idx,
                                         &response.read_packet);
      break;
    case Tmem::PortRequestKind::RegionWrite:
      success = tmem_.region_write_packet(request.col_base,
                                          request.col_span,
                                          request.packet_idx,
                                          pending.port_request.write_packet);
      break;
    default:
      std::cerr << "TmemSystem error: unsupported PTX TMEM request"
                << " request_id=" << request_id
                << " kind=" << static_cast<uint32_t>(request.kind)
                << " packet=" << request.packet_idx
                << std::endl;
      std::abort();
    }
    if (!success) {
      std::cerr << "TmemSystem error: PTX TMEM packet access failed"
                << " request_id=" << request_id
                << " kind=" << static_cast<uint32_t>(request.kind)
                << " col_base=" << request.col_base
                << " col_span=" << request.col_span
                << " packet=" << request.packet_idx
                << std::endl;
      std::abort();
    }
    response_port.push(response, 0);
    completed_request_ids.push_back(request_id);
  }

  for (auto request_id : completed_request_ids) {
    pending_requests.erase(request_id);
  }
}

void TmemSystem::initialize_shift_transaction(ShiftTransaction& transaction) {
  if (transaction.initialized) {
    return;
  }

  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(transaction.taddr, &allocation)) {
    transaction.completed = true;
    transaction.initialized = true;
    return;
  }

  uint32_t size_bytes = 0;
  if (!region_query(allocation->payload_col_base, allocation->col_span, &size_bytes)) {
    transaction.completed = true;
    transaction.initialized = true;
    return;
  }
  transaction.packet_count = (size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
  transaction.initialized = true;
}

void TmemSystem::issue_shift_requests(ShiftTransaction& transaction) {
  if (transaction.completed || transaction.next_packet_idx >= transaction.packet_count) {
    return;
  }
  if (transaction.pending_read_request_tag != 0
   || transaction.pending_write_request_tag != 0) {
    return;
  }

  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(transaction.taddr, &allocation)) {
    transaction.completed = true;
    return;
  }

  auto cycle = SimPlatform::instance().cycles();
  auto packet_idx = transaction.next_packet_idx;
  auto request_age = [&transaction](uint32_t ordinal) -> uint64_t {
    return (static_cast<uint64_t>(transaction.async_id) << 32) | ordinal;
  };

  Tmem::PortRequestDesc read_request{};
  read_request.kind = Tmem::PortRequestKind::RegionRead;
  read_request.age = request_age(packet_idx * 2);
  read_request.packet_idx = packet_idx;
  read_request.col_base = allocation->payload_col_base;
  read_request.col_span = allocation->col_span;
  transaction.pending_read_request_tag = tmem_.enqueue_port_request(cycle, read_request);

  Tmem::PortRequestDesc write_request{};
  write_request.kind = Tmem::PortRequestKind::RegionWrite;
  write_request.age = request_age(packet_idx * 2 + 1);
  write_request.packet_idx = packet_idx;
  write_request.col_base = allocation->payload_col_base;
  write_request.col_span = allocation->col_span;
  transaction.pending_write_request_tag = tmem_.enqueue_port_request(cycle, write_request);
}

void TmemSystem::complete_shift_transaction_step(ShiftTransaction& transaction) {
  if (transaction.completed) {
    return;
  }

  if (transaction.pending_read_request_tag != 0
   && tmem_.request_granted(transaction.pending_read_request_tag)) {
    tmem_.consume_request_grant(transaction.pending_read_request_tag);
    transaction.pending_read_request_tag = 0;
    transaction.read_complete = true;
    ++core_->mutable_perf_stats().tmem_read_packets;
  }
  if (transaction.pending_write_request_tag != 0
   && tmem_.request_granted(transaction.pending_write_request_tag)) {
    tmem_.consume_request_grant(transaction.pending_write_request_tag);
    transaction.pending_write_request_tag = 0;
    transaction.write_complete = true;
    ++core_->mutable_perf_stats().tmem_write_packets;
  }
  if (!transaction.read_complete || !transaction.write_complete) {
    return;
  }

  transaction.read_complete = false;
  transaction.write_complete = false;
  ++transaction.next_packet_idx;
  if (transaction.next_packet_idx < transaction.packet_count) {
    return;
  }

  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(transaction.taddr, &allocation)) {
    transaction.completed = true;
    return;
  }
  if (!tmem_.region_shift_down(allocation->payload_col_base,
                               allocation->col_span,
                               allocation->row_bytes)) {
    transaction.completed = true;
    return;
  }
  set_payload_ready(transaction.taddr, true);
  transaction.completed = true;
  AsyncOpCompletionOut.push({transaction.async_id}, 1);
}

void TmemSystem::advance_shift_engine() {
  compact_shift_transaction_fifo();
  for (auto async_id : shift_transaction_fifo_) {
    auto it = shift_transactions_.find(async_id);
    if (it == shift_transactions_.end()) {
      continue;
    }
    auto& transaction = it->second;
    if (transaction.completed
     || SimPlatform::instance().cycles() < transaction.first_service_cycle) {
      continue;
    }
    initialize_shift_transaction(transaction);
    if (transaction.completed) {
      live_shift_busy_taddrs_.erase(transaction.taddr);
      AsyncOpCompletionOut.push({transaction.async_id}, 1);
      continue;
    }
    issue_shift_requests(transaction);
  }
}

void TmemSystem::tick() {
  publish_visible_state();
  tmem_.ensure_port_budgets(SimPlatform::instance().cycles());
  drain_one_inbound_request(TensorExecuteReqIn, pending_tensor_execute_requests_);
  drain_one_inbound_request(CoreTransferReqIn, pending_core_transfer_requests_);
  advance_shift_engine();
  tmem_.arbitrate_requests(SimPlatform::instance().cycles());
  complete_granted_port_requests(pending_tensor_execute_requests_, TensorExecuteRspOut);
  complete_granted_port_requests(pending_core_transfer_requests_, CoreTransferRspOut);
  for (auto async_id : shift_transaction_fifo_) {
    auto it = shift_transactions_.find(async_id);
    if (it == shift_transactions_.end() || it->second.completed) {
      continue;
    }
    complete_shift_transaction_step(it->second);
    if (it->second.completed) {
      live_shift_busy_taddrs_.erase(it->second.taddr);
    }
  }
  compact_shift_transaction_fifo();
}

uint32_t TmemSystem::alloc(uint32_t col_span) { return tmem_.alloc(col_span); }
bool TmemSystem::free(uint32_t taddr) { return tmem_.free(taddr); }
void TmemSystem::seal_allocator() { tmem_.seal_allocator(); }
bool TmemSystem::allocator_sealed() const { return tmem_.allocator_sealed(); }
bool TmemSystem::lookup_allocation(uint32_t taddr, TmemAllocation** allocation) { return tmem_.lookup_allocation(taddr, allocation); }
bool TmemSystem::lookup_allocation(uint32_t taddr, const TmemAllocation** allocation) const { return tmem_.lookup_allocation(taddr, allocation); }
bool TmemSystem::region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const { return tmem_.region_query(col_base, col_span, size_bytes); }
bool TmemSystem::query(uint32_t taddr, uint32_t* col_span, uint32_t* size_bytes) const { return tmem_.query(taddr, col_span, size_bytes); }
bool TmemSystem::set_row_bytes(uint32_t taddr, uint32_t row_bytes) { return tmem_.set_row_bytes(taddr, row_bytes); }
bool TmemSystem::row_bytes(uint32_t taddr, uint32_t* row_bytes) const { return tmem_.row_bytes(taddr, row_bytes); }

void TmemSystem::set_payload_ready(uint32_t taddr, bool ready, bool update_visible_now) {
  tmem_.set_payload_ready(taddr, ready);
  if (!update_visible_now) {
    return;
  }
  if (TmemAllocation* allocation = nullptr; tmem_.lookup_allocation(taddr, &allocation)) {
    allocation->visible_payload_ready = ready;
  }
}

void TmemSystem::set_meta_ready(uint32_t taddr, bool ready, bool update_visible_now) {
  tmem_.set_meta_ready(taddr, ready);
  if (!update_visible_now) {
    return;
  }
  if (TmemAllocation* allocation = nullptr; tmem_.lookup_allocation(taddr, &allocation)) {
    allocation->visible_meta_ready = ready;
  }
}

void TmemSystem::set_meta_region(uint32_t taddr, uint32_t meta_col_base, uint32_t meta_col_span) {
  tmem_.set_meta_region(taddr, meta_col_base, meta_col_span);
}

uint64_t TmemSystem::enqueue_port_request(const Tmem::PortRequestDesc& desc, uint64_t arbitration_age) {
  auto request = desc;
  request.age = arbitration_age;
  return tmem_.enqueue_port_request(SimPlatform::instance().cycles(), request);
}

bool TmemSystem::request_granted(uint64_t request_tag) const { return tmem_.request_granted(request_tag); }
void TmemSystem::consume_request_grant(uint64_t request_tag) { tmem_.consume_request_grant(request_tag); }

bool TmemSystem::issue_shift(uint32_t async_id,
                             uint32_t wid,
                             uint32_t wgid,
                             uint32_t taddr,
                             uint64_t issue_cycle) {
  ShiftTransaction transaction{};
  transaction.async_id = async_id;
  transaction.wid = wid;
  transaction.wgid = wgid;
  transaction.taddr = taddr;
  transaction.issue_cycle = issue_cycle;
  transaction.first_service_cycle = SimPlatform::instance().cycles() + 1;
  shift_transactions_[async_id] = std::move(transaction);
  shift_transaction_fifo_.push_back(async_id);
  live_shift_busy_taddrs_.insert(taddr);
  set_payload_ready(taddr, false);
  return true;
}

bool TmemSystem::visible_payload_ready(uint32_t taddr) const {
  const TmemAllocation* allocation = nullptr;
  return tmem_.lookup_allocation(taddr, &allocation) && allocation->visible_payload_ready;
}

bool TmemSystem::visible_meta_ready(uint32_t taddr) const {
  const TmemAllocation* allocation = nullptr;
  return tmem_.lookup_allocation(taddr, &allocation) && allocation->visible_meta_ready;
}

void TmemSystem::dump_debug_state(std::ostream& os) const {
  os << "[TmemSystem] tensor_execute_pending=" << pending_tensor_execute_requests_.size()
     << " core_transfer_pending=" << pending_core_transfer_requests_.size()
     << " shift_transactions=" << shift_transactions_.size()
     << " shift_fifo_depth=" << shift_transaction_fifo_.size()
     << " live_shift_busy=" << live_shift_busy_taddrs_.size()
     << "\n";
  for (const auto& entry : shift_transactions_) {
    const auto& transaction = entry.second;
    os << "  shift_async_id=" << transaction.async_id
       << " taddr=" << transaction.taddr
       << " completed=" << transaction.completed
       << " initialized=" << transaction.initialized
       << " packet=" << transaction.next_packet_idx << "/" << transaction.packet_count
       << " pend_read_tag=" << transaction.pending_read_request_tag
       << " pend_write_tag=" << transaction.pending_write_request_tag
       << "\n";
  }
}
