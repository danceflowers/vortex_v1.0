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
#include "tensor_mem_layout_utils.h"

using namespace vortex;

namespace {

uint32_t tma_payload_size_bytes(const TmaDescriptor& desc) {
  uint32_t matrix_bytes = desc.rows * desc.cols * desc.elem_bytes;
  uint32_t requested_size = desc.size_bytes ? desc.size_bytes : matrix_bytes;
  return requested_size ? requested_size : matrix_bytes;
}

void build_math_row_packet_ids(const TmemWindowPlan& window,
                               uint32_t math_row,
                               std::vector<uint32_t>* packet_ids) {
  packet_ids->clear();
  auto packet_count = window.tile_count * window.packets_per_tile;
  for (uint32_t packet_idx = 0; packet_idx < packet_count; ++packet_idx) {
    TmemMathPacketRegion region{};
    if (!TmemWindowPlanner::packet_math_region(window, packet_idx, &region)) {
      continue;
    }
    if (math_row >= region.math_row_base && math_row < (region.math_row_base + region.packet_rows)) {
      packet_ids->push_back(packet_idx);
    }
  }
}

} // namespace

TmemSystem::TmemSystem(const SimContext& ctx, const char* name, Core* core)
  : SimObject(ctx, name)
  , TensorExecuteReqIn(this)
  , TensorExecuteRspOut(this)
  , TmaFrontendReqIn(this)
  , TmaFrontendRspOut(this)
  , RefillChunkReqOut(this)
  , RefillChunkRspIn(this)
  , AsyncOpCompletionOut(this)
  , core_(core)
{
  (void)name;
}

void TmemSystem::reset() {
  tmem_.reset();
  pending_tensor_execute_requests_.clear();
  pending_tma_frontend_requests_.clear();
  pending_refill_request_to_transaction_.clear();
  shift_transactions_.clear();
  shift_transaction_fifo_.clear();
  live_shift_busy_handles_.clear();
  visible_shift_busy_handles_.clear();
  next_refill_request_id_ = 1;
}

void TmemSystem::publish_visible_state() {
  visible_shift_busy_handles_ = live_shift_busy_handles_;
  tmem_.publish_visible_state();
}

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

void TmemSystem::drain_one_inbound_request() {
  auto have_tensor_execute_req = !TensorExecuteReqIn.empty();
  auto have_tma_frontend_req = !TmaFrontendReqIn.empty();
  if (!have_tensor_execute_req && !have_tma_frontend_req) {
    return;
  }

  bool take_tensor_execute_req = have_tensor_execute_req;
  if (have_tensor_execute_req && have_tma_frontend_req) {
    const auto& tensor_execute_req = TensorExecuteReqIn.front();
    const auto& tma_frontend_req = TmaFrontendReqIn.front();
    take_tensor_execute_req = tensor_execute_req.arbitration_age <= tma_frontend_req.arbitration_age;
  }

  PendingPortRequest pending{};
  if (take_tensor_execute_req) {
    auto request = TensorExecuteReqIn.front();
    pending.port_request = request;
    pending.tmem_request_tag = tmem_.enqueue_port_request(SimPlatform::instance().cycles(), request.port_request);
    pending_tensor_execute_requests_[request.request_id] = pending;
    TensorExecuteReqIn.pop();
    return;
  }

  auto request = TmaFrontendReqIn.front();
  pending.port_request = request;
  pending.tmem_request_tag = tmem_.enqueue_port_request(SimPlatform::instance().cycles(), request.port_request);
  pending_tma_frontend_requests_[request.request_id] = pending;
  TmaFrontendReqIn.pop();
}

void TmemSystem::drain_refill_chunk_responses() {
  if (RefillChunkRspIn.empty()) {
    return;
  }

  auto response = RefillChunkRspIn.front();
  RefillChunkRspIn.pop();

  auto pending_it = pending_refill_request_to_transaction_.find(response.request_id);
  if (pending_it == pending_refill_request_to_transaction_.end()) {
    return;
  }

  auto transaction_it = shift_transactions_.find(pending_it->second);
  if (transaction_it == shift_transactions_.end()) {
    pending_refill_request_to_transaction_.erase(pending_it);
    return;
  }

  auto& transaction = transaction_it->second;
  transaction.pending_refill_fetch_request_id = 0;
  if (!response.success) {
    transaction.completed = true;
    pending_refill_request_to_transaction_.erase(pending_it);
    return;
  }

  transaction.pending_refill_packet = response.packet;
  transaction.pending_refill_packet_valid = true;
  pending_refill_request_to_transaction_.erase(pending_it);
}

void TmemSystem::complete_granted_tensor_execute_requests() {
  std::vector<uint64_t> completed_request_ids;
  for (auto& entry : pending_tensor_execute_requests_) {
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
    case Tmem::PortRequestKind::WindowRead:
      success = tmem_.read_window_packet(request.handle, request.window_id, request.packet_idx, &response.read_packet);
      break;
    case Tmem::PortRequestKind::WindowWrite:
      success = tmem_.write_window_packet(request.handle, request.window_id, request.packet_idx, pending.port_request.write_packet);
      break;
    default:
      std::cerr << "TmemSystem error: tensor execute ingress received non-window TMEM request"
                << " request_id=" << request_id
                << " kind=" << static_cast<uint32_t>(request.kind)
                << " handle=" << request.handle
                << " window=" << request.window_id
                << " packet=" << request.packet_idx
                << std::endl;
      std::abort();
    }
    if (!success) {
      std::abort();
    }
    TensorExecuteRspOut.push(response, 0);
    completed_request_ids.push_back(request_id);
  }

  for (auto request_id : completed_request_ids) {
    pending_tensor_execute_requests_.erase(request_id);
  }
}

void TmemSystem::complete_granted_tma_frontend_requests() {
  std::vector<uint64_t> completed_request_ids;
  for (auto& entry : pending_tma_frontend_requests_) {
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
    case Tmem::PortRequestKind::RegionWrite:
      success = tmem_.region_write_packet(request.col_base, request.col_span, request.packet_idx, pending.port_request.write_packet);
      break;
    case Tmem::PortRequestKind::WindowWrite:
      success = tmem_.write_window_packet(request.handle, request.window_id, request.packet_idx, pending.port_request.write_packet);
      break;
    case Tmem::PortRequestKind::WindowLinearWrite:
      success = tmem_.write_window_linear_packet(request.handle, request.window_id, request.packet_idx, pending.port_request.write_packet);
      break;
    case Tmem::PortRequestKind::WindowRead:
      success = tmem_.read_window_packet(request.handle, request.window_id, request.packet_idx, &response.read_packet);
      break;
    case Tmem::PortRequestKind::WindowLinearRead:
      success = tmem_.read_window_linear_packet(request.handle, request.window_id, request.packet_idx, &response.read_packet);
      break;
    case Tmem::PortRequestKind::WindowLineWrite:
      success = tmem_.write_window_line_chunk(request.handle,
                                              request.window_id,
                                              request.line_idx,
                                              request.chunk_idx,
                                              pending.port_request.write_packet);
      break;
    default:
      std::abort();
    }
    if (!success) {
      std::abort();
    }
    TmaFrontendRspOut.push(response, 0);
    completed_request_ids.push_back(request_id);
  }

  for (auto request_id : completed_request_ids) {
    pending_tma_frontend_requests_.erase(request_id);
  }
}

void TmemSystem::initialize_shift_transaction(ShiftTransaction& transaction) {
  if (transaction.transaction_initialized) {
    return;
  }

  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(transaction.handle, &allocation)) {
    transaction.completed = true;
    transaction.transaction_initialized = true;
    return;
  }

  const TmemWindowPlan* window = nullptr;
  if (lookup_window(transaction.handle, transaction.window_id, &window)
   && nullptr != window
   && TmemWindowPlanner::uses_math_packet_adapter(*window)
   && !window->elem_shape.empty()) {
    transaction.uses_math_row_shift = true;
    auto rows = std::max<uint32_t>(1, window->elem_shape.rows);
    auto row_bytes = window->elem_shape.cols * fmt_elem_bytes(window->fmt);
    if (0 == row_bytes) {
      row_bytes = window->logical_col_span;
    }
    transaction.shift_math_row_bytes = row_bytes;
    transaction.shift_row_reorder_buffer.assign(row_bytes, 0);
    if (rows > 1) {
      transaction.current_shift_math_row = rows - 1;
      (void)prepare_next_shift_math_row(transaction);
    } else {
      transaction.main_shift_body_complete = true;
      transaction.remaining_tmem_read_packets = 0;
      transaction.remaining_tmem_write_packets = 0;
      transaction.current_shift_row_packet_ids.clear();
    }
  } else if (lookup_window(transaction.handle, transaction.window_id, &window)
          && nullptr != window
          && window->logical_col_span != 0
          && window->logical_line_span != 0) {
    transaction.uses_logical_line_shift = true;
    if (window->logical_line_span > 1) {
      transaction.current_shift_logical_line = window->logical_line_span - 1;
      if (!prepare_next_shift_logical_line(transaction)) {
        transaction.completed = true;
        transaction.transaction_initialized = true;
        return;
      }
    } else {
      transaction.main_shift_body_complete = true;
      transaction.remaining_tmem_read_packets = 0;
      transaction.remaining_tmem_write_packets = 0;
      transaction.shift_line_chunk_count = 0;
    }
  } else {
    uint32_t packets = 0;
    if (!tmem_.window_packet_count(transaction.handle, transaction.window_id, &packets)) {
      uint32_t size_bytes = 0;
      if (!region_query(allocation->payload_col_base, allocation->col_span, &size_bytes)) {
        transaction.completed = true;
        transaction.transaction_initialized = true;
        return;
      }
      packets = (size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    transaction.remaining_tmem_read_packets = packets;
    transaction.remaining_tmem_write_packets = packets;
  }

  if ((transaction.refill_descriptor_id != 0) || transaction.uses_logical_line_shift) {
    if (!lookup_window(transaction.handle, transaction.window_id, &window) || window->logical_col_span == 0) {
      transaction.completed = true;
      transaction.transaction_initialized = true;
      return;
    }
    uint32_t refill_row_bytes = window->logical_col_span;
    if (transaction.uses_math_row_shift) {
      refill_row_bytes = window->elem_shape.cols * fmt_elem_bytes(window->fmt);
      if (0 == refill_row_bytes) {
        refill_row_bytes = window->logical_col_span;
      }
    }
    if (transaction.refill_descriptor_id != 0) {
      TmaDescriptor desc{};
      if (!core_->read_tma_descriptor(transaction.refill_descriptor_id, &desc) || desc.rows != 1) {
        transaction.completed = true;
        transaction.transaction_initialized = true;
        return;
      }
      transaction.refill_math_row_bytes = std::min<uint32_t>(tma_payload_size_bytes(desc), refill_row_bytes);
    } else {
      transaction.refill_math_row_bytes = refill_row_bytes;
    }
    transaction.refill_row_reorder_buffer.assign(transaction.refill_math_row_bytes, 0);
    transaction.remaining_refill_line_packets =
      (transaction.refill_math_row_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
  }

  transaction.transaction_initialized = true;
}

bool TmemSystem::prepare_next_shift_math_row(ShiftTransaction& transaction) {
  const TmemWindowPlan* window = nullptr;
  if (!lookup_window(transaction.handle, transaction.window_id, &window) || nullptr == window) {
    return false;
  }
  build_math_row_packet_ids(*window,
                            transaction.current_shift_math_row,
                            &transaction.current_shift_row_packet_ids);
  transaction.next_shift_body_packet_ordinal = 0;
  transaction.remaining_tmem_read_packets = transaction.current_shift_row_packet_ids.size();
  transaction.remaining_tmem_write_packets = transaction.current_shift_row_packet_ids.size();
  transaction.current_shift_body_read_complete = false;
  transaction.current_shift_body_write_complete = false;
  transaction.pending_shift_read_request_tag = 0;
  transaction.pending_shift_write_request_tag = 0;
  return true;
}

bool TmemSystem::prepare_next_shift_logical_line(ShiftTransaction& transaction) {
  const TmemWindowPlan* window = nullptr;
  uint32_t chunk_count = 0;
  if (!lookup_window(transaction.handle, transaction.window_id, &window)
   || nullptr == window
   || !tmem_.window_line_chunk_count(transaction.handle, transaction.window_id, &chunk_count)) {
    return false;
  }
  transaction.shift_line_chunk_count = chunk_count;
  transaction.next_shift_body_packet_ordinal = 0;
  transaction.remaining_tmem_read_packets = chunk_count;
  transaction.remaining_tmem_write_packets = chunk_count;
  transaction.current_shift_body_read_complete = false;
  transaction.current_shift_body_write_complete = false;
  transaction.pending_shift_read_request_tag = 0;
  transaction.pending_shift_write_request_tag = 0;
  return true;
}

void TmemSystem::issue_shift_requests(ShiftTransaction& transaction) {
  if (transaction.completed) {
    return;
  }
  auto cycle = SimPlatform::instance().cycles();
  auto request_age = [&transaction](uint32_t ordinal) -> uint64_t {
    return (static_cast<uint64_t>(transaction.async_id) << 32) | ordinal;
  };
  bool use_window = (transaction.window_id != 0) || tmem_.lookup_window(transaction.handle, transaction.window_id, nullptr);
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(transaction.handle, &allocation)) {
    transaction.completed = true;
    return;
  }

  if (!transaction.main_shift_body_complete) {
    auto packet_idx = transaction.uses_math_row_shift
                    ? transaction.current_shift_row_packet_ids.at(transaction.next_shift_body_packet_ordinal)
                    : transaction.next_shift_body_packet_ordinal;
    if (transaction.remaining_tmem_read_packets != 0
     && !transaction.current_shift_body_read_complete
     && transaction.pending_shift_read_request_tag == 0) {
      Tmem::PortRequestDesc request{};
      request.kind = (use_window && transaction.uses_logical_line_shift)
                   ? Tmem::PortRequestKind::WindowLineRead
                   : (use_window ? Tmem::PortRequestKind::WindowRead
                                 : Tmem::PortRequestKind::RegionRead);
      request.age = request_age(packet_idx * 2);
      request.handle = transaction.handle;
      request.window_id = transaction.window_id;
      request.packet_idx = packet_idx;
      request.line_idx = transaction.current_shift_logical_line - 1;
      request.chunk_idx = packet_idx;
      request.col_base = allocation->payload_col_base;
      request.col_span = allocation->col_span;
      transaction.pending_shift_read_request_tag = tmem_.enqueue_port_request(cycle, request);
    }
    if (transaction.remaining_tmem_write_packets != 0
     && !transaction.current_shift_body_write_complete
     && transaction.pending_shift_write_request_tag == 0) {
      Tmem::PortRequestDesc request{};
      request.kind = (use_window && transaction.uses_logical_line_shift)
                   ? Tmem::PortRequestKind::WindowLineWrite
                   : (use_window ? Tmem::PortRequestKind::WindowWrite
                                 : Tmem::PortRequestKind::RegionWrite);
      request.age = request_age(packet_idx * 2 + 1);
      request.handle = transaction.handle;
      request.window_id = transaction.window_id;
      request.packet_idx = packet_idx;
      request.line_idx = transaction.current_shift_logical_line;
      request.chunk_idx = packet_idx;
      request.col_base = allocation->payload_col_base;
      request.col_span = allocation->col_span;
      transaction.pending_shift_write_request_tag = tmem_.enqueue_port_request(cycle, request);
    }
    return;
  }

  if (transaction.remaining_refill_line_packets != 0
   && transaction.refill_descriptor_id != 0
   && transaction.pending_refill_fetch_request_id == 0
   && !transaction.pending_refill_packet_valid
   && transaction.pending_refill_write_request_tag == 0) {
    auto request_id = next_refill_request_id_++;
    auto chunk_idx = transaction.next_refill_line_packet_idx;
    pending_refill_request_to_transaction_[request_id] = transaction.async_id;
    transaction.pending_refill_fetch_request_id = request_id;
    RefillChunkReqOut.push({request_id,
                            transaction.refill_descriptor_id,
                            chunk_idx,
                            transaction.refill_math_row_bytes});
    return;
  }

  if (transaction.remaining_refill_line_packets != 0
   && (transaction.pending_refill_packet_valid || transaction.refill_descriptor_id == 0)
   && transaction.pending_refill_write_request_tag == 0) {
    auto chunk_idx = transaction.next_refill_line_packet_idx;
    auto refill_offset = chunk_idx * Tmem::kPacketBytes;
    if (transaction.pending_refill_packet_valid && refill_offset < transaction.refill_row_reorder_buffer.size()) {
      auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes,
                                            transaction.refill_row_reorder_buffer.size() - refill_offset);
      std::copy_n(transaction.pending_refill_packet.bytes.data(),
                  valid_bytes,
                  transaction.refill_row_reorder_buffer.data() + refill_offset);
    }
    Tmem::PortRequestDesc request{};
    request.kind = use_window ? Tmem::PortRequestKind::WindowLineWrite
                              : Tmem::PortRequestKind::RegionWrite;
    request.age = request_age(chunk_idx);
    request.handle = transaction.handle;
    request.window_id = transaction.window_id;
    request.packet_idx = chunk_idx;
    request.line_idx = 0;
    request.chunk_idx = chunk_idx;
    request.col_base = allocation->payload_col_base;
    request.col_span = allocation->col_span;
    transaction.pending_refill_write_request_tag = tmem_.enqueue_port_request(cycle, request);
    transaction.pending_refill_packet_valid = false;
  }
}

void TmemSystem::complete_shift_transaction_step(ShiftTransaction& transaction) {
  if (transaction.completed) {
    return;
  }
  bool use_window = (transaction.window_id != 0) || tmem_.lookup_window(transaction.handle, transaction.window_id, nullptr);
  TmemAllocation* allocation = nullptr;
  if (!lookup_allocation(transaction.handle, &allocation)) {
    transaction.completed = true;
    return;
  }

  if (!transaction.main_shift_body_complete) {
    if (transaction.pending_shift_read_request_tag != 0) {
      if (!tmem_.request_granted(transaction.pending_shift_read_request_tag)) {
        return;
      }
      tmem_.consume_request_grant(transaction.pending_shift_read_request_tag);
      transaction.pending_shift_read_request_tag = 0;
      --transaction.remaining_tmem_read_packets;
      transaction.current_shift_body_read_complete = true;
      ++core_->mutable_perf_stats().tmem_read_packets;
    }
    if (transaction.pending_shift_write_request_tag != 0) {
      if (!tmem_.request_granted(transaction.pending_shift_write_request_tag)) {
        return;
      }
      tmem_.consume_request_grant(transaction.pending_shift_write_request_tag);
      transaction.pending_shift_write_request_tag = 0;
      --transaction.remaining_tmem_write_packets;
      transaction.current_shift_body_write_complete = true;
      ++core_->mutable_perf_stats().tmem_write_packets;
    }
    if (!(transaction.current_shift_body_read_complete && transaction.current_shift_body_write_complete)) {
      return;
    }
    auto completed_chunk_idx = transaction.next_shift_body_packet_ordinal;
    if (transaction.uses_logical_line_shift) {
      TmemPacket line_chunk{};
      if (!tmem_.read_window_line_chunk(transaction.handle,
                                        transaction.window_id,
                                        transaction.current_shift_logical_line - 1,
                                        completed_chunk_idx,
                                        &line_chunk)) {
        transaction.completed = true;
        return;
      }
      if (!tmem_.write_window_line_chunk(transaction.handle,
                                         transaction.window_id,
                                         transaction.current_shift_logical_line,
                                         completed_chunk_idx,
                                         line_chunk)) {
        transaction.completed = true;
        return;
      }
    }
    transaction.current_shift_body_read_complete = false;
    transaction.current_shift_body_write_complete = false;
    ++transaction.next_shift_body_packet_ordinal;
    if (transaction.remaining_tmem_read_packets != 0 || transaction.remaining_tmem_write_packets != 0) {
      return;
    }
    if (transaction.uses_math_row_shift) {
      if (!tmem_.read_window_math_row(transaction.handle,
                                      transaction.window_id,
                                      transaction.current_shift_math_row - 1,
                                      transaction.shift_row_reorder_buffer.data(),
                                      transaction.shift_math_row_bytes)) {
        transaction.completed = true;
        return;
      }
      if (!tmem_.write_window_math_row(transaction.handle,
                                       transaction.window_id,
                                       transaction.current_shift_math_row,
                                       transaction.shift_row_reorder_buffer.data(),
                                       transaction.shift_math_row_bytes)) {
        transaction.completed = true;
        return;
      }
      if (transaction.current_shift_math_row > 1) {
        --transaction.current_shift_math_row;
        if (!prepare_next_shift_math_row(transaction)) {
          transaction.completed = true;
        }
        return;
      }
      transaction.main_shift_body_complete = true;
    } else if (transaction.uses_logical_line_shift) {
      if (transaction.current_shift_logical_line > 1) {
        --transaction.current_shift_logical_line;
        if (!prepare_next_shift_logical_line(transaction)) {
          transaction.completed = true;
        }
        return;
      }
      transaction.main_shift_body_complete = true;
    } else {
      transaction.main_shift_body_complete = true;
      bool shifted = use_window
                   ? tmem_.shift_window_math_rows_down(transaction.handle, transaction.window_id, nullptr, 0)
                   : tmem_.region_shift_down(allocation->payload_col_base, allocation->col_span, allocation->row_bytes);
      if (!shifted) {
        transaction.completed = true;
        return;
      }
    }

    if (transaction.remaining_refill_line_packets == 0) {
      if (transaction.uses_math_row_shift) {
        std::fill(transaction.shift_row_reorder_buffer.begin(),
                  transaction.shift_row_reorder_buffer.end(),
                  0);
        if (!tmem_.write_window_math_row(transaction.handle,
                                         transaction.window_id,
                                         0,
                                         transaction.shift_row_reorder_buffer.data(),
                                         transaction.shift_math_row_bytes)) {
          transaction.completed = true;
          return;
        }
      }
      set_payload_ready(transaction.handle, true);
      (void)tmem_.bump_window_epoch(transaction.handle);
      transaction.completed = true;
      AsyncOpCompletionOut.push({transaction.async_id}, 1);
    }
    return;
  }

  if (transaction.pending_refill_write_request_tag != 0) {
    if (!tmem_.request_granted(transaction.pending_refill_write_request_tag)) {
      return;
    }
    tmem_.consume_request_grant(transaction.pending_refill_write_request_tag);
    transaction.pending_refill_write_request_tag = 0;
    ++core_->mutable_perf_stats().tmem_write_packets;
    ++transaction.next_refill_line_packet_idx;
    --transaction.remaining_refill_line_packets;
    if (transaction.uses_logical_line_shift) {
      auto chunk_idx = transaction.next_refill_line_packet_idx - 1;
      TmemPacket refill_chunk{};
      auto refill_offset = chunk_idx * Tmem::kPacketBytes;
      auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes,
                                            static_cast<uint32_t>(transaction.refill_row_reorder_buffer.size() - refill_offset));
      std::copy_n(transaction.refill_row_reorder_buffer.data() + refill_offset,
                  valid_bytes,
                  refill_chunk.bytes.data());
      if (!tmem_.write_window_line_chunk(transaction.handle,
                                         transaction.window_id,
                                         0,
                                         chunk_idx,
                                         refill_chunk)) {
        transaction.completed = true;
        return;
      }
    }
  }
  if (transaction.remaining_refill_line_packets != 0) {
    return;
  }

  if (transaction.uses_math_row_shift) {
    if (!tmem_.write_window_math_row(transaction.handle,
                                     transaction.window_id,
                                     0,
                                     transaction.refill_row_reorder_buffer.data(),
                                     transaction.refill_math_row_bytes)) {
      transaction.completed = true;
      return;
    }
  } else {
    bool shifted = use_window
                 ? tmem_.shift_window_math_rows_down(transaction.handle,
                                                    transaction.window_id,
                                                    transaction.refill_row_reorder_buffer.data(),
                                                    transaction.refill_math_row_bytes)
                 : tmem_.region_shift_down(allocation->payload_col_base, allocation->col_span, allocation->row_bytes);
    if (!shifted) {
      transaction.completed = true;
      return;
    }
  }
  set_payload_ready(transaction.handle, true);
  (void)tmem_.bump_window_epoch(transaction.handle);
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
    if (transaction.completed || SimPlatform::instance().cycles() < transaction.first_service_cycle) {
      continue;
    }
    initialize_shift_transaction(transaction);
    if (transaction.completed) {
      live_shift_busy_handles_.erase(transaction.handle);
      AsyncOpCompletionOut.push({transaction.async_id}, 1);
      continue;
    }
    issue_shift_requests(transaction);
  }
}

void TmemSystem::tick() {
  publish_visible_state();
  tmem_.ensure_port_budgets(SimPlatform::instance().cycles());
  drain_one_inbound_request();
  drain_refill_chunk_responses();
  advance_shift_engine();
  tmem_.arbitrate_requests(SimPlatform::instance().cycles());
  complete_granted_tensor_execute_requests();
  complete_granted_tma_frontend_requests();
  for (auto async_id : shift_transaction_fifo_) {
    auto it = shift_transactions_.find(async_id);
    if (it == shift_transactions_.end() || it->second.completed) {
      continue;
    }
    complete_shift_transaction_step(it->second);
    if (it->second.completed) {
      live_shift_busy_handles_.erase(it->second.handle);
    }
  }
  compact_shift_transaction_fifo();
}

uint32_t TmemSystem::alloc(uint32_t col_span) { return tmem_.alloc(col_span); }
bool TmemSystem::free(uint32_t handle) { return tmem_.free(handle); }
void TmemSystem::seal_allocator() { tmem_.seal_allocator(); }
bool TmemSystem::allocator_sealed() const { return tmem_.allocator_sealed(); }
bool TmemSystem::lookup_allocation(uint32_t handle, TmemAllocation** allocation) { return tmem_.lookup_allocation(handle, allocation); }
bool TmemSystem::lookup_allocation(uint32_t handle, const TmemAllocation** allocation) const { return tmem_.lookup_allocation(handle, allocation); }
bool TmemSystem::region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const { return tmem_.region_query(col_base, col_span, size_bytes); }
bool TmemSystem::query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const { return tmem_.query(handle, col_span, size_bytes); }
bool TmemSystem::lookup_window(uint32_t handle, uint32_t window_id, const TmemWindowPlan** out) const { return tmem_.lookup_window(handle, window_id, out); }
bool TmemSystem::lookup_window(uint32_t handle, uint32_t window_id, TmemWindowPlan** out) { return tmem_.lookup_window(handle, window_id, out); }
bool TmemSystem::bind_layout(uint32_t handle, const TmemLayoutPlan& layout) { return tmem_.bind_layout(handle, layout); }
bool TmemSystem::upsert_window(uint32_t handle, const TmemWindowPlan& window) { return tmem_.upsert_window(handle, window); }
bool TmemSystem::window_epoch(uint32_t handle, uint32_t* epoch) const { return tmem_.window_epoch(handle, epoch); }
bool TmemSystem::bump_window_epoch(uint32_t handle) { return tmem_.bump_window_epoch(handle); }
bool TmemSystem::set_row_bytes(uint32_t handle, uint32_t row_bytes) { return tmem_.set_row_bytes(handle, row_bytes); }
bool TmemSystem::row_bytes(uint32_t handle, uint32_t* row_bytes) const { return tmem_.row_bytes(handle, row_bytes); }
bool TmemSystem::read_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const { return tmem_.read_window_packet(handle, window_id, packet_idx, out); }
bool TmemSystem::read_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const { return tmem_.read_window_linear_packet(handle, window_id, packet_idx, out); }
bool TmemSystem::write_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in) { return tmem_.write_window_packet(handle, window_id, packet_idx, in); }
bool TmemSystem::write_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in) { return tmem_.write_window_linear_packet(handle, window_id, packet_idx, in); }
bool TmemSystem::region_write_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, const TmemPacket& in) { return tmem_.region_write_packet(col_base, col_span, packet_idx, in); }
bool TmemSystem::window_packet_count(uint32_t handle, uint32_t window_id, uint32_t* count) const { return tmem_.window_packet_count(handle, window_id, count); }
bool TmemSystem::shift_window_math_rows_down(uint32_t handle, uint32_t window_id, const uint8_t* refill_math_row, uint32_t refill_math_row_bytes) { return tmem_.shift_window_math_rows_down(handle, window_id, refill_math_row, refill_math_row_bytes); }
bool TmemSystem::region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes) { return tmem_.region_shift_down(col_base, col_span, row_bytes); }

void TmemSystem::set_payload_ready(uint32_t handle, bool ready, bool update_visible_now) {
  tmem_.set_payload_ready(handle, ready);
  if (!update_visible_now) {
    return;
  }
  if (TmemAllocation* allocation = nullptr; tmem_.lookup_allocation(handle, &allocation)) {
    allocation->visible_payload_ready = ready;
  }
}

void TmemSystem::set_meta_ready(uint32_t handle, bool ready, bool update_visible_now) {
  tmem_.set_meta_ready(handle, ready);
  if (!update_visible_now) {
    return;
  }
  if (TmemAllocation* allocation = nullptr; tmem_.lookup_allocation(handle, &allocation)) {
    allocation->visible_meta_ready = ready;
  }
}

void TmemSystem::set_meta_region(uint32_t handle, uint32_t meta_col_base, uint32_t meta_col_span) { tmem_.set_meta_region(handle, meta_col_base, meta_col_span); }
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
                             uint32_t handle,
                             uint32_t window_id,
                             uint32_t refill_descriptor_id,
                             uint64_t issue_cycle) {
  ShiftTransaction transaction{};
  transaction.async_id = async_id;
  transaction.wid = wid;
  transaction.wgid = wgid;
  transaction.handle = handle;
  transaction.window_id = window_id;
  transaction.refill_descriptor_id = refill_descriptor_id;
  transaction.issue_cycle = issue_cycle;
  transaction.first_service_cycle = SimPlatform::instance().cycles() + 1;
  shift_transactions_[async_id] = std::move(transaction);
  shift_transaction_fifo_.push_back(async_id);
  live_shift_busy_handles_.insert(handle);
  set_payload_ready(handle, false);
  return true;
}

bool TmemSystem::visible_payload_ready(uint32_t handle) const {
  const TmemAllocation* allocation = nullptr;
  return tmem_.lookup_allocation(handle, &allocation) && allocation->visible_payload_ready;
}

bool TmemSystem::visible_meta_ready(uint32_t handle) const {
  const TmemAllocation* allocation = nullptr;
  return tmem_.lookup_allocation(handle, &allocation) && allocation->visible_meta_ready;
}

void TmemSystem::dump_debug_state(std::ostream& os) const {
  os << "[TmemSystem] tensor_execute_pending=" << pending_tensor_execute_requests_.size()
     << " tma_frontend_pending=" << pending_tma_frontend_requests_.size()
     << " shift_transactions=" << shift_transactions_.size()
     << " shift_fifo_depth=" << shift_transaction_fifo_.size()
     << " live_shift_busy=" << live_shift_busy_handles_.size()
     << "\n";
  for (const auto& entry : shift_transactions_) {
    const auto& transaction = entry.second;
    os << "  shift_async_id=" << transaction.async_id
       << " handle=" << transaction.handle
       << " window=" << transaction.window_id
       << " completed=" << transaction.completed
       << " initialized=" << transaction.transaction_initialized
       << " main_done=" << transaction.main_shift_body_complete
       << " rem_read_pkts=" << transaction.remaining_tmem_read_packets
       << " rem_write_pkts=" << transaction.remaining_tmem_write_packets
       << " rem_refill_pkts=" << transaction.remaining_refill_line_packets
       << " next_shift_pkt_ord=" << transaction.next_shift_body_packet_ordinal
       << " next_refill_pkt=" << transaction.next_refill_line_packet_idx
       << " pend_read_tag=" << transaction.pending_shift_read_request_tag
       << " pend_write_tag=" << transaction.pending_shift_write_request_tag
       << " pend_refill_tag=" << transaction.pending_refill_write_request_tag
       << "\n";
  }
}
