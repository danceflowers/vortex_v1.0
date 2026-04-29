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
#include "tma_frontend.h"
#include "core.h"
#include "tmem_system.h"
#include "tensor_mem_port_types.h"
#include "tensor_mem_layout_utils.h"

using namespace vortex;

namespace {

Tmem::PortRequestKind load_payload_request_kind(bool use_math_window) {
  return use_math_window ? Tmem::PortRequestKind::WindowWrite
                         : Tmem::PortRequestKind::WindowLinearWrite;
}

Tmem::PortRequestKind store_payload_request_kind(bool use_math_window) {
  return use_math_window ? Tmem::PortRequestKind::WindowRead
                         : Tmem::PortRequestKind::WindowLinearRead;
}

} // namespace

TmaFrontend::TmaFrontend(const SimContext& ctx,
                         const char* name,
                         Core* core,
                         bool realistic_load)
  : SimObject(ctx, name)
  , TmemReqOut(this)
  , TmemRspIn(this)
  , RefillChunkReqIn(this)
  , RefillChunkRspOut(this)
  , AsyncOpCompletionOut(this)
  , core_(core)
  , tma_model_(realistic_load)
  , next_request_id_(1) {
  (void)name;
}

void TmaFrontend::reset() {
  tma_model_.reset();
  transactions_.clear();
  transaction_fifo_.clear();
  pending_request_to_transaction_.clear();
  live_load_busy_handles_.clear();
  visible_load_busy_handles_.clear();
  live_store_busy_handles_.clear();
  visible_store_busy_handles_.clear();
  next_request_id_ = 1;
}

bool TmaFrontend::read_tma_descriptor(uint32_t desc_id, TmaDescriptor* out) {
  return tma_model_.read_tma_descriptor(core_->startup_arg(),
                                        desc_id,
                                        out,
                                        [this](void* data, uint64_t addr, uint32_t size) {
                                          core_->dcache_read(data, addr, size);
                                        });
}

bool TmaFrontend::read_idescriptor(uint32_t desc_id, IDescriptor* out) {
  return tma_model_.read_idescriptor(core_->startup_arg(),
                                        desc_id,
                                        out,
                                        [this](void* data, uint64_t addr, uint32_t size) {
                                          core_->dcache_read(data, addr, size);
                                        });
}

uint32_t TmaFrontend::estimate_load_latency(const TmaDescriptor& desc) const {
  return tma_model_.estimate_load_latency(desc);
}

uint32_t TmaFrontend::payload_size_bytes(const TmaDescriptor& desc) const {
  return tma_model_.payload_size_bytes(desc);
}

bool TmaFrontend::issue_load(uint32_t async_id,
                             uint32_t wid,
                             uint32_t wgid,
                             uint32_t handle,
                             uint32_t descriptor_id,
                             uint32_t window_id,
                             uint64_t issue_cycle) {
  FrontendTransaction transaction{};
  transaction.async_id = async_id;
  transaction.type = TransactionType::Load;
  transaction.wid = wid;
  transaction.wgid = wgid;
  transaction.handle = handle;
  transaction.descriptor_id = descriptor_id;
  transaction.window_id = window_id;
  transaction.issue_cycle = issue_cycle;
  transaction.first_service_cycle = SimPlatform::instance().cycles() + 1;
  transactions_[async_id] = std::move(transaction);
  transaction_fifo_.push_back(async_id);
  live_load_busy_handles_.insert(handle);
  TmaDescriptor descriptor{};
  if (read_tma_descriptor(descriptor_id, &descriptor)) {
    auto payload_kind = static_cast<TcuPayloadKind>(descriptor.payload_kind);
    core_->tmem_set_meta_region(handle, descriptor.meta_tmem_base, descriptor.meta_col_span);
    if (payload_kind == TcuPayloadKind::SparseMeta) {
      core_->tmem_set_meta_ready(handle, false);
    } else {
      core_->tmem_set_payload_ready(handle, false);
    }
    if (payload_kind != TcuPayloadKind::SparseMeta
     && descriptor.meta_addr != 0 && descriptor.meta_size_bytes != 0 && descriptor.meta_col_span != 0) {
      core_->tmem_set_meta_ready(handle, false);
    }
  }
  return true;
}

bool TmaFrontend::issue_store(uint32_t async_id,
                              uint32_t wid,
                              uint32_t wgid,
                              uint32_t handle,
                              uint32_t descriptor_id,
                              uint32_t window_id,
                              uint64_t issue_cycle) {
  FrontendTransaction transaction{};
  transaction.async_id = async_id;
  transaction.type = TransactionType::Store;
  transaction.wid = wid;
  transaction.wgid = wgid;
  transaction.handle = handle;
  transaction.descriptor_id = descriptor_id;
  transaction.window_id = window_id;
  transaction.issue_cycle = issue_cycle;
  transaction.first_service_cycle = SimPlatform::instance().cycles() + 1;
  transactions_[async_id] = std::move(transaction);
  transaction_fifo_.push_back(async_id);
  live_store_busy_handles_.insert(handle);
  TmaDescriptor descriptor{};
  if (read_tma_descriptor(descriptor_id, &descriptor)) {
    if (static_cast<TcuPayloadKind>(descriptor.payload_kind) == TcuPayloadKind::SparseMeta) {
      core_->tmem_set_meta_ready(handle, false);
    } else {
      core_->tmem_set_payload_ready(handle, false);
    }
  }
  return true;
}

void TmaFrontend::publish_visible_state() {
  visible_load_busy_handles_ = live_load_busy_handles_;
  visible_store_busy_handles_ = live_store_busy_handles_;
}

void TmaFrontend::compact_transaction_fifo() {
  std::deque<uint32_t> compacted_fifo;
  for (auto async_id : transaction_fifo_) {
    auto it = transactions_.find(async_id);
    if (it == transactions_.end() || it->second.completed) {
      continue;
    }
    compacted_fifo.push_back(async_id);
  }
  transaction_fifo_.swap(compacted_fifo);
}

void TmaFrontend::drain_tmem_responses() {
  if (TmemRspIn.empty()) {
    return;
  }
  auto response = TmemRspIn.front();
  TmemRspIn.pop();
  auto pending_it = pending_request_to_transaction_.find(response.request_id);
  if (pending_it == pending_request_to_transaction_.end()) {
    return;
  }
  auto transaction_it = transactions_.find(pending_it->second);
  if (transaction_it == transactions_.end()) {
    pending_request_to_transaction_.erase(pending_it);
    return;
  }

  auto& transaction = transaction_it->second;
  if (transaction.type == TransactionType::Store && response.access_type == TensorMemPortReq::AccessType::Read) {
    const TmemWindowPlan* payload_window = nullptr;
    bool use_math_payload_window = core_->lookup_tmem_window(transaction.handle, transaction.window_id, &payload_window)
                                && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);
    tma_model_.store_payload_packet(transaction.tma_descriptor,
                                    transaction.payload_size_bytes,
                                    use_math_payload_window ? payload_window : nullptr,
                                    transaction.next_payload_packet_idx,
                                    response.read_packet,
                                    [this](const void* data, uint64_t addr, uint32_t size) {
                                      core_->dcache_write(data, addr, size);
                                    });
    ++transaction.next_payload_packet_idx;
    --transaction.remaining_tmem_read_packets;
    transaction.pending_tmem_request_id = 0;
  } else if (transaction.type == TransactionType::Load && response.access_type == TensorMemPortReq::AccessType::Write) {
    --transaction.remaining_tmem_write_packets;
    transaction.pending_tmem_request_id = 0;
  }

  if (transaction.type == TransactionType::Load && transaction.remaining_tmem_write_packets == 0) {
    if (transaction.payload_packet_count != 0) {
      core_->tmem_set_payload_ready(transaction.handle, true);
      uint32_t row_bytes = transaction.tma_descriptor.stride_bytes
                         ? transaction.tma_descriptor.stride_bytes
                         : (transaction.tma_descriptor.cols * transaction.tma_descriptor.elem_bytes);
      if (row_bytes != 0) {
        (void)core_->tmem_set_row_bytes(transaction.handle, row_bytes);
      }
    }
    if (transaction.meta_packet_count != 0) {
      core_->tmem_set_meta_ready(transaction.handle, true);
    }
    (void)core_->tmem_bump_window_epoch(transaction.handle);
    transaction.completed = true;
    live_load_busy_handles_.erase(transaction.handle);
    AsyncOpCompletionOut.push({transaction.async_id}, 1);
  } else if (transaction.type == TransactionType::Store && transaction.remaining_tmem_read_packets == 0) {
    if (transaction.use_meta_region) {
      core_->tmem_set_meta_ready(transaction.handle, true);
    } else {
      core_->tmem_set_payload_ready(transaction.handle, true);
    }
    transaction.completed = true;
    live_store_busy_handles_.erase(transaction.handle);
    AsyncOpCompletionOut.push({transaction.async_id}, 1);
  }

  pending_request_to_transaction_.erase(pending_it);
}

void TmaFrontend::initialize_transaction(FrontendTransaction& transaction) {
  if (transaction.transaction_initialized) {
    return;
  }

  TmaDescriptor descriptor{};
  if (!read_tma_descriptor(transaction.descriptor_id, &descriptor)) {
    transaction.completed = true;
    transaction.transaction_initialized = true;
    return;
  }
  transaction.tma_descriptor = descriptor;

  auto payload_kind = static_cast<TcuPayloadKind>(descriptor.payload_kind);
  if (payload_kind == TcuPayloadKind::SparseMeta) {
    transaction.use_meta_region = true;
    transaction.meta_size_bytes = descriptor.meta_size_bytes;
    transaction.meta_region_col_base = descriptor.meta_tmem_base;
    transaction.meta_region_col_span = descriptor.meta_col_span;
  }

  auto configure_windows = [this, &transaction, &descriptor]() -> bool {
    uint32_t transfer_region_col_base = 0;
    uint32_t transfer_region_col_span = 0;
    if (!core_->tmem_transfer_region(transaction.handle, &transfer_region_col_base, &transfer_region_col_span)) {
      return false;
    }
    transaction.transfer_region_col_base = transfer_region_col_base;
    transaction.transfer_region_col_span = transfer_region_col_span;

    auto legacy_window = build_legacy_window_plan(transaction.window_id,
                                                  descriptor,
                                                  transaction.transfer_region_col_base,
                                                  transaction.transfer_region_col_span);
    const TmemWindowPlan* existing_window = nullptr;
    bool preserve_payload_window = core_->lookup_tmem_window(transaction.handle, transaction.window_id, &existing_window)
                                && preserve_existing_math_window(existing_window, descriptor);
    if (!preserve_payload_window) {
      if (descriptor.tile_role != 0
       && fmt_elem_bytes(infer_window_fmt(descriptor)) != 0
       && descriptor.rows != 0
       && descriptor.cols != 0) {
        (void)TmemWindowPlanner::build_single_dense_window(map_window_target(descriptor),
                                                           {descriptor.rows, descriptor.cols},
                                                           infer_window_fmt(descriptor),
                                                           0,
                                                           transaction.transfer_region_col_span,
                                                           transaction.window_id,
                                                           &legacy_window,
                                                           nullptr);
        const TmemAllocation* allocation = nullptr;
        if (!core_->tmem_lookup_allocation(transaction.handle, &allocation)
         || nullptr == allocation
         || !place_window_after_existing(*allocation, &legacy_window)) {
          return false;
        }
      } else if (nullptr != existing_window) {
        adjust_legacy_subwindow_from_existing(descriptor, existing_window, &legacy_window);
      }
      (void)core_->tmem_upsert_window(transaction.handle, legacy_window);
    }
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(descriptor, transaction.window_id, &meta_window)) {
      meta_window.logical_col_base = descriptor.meta_tmem_base;
      const TmemWindowPlan* existing_meta_window = nullptr;
      bool preserve_meta_window = core_->lookup_tmem_window(transaction.handle,
                                                            meta_shadow_window_id(transaction.window_id),
                                                            &existing_meta_window)
                               && existing_meta_window->target == TmemWindowTarget::Meta
                               && TmemWindowPlanner::uses_math_packet_adapter(*existing_meta_window);
      if (!preserve_meta_window) {
        (void)core_->tmem_upsert_window(transaction.handle, meta_window);
      }
    }
    return true;
  };

  if (!configure_windows()) {
    transaction.completed = true;
    transaction.transaction_initialized = true;
    return;
  }

  if (transaction.type == TransactionType::Load) {
    uint32_t capacity = 0;
    if (!core_->tmem_query(transaction.handle, nullptr, &capacity)) {
      transaction.completed = true;
      transaction.transaction_initialized = true;
      return;
    }
    transaction.payload_size_bytes = std::min<uint32_t>(tma_model_.payload_size_bytes(descriptor), capacity);
    if (descriptor.meta_addr != 0 && descriptor.meta_size_bytes != 0 && descriptor.meta_col_span != 0) {
      transaction.meta_size_bytes = descriptor.meta_size_bytes;
      transaction.use_meta_region = true;
      transaction.meta_region_col_base = descriptor.meta_tmem_base;
      transaction.meta_region_col_span = descriptor.meta_col_span;
    }
    uint32_t payload_packets = 0;
    const TmemWindowPlan* payload_window = nullptr;
    bool use_math_payload_window = core_->lookup_tmem_window(transaction.handle, transaction.window_id, &payload_window)
                                && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);
    if (use_math_payload_window) {
      if (!core_->tmem_window_packet_count(transaction.handle, transaction.window_id, &payload_packets)) {
        transaction.completed = true;
        transaction.transaction_initialized = true;
        return;
      }
    } else {
      payload_packets = (transaction.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    const TmemWindowPlan* meta_window = nullptr;
    bool use_math_meta_window = core_->lookup_tmem_window(transaction.handle,
                                                          meta_shadow_window_id(transaction.window_id),
                                                          &meta_window)
                             && TmemWindowPlanner::uses_math_packet_adapter(*meta_window);
    uint32_t meta_packets = 0;
    if (transaction.meta_size_bytes != 0) {
      if (use_math_meta_window) {
        if (!core_->tmem_window_packet_count(transaction.handle,
                                             meta_shadow_window_id(transaction.window_id),
                                             &meta_packets)) {
          transaction.completed = true;
          transaction.transaction_initialized = true;
          return;
        }
      } else {
        meta_packets = (transaction.meta_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
      }
    }
    transaction.payload_packet_count = payload_packets;
    transaction.meta_packet_count = meta_packets;
    transaction.remaining_launch_cycles = tma_model_.estimate_load_latency(descriptor);
    transaction.remaining_tmem_write_packets = payload_packets + meta_packets;
  } else {
    uint32_t capacity = 0;
    if (!core_->tmem_query(transaction.handle, nullptr, &capacity)) {
      transaction.completed = true;
      transaction.transaction_initialized = true;
      return;
    }
    transaction.payload_size_bytes = std::min<uint32_t>(tma_model_.payload_size_bytes(descriptor), capacity);
    uint32_t payload_packets = 0;
    const TmemWindowPlan* payload_window = nullptr;
    bool use_math_payload_window = core_->lookup_tmem_window(transaction.handle, transaction.window_id, &payload_window)
                                && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);
    if (use_math_payload_window) {
      if (!core_->tmem_window_packet_count(transaction.handle, transaction.window_id, &payload_packets)) {
        transaction.completed = true;
        transaction.transaction_initialized = true;
        return;
      }
    } else {
      payload_packets = (transaction.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    transaction.payload_packet_count = payload_packets;
    transaction.remaining_launch_cycles = TmaModel::kStoreBaseLatency;
    transaction.remaining_tmem_read_packets = payload_packets;
  }

  transaction.transaction_initialized = true;
}

bool TmaFrontend::advance_one_transaction(FrontendTransaction& transaction) {
  initialize_transaction(transaction);
  if (transaction.completed) {
    if (transaction.type == TransactionType::Load) {
      live_load_busy_handles_.erase(transaction.handle);
    } else {
      live_store_busy_handles_.erase(transaction.handle);
    }
    AsyncOpCompletionOut.push({transaction.async_id}, 1);
    return true;
  }
  if (transaction.remaining_launch_cycles != 0) {
    --transaction.remaining_launch_cycles;
    return true;
  }

  if (transaction.pending_tmem_request_id != 0) {
    return true;
  }

  const TmemWindowPlan* payload_window = nullptr;
  bool use_math_payload_window = core_->lookup_tmem_window(transaction.handle, transaction.window_id, &payload_window)
                              && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);

  if (transaction.type == TransactionType::Load) {
    if (transaction.remaining_tmem_write_packets == 0) {
      transaction.completed = true;
      live_load_busy_handles_.erase(transaction.handle);
      AsyncOpCompletionOut.push({transaction.async_id}, 1);
      return true;
    }

    TensorMemPortReq request{};
    request.request_id = next_request_id_++;
    request.arbitration_age = (static_cast<uint64_t>(transaction.async_id) << 32)
                            | (transaction.next_payload_packet_idx + transaction.next_meta_packet_idx);
    request.access_type = TensorMemPortReq::AccessType::Write;

    if (transaction.next_payload_packet_idx < transaction.payload_packet_count) {
      request.port_request.kind = load_payload_request_kind(use_math_payload_window);
      request.port_request.handle = transaction.handle;
      request.port_request.window_id = transaction.window_id;
      request.port_request.packet_idx = transaction.next_payload_packet_idx;
      request.port_request.age = request.arbitration_age;
      TmemPacket packet{};
      if (!tma_model_.load_payload_packet(transaction.tma_descriptor,
                                          transaction.payload_size_bytes,
                                          use_math_payload_window ? payload_window : nullptr,
                                          transaction.next_payload_packet_idx,
                                          &packet,
                                          [this](void* data, uint64_t addr, uint32_t size) {
                                            core_->dcache_read(data, addr, size);
                                          })) {
        transaction.completed = true;
        live_load_busy_handles_.erase(transaction.handle);
        AsyncOpCompletionOut.push({transaction.async_id}, 1);
        return true;
      }
      request.write_packet = packet;
      ++transaction.next_payload_packet_idx;
    } else {
      const TmemWindowPlan* meta_window = nullptr;
      bool use_meta_window = core_->lookup_tmem_window(transaction.handle,
                                                       meta_shadow_window_id(transaction.window_id),
                                                       &meta_window);
      bool use_math_meta_window = use_meta_window && TmemWindowPlanner::uses_math_packet_adapter(*meta_window);
      request.port_request.kind = use_meta_window
                                ? (use_math_meta_window ? Tmem::PortRequestKind::WindowWrite
                                                        : Tmem::PortRequestKind::WindowLinearWrite)
                                : Tmem::PortRequestKind::RegionWrite;
      request.port_request.handle = transaction.handle;
      request.port_request.window_id = use_meta_window ? meta_shadow_window_id(transaction.window_id) : 0;
      request.port_request.packet_idx = transaction.next_meta_packet_idx;
      request.port_request.col_base = transaction.meta_region_col_base;
      request.port_request.col_span = transaction.meta_region_col_span;
      request.port_request.age = request.arbitration_age;
      TmemPacket packet{};
      if (!tma_model_.load_meta_packet(transaction.tma_descriptor,
                                       transaction.meta_size_bytes,
                                       use_math_meta_window ? meta_window : nullptr,
                                       transaction.next_meta_packet_idx,
                                       &packet,
                                       [this](void* data, uint64_t addr, uint32_t size) {
                                         core_->dcache_read(data, addr, size);
                                       })) {
        transaction.completed = true;
        live_load_busy_handles_.erase(transaction.handle);
        AsyncOpCompletionOut.push({transaction.async_id}, 1);
        return true;
      }
      request.write_packet = packet;
      ++transaction.next_meta_packet_idx;
    }

    pending_request_to_transaction_[request.request_id] = transaction.async_id;
    transaction.pending_tmem_request_id = request.request_id;
    TmemReqOut.push(request);
    return true;
  }

  if (transaction.remaining_tmem_read_packets == 0) {
    return true;
  }

  TensorMemPortReq request{};
  request.request_id = next_request_id_++;
  request.arbitration_age = (static_cast<uint64_t>(transaction.async_id) << 32) | transaction.next_payload_packet_idx;
  request.access_type = TensorMemPortReq::AccessType::Read;
  request.port_request.kind = store_payload_request_kind(use_math_payload_window);
  request.port_request.handle = transaction.handle;
  request.port_request.window_id = transaction.window_id;
  request.port_request.packet_idx = transaction.next_payload_packet_idx;
  request.port_request.age = request.arbitration_age;
  pending_request_to_transaction_[request.request_id] = transaction.async_id;
  transaction.pending_tmem_request_id = request.request_id;
  TmemReqOut.push(request);
  return true;
}

void TmaFrontend::drain_refill_chunk_requests() {
  if (RefillChunkReqIn.empty()) {
    return;
  }

  auto request = RefillChunkReqIn.front();
  RefillChunkReqIn.pop();

  TmaRefillChunkRsp response{};
  response.request_id = request.request_id;

  TmaDescriptor descriptor{};
  if (!read_tma_descriptor(request.descriptor_id, &descriptor) || descriptor.rows != 1) {
    response.success = false;
    RefillChunkRspOut.push(response);
    return;
  }

  auto row_stride_bytes = descriptor.stride_bytes ? descriptor.stride_bytes : request.row_bytes;
  response.success = tma_model_.load_row_chunk_packet(descriptor.addr,
                                                      row_stride_bytes,
                                                      0,
                                                      request.chunk_idx,
                                                      request.row_bytes,
                                                      &response.packet,
                                                      [this](void* data, uint64_t addr, uint32_t size) {
                                                        core_->dcache_read(data, addr, size);
                                                      });
  RefillChunkRspOut.push(response);
}

void TmaFrontend::tick() {
  publish_visible_state();
  drain_refill_chunk_requests();
  drain_tmem_responses();
  compact_transaction_fifo();
  for (auto async_id : transaction_fifo_) {
    auto it = transactions_.find(async_id);
    if (it == transactions_.end()) {
      continue;
    }
    auto& transaction = it->second;
    if (transaction.completed || SimPlatform::instance().cycles() < transaction.first_service_cycle) {
      continue;
    }
    if (transaction.pending_tmem_request_id != 0) {
      continue;
    }
    if (!advance_one_transaction(transaction)) {
      break;
    }
  }
  compact_transaction_fifo();
}

void TmaFrontend::dump_debug_state(std::ostream& os) const {
  os << "[TmaFrontend] transactions=" << transactions_.size()
     << " fifo_depth=" << transaction_fifo_.size()
     << " pending_requests=" << pending_request_to_transaction_.size()
     << " live_load_busy=" << live_load_busy_handles_.size()
     << " live_store_busy=" << live_store_busy_handles_.size()
     << "\n";
  for (const auto& entry : transactions_) {
    const auto& transaction = entry.second;
    os << "  async_id=" << transaction.async_id
       << " type=" << static_cast<uint32_t>(transaction.type)
       << " handle=" << transaction.handle
       << " window=" << transaction.window_id
       << " initialized=" << transaction.transaction_initialized
       << " completed=" << transaction.completed
       << " launch=" << transaction.remaining_launch_cycles
       << " rem_read_pkts=" << transaction.remaining_tmem_read_packets
       << " rem_write_pkts=" << transaction.remaining_tmem_write_packets
       << " next_payload_pkt=" << transaction.next_payload_packet_idx
       << " next_meta_pkt=" << transaction.next_meta_packet_idx
       << " pending_req=" << transaction.pending_tmem_request_id
       << "\n";
  }
}
