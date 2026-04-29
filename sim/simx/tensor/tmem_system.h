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

#include <deque>
#include <iosfwd>
#include <unordered_map>
#include <unordered_set>
#include <simobject.h>
#include "tensor_mem_port_types.h"
#include "tmem.h"

namespace vortex {

class Core;

// TMEM system module.
//
// Hardware view:
// - owns the scratchpad allocation table, window/layout state and banked SRAM
// - arbitrates all packet traffic through one shared request FIFO
// - services both TensorExecuteSystem requests and TMA-front-end requests
// - owns TMEM_SHIFT because shift mutates internal TMEM window state
class TmemSystem : public SimObject<TmemSystem> {
public:
  SimPort<TensorMemPortReq> TensorExecuteReqIn;
  SimPort<TensorMemPortRsp> TensorExecuteRspOut;
  SimPort<TensorMemPortReq> TmaFrontendReqIn;
  SimPort<TensorMemPortRsp> TmaFrontendRspOut;
  SimPort<TmaRefillChunkReq> RefillChunkReqOut;
  SimPort<TmaRefillChunkRsp> RefillChunkRspIn;
  SimPort<TensorAsyncOpCompletion> AsyncOpCompletionOut;

  TmemSystem(const SimContext& ctx, const char* name, Core* core);

  void reset();
  void tick();

  uint32_t alloc(uint32_t col_span);
  bool free(uint32_t handle);
  void seal_allocator();
  bool allocator_sealed() const;

  bool lookup_allocation(uint32_t handle, TmemAllocation** allocation);
  bool lookup_allocation(uint32_t handle, const TmemAllocation** allocation) const;
  bool region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const;
  bool query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const;
  bool lookup_window(uint32_t handle, uint32_t window_id, const TmemWindowPlan** out) const;
  bool lookup_window(uint32_t handle, uint32_t window_id, TmemWindowPlan** out);
  bool bind_layout(uint32_t handle, const TmemLayoutPlan& layout);
  bool upsert_window(uint32_t handle, const TmemWindowPlan& window);
  bool window_epoch(uint32_t handle, uint32_t* epoch) const;
  bool bump_window_epoch(uint32_t handle);
  bool set_row_bytes(uint32_t handle, uint32_t row_bytes);
  bool row_bytes(uint32_t handle, uint32_t* row_bytes) const;

  bool read_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const;
  bool read_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) const;

  // Phase-3.3.1 GAP-4: handle-based byte-range R/W (delegates to internal Tmem).
  bool handle_region_read_bytes(uint32_t handle, uint32_t byte_offset, uint8_t* dst, uint32_t bytes) const {
    return tmem_.handle_region_read_bytes(handle, byte_offset, dst, bytes);
  }
  bool handle_region_write_bytes(uint32_t handle, uint32_t byte_offset, const uint8_t* src, uint32_t bytes) {
    return tmem_.handle_region_write_bytes(handle, byte_offset, src, bytes);
  }
  // Phase-3.4 Stage 0: PTX TADDR lane → owning allocation's col_base lookup.
  bool find_allocation_by_lane(uint32_t lane, uint32_t* col_base) const {
    return tmem_.find_allocation_by_lane(lane, col_base);
  }
  bool write_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in);
  bool write_window_linear_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in);
  bool region_write_packet(uint32_t col_base, uint32_t col_span, uint32_t packet_idx, const TmemPacket& in);
  bool window_packet_count(uint32_t handle, uint32_t window_id, uint32_t* count) const;
  bool shift_window_math_rows_down(uint32_t handle,
                                   uint32_t window_id,
                                   const uint8_t* refill_math_row,
                                   uint32_t refill_math_row_bytes);
  bool region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes);
  void set_payload_ready(uint32_t handle, bool ready, bool update_visible_now = false);
  void set_meta_ready(uint32_t handle, bool ready, bool update_visible_now = false);
  void set_meta_region(uint32_t handle, uint32_t meta_col_base, uint32_t meta_col_span);

  uint64_t enqueue_port_request(const Tmem::PortRequestDesc& desc, uint64_t arbitration_age);
  bool request_granted(uint64_t request_tag) const;
  void consume_request_grant(uint64_t request_tag);

  bool issue_shift(uint32_t async_id,
                   uint32_t wid,
                   uint32_t wgid,
                   uint32_t handle,
                   uint32_t window_id,
                   uint32_t refill_descriptor_id,
                   uint64_t issue_cycle);

  bool visible_payload_ready(uint32_t handle) const;
  bool visible_meta_ready(uint32_t handle) const;
  bool visible_shift_busy(uint32_t handle) const {
    return visible_shift_busy_handles_.count(handle) != 0;
  }

  void publish_visible_state();
  void dump_debug_state(std::ostream& os) const;

private:
  struct PendingPortRequest {
    TensorMemPortReq port_request = {};
    uint64_t tmem_request_tag = 0;
  };

  struct ShiftTransaction {
    uint32_t async_id = 0;
    uint32_t wid = 0;
    uint32_t wgid = 0;
    uint32_t handle = 0;
    uint32_t window_id = 0;
    uint32_t refill_descriptor_id = 0;
    uint64_t issue_cycle = 0;
    uint64_t first_service_cycle = 0;
    bool completed = false;
    bool transaction_initialized = false;
    bool main_shift_body_complete = false;
    bool uses_math_row_shift = false;
    bool uses_logical_line_shift = false;
    uint32_t shift_math_row_bytes = 0;
    uint32_t current_shift_math_row = 0;
    uint32_t current_shift_logical_line = 0;
    uint32_t shift_line_chunk_count = 0;
    uint32_t remaining_tmem_read_packets = 0;
    uint32_t remaining_tmem_write_packets = 0;
    uint32_t next_shift_body_packet_ordinal = 0;
    std::vector<uint32_t> current_shift_row_packet_ids;
    uint64_t pending_shift_read_request_tag = 0;
    uint64_t pending_shift_write_request_tag = 0;
    bool current_shift_body_read_complete = false;
    bool current_shift_body_write_complete = false;
    std::vector<uint8_t> shift_row_reorder_buffer;
    uint32_t refill_math_row_bytes = 0;
    uint32_t remaining_refill_line_packets = 0;
    uint32_t next_refill_line_packet_idx = 0;
    uint64_t pending_refill_fetch_request_id = 0;
    uint64_t pending_refill_write_request_tag = 0;
    bool pending_refill_packet_valid = false;
    TmemPacket pending_refill_packet = {};
    std::vector<uint8_t> refill_row_reorder_buffer;
  };

  Core* core_;
  Tmem tmem_;
  std::unordered_map<uint64_t, PendingPortRequest> pending_tensor_execute_requests_;
  std::unordered_map<uint64_t, PendingPortRequest> pending_tma_frontend_requests_;
  std::unordered_map<uint64_t, uint32_t> pending_refill_request_to_transaction_;
  std::unordered_map<uint32_t, ShiftTransaction> shift_transactions_;
  std::deque<uint32_t> shift_transaction_fifo_;
  std::unordered_set<uint32_t> live_shift_busy_handles_;
  std::unordered_set<uint32_t> visible_shift_busy_handles_;
  uint64_t next_refill_request_id_ = 1;

  void compact_shift_transaction_fifo();
  void drain_one_inbound_request();
  void drain_refill_chunk_responses();
  void complete_granted_tensor_execute_requests();
  void complete_granted_tma_frontend_requests();
  void initialize_shift_transaction(ShiftTransaction& transaction);
  bool prepare_next_shift_math_row(ShiftTransaction& transaction);
  bool prepare_next_shift_logical_line(ShiftTransaction& transaction);
  void issue_shift_requests(ShiftTransaction& transaction);
  void complete_shift_transaction_step(ShiftTransaction& transaction);
  void advance_shift_engine();
};

} // namespace vortex
