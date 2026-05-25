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

// PTX TMEM system module.
//
// Hardware view:
// - owns the scratchpad allocation table and banked SRAM
// - arbitrates all 64 B packet traffic through one shared request FIFO
// - services TensorUnit region packet requests
// - owns tcgen05.shift because shift mutates TMEM state asynchronously
class TmemSystem : public SimObject<TmemSystem> {
public:
  SimPort<TensorMemPortReq> TensorExecuteReqIn;
  SimPort<TensorMemPortRsp> TensorExecuteRspOut;
  SimPort<TensorMemPortReq> CoreTransferReqIn;
  SimPort<TensorMemPortRsp> CoreTransferRspOut;
  SimPort<TensorAsyncOpCompletion> AsyncOpCompletionOut;

  TmemSystem(const SimContext& ctx, const char* name, Core* core);

  // Reset TMEM storage plus all in-flight SimPort and shift-engine state.
  void reset();

  // Publish visible state, drain ports, arbitrate TMEM packets, and advance shifts.
  void tick();

  uint32_t alloc(uint32_t col_span);
  bool free(uint32_t taddr);
  void seal_allocator();
  bool allocator_sealed() const;

  bool lookup_allocation(uint32_t taddr, TmemAllocation** allocation);
  bool lookup_allocation(uint32_t taddr, const TmemAllocation** allocation) const;
  bool region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const;
  bool query(uint32_t taddr, uint32_t* col_span, uint32_t* size_bytes) const;
  bool set_row_bytes(uint32_t taddr, uint32_t row_bytes);
  bool row_bytes(uint32_t taddr, uint32_t* row_bytes) const;

  bool taddr_read_bytes(uint32_t taddr, uint32_t byte_offset, uint8_t* dst, uint32_t bytes) const {
    return tmem_.taddr_read_bytes(taddr, byte_offset, dst, bytes);
  }
  bool taddr_write_bytes(uint32_t taddr, uint32_t byte_offset, const uint8_t* src, uint32_t bytes) {
    return tmem_.taddr_write_bytes(taddr, byte_offset, src, bytes);
  }
  bool find_allocation_by_lane(uint32_t lane, uint32_t* col_base) const {
    return tmem_.find_allocation_by_lane(lane, col_base);
  }

  void set_payload_ready(uint32_t taddr, bool ready, bool update_visible_now = false);
  void set_meta_ready(uint32_t taddr, bool ready, bool update_visible_now = false);
  void set_meta_region(uint32_t taddr, uint32_t meta_col_base, uint32_t meta_col_span);

  uint64_t enqueue_port_request(const Tmem::PortRequestDesc& desc, uint64_t arbitration_age);
  bool request_granted(uint64_t request_tag) const;
  void consume_request_grant(uint64_t request_tag);

  // Launch an asynchronous tcgen05.shift operation through the shared TMEM ports.
  bool issue_shift(uint32_t async_id,
                   uint32_t wid,
                   uint32_t taddr,
                   uint64_t issue_cycle);

  bool visible_payload_ready(uint32_t taddr) const;
  bool visible_meta_ready(uint32_t taddr) const;
  bool visible_shift_busy(uint32_t taddr) const {
    return visible_shift_busy_taddrs_.count(taddr) != 0;
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
    uint32_t taddr = 0;
    uint64_t issue_cycle = 0;
    uint64_t first_service_cycle = 0;
    bool completed = false;
    bool initialized = false;
    uint32_t packet_count = 0;
    uint32_t next_packet_idx = 0;
    uint64_t pending_read_request_tag = 0;
    uint64_t pending_write_request_tag = 0;
    bool read_complete = false;
    bool write_complete = false;
  };

  Core* core_;
  Tmem tmem_;
  std::unordered_map<uint64_t, PendingPortRequest> pending_tensor_execute_requests_;
  std::unordered_map<uint64_t, PendingPortRequest> pending_core_transfer_requests_;
  std::unordered_map<uint32_t, ShiftTransaction> shift_transactions_;
  std::deque<uint32_t> shift_transaction_fifo_;
  std::unordered_set<uint32_t> live_shift_busy_taddrs_;
  std::unordered_set<uint32_t> visible_shift_busy_taddrs_;

  // Remove completed or erased shift ids from the front of the service FIFO.
  void compact_shift_transaction_fifo();

  // Bridge inbound SimPort packet requests into the TMEM arbitration queue.
  void drain_one_inbound_request(SimPort<TensorMemPortReq>& request_port,
                                 std::unordered_map<uint64_t, PendingPortRequest>& pending_requests);

  // Return granted packet requests to the original requester port.
  void complete_granted_port_requests(std::unordered_map<uint64_t, PendingPortRequest>& pending_requests,
                                      SimPort<TensorMemPortRsp>& response_port);

  // Set up packet count and metadata for a shift transaction on first service.
  void initialize_shift_transaction(ShiftTransaction& transaction);

  // Issue one read/write packet pair for a shift transaction when ports are free.
  void issue_shift_requests(ShiftTransaction& transaction);

  // Consume granted packets and finish the data movement for a shift transaction.
  void complete_shift_transaction_step(ShiftTransaction& transaction);

  // Progress all shift transactions that are ready to be serviced this cycle.
  void advance_shift_engine();
};

} // namespace vortex
