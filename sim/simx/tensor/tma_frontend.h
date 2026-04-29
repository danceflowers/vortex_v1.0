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
#include "tma.h"

namespace vortex {

class Core;

// TMA front-end module.
//
// Hardware view:
// - owns descriptor fetch and launch/setup timing
// - marshals row-major math windows into packet-sized reorder buffers
// - emits TMEM packet requests through the TensorMemSystem ingress port
// - receives packet responses from TMEM for store direction
// - reports architecturally visible completion back to Core
class TmaFrontend : public SimObject<TmaFrontend> {
public:
  SimPort<TensorMemPortReq> TmemReqOut;
  SimPort<TensorMemPortRsp> TmemRspIn;
  SimPort<TmaRefillChunkReq> RefillChunkReqIn;
  SimPort<TmaRefillChunkRsp> RefillChunkRspOut;
  SimPort<TensorAsyncOpCompletion> AsyncOpCompletionOut;

  TmaFrontend(const SimContext& ctx,
              const char* name,
              Core* core,
              bool realistic_load);

  void reset();
  void tick();

  bool read_tma_descriptor(uint32_t desc_id, TmaDescriptor* out);
  bool read_idescriptor(uint32_t desc_id, IDescriptor* out);
  uint32_t estimate_load_latency(const TmaDescriptor& desc) const;
  uint32_t payload_size_bytes(const TmaDescriptor& desc) const;
  bool issue_load(uint32_t async_id,
                  uint32_t wid,
                  uint32_t wgid,
                  uint32_t handle,
                  uint32_t descriptor_id,
                  uint32_t window_id,
                  uint64_t issue_cycle);

  bool issue_store(uint32_t async_id,
                   uint32_t wid,
                   uint32_t wgid,
                   uint32_t handle,
                   uint32_t descriptor_id,
                   uint32_t window_id,
                   uint64_t issue_cycle);

  bool visible_load_busy(uint32_t handle) const {
    return visible_load_busy_handles_.count(handle) != 0;
  }

  bool visible_store_busy(uint32_t handle) const {
    return visible_store_busy_handles_.count(handle) != 0;
  }

  void dump_debug_state(std::ostream& os) const;

private:
  enum class TransactionType : uint8_t {
    Load = 0,
    Store,
  };

  struct FrontendTransaction {
    uint32_t async_id = 0;
    TransactionType type = TransactionType::Load;
    uint32_t wid = 0;
    uint32_t wgid = 0;
    uint32_t handle = 0;
    uint32_t descriptor_id = 0;
    uint32_t window_id = 0;
    uint64_t issue_cycle = 0;
    uint64_t first_service_cycle = 0;
    bool completed = false;
    bool transaction_initialized = false;
    uint32_t remaining_launch_cycles = 0;
    uint32_t remaining_tmem_read_packets = 0;
    uint32_t remaining_tmem_write_packets = 0;
    uint32_t next_payload_packet_idx = 0;
    uint32_t next_meta_packet_idx = 0;
    uint64_t pending_tmem_request_id = 0;
    uint32_t payload_size_bytes = 0;
    uint32_t meta_size_bytes = 0;
    uint32_t transfer_region_col_base = 0;
    uint32_t transfer_region_col_span = 0;
    uint32_t meta_region_col_base = 0;
    uint32_t meta_region_col_span = 0;
    uint32_t payload_packet_count = 0;
    uint32_t meta_packet_count = 0;
    bool use_meta_region = false;
    TmaDescriptor tma_descriptor = {};
  };

  Core* core_;
  TmaModel tma_model_;
  std::unordered_map<uint32_t, FrontendTransaction> transactions_;
  std::deque<uint32_t> transaction_fifo_;
  std::unordered_map<uint64_t, uint32_t> pending_request_to_transaction_;
  std::unordered_set<uint32_t> live_load_busy_handles_;
  std::unordered_set<uint32_t> visible_load_busy_handles_;
  std::unordered_set<uint32_t> live_store_busy_handles_;
  std::unordered_set<uint32_t> visible_store_busy_handles_;
  uint64_t next_request_id_;

  void publish_visible_state();
  void compact_transaction_fifo();
  void drain_tmem_responses();
  void drain_refill_chunk_requests();
  void initialize_transaction(FrontendTransaction& transaction);
  bool advance_one_transaction(FrontendTransaction& transaction);
};

} // namespace vortex
