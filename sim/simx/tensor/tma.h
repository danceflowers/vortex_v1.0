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

#include <cstdint>
#include <iosfwd>
#include <unordered_map>
#include <vector>
#include <simobject.h>
#include "tensor_mem_port_types.h"
#include "types.h"

namespace vortex {

class Core;

// Result metadata returned by the functional cp.async.bulk.tensor path.
struct cpabulk_transfer_result_t {
  uint32_t payload_size_bytes = 0;
  uint64_t tx_bound_mbar = 0;
  uint32_t tx_bytes = 0;
};

// Tensor Memory Accelerator front-end.
//
// Hardware view:
// - owns the TMA-facing data movement helpers that were previously embedded in
//   Core
// - talks to TMEM through packet-level SimPorts
// - talks to the socket-to-L2 path through CacheReqOut/CacheRspIn; descriptor
//   and payload bytes become functionally visible only after timing responses
class Tma : public SimObject<Tma> {
public:
  SimPort<TensorMemPortReq> TmemReqOut;
  SimPort<TensorMemPortRsp> TmemRspIn;
  SimPort<TensorAsyncOpCompletion> AsyncOpCompletionOut;
  std::vector<SimPort<MemReq>> CacheReqOut;
  std::vector<SimPort<MemRsp>> CacheRspIn;

  Tma(const SimContext& ctx, const char* name, Core* core);

  // Reset all outstanding TMA-side transactions and response bookkeeping.
  void reset();

  // Advance the TMA engine by one simulator cycle.
  void tick();

  // Return true when addr points into the Core local-memory address window.
  static bool is_lmem_addr(uint64_t addr);

  // Decode the CUtensorMap element type into the byte size used by this CModel.
  static uint32_t element_type_bytes(uint8_t element_type);

  // Execute cp.async.bulk.tensor.global.shared: DRAM tensor map -> LMEM payload.
  cpabulk_transfer_result_t cpabulk_tensor_load(uint32_t async_id,
                                                uint64_t tensor_map_addr,
                                                uint64_t args_lmem_ptr,
                                                bool complete_tx);

  // Execute cp.async.bulk.tensor.shared.global: LMEM payload -> DRAM tensor map.
  cpabulk_transfer_result_t cpabulk_tensor_store(uint32_t async_id,
                                                 uint64_t tensor_map_addr,
                                                 uint64_t args_lmem_ptr);

  // Start an asynchronous LMEM -> TMEM packet copy used by tcgen05.cp.
  bool issue_lmem_to_tmem_copy(uint32_t async_id,
                               uint32_t wid,
                               uint32_t col_base,
                               uint32_t col_span,
                               uint64_t lmem_addr,
                               uint32_t byte_offset,
                               uint32_t total_bytes);

  void dump_debug_state(std::ostream& os) const;

private:
  enum class lmem_to_tmem_copy_stage_t : uint8_t {
    Ready = 0,
    WaitRead,
    WaitWrite,
  };

  enum class cpabulk_cache_stage_t : uint8_t {
    Descriptor = 0,
    Payload,
  };

  struct pending_cpabulk_cache_request_t {
    uint32_t async_id = 0;
    cpabulk_cache_stage_t stage = cpabulk_cache_stage_t::Descriptor;
    uint32_t offset = 0;
    uint32_t bytes = 0;
  };

  struct pending_cpabulk_transfer_t {
    uint32_t async_id = 0;
    bool is_store = false;
    bool descriptor_ready = false;
    uint64_t tensor_map_addr = 0;
    uint64_t global_addr = 0;
    uint64_t lmem_addr = 0;
    uint64_t tx_bound_mbar = 0;
    uint32_t coords[5] = {};
    uint32_t total_bytes = 0;
    uint32_t descriptor_next_offset = 0;
    uint32_t descriptor_completed_bytes = 0;
    uint32_t payload_next_offset = 0;
    uint32_t payload_completed_bytes = 0;
    uint32_t inflight_cache_requests = 0;
  };

  struct pending_lmem_to_tmem_copy_t {
    uint32_t async_id = 0;
    uint32_t wid = 0;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
    uint64_t lmem_addr = 0;
    uint32_t byte_offset = 0;
    uint32_t total_bytes = 0;
    uint32_t cursor = 0;
    lmem_to_tmem_copy_stage_t stage = lmem_to_tmem_copy_stage_t::Ready;
    uint64_t request_id = 0;
    uint32_t packet_idx = 0;
    uint32_t packet_offset = 0;
    uint32_t packet_bytes = 0;
    TmemPacket packet = {};
  };

  Core* core_;
  std::unordered_map<uint32_t, pending_cpabulk_transfer_t> pending_cpabulk_transfers_;
  std::unordered_map<uint64_t, pending_cpabulk_cache_request_t> pending_cpabulk_cache_requests_;
  std::unordered_map<uint32_t, pending_lmem_to_tmem_copy_t> pending_lmem_to_tmem_copies_;
  std::unordered_map<uint64_t, TensorMemPortRsp> completed_lmem_to_tmem_responses_;
  uint64_t next_cache_request_id_;
  uint64_t next_lmem_to_tmem_request_id_;

  void issue_cache_timing_request(pending_cpabulk_transfer_t& op,
                                  cpabulk_cache_stage_t stage,
                                  uint64_t addr,
                                  uint32_t offset,
                                  uint32_t bytes);

  // Retire completed L2/global timing responses back into cp.async.bulk ops.
  void drain_cache_responses();

  // Issue descriptor and payload requests for in-flight cp.async.bulk ops.
  void advance_cpabulk_transfer_ops();

  // Collect completed TMEM packet responses and make them visible to copy ops.
  void drain_tmem_responses();

  // Drive one step of each in-flight LMEM -> TMEM packet-copy state machine.
  void advance_lmem_to_tmem_copy_ops();
};

} // namespace vortex
