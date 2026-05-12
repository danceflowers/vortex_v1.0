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

struct TmaCpAsyncBulkResult {
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
// - keeps the existing functional DRAM/LMEM behavior by using Core's memory
//   callbacks; CacheReqOut/CacheRspIn are exposed for future cache-timing
//   wiring without changing the current data path semantics
class Tma : public SimObject<Tma> {
public:
  SimPort<TensorMemPortReq> TmemReqOut;
  SimPort<TensorMemPortRsp> TmemRspIn;
  SimPort<TensorAsyncOpCompletion> AsyncOpCompletionOut;
  std::vector<SimPort<MemReq>> CacheReqOut;
  std::vector<SimPort<MemRsp>> CacheRspIn;

  Tma(const SimContext& ctx, const char* name, Core* core);

  void reset();
  void tick();

  static bool is_lmem_addr(uint64_t addr);
  static uint32_t element_type_bytes(uint8_t element_type);

  TmaCpAsyncBulkResult cpabulk_tensor_load(uint64_t tensor_map_addr,
                                           uint64_t args_lmem_ptr,
                                           bool complete_tx);

  TmaCpAsyncBulkResult cpabulk_tensor_store(uint64_t tensor_map_addr,
                                            uint64_t args_lmem_ptr);

  bool issue_lmem_to_tmem_copy(uint32_t async_id,
                               uint32_t wid,
                               uint32_t wgid,
                               uint32_t col_base,
                               uint32_t col_span,
                               uint64_t lmem_addr,
                               uint32_t byte_offset,
                               uint32_t total_bytes);

  void dump_debug_state(std::ostream& os) const;

private:
  enum class LmemToTmemCopyStage : uint8_t {
    Ready = 0,
    WaitRead,
    WaitWrite,
  };

  struct PendingLmemToTmemCopyOp {
    uint32_t async_id = 0;
    uint32_t wid = 0;
    uint32_t wgid = 0;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
    uint64_t lmem_addr = 0;
    uint32_t byte_offset = 0;
    uint32_t total_bytes = 0;
    uint32_t cursor = 0;
    LmemToTmemCopyStage stage = LmemToTmemCopyStage::Ready;
    uint64_t request_id = 0;
    uint32_t packet_idx = 0;
    uint32_t packet_offset = 0;
    uint32_t packet_bytes = 0;
    TmemPacket packet = {};
  };

  Core* core_;
  std::unordered_map<uint32_t, PendingLmemToTmemCopyOp> pending_lmem_to_tmem_copy_ops_;
  std::unordered_map<uint64_t, TensorMemPortRsp> completed_lmem_to_tmem_transfer_responses_;
  uint64_t next_lmem_to_tmem_request_id_;

  void drain_tmem_responses();
  void advance_lmem_to_tmem_copy_ops();
};

} // namespace vortex
