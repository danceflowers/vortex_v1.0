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

// ============================================================================
// TensorUnit -- near-core TensorCore frontend.
// ============================================================================
//
// TensorUnit sits between execute stage and the TensorCore datapath. It decodes
// tcgen05.mma descriptor operands, moves TMEM packets into local operand SRAMs,
// expands one macro MMA into 8 primitive TensorCore issues, drains the D result
// back to TMEM, and emits TensorAsyncOpCompletion when the macro operation
// completes. The implementation is hidden behind Impl to keep this public
// header stable and avoid exposing the internal queues and SRAM state.
// ============================================================================

#include <algorithm>
#include <iosfwd>
#include <simobject.h>
#include "instr_trace.h"
#include "tensor_mem_port_types.h"
#include "open_tensorcore/tensor_helper/tensor_debug_utils.h"

namespace vortex {

class Core;

class TensorUnit : public SimObject<TensorUnit> {
public:

  // Execution trace payload used by the scheduler for writeback/retry feedback.
  struct ExeTraceData : public ITraceData {
    using Ptr = std::shared_ptr<ExeTraceData>;
    bool rd_write = false;   // Whether the instruction writes a destination register.
    bool retry = false;      // Whether the scheduler should reissue later.
  };

  // TensorUnit performance counters. They track useful pipeline occupancy and
  // scheduler-visible stalls while keeping setup/epilogue cycle markers internal.
	struct PerfStats {
		uint64_t latency;
    uint64_t tc_active_cycles;
    uint64_t issued_primitive_tiles;
    uint64_t retired_primitive_tiles;
    uint64_t setup_end_cycle;
    uint64_t epilogue_begin_cycle;
    uint64_t stall_a_not_ready;
    uint64_t stall_b_not_ready;
    uint64_t stall_c_not_ready;
    uint64_t stall_tc_busy;
    uint64_t stall_no_wmma_job_ready;
    uint64_t stall_no_tensor_instr_candidate;
    uint64_t stall_mma_load_taddr_not_ready;
    uint64_t stall_taddr_reuse;
    uint64_t stall_taddr_busy_due_to_tma_load;
    uint64_t stall_taddr_busy_due_to_tma_store_or_shift;
    uint64_t stall_slot_busy;
    uint64_t stall_tmem_read_port_busy;
    uint64_t stall_tmem_write_port_busy;
    uint64_t issued_macro_wmma;
    uint64_t retired_macro_wmma;
    uint64_t pending_wmma_jobs_max;
    uint64_t stall_a_meta_not_ready;
    uint64_t stall_no_wmma_job_builder_empty;
    uint64_t stall_no_wmma_waiting_for_mma_load;
    uint64_t stall_no_wmma_waiting_for_taddr_alloc;
    uint64_t stall_no_wmma_waiting_for_slot_release;

		PerfStats()
			: latency(0)
      , tc_active_cycles(0)
      , issued_primitive_tiles(0)
      , retired_primitive_tiles(0)
      , setup_end_cycle(0)
      , epilogue_begin_cycle(0)
      , stall_a_not_ready(0)
      , stall_b_not_ready(0)
      , stall_c_not_ready(0)
      , stall_tc_busy(0)
      , stall_no_wmma_job_ready(0)
      , stall_no_tensor_instr_candidate(0)
      , stall_mma_load_taddr_not_ready(0)
      , stall_taddr_reuse(0)
      , stall_taddr_busy_due_to_tma_load(0)
      , stall_taddr_busy_due_to_tma_store_or_shift(0)
      , stall_slot_busy(0)
      , stall_tmem_read_port_busy(0)
      , stall_tmem_write_port_busy(0)
      , issued_macro_wmma(0)
      , retired_macro_wmma(0)
      , pending_wmma_jobs_max(0)
      , stall_a_meta_not_ready(0)
      , stall_no_wmma_job_builder_empty(0)
      , stall_no_wmma_waiting_for_mma_load(0)
      , stall_no_wmma_waiting_for_taddr_alloc(0)
      , stall_no_wmma_waiting_for_slot_release(0)
		{}

		PerfStats& operator+=(const PerfStats& rhs) {
			this->latency += rhs.latency;
      this->tc_active_cycles += rhs.tc_active_cycles;
      this->issued_primitive_tiles += rhs.issued_primitive_tiles;
      this->retired_primitive_tiles += rhs.retired_primitive_tiles;
      if (0 == this->setup_end_cycle || (rhs.setup_end_cycle != 0 && rhs.setup_end_cycle < this->setup_end_cycle))
        this->setup_end_cycle = rhs.setup_end_cycle;
      this->epilogue_begin_cycle = std::max(this->epilogue_begin_cycle, rhs.epilogue_begin_cycle);
      this->stall_a_not_ready += rhs.stall_a_not_ready;
      this->stall_b_not_ready += rhs.stall_b_not_ready;
      this->stall_c_not_ready += rhs.stall_c_not_ready;
      this->stall_tc_busy += rhs.stall_tc_busy;
      this->stall_no_wmma_job_ready += rhs.stall_no_wmma_job_ready;
      this->stall_no_tensor_instr_candidate += rhs.stall_no_tensor_instr_candidate;
      this->stall_mma_load_taddr_not_ready += rhs.stall_mma_load_taddr_not_ready;
      this->stall_taddr_reuse += rhs.stall_taddr_reuse;
      this->stall_taddr_busy_due_to_tma_load += rhs.stall_taddr_busy_due_to_tma_load;
      this->stall_taddr_busy_due_to_tma_store_or_shift += rhs.stall_taddr_busy_due_to_tma_store_or_shift;
      this->stall_slot_busy += rhs.stall_slot_busy;
      this->stall_tmem_read_port_busy += rhs.stall_tmem_read_port_busy;
      this->stall_tmem_write_port_busy += rhs.stall_tmem_write_port_busy;
      this->issued_macro_wmma += rhs.issued_macro_wmma;
      this->retired_macro_wmma += rhs.retired_macro_wmma;
      this->pending_wmma_jobs_max = std::max(this->pending_wmma_jobs_max, rhs.pending_wmma_jobs_max);
      this->stall_a_meta_not_ready += rhs.stall_a_meta_not_ready;
      this->stall_no_wmma_job_builder_empty += rhs.stall_no_wmma_job_builder_empty;
      this->stall_no_wmma_waiting_for_mma_load += rhs.stall_no_wmma_waiting_for_mma_load;
      this->stall_no_wmma_waiting_for_taddr_alloc += rhs.stall_no_wmma_waiting_for_taddr_alloc;
      this->stall_no_wmma_waiting_for_slot_release += rhs.stall_no_wmma_waiting_for_slot_release;
			return *this;
		}
	};

  // Scheduler-visible reason why TensorUnit rejected an issue attempt.
  enum class IssueBlockReason : uint8_t {
    None = 0,
    ANotReady,
    BNotReady,
    CNotReady,
    TcBusy,
    NoTensorInstrCandidate,
    MmaLoadTaddrNotReady,
    TaddrBusyDueToTmaLoad,
    TaddrBusyDueToTmaStoreOrShift,
    TaddrReuse,
    SlotBusy,
    AMetaNotReady,
  };

  // Instruction input ports from execute stage, one per issue lane.
  std::vector<SimPort<instr_trace_t*>> Inputs;
  // Delayed instruction output ports, one per issue lane.
	std::vector<SimPort<instr_trace_t*>> Outputs;
//#ifdef EXT_TCU_ENABLE
  // Packet-level boundary between TensorUnit and TmemSystem.
  SimPort<TensorMemPortReq> TensorMemReqOut;           // TMEM read/write requests.
  SimPort<TensorMemPortRsp> TensorMemRspIn;            // TMEM read/write responses.
  SimPort<TensorAsyncOpCompletion> TensorAsyncOpCompletionOut; // Async completion events.
//#endif

  TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core);
  virtual ~TensorUnit();

  virtual void reset();

  virtual void tick();

  // Unified coprocessor dispatch path. The core forwards raw fields and
  // TensorUnit uses TcDecode to interpret instruction semantics.
  // PTX path: each tcgen05.mma carries its idesc directly.

  /// Phase-2 PTX-aligned tcgen05.mma dispatch (single-instruction fan-out).
  /// rs1_value carries i_descriptor_t, rs2_value points to operand_block_t in
  /// LMEM, and qualifier carries funct7 mode bits. TensorUnit decodes those
  /// fields and runs the fill -> compute -> drain state machine.
  void dispatch_tcu_mma(uint32_t wid,
                        uint32_t rs1_value,
                        uint32_t rs2_value,
                        uint32_t qualifier,
                        ExeTraceData* trace_data);

  // Query hooks used by the warp scheduler before issuing tensor instructions.

  uint32_t scheduler_score(uint32_t wid, TcuType tcu_type) const;
  IssueBlockReason classify_issue_block(uint32_t wid, TcuType tcu_type) const;
  void record_issue_stall(IssueBlockReason reason);
  void record_no_tensor_instr_candidate_stall();

	const PerfStats& perf_stats() const;

  void dump_debug_state(std::ostream& os) const;

private:
  // Impl owns local SRAMs, queues, TensorCore instance, and all transient state.
	class Impl;
	Impl* impl_;
};

} // namespace vortex
