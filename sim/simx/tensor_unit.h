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

#include <algorithm>
#include <simobject.h>
#include "instr_trace.h"

namespace vortex {

class Core;

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

class TensorUnit : public SimObject<TensorUnit> {
public:

  struct ExeTraceData : public ITraceData {
    using Ptr = std::shared_ptr<ExeTraceData>;
    bool rd_write = false;
    bool retry = false;
  };

	struct PerfStats {
		uint64_t latency;
    uint64_t tc_active_cycles;
    uint64_t tc_idle_cycles;
    uint64_t issued_primitive_tiles;
    uint64_t retired_primitive_tiles;
    uint64_t first_tc_issue_cycle;
    uint64_t last_tc_issue_cycle;
    uint64_t first_tc_retire_cycle;
    uint64_t last_tc_retire_cycle;
    uint64_t stall_a_not_ready;
    uint64_t stall_b_not_ready;
    uint64_t stall_c_not_ready;
    uint64_t stall_tc_busy;
    uint64_t stall_no_wmma_job_ready;
    uint64_t stall_no_tensor_instr_candidate;
    uint64_t stall_mma_load_handle_not_ready;
    uint64_t stall_handle_reuse;
    uint64_t stall_handle_busy_due_to_tma_load;
    uint64_t stall_handle_busy_due_to_tma_store_or_shift;
    uint64_t stall_slot_busy;
    uint64_t stall_tmem_read_port_busy;
    uint64_t stall_tmem_write_port_busy;
    uint64_t stall_amem_port_busy;
    uint64_t stall_bmem_port_busy;
    uint64_t stall_cmem_port_busy;
    uint64_t stall_meta_port_busy;
    uint64_t issued_macro_wmma;
    uint64_t retired_macro_wmma;
    uint64_t pending_wmma_jobs_max;
    uint64_t pending_wmma_depth_cycles_0;
    uint64_t pending_wmma_depth_cycles_1;
    uint64_t pending_wmma_depth_cycles_2;
    uint64_t pending_wmma_depth_cycles_3plus;
    uint64_t mem_queue_max;
    uint64_t stall_a_meta_not_ready;
    uint64_t stall_no_wmma_job_builder_empty;
    uint64_t stall_no_wmma_waiting_for_mma_load;
    uint64_t stall_no_wmma_waiting_for_handle_alloc;
    uint64_t stall_no_wmma_waiting_for_slot_release;
    uint64_t stall_no_wmma_waiting_for_c_wmma_inflight_drain;
    uint64_t stall_no_wmma_waiting_for_accum_live_only;
    uint64_t stall_no_wmma_waiting_for_dirty_flush_only;
    uint64_t stall_no_wmma_waiting_for_store_pending;
    uint64_t stall_no_wmma_waiting_for_ab_wmma_pending_clear;

		PerfStats()
			: latency(0)
      , tc_active_cycles(0)
      , tc_idle_cycles(0)
      , issued_primitive_tiles(0)
      , retired_primitive_tiles(0)
      , first_tc_issue_cycle(0)
      , last_tc_issue_cycle(0)
      , first_tc_retire_cycle(0)
      , last_tc_retire_cycle(0)
      , stall_a_not_ready(0)
      , stall_b_not_ready(0)
      , stall_c_not_ready(0)
      , stall_tc_busy(0)
      , stall_no_wmma_job_ready(0)
      , stall_no_tensor_instr_candidate(0)
      , stall_mma_load_handle_not_ready(0)
      , stall_handle_reuse(0)
      , stall_handle_busy_due_to_tma_load(0)
      , stall_handle_busy_due_to_tma_store_or_shift(0)
      , stall_slot_busy(0)
      , stall_tmem_read_port_busy(0)
      , stall_tmem_write_port_busy(0)
      , stall_amem_port_busy(0)
      , stall_bmem_port_busy(0)
      , stall_cmem_port_busy(0)
      , stall_meta_port_busy(0)
      , issued_macro_wmma(0)
      , retired_macro_wmma(0)
      , pending_wmma_jobs_max(0)
      , pending_wmma_depth_cycles_0(0)
      , pending_wmma_depth_cycles_1(0)
      , pending_wmma_depth_cycles_2(0)
      , pending_wmma_depth_cycles_3plus(0)
      , mem_queue_max(0)
      , stall_a_meta_not_ready(0)
      , stall_no_wmma_job_builder_empty(0)
      , stall_no_wmma_waiting_for_mma_load(0)
      , stall_no_wmma_waiting_for_handle_alloc(0)
      , stall_no_wmma_waiting_for_slot_release(0)
      , stall_no_wmma_waiting_for_c_wmma_inflight_drain(0)
      , stall_no_wmma_waiting_for_accum_live_only(0)
      , stall_no_wmma_waiting_for_dirty_flush_only(0)
      , stall_no_wmma_waiting_for_store_pending(0)
      , stall_no_wmma_waiting_for_ab_wmma_pending_clear(0)
		{}

		PerfStats& operator+=(const PerfStats& rhs) {
			this->latency += rhs.latency;
      this->tc_active_cycles += rhs.tc_active_cycles;
      this->tc_idle_cycles += rhs.tc_idle_cycles;
      this->issued_primitive_tiles += rhs.issued_primitive_tiles;
      this->retired_primitive_tiles += rhs.retired_primitive_tiles;
      if (0 == this->first_tc_issue_cycle || (rhs.first_tc_issue_cycle != 0 && rhs.first_tc_issue_cycle < this->first_tc_issue_cycle))
        this->first_tc_issue_cycle = rhs.first_tc_issue_cycle;
      this->last_tc_issue_cycle = std::max(this->last_tc_issue_cycle, rhs.last_tc_issue_cycle);
      if (0 == this->first_tc_retire_cycle || (rhs.first_tc_retire_cycle != 0 && rhs.first_tc_retire_cycle < this->first_tc_retire_cycle))
        this->first_tc_retire_cycle = rhs.first_tc_retire_cycle;
      this->last_tc_retire_cycle = std::max(this->last_tc_retire_cycle, rhs.last_tc_retire_cycle);
      this->stall_a_not_ready += rhs.stall_a_not_ready;
      this->stall_b_not_ready += rhs.stall_b_not_ready;
      this->stall_c_not_ready += rhs.stall_c_not_ready;
      this->stall_tc_busy += rhs.stall_tc_busy;
      this->stall_no_wmma_job_ready += rhs.stall_no_wmma_job_ready;
      this->stall_no_tensor_instr_candidate += rhs.stall_no_tensor_instr_candidate;
      this->stall_mma_load_handle_not_ready += rhs.stall_mma_load_handle_not_ready;
      this->stall_handle_reuse += rhs.stall_handle_reuse;
      this->stall_handle_busy_due_to_tma_load += rhs.stall_handle_busy_due_to_tma_load;
      this->stall_handle_busy_due_to_tma_store_or_shift += rhs.stall_handle_busy_due_to_tma_store_or_shift;
      this->stall_slot_busy += rhs.stall_slot_busy;
      this->stall_tmem_read_port_busy += rhs.stall_tmem_read_port_busy;
      this->stall_tmem_write_port_busy += rhs.stall_tmem_write_port_busy;
      this->stall_amem_port_busy += rhs.stall_amem_port_busy;
      this->stall_bmem_port_busy += rhs.stall_bmem_port_busy;
      this->stall_cmem_port_busy += rhs.stall_cmem_port_busy;
      this->stall_meta_port_busy += rhs.stall_meta_port_busy;
      this->issued_macro_wmma += rhs.issued_macro_wmma;
      this->retired_macro_wmma += rhs.retired_macro_wmma;
      this->pending_wmma_jobs_max = std::max(this->pending_wmma_jobs_max, rhs.pending_wmma_jobs_max);
      this->pending_wmma_depth_cycles_0 += rhs.pending_wmma_depth_cycles_0;
      this->pending_wmma_depth_cycles_1 += rhs.pending_wmma_depth_cycles_1;
      this->pending_wmma_depth_cycles_2 += rhs.pending_wmma_depth_cycles_2;
      this->pending_wmma_depth_cycles_3plus += rhs.pending_wmma_depth_cycles_3plus;
      this->mem_queue_max = std::max(this->mem_queue_max, rhs.mem_queue_max);
      this->stall_a_meta_not_ready += rhs.stall_a_meta_not_ready;
      this->stall_no_wmma_job_builder_empty += rhs.stall_no_wmma_job_builder_empty;
      this->stall_no_wmma_waiting_for_mma_load += rhs.stall_no_wmma_waiting_for_mma_load;
      this->stall_no_wmma_waiting_for_handle_alloc += rhs.stall_no_wmma_waiting_for_handle_alloc;
      this->stall_no_wmma_waiting_for_slot_release += rhs.stall_no_wmma_waiting_for_slot_release;
      this->stall_no_wmma_waiting_for_c_wmma_inflight_drain += rhs.stall_no_wmma_waiting_for_c_wmma_inflight_drain;
      this->stall_no_wmma_waiting_for_accum_live_only += rhs.stall_no_wmma_waiting_for_accum_live_only;
      this->stall_no_wmma_waiting_for_dirty_flush_only += rhs.stall_no_wmma_waiting_for_dirty_flush_only;
      this->stall_no_wmma_waiting_for_store_pending += rhs.stall_no_wmma_waiting_for_store_pending;
      this->stall_no_wmma_waiting_for_ab_wmma_pending_clear += rhs.stall_no_wmma_waiting_for_ab_wmma_pending_clear;
			return *this;
		}
	};

  enum class IssueBlockReason : uint8_t {
    None = 0,
    ANotReady,
    BNotReady,
    CNotReady,
    TcBusy,
    NoTensorInstrCandidate,
    MmaLoadHandleNotReady,
    HandleBusyDueToTmaLoad,
    HandleBusyDueToTmaStoreOrShift,
    HandleReuse,
    SlotBusy,
    AMetaNotReady,
  };

  std::vector<SimPort<instr_trace_t*>> Inputs;
	std::vector<SimPort<instr_trace_t*>> Outputs;

  TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core);
  virtual ~TensorUnit();

  virtual void reset();

  virtual void tick();

  void mma_load(uint32_t wid,
                uint32_t handle,
                IntrTcuArgs args,
                ExeTraceData* trace_data);

  void mma_store(uint32_t wid,
                 uint32_t handle,
                 IntrTcuArgs args,
                 ExeTraceData* trace_data);

  void wmma(uint32_t wid,
            IntrTcuArgs args,
	          const std::vector<reg_data_t>& rs1_data,
	          const std::vector<reg_data_t>& rs2_data,
	          const std::vector<reg_data_t>& rs3_data,
	          std::vector<reg_data_t>& rd_data,
	          ExeTraceData* trace_data);

  uint32_t scheduler_score(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const;
  IssueBlockReason classify_issue_block(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const;
  void record_issue_stall(IssueBlockReason reason);
  void record_no_tensor_instr_candidate_stall();

	const PerfStats& perf_stats() const;

private:
	class Impl;
	Impl* impl_;
};

} // namespace vortex
