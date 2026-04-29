// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// tensor_unit.cpp —— TensorUnit 实现（单实例简化版）
// ============================================================================

#include "tensor_unit.h"
#include "open_tensorcore/tensor_control/tc_decode.h"
#include "open_tensorcore/tensor_control/tensor_async_frontend.h"
#include "open_tensorcore/tensor_control/tensor_issue_policy.h"
#include "open_tensorcore/tensor_control/tensor_mem_manager.h"
#include "open_tensorcore/tensor_control/tensor_wmma_issue_engine.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/tensor_control/tensor_wmma_retire_unit.h"
#include "open_tensorcore/local_memory/tensor_local_mem_arbiter.h"
#include "open_tensorcore/local_memory/tensor_mem_pipeline.h"
#include "open_tensorcore/tensor_top/tensor_runtime_utils.h"
#include "open_tensorcore/tensor_top/tensor_tick_utils.h"

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <vector>

#include "core.h"
#include "tensor_cfg.h"
#include "tmem_window_planner.h"
#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/dmem.h"
#include "open_tensorcore/local_memory/meta_mem.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"

using namespace vortex;
using namespace vortex::tensor_unit_detail;

namespace {

void reset_trace_for_async_dispatch(TensorUnit::ExeTraceData* trace_data) {
  if (trace_data) {
    trace_data->rd_write = false;
    trace_data->retry = false;
  }
}

} // namespace

class TensorUnit::Impl {
public:
  friend class TensorUnit;
  static constexpr uint32_t kWmmaPrimitiveCount = tensor_unit_detail::kWmmaPrimitiveCount;
  static constexpr uint32_t kSubtilesPerTile = tensor_unit_detail::kSubtilesPerTile;
  static constexpr uint32_t kPrimitiveDim = tensor_unit_detail::kPrimitiveDim;
  static constexpr uint32_t kMaxConversionInputPackets = tensor_unit_detail::kMaxConversionInputPackets;
  static constexpr uint32_t kOutputPacketBufferDepth = tensor_unit_detail::kOutputPacketBufferDepth;
  static constexpr uint32_t kAmemWriteLinesPerCycle = 1;
  static constexpr uint32_t kBmemWriteLinesPerCycle = 1;
  static constexpr uint32_t kCmemWriteSubtilesPerCycle = 1;
  static constexpr uint32_t kCmemReadSubtilesPerCycle = 1;
  static constexpr uint32_t kMetaWritePacketsPerCycle = 1;

  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_() {
    reset();
  }

  void reset() {
    vortex::reset_all_tensor_unit_state(&perf_stats_,
                                        &amem_, &bmem_, &cmem_, &dmem_, &metamem_,
                                        &amem_state_, &bmem_state_, &cmem_state_, &dmem_state_,
                                        &pub_amem_state_, &pub_bmem_state_, &pub_cmem_state_, &pub_dmem_state_,
                                        &mem_ops_,
                                        &pending_mem_ops_,
                                        &pending_wmma_uops_,
                                        &pending_tensorcore_retires_,
                                        &pending_wmma_jobs_,
                                        &active_wmma_job_,
                                        &active_wmma_job_valid_,
                                        &prev_or_,
                                        &tensorcore_,
                                        &mem_arbiter_,
                                        &next_tensor_mem_request_id_,
                                        &completed_tensor_mem_responses_);
    tc_decode_.reset();
  }

  void tick() {
    // 阶段 0: 发布 Mem 状态快照
    TensorMemManager::snapshot_for_scheduler(amem_state_, bmem_state_, cmem_state_, dmem_state_,
                                             &pub_amem_state_, &pub_bmem_state_,
                                             &pub_cmem_state_, &pub_dmem_state_);

    // 阶段 1: 排空 TMEM 响应端口
    TensorMemPipeline::receive_one_tmem_response(&simobject_->TensorMemRspIn,
                                                  &completed_tensor_mem_responses_);

    // 阶段 2: 锁存张量指令到输出延迟管线
    vortex::latch_tensor_instructions_into_output_pipe(simobject_);

    // 阶段 3: 推进 TMEM↔SRAM 搬运流水线
    vortex::tick_tmem_to_local_sram_transfer_pipeline(core_,
                                              &simobject_->TensorMemReqOut,
                                              &simobject_->TensorAsyncOpCompletionOut,
                                              &mem_arbiter_,
                                              kAmemWriteLinesPerCycle,
                                              kBmemWriteLinesPerCycle,
                                              kCmemWriteSubtilesPerCycle,
                                              kCmemReadSubtilesPerCycle,
                                              kMetaWritePacketsPerCycle,
                                              &amem_, &bmem_, &cmem_, &dmem_, &metamem_,
                                              &mem_ops_,
                                              &pending_mem_ops_,
                                              &completed_tensor_mem_responses_,
                                              &next_tensor_mem_request_id_,
                                              &amem_state_, &bmem_state_, &cmem_state_, &dmem_state_,
                                              &perf_stats_,
                                              &tmem_bus_rr_index_);

    // 阶段 4: 展开宏 WMMA 并发射原语
    TensorWmmaIssueEngine::issue_wmma_primitives(core_,
                                                 arch_,
                                                 &mem_arbiter_,
                                                 &amem_, &bmem_, &cmem_, &dmem_, &metamem_,
                                                 &tensorcore_,
                                                 &amem_state_, &bmem_state_, &cmem_state_, &dmem_state_,
                                                 mem_ops_,
                                                 &pending_wmma_uops_,
                                                 &pending_wmma_jobs_,
                                                 &active_wmma_job_,
                                                 &active_wmma_job_valid_,
                                                 &perf_stats_);
    TensorWmmaIssueEngine::sample_pending_wmma_depth(pending_wmma_jobs_,
                                                     active_wmma_job_valid_,
                                                     &perf_stats_);
    if (tensorcore_.active()) {
      ++perf_stats_.tc_active_cycles;
    }

    // 阶段 5: 推进 TensorCore 计算管线并退休
    TensorWmmaRetireUnit::advance_tensorcore_pipeline(core_,
                                                      &tensorcore_,
                                                      &mem_arbiter_,
                                                      &cmem_state_, &dmem_state_,
                                                      &pending_wmma_uops_,
                                                      &pending_tensorcore_retires_,
                                                      &cmem_, &dmem_,
                                                      &perf_stats_,
                                                      &simobject_->TensorAsyncOpCompletionOut);
  }

  void dump_debug_state(std::ostream& os) const {
    vortex::dump_tensor_unit_state(os,
                                   mem_ops_,
                                   pending_mem_ops_,
                                   pending_wmma_jobs_,
                                   active_wmma_job_valid_,
                                   active_wmma_job_,
                                   pending_wmma_uops_,
                                   completed_tensor_mem_responses_,
                                   amem_state_, bmem_state_, cmem_state_, dmem_state_,
                                   pub_amem_state_, pub_bmem_state_, pub_cmem_state_, pub_dmem_state_);
  }

  void mma_load(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    reset_trace_for_async_dispatch(trace_data);
    if (!args.macro_op) {
      std::abort();
    }
    TensorAsyncFrontend::enqueue_async_mma_load(core_, arch_, wid, handle, args,
                                                &amem_state_, &bmem_state_, &cmem_state_, &dmem_state_,
                                                &pub_amem_state_, &pub_bmem_state_, &pub_cmem_state_, &pub_dmem_state_,
                                                &amem_, &bmem_, &cmem_, &dmem_, &metamem_,
                                                &mem_ops_,
                                                &pending_mem_ops_,
                                                &perf_stats_,
                                                trace_data);
  }

  void mma_store(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    reset_trace_for_async_dispatch(trace_data);
    if (!args.macro_op) {
      std::abort();
    }
    TensorAsyncFrontend::enqueue_async_mma_store(core_, arch_, wid, handle, args,
                                                 &cmem_state_, &dmem_state_,
                                                 &pub_cmem_state_, &pub_dmem_state_,
                                                 &cmem_, &dmem_,
                                                 &mem_ops_,
                                                 &perf_stats_,
                                                 trace_data);
  }

  void wmma(uint32_t wid,
            IntrTcuArgs args,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    reset_trace_for_async_dispatch(trace_data);
    if (!args.macro_op) {
      std::abort();
    }
    TensorAsyncFrontend::enqueue_async_wmma(core_, arch_, wid, args,
                                            args.fmt_a, args.fmt_b, args.fmt_c,
                                            &amem_state_, &bmem_state_, &cmem_state_, &dmem_state_,
                                            &pub_amem_state_, &pub_bmem_state_, &pub_cmem_state_, &pub_dmem_state_,
                                            &pending_wmma_uops_,
                                            &pending_wmma_jobs_,
                                            &prev_or_,
                                            active_wmma_job_valid_,
                                            &perf_stats_,
                                            trace_data);
    (void)rs1_data; (void)rs2_data; (void)rs3_data; (void)rd_data;
  }

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;

  // 单实例 Mem 状态
  AMemState amem_state_;
  BMemState bmem_state_;
  CMemState cmem_state_;
  DMemState dmem_state_;
  AMemState pub_amem_state_;
  BMemState pub_bmem_state_;
  CMemState pub_cmem_state_;
  DMemState pub_dmem_state_;

  // 本地 SRAM
  AMem amem_;
  BMem bmem_;
  CMem cmem_;
  DMem dmem_;
  MetaMem metamem_;

  // 操作队列
  std::deque<MemUop> mem_ops_;
  std::unordered_map<uint32_t, uint32_t> pending_mem_ops_;
  std::unordered_map<uint32_t, uint32_t> pending_wmma_uops_;
  std::deque<TensorCoreRetire> pending_tensorcore_retires_;
  std::deque<PendingWmmaJob> pending_wmma_jobs_;
  PendingWmmaJob active_wmma_job_;
  bool active_wmma_job_valid_ = false;

  // 累加序列上升沿检测寄存器
  bool prev_or_ = false;

  TensorCoreTop tensorcore_;
  TensorLocalMemArbiter mem_arbiter_;

  TcDecode tc_decode_;

  uint32_t tmem_bus_rr_index_ = 0;
  uint64_t next_tensor_mem_request_id_ = 1;
  std::unordered_map<uint64_t, TensorMemPortRsp> completed_tensor_mem_responses_;
};

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
  : SimObject<TensorUnit>(ctx, name)
  , Inputs(ISSUE_WIDTH, this)
  , Outputs(ISSUE_WIDTH, this)
  , TensorMemReqOut(this)
  , TensorMemRspIn(this)
  , TensorAsyncOpCompletionOut(this)
  , impl_(new Impl(this, arch, core))
{}

TensorUnit::~TensorUnit() {
  delete impl_;
}

void TensorUnit::reset() {
  impl_->reset();
}

void TensorUnit::tick() {
  impl_->tick();
}

// Phase-2: legacy TensorUnit::set_descriptor() and the multi-type
// dispatch(TcuType, ...) entry are removed. The new public surface is
// dispatch_tcu_mma() (PTX tcgen05.mma fan-out) below.

void TensorUnit::dispatch_tcu_mma(uint32_t wid,
                                  uint32_t rs1_value,
                                  uint32_t rs2_value,
                                  uint32_t qualifier,
                                  ExeTraceData* trace_data) {
  reset_trace_for_async_dispatch(trace_data);

  // Stage 1: tc_decode -> TcDecodedMmaCmd (reads idesc + operand_block_t).
  TcDecodedMmaCmd mma_cmd;
  if (!impl_->tc_decode_.decode_tcu_mma(impl_->core_, wid,
                                        rs1_value, rs2_value,
                                        qualifier, &mma_cmd)) {
    if (trace_data) { trace_data->retry = true; }
    return;
  }

  // Stage 2: synthesize an IntrTcuArgs that the existing fan-out path expects.
  // We translate the PTX-level cmd back into the legacy field set used by
  // enqueue_async_tcu_mma's stage helpers.
  IntrTcuArgs args{};
  args.fmt_a           = mma_cmd.fmt_a;
  args.fmt_b           = mma_cmd.fmt_b;
  args.fmt_c           = mma_cmd.fmt_c;
  args.fmt_d           = mma_cmd.fmt_d;
  args.a_sparse_mode   = mma_cmd.sparsity_kind;
  args.output_resident = mma_cmd.enable_input_d;
  args.ws              = mma_cmd.ws;
  args.sp              = mma_cmd.sp;
  args.cta_group       = mma_cmd.cta_group;
  args.collector_a_fill = mma_cmd.collector_a_state;
  args.multicast       = mma_cmd.multicast;
  args.transpose_a     = mma_cmd.transpose_a;
  args.transpose_b     = mma_cmd.transpose_b;
  args.macro_op        = 1;
  args.descriptor      = rs1_value;  // idesc value as opaque descriptor key

  // Operand routing: the legacy fill helpers expect (handle, target, tile_id).
  // PTX semantics: A from TMEM (a_taddr); B is always 64-bit sdesc;
  // C input source = d_taddr (read-modify-write when enable_input_d=1).
  uint32_t a_handle = mma_cmd.a_taddr;
  uint32_t b_handle = static_cast<uint32_t>(mma_cmd.b_sdesc & 0xFFFFFFFFu);
  uint32_t c_handle = mma_cmd.d_taddr;   // PTX: D location doubles as C input
  uint32_t d_handle = mma_cmd.d_taddr;

  // Phase-3.3.7 Step A (per user): drive an outer 16x16 sub-tile loop from
  // idesc.shape_m / shape_n. PTX shape_m/shape_n holds the M/N dimension
  // (must be a multiple of 16 to align to the underlying 16x16 sub-array).
  // shape_m=0 fallback to 16 for back-compat with kernels that don't fill it.
  uint32_t shape_m = (mma_cmd.shape_m != 0) ? mma_cmd.shape_m : 16u;
  uint32_t shape_n = (mma_cmd.shape_n != 0) ? mma_cmd.shape_n : 16u;
  if (shape_m % 16 != 0) shape_m = ((shape_m + 15) / 16) * 16;
  if (shape_n % 16 != 0) shape_n = ((shape_n + 15) / 16) * 16;
  uint32_t m_tiles = shape_m / 16;
  uint32_t n_tiles = shape_n / 16;
  if (m_tiles == 0) m_tiles = 1;
  if (n_tiles == 0) n_tiles = 1;

  for (uint32_t m_idx = 0; m_idx < m_tiles; ++m_idx) {
    for (uint32_t n_idx = 0; n_idx < n_tiles; ++n_idx) {
      uint32_t tile_id = m_idx * n_tiles + n_idx;
      TensorAsyncFrontend::enqueue_async_tcu_mma(
          impl_->core_, impl_->arch_, wid, args,
          mma_cmd.fmt_a, mma_cmd.fmt_b, mma_cmd.fmt_c, mma_cmd.fmt_d,
          a_handle, /*a_tile_id=*/tile_id,
          b_handle, /*b_tile_id=*/tile_id,
          c_handle, /*c_tile_id=*/tile_id,
          d_handle, /*d_tile_id=*/tile_id,
          &impl_->amem_state_, &impl_->bmem_state_, &impl_->cmem_state_, &impl_->dmem_state_,
          &impl_->pub_amem_state_, &impl_->pub_bmem_state_, &impl_->pub_cmem_state_, &impl_->pub_dmem_state_,
          &impl_->amem_, &impl_->bmem_, &impl_->cmem_, &impl_->dmem_, &impl_->metamem_,
          &impl_->mem_ops_,
          &impl_->pending_mem_ops_,
          &impl_->pending_wmma_uops_,
          &impl_->pending_wmma_jobs_,
          &impl_->prev_or_,
          impl_->active_wmma_job_valid_,
          &impl_->perf_stats_,
          trace_data);
      if (trace_data && trace_data->retry) return;
    }
  }
}

uint32_t TensorUnit::scheduler_score(uint32_t wid, TcuType tcu_type) const {
  // Phase-2: descriptor-cache lookup is gone (each TCU_MMA carries its own
  // 32-bit idesc in rs1). Issue-side scoring now operates on the published
  // operand state plus a synthetic args (descriptor=0xffffffff means no
  // cached descriptor). Issue policy returns 1 for non-TCU_MMA types.
  IntrTcuArgs args{};
  args.macro_op = 1;
  return TensorIssuePolicy::scheduler_score(impl_->core_,
                                            wid,
                                            tcu_type,
                                            args,
                                            impl_->arch_,
                                            impl_->pub_amem_state_,
                                            impl_->pub_bmem_state_,
                                            impl_->pub_cmem_state_,
                                            impl_->pub_dmem_state_);
}

TensorUnit::IssueBlockReason TensorUnit::classify_issue_block(uint32_t wid, TcuType tcu_type) const {
  IntrTcuArgs args{};
  args.macro_op = 1;
  return TensorIssuePolicy::classify_issue_block(impl_->core_,
                                                 wid,
                                                 tcu_type,
                                                 args,
                                                 impl_->arch_,
                                                 impl_->pub_amem_state_,
                                                 impl_->pub_bmem_state_,
                                                 impl_->pub_cmem_state_,
                                                 impl_->pub_dmem_state_);
}

void TensorUnit::record_issue_stall(IssueBlockReason reason) {
  TensorIssuePolicy::record_issue_stall(reason, &impl_->perf_stats_);
}

void TensorUnit::record_no_tensor_instr_candidate_stall() {
  TensorIssuePolicy::record_issue_stall(IssueBlockReason::NoTensorInstrCandidate, &impl_->perf_stats_);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats_;
}

void TensorUnit::dump_debug_state(std::ostream& os) const {
  impl_->dump_debug_state(os);
}
