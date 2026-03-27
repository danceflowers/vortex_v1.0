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

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string.h>
#include <algorithm>
#include <assert.h>
#include <util.h>
#include "types.h"
#include "arch.h"
#include "mem.h"
#include "core.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

namespace {

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

bool env_flag_enabled(const char* name, bool default_value) {
  auto value = std::getenv(name);
  if (nullptr == value || '\0' == value[0]) {
    return default_value;
  }
  return std::atoi(value) != 0;
}

TmemWindowTarget map_window_target(const TmaDescriptor& desc) {
  if (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta) {
    return TmemWindowTarget::Meta;
  }
  switch (static_cast<TcuTarget>(desc.tile_role)) {
  case TcuTarget::A:
    return TmemWindowTarget::A;
  case TcuTarget::B:
    return TmemWindowTarget::B;
  case TcuTarget::C:
    return TmemWindowTarget::C;
  default:
    return TmemWindowTarget::A;
  }
}

uint32_t infer_window_fmt(const TmaDescriptor& desc) {
  switch (desc.elem_bytes) {
  case 1:
    return vortex::tensor::fp8::id;
  case 2:
    return vortex::tensor::fp16::id;
  case 4:
    return vortex::tensor::fp32::id;
  default:
    return 0;
  }
}

uint32_t fmt_elem_bytes(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
    return 1;
  case vortex::tensor::fp16::id:
  case vortex::tensor::bf16::id:
    return 2;
  case vortex::tensor::fp32::id:
    return 4;
  default:
    return 0;
  }
}

TensorShape2D shape_for_target(const MmaDescriptor& desc, TcuTarget target) {
  switch (target) {
  case TcuTarget::A:
    return {desc.a_rows, desc.a_cols};
  case TcuTarget::B:
    return {desc.b_rows, desc.b_cols};
  case TcuTarget::C:
    return {desc.c_rows, desc.c_cols};
  default:
    return {};
  }
}

uint32_t fmt_for_target(const MmaDescriptor& desc, TcuTarget target) {
  switch (target) {
  case TcuTarget::A:
    return desc.fmt_a;
  case TcuTarget::B:
    return desc.fmt_b;
  case TcuTarget::C:
    return desc.fmt_c;
  default:
    return 0;
  }
}

TmemWindowTarget planner_target(TcuTarget target) {
  switch (target) {
  case TcuTarget::A:
    return TmemWindowTarget::A;
  case TcuTarget::B:
    return TmemWindowTarget::B;
  case TcuTarget::C:
    return TmemWindowTarget::C;
  default:
    return TmemWindowTarget::A;
  }
}

uint32_t meta_shadow_window_id(uint32_t window_id) {
  return window_id | 0x80000000u;
}

bool build_single_target_window_plan(const TmemAllocation& allocation,
                                     const MmaDescriptor& desc,
                                     TcuTarget target,
                                     uint32_t window_id,
                                     TmemWindowPlan* out) {
  (void)allocation;
  if (nullptr == out) {
    return false;
  }
  auto shape = shape_for_target(desc, target);
  auto fmt = fmt_for_target(desc, target);
  if (shape.empty() || fmt_elem_bytes(fmt) == 0) {
    return false;
  }
  return TmemWindowPlanner::build_single_dense_window(planner_target(target),
                                                      shape,
                                                      fmt,
                                                      desc.sparse_mode,
                                                      window_id,
                                                      out,
                                                      nullptr);
}

bool build_sparse_meta_window_plan(const MmaDescriptor& desc,
                                   uint32_t window_id,
                                   TmemWindowPlan* out) {
  if (nullptr == out || desc.sparse_mode == vortex::tensor::sparse_none || desc.a_rows == 0 || desc.a_cols == 0) {
    return false;
  }
  TmemWindowPlan window{};
  window.window_id = meta_shadow_window_id(window_id);
  window.target = TmemWindowTarget::Meta;
  window.elem_shape = {static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.a_rows), 16u) * 4u),
                       static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.a_cols), 16u) * 16u)};
  window.fmt = vortex::tensor::uint8::id;
  window.sparse_mode = desc.sparse_mode;
  window.logical_line_base = 0;
  window.logical_col_span = window.elem_shape.cols;
  window.logical_line_span = window.elem_shape.rows;
  window.logical_tile_col_span = 16;
  window.logical_tile_line_span = 4;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, desc.a_rows), 16u);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, desc.a_cols), 16u);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.packet_cols = 16;
  window.packet_rows = 4;
  window.logical_packet_col_span = 16;
  window.logical_packet_line_span = 4;
  window.packets_per_tile = 1;
  *out = window;
  return true;
}

bool build_sparse_meta_window_plan(const TmaDescriptor& desc,
                                   uint32_t window_id,
                                   TmemWindowPlan* out) {
  if (nullptr == out
   || desc.meta_addr == 0
   || desc.meta_size_bytes == 0
   || desc.meta_col_span == 0
   || static_cast<TcuTarget>(desc.tile_role) != TcuTarget::A) {
    return false;
  }
  TmemWindowPlan window{};
  window.window_id = meta_shadow_window_id(window_id);
  window.target = TmemWindowTarget::Meta;
  window.elem_shape = {static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.rows), 16u) * 4u),
                       static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.cols), 16u) * 16u)};
  window.fmt = vortex::tensor::uint8::id;
  window.logical_line_base = 0;
  window.logical_col_span = window.elem_shape.cols;
  window.logical_line_span = window.elem_shape.rows;
  window.logical_tile_col_span = 16;
  window.logical_tile_line_span = 4;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, desc.rows), 16u);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, desc.cols), 16u);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.packet_cols = 16;
  window.packet_rows = 4;
  window.logical_packet_col_span = 16;
  window.logical_packet_line_span = 4;
  window.packets_per_tile = 1;
  *out = window;
  return true;
}

TmemWindowPlan build_legacy_window_plan(uint32_t window_id,
                                        const TmaDescriptor& desc,
                                        uint32_t transfer_col_base,
                                        uint32_t transfer_col_span) {
  TmemWindowPlan window{};
  auto shape = TensorShape2D{desc.rows, desc.cols};
  auto fmt = infer_window_fmt(desc);
  if (!shape.empty() && fmt != 0) {
    (void)TmemWindowPlanner::build_single_dense_window(map_window_target(desc),
                                                       shape,
                                                       fmt,
                                                       0,
                                                       window_id,
                                                       &window,
                                                       nullptr);
  } else {
    window.window_id = window_id;
    window.target = map_window_target(desc);
  }
  window.logical_col_base = transfer_col_base;
  window.logical_line_base = 0;
  window.logical_col_span = transfer_col_span;
  window.logical_line_span = std::max<uint32_t>(1, desc.rows);
  return window;
}

}

Core::Core(const SimContext& ctx,
           uint32_t core_id,
           Socket* socket,
           const Arch &arch,
           const DCRS &dcrs
           )
  : SimObject(ctx, StrFormat("core%d", core_id))
  , icache_req_ports(1, this)
  , icache_rsp_ports(1, this)
  , dcache_req_ports(DCACHE_NUM_REQS, this)
  , dcache_rsp_ports(DCACHE_NUM_REQS, this)
  , core_id_(core_id)
  , socket_(socket)
  , arch_(arch)
#ifdef EXT_TCU_ENABLE
  , tensor_unit_(TensorUnit::Create("tcu", arch, this))
#endif
#ifdef EXT_V_ENABLE
  , vec_unit_(VecUnit::Create("vpu", arch, this))
#endif
  , emulator_(arch, dcrs, this)
  , ibuffers_(arch.num_warps(), IBUF_SIZE)
  , scoreboard_(arch_)
  , operands_(ISSUE_WIDTH)
  , dispatchers_((uint32_t)FUType::Count)
  , func_units_((uint32_t)FUType::Count)
  , lmem_switch_(NUM_LSU_BLOCKS)
  , mem_coalescers_(NUM_LSU_BLOCKS)
  , pending_icache_(arch_.num_warps())
  , commit_arbs_(ISSUE_WIDTH)
  , ibuffer_arbs_(ISSUE_WIDTH, {ArbiterType::RoundRobin, PER_ISSUE_WARPS})
#ifdef EXT_TCU_ENABLE
  , tma_(env_flag_enabled("VORTEX_SIMX_TMA_LOAD_REALISTIC", true))
  , mbarriers_(arch.num_barriers())
  , mbarrier_wait_targets_(arch.num_warps())
  , fence_wait_states_(arch.num_warps())
  , next_async_id_(1)
#endif
{
  char sname[100];

  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    operands_.at(iw) = Operands::Create(this);
  }

  // create the memory coalescer
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-coalescer%d", this->name().c_str(), b);
    mem_coalescers_.at(b) = MemCoalescer::Create(sname, LSU_CHANNELS, DCACHE_CHANNELS, DCACHE_WORD_SIZE, LSUQ_OUT_SIZE, 1);
  }

  // create local memory
  snprintf(sname, 100, "%s-lmem", this->name().c_str());
  local_mem_ = LocalMem::Create(sname, LocalMem::Config{
    (1 << LMEM_LOG_SIZE),
    LSU_WORD_SIZE,
    LSU_CHANNELS,
    log2ceil(LMEM_NUM_BANKS),
    false
  });

  // create lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lmem_switch%d", this->name().c_str(), b);
    lmem_switch_.at(b) = LocalMemSwitch::Create(sname, 1);
  }

  // create dcache adapter
  std::vector<LsuMemAdapter::Ptr> lsu_dcache_adapter(NUM_LSU_BLOCKS);
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    snprintf(sname, 100, "%s-lsu_dcache_adapter%d", this->name().c_str(), b);
    lsu_dcache_adapter.at(b) = LsuMemAdapter::Create(sname, DCACHE_CHANNELS, 1);
  }

  // create lmem arbiter
  snprintf(sname, 100, "%s-lmem_arb", this->name().c_str());
  auto lmem_arb = LsuArbiter::Create(sname, ArbiterType::RoundRobin, NUM_LSU_BLOCKS, 1);

  // create lmem adapter
  snprintf(sname, 100, "%s-lsu_lmem_adapter", this->name().c_str());
  auto lsu_lmem_adapter = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);

  // connect lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    lmem_switch_.at(b)->ReqDC.bind(&mem_coalescers_.at(b)->ReqIn);
    lmem_switch_.at(b)->ReqLmem.bind(&lmem_arb->ReqIn.at(b));

    mem_coalescers_.at(b)->RspIn.bind(&lmem_switch_.at(b)->RspDC);
    lmem_arb->RspIn.at(b).bind(&lmem_switch_.at(b)->RspLmem);
  }

  // connect lmem arbiter
  lmem_arb->ReqOut.at(0).bind(&lsu_lmem_adapter->ReqIn);
  lsu_lmem_adapter->RspIn.bind(&lmem_arb->RspOut.at(0));

  // connect lmem adapter
  for (uint32_t c = 0; c < LSU_CHANNELS; ++c) {
    lsu_lmem_adapter->ReqOut.at(c).bind(&local_mem_->Inputs.at(c));
    local_mem_->Outputs.at(c).bind(&lsu_lmem_adapter->RspOut.at(c));
  }

  // connect dcache coalescer
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    mem_coalescers_.at(b)->ReqOut.bind(&lsu_dcache_adapter.at(b)->ReqIn);
    lsu_dcache_adapter.at(b)->RspIn.bind(&mem_coalescers_.at(b)->RspOut);
  }

  // connect dcache adapter
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    for (uint32_t c = 0; c < DCACHE_CHANNELS; ++c) {
      uint32_t p = b * DCACHE_CHANNELS + c;
      lsu_dcache_adapter.at(b)->ReqOut.at(c).bind(&dcache_req_ports.at(p));
      dcache_rsp_ports.at(p).bind(&lsu_dcache_adapter.at(b)->RspOut.at(c));
    }
  }

  // initialize dispatchers
  dispatchers_.at((int)FUType::ALU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_ALU_BLOCKS, NUM_ALU_LANES);
  dispatchers_.at((int)FUType::FPU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_FPU_BLOCKS, NUM_FPU_LANES);
  dispatchers_.at((int)FUType::LSU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_LSU_BLOCKS, NUM_LSU_LANES);
  dispatchers_.at((int)FUType::SFU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_SFU_BLOCKS, NUM_SFU_LANES);
#ifdef EXT_V_ENABLE
  dispatchers_.at((int)FUType::VPU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_VPU_BLOCKS, NUM_VPU_LANES);
#endif
#ifdef EXT_TCU_ENABLE
  dispatchers_.at((int)FUType::TCU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_TCU_BLOCKS, NUM_TCU_LANES);
#endif

  // initialize execute units
  func_units_.at((int)FUType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  func_units_.at((int)FUType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);
  func_units_.at((int)FUType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  func_units_.at((int)FUType::SFU) = SimPlatform::instance().create_object<SfuUnit>(this);
#ifdef EXT_V_ENABLE
  func_units_.at((int)FUType::VPU) = SimPlatform::instance().create_object<VpuUnit>(this);
#endif
#ifdef EXT_TCU_ENABLE
  func_units_.at((int)FUType::TCU) = SimPlatform::instance().create_object<TcuUnit>(this);
#endif

  // bind commit arbiters
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    snprintf(sname, 100, "%s-commit-arb%d", this->name().c_str(), iw);
    auto arbiter = TraceArbiter::Create(sname, ArbiterType::RoundRobin, (uint32_t)FUType::Count, 1);
    for (uint32_t fu = 0; fu < (uint32_t)FUType::Count; ++fu) {
      func_units_.at(fu)->Outputs.at(iw).bind(&arbiter->Inputs.at(fu));
    }
    commit_arbs_.at(iw) = arbiter;
  }

  this->reset();
}

Core::~Core() {
  //--
}

void Core::reset() {

  emulator_.reset();

  for (auto& commit_arb : commit_arbs_) {
    commit_arb->reset();
  }

  for (auto& ibuf : ibuffers_) {
    ibuf.reset();
  }

  scoreboard_.reset();
  fetch_latch_.reset();
  decode_latch_.reset();
  pending_icache_.clear();

  for (auto& arb : ibuffer_arbs_) {
    arb.reset();
  }

  pending_instrs_.clear();
  pending_ifetches_ = 0;

#ifdef EXT_TCU_ENABLE
  tensor_unit_->reset();
  tmem_.reset();
  tma_.reset();
  async_tensor_ops_.clear();
  async_tensor_waiters_.clear();
  for (auto& mbarrier : mbarriers_) {
    mbarrier = {};
  }
  for (auto& wait_targets : mbarrier_wait_targets_) {
    wait_targets.clear();
  }
  for (auto& fence_wait : fence_wait_states_) {
    fence_wait = {};
  }
  next_async_id_ = 1;
#endif

  perf_stats_ = PerfStats();
}

void Core::tick() {
#ifdef EXT_TCU_ENABLE
  advance_async_tensor_ops();
#endif
  this->commit();
  this->execute();
  this->issue();
  this->decode();
  this->fetch();
  this->schedule();

  ++perf_stats_.cycles;
  DPN(2, std::flush);
}

void Core::schedule() {
  auto trace = emulator_.step();
  if (trace == nullptr) {
    ++perf_stats_.sched_idle;
    return;
  }

  // suspend warp until decode
  emulator_.suspend(trace->wid, WarpStallReason::Pipeline);

  DT(3, "pipeline-schedule: " << *trace);

  // advance to fetch stage
  fetch_latch_.push(trace);
  pending_instrs_.push_back(trace);
}

void Core::fetch() {
  perf_stats_.ifetch_latency += pending_ifetches_;

  // handle icache response
  auto& icache_rsp_port = icache_rsp_ports.at(0);
  if (!icache_rsp_port.empty()){
    auto& mem_rsp = icache_rsp_port.front();
    auto trace = pending_icache_.at(mem_rsp.tag);
    decode_latch_.push(trace);
    DT(3, "icache-rsp: addr=0x" << std::hex << trace->PC << ", tag=0x" << mem_rsp.tag << std::dec << ", " << *trace);
    pending_icache_.release(mem_rsp.tag);
    icache_rsp_port.pop();
    --pending_ifetches_;
  }

  // send icache request
  if (fetch_latch_.empty())
    return;
  auto trace = fetch_latch_.front();
  MemReq mem_req;
  mem_req.addr  = trace->PC;
  mem_req.write = false;
  mem_req.tag   = pending_icache_.allocate(trace);
  mem_req.cid   = trace->cid;
  mem_req.uuid  = trace->uuid;
  icache_req_ports.at(0).push(mem_req, 2);
  DT(3, "icache-req: addr=0x" << std::hex << mem_req.addr << ", tag=0x" << mem_req.tag << std::dec << ", " << *trace);
  fetch_latch_.pop();
  ++perf_stats_.ifetches;
  ++pending_ifetches_;
}

void Core::decode() {
  if (decode_latch_.empty())
    return;

  auto trace = decode_latch_.front();

  // check ibuffer capacity
  auto& ibuffer = ibuffers_.at(trace->wid);
  if (ibuffer.full()) {
    if (!trace->log_once(true)) {
      DT(4, "*** ibuffer-stall: " << *trace);
    }
    ++perf_stats_.ibuf_stalls;
    return;
  } else {
    trace->log_once(false);
  }

  // release warp
  if (!trace->fetch_stall) {
    emulator_.resume(trace->wid);
  }

  DT(3, "pipeline-decode: " << *trace);

  // insert to ibuffer
  ibuffer.push(trace);

  decode_latch_.pop();
}

void Core::issue() {
  // dispatch operands
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& operand = operands_.at(iw);
    if (operand->Output.empty())
      continue;
    auto trace = operand->Output.front();
    dispatchers_.at((int)trace->fu_type)->Inputs.at(iw).push(trace);
    operand->Output.pop();
  }

  // issue ibuffer instructions
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    bool has_instrs = false;
    BitVector<> ready_set(PER_ISSUE_WARPS);
    for (uint32_t w = 0; w < PER_ISSUE_WARPS; ++w) {
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      if (ibuffer.empty())
        continue;
      // check scoreboard
      has_instrs = true;
      auto trace = ibuffer.top();
      if (scoreboard_.in_use(trace)) {
        auto uses = scoreboard_.get_uses(trace);
        if (!trace->log_once(true)) {
          DTH(4, "*** scoreboard-stall: dependents={");
          for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
            auto& use = uses.at(j);
            __unused (use);
            if (j) DTN(4, ", ");
            DTN(4, use.reg_type << use.reg_id << " (#" << use.uuid << ")");
          }
          DTN(4, "}, " << *trace << std::endl);
        }
        for (uint32_t j = 0, n = uses.size(); j < n; ++j) {
          auto& use = uses.at(j);
          switch (use.fu_type) {
          case FUType::ALU: ++perf_stats_.scrb_alu; break;
          case FUType::FPU: ++perf_stats_.scrb_fpu; break;
          case FUType::LSU: ++perf_stats_.scrb_lsu; break;
          case FUType::SFU: {
            ++perf_stats_.scrb_sfu;
            if (std::get_if<WctlType>(&use.op_type)) {
              ++perf_stats_.scrb_wctl;
            } else if (std::get_if<CsrType>(&use.op_type)) {
              ++perf_stats_.scrb_csrs;
            }
          } break;
        #ifdef EXT_V_ENABLE
          case FUType::VPU: ++perf_stats_.scrb_vpu; break;
        #endif
        #ifdef EXT_TCU_ENABLE
          case FUType::TCU: ++perf_stats_.scrb_tcu; break;
        #endif
          default: assert(false);
          }
        }
      } else {
        trace->log_once(false);
        ready_set.set(w); // mark instruction as ready
      }
    }

    if (ready_set.any()) {
      // select one instruction from ready set
      auto w = ibuffer_arbs_.at(iw).grant(ready_set);
      uint32_t wid = w * ISSUE_WIDTH + iw;
      auto& ibuffer = ibuffers_.at(wid);
      auto trace = ibuffer.top();
      // update scoreboard
      DT(3, "pipeline-ibuffer: " << *trace);
      if (trace->wb) {
        scoreboard_.reserve(trace);
      }
      // to operand stage
      operands_.at(iw)->Input.push(trace, 1);
      ibuffer.pop();
    }

    // track scoreboard stalls
    if (has_instrs && !ready_set.any()) {
      ++perf_stats_.scrb_stalls;
    }
  }
}

void Core::execute() {
  for (uint32_t fu = 0; fu < (uint32_t)FUType::Count; ++fu) {
    auto& dispatch = dispatchers_.at(fu);
    auto& func_unit = func_units_.at(fu);
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      if (dispatch->Outputs.at(iw).empty())
        continue;
      auto trace = dispatch->Outputs.at(iw).front();
      func_unit->Inputs.at(iw).push(trace, 2);
      dispatch->Outputs.at(iw).pop();
    }
  }
}

void Core::commit() {
  // process completed instructions
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& commit_arb = commit_arbs_.at(iw);
    if (commit_arb->Outputs.at(0).empty())
      continue;
    auto trace = commit_arb->Outputs.at(0).front().data;

    // advance to commit stage
    DT(3, "pipeline-commit: " << *trace);
    assert(trace->cid == core_id_);

    // update scoreboard
    if (trace->eop) {
      if (trace->wb) {
        operands_.at(iw)->writeback(trace);
        scoreboard_.release(trace);
      }
      auto orig_size = pending_instrs_.size();
      pending_instrs_.remove(trace);
      if (pending_instrs_.size() != orig_size) {
        perf_stats_.instrs += trace->tmask.count();
      #ifdef EXT_V_ENABLE
        if (std::get_if<VsetType>(&trace->op_type)
         || std::get_if<VlsType>(&trace->op_type)
         || std::get_if<VopType>(&trace->op_type)) {
          perf_stats_.vinstrs += trace->tmask.count();
        }
      #endif
      }
    }

    // delete the trace
    trace_pool_.deallocate(trace, 1);

    commit_arb->Outputs.at(0).pop();
  }
}

int Core::get_exitcode() const {
  return emulator_.get_exitcode();
}

bool Core::running() const {
  if (emulator_.running() || !pending_instrs_.empty()) {
  #ifndef NDEBUG
    for (auto& trace : pending_instrs_) {
      DT(5, "pipeline-pending: " << *trace);
    }
  #endif
    return true;
  }
  return false;
}

void Core::resume(uint32_t wid) {
  emulator_.resume(wid);
}

void Core::suspend(uint32_t wid, WarpStallReason reason) {
  emulator_.suspend(wid, reason);
}

void Core::set_stall_reason(uint32_t wid, WarpStallReason reason) {
  emulator_.set_stall_reason(wid, reason);
}

bool Core::barrier(uint32_t bar_id, uint32_t count, uint32_t wid) {
  return emulator_.barrier(bar_id, count, wid);
}

bool Core::wspawn(uint32_t num_warps, Word nextPC) {
  return emulator_.wspawn(num_warps, nextPC);
}

void Core::attach_ram(RAM* ram) {
  emulator_.attach_ram(ram);
}

const Core::PerfStats& Core::perf_stats() const {
  perf_stats_.opds_stalls = 0;
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    perf_stats_.opds_stalls += operands_.at(iw)->total_stalls();
  }
  return perf_stats_;
}

#ifdef VM_ENABLE
void Core::set_satp(uint64_t satp) {
  emulator_.set_satp(satp); //JAEWON wit, tid???
  // emulator_.set_csr(VX_CSR_SATP,satp,0,0); //JAEWON wit, tid???
}
#endif

#ifdef EXT_TCU_ENABLE

bool Core::lookup_tmem_allocation(uint32_t handle, TmemAllocation** allocation) {
  return tmem_.lookup_allocation(handle, allocation);
}

bool Core::lookup_tmem_allocation(uint32_t handle, const TmemAllocation** allocation) const {
  return tmem_.lookup_allocation(handle, allocation);
}

bool Core::read_tma_descriptor(uint32_t desc_id, TmaDescriptor* out) {
  return tma_.read_tma_descriptor(emulator_.startup_arg(),
                                  desc_id,
                                  out,
                                  [this](void* data, uint64_t addr, uint32_t size) {
                                    this->dcache_read(data, addr, size);
                                  });
}

bool Core::read_mma_descriptor(uint32_t desc_id, MmaDescriptor* out) {
  return tma_.read_mma_descriptor(emulator_.startup_arg(),
                                  desc_id,
                                  out,
                                  [this](void* data, uint64_t addr, uint32_t size) {
                                    this->dcache_read(data, addr, size);
                                  });
}

void Core::reset_tmem_port_budgets() {
  tmem_.reset_port_budgets(perf_stats_.cycles);
}

void Core::ensure_tmem_port_budgets() {
  tmem_.ensure_port_budgets(perf_stats_.cycles);
}

bool Core::try_acquire_tmem_read_port(uint32_t handle, uint32_t packet_idx) {
  if (!tmem_.try_acquire_read_packet(perf_stats_.cycles, handle, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_read_packets;
  return true;
}

bool Core::try_acquire_tmem_window_read_port(uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  if (!tmem_.try_acquire_window_read_packet(perf_stats_.cycles, handle, window_id, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_read_packets;
  return true;
}

bool Core::try_acquire_tmem_window_linear_read_port(uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  if (!tmem_.try_acquire_window_linear_read_packet(perf_stats_.cycles, handle, window_id, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_read_packets;
  return true;
}

bool Core::try_acquire_tmem_read_meta_port(uint32_t handle, uint32_t packet_idx) {
  if (!tmem_.try_acquire_meta_read_packet(perf_stats_.cycles, handle, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_read_packets;
  return true;
}

bool Core::try_acquire_tmem_write_port(uint32_t handle, uint32_t packet_idx) {
  if (!tmem_.try_acquire_write_packet(perf_stats_.cycles, handle, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_write_packets;
  return true;
}

bool Core::try_acquire_tmem_window_write_port(uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  if (!tmem_.try_acquire_window_write_packet(perf_stats_.cycles, handle, window_id, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_write_packets;
  return true;
}

bool Core::try_acquire_tmem_window_linear_write_port(uint32_t handle, uint32_t window_id, uint32_t packet_idx) {
  if (!tmem_.try_acquire_window_linear_write_packet(perf_stats_.cycles, handle, window_id, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_write_packets;
  return true;
}

bool Core::try_acquire_tmem_region_read_port(uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  if (!tmem_.try_acquire_region_read_packet(perf_stats_.cycles, col_base, col_span, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_read_packets;
  return true;
}

bool Core::try_acquire_tmem_region_write_port(uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  if (!tmem_.try_acquire_region_write_packet(perf_stats_.cycles, col_base, col_span, packet_idx)) {
    return false;
  }
  ++perf_stats_.tmem_write_packets;
  return true;
}

void Core::refund_tmem_region_read_port(uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  if (perf_stats_.tmem_read_packets != 0) {
    --perf_stats_.tmem_read_packets;
  }
  tmem_.refund_region_read_packet(perf_stats_.cycles, col_base, col_span, packet_idx);
}

void Core::refund_tmem_region_write_port(uint32_t col_base, uint32_t col_span, uint32_t packet_idx) {
  if (perf_stats_.tmem_write_packets != 0) {
    --perf_stats_.tmem_write_packets;
  }
  tmem_.refund_region_write_packet(perf_stats_.cycles, col_base, col_span, packet_idx);
}

void Core::init_async_tensor_op_progress(AsyncTensorOp& op) {
  if (op.txn_initialized) {
    return;
  }

  switch (op.type) {
  case AsyncTensorOpType::TmaLoad: {
    TmaDescriptor desc;
    if (!read_tma_descriptor(op.descriptor_id, &desc)) {
      op.completed = true;
      op.txn_initialized = true;
      return;
    }
    op.tma_desc = desc;
    op.use_meta_region = (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta);
    op.transfer_col_base = op.use_meta_region ? desc.meta_tmem_base : desc.tmem_base;
    op.transfer_col_span = op.use_meta_region ? desc.meta_col_span : desc.bank_span;
    if (0 == op.transfer_col_span) {
      const TmemAllocation* allocation = nullptr;
      if (!lookup_tmem_allocation(op.handle, &allocation)) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      op.transfer_col_base = allocation->payload_col_base;
      op.transfer_col_span = allocation->col_span;
    }

    uint32_t capacity = 0;
    if (!tmem_region_query(op.transfer_col_base, op.transfer_col_span, &capacity)) {
      op.completed = true;
      op.txn_initialized = true;
      return;
    }

    (void)tmem_.upsert_window(op.handle,
                              build_legacy_window_plan(op.window_id, op.tma_desc,
                                                       op.transfer_col_base, op.transfer_col_span));
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(op.tma_desc, op.window_id, &meta_window)) {
      meta_window.logical_col_base = op.tma_desc.meta_tmem_base;
      (void)tmem_.upsert_window(op.handle, meta_window);
    }

    op.payload_size_bytes = std::min<uint32_t>(tma_.payload_size_bytes(desc), capacity);
    if (!tma_.load_payload(desc,
                           op.transpose_b,
                           capacity,
                           &op.payload_buffer,
                           [this](void* data, uint64_t addr, uint32_t size) {
                             this->dcache_read(data, addr, size);
                           })) {
      op.completed = true;
      op.txn_initialized = true;
      return;
    }

    uint32_t meta_packets = 0;
    op.meta_col_base = desc.meta_tmem_base;
    op.meta_col_span = desc.meta_col_span;
    op.meta_size_bytes = 0;
    op.meta_buffer.clear();
    if (!op.use_meta_region && desc.meta_addr != 0 && desc.meta_size_bytes != 0 && desc.meta_col_span != 0) {
      uint32_t meta_capacity = 0;
      if (!tmem_region_query(op.meta_col_base, op.meta_col_span, &meta_capacity)
       || !tma_.load_meta(desc,
                          meta_capacity,
                          &op.meta_buffer,
                          [this](void* data, uint64_t addr, uint32_t size) {
                            this->dcache_read(data, addr, size);
                          })) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      op.meta_size_bytes = std::min<uint32_t>(desc.meta_size_bytes, meta_capacity);
      meta_packets = (op.meta_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }

    uint32_t payload_packets = (op.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    op.remaining_base_cycles = kTmaLoadBaseLatency + ((op.transpose_b || ((desc.flags & 0x1) != 0)) ? kTmaTransposePenalty : 0);
    op.remaining_tmem_write_packets = payload_packets + meta_packets;
  } break;
  case AsyncTensorOpType::TmaStore: {
    TmaDescriptor desc;
    if (!read_tma_descriptor(op.descriptor_id, &desc)) {
      op.completed = true;
      op.txn_initialized = true;
      return;
    }
    op.tma_desc = desc;
    op.use_meta_region = (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta);
    op.transfer_col_base = op.use_meta_region ? desc.meta_tmem_base : desc.tmem_base;
    op.transfer_col_span = op.use_meta_region ? desc.meta_col_span : desc.bank_span;
    if (0 == op.transfer_col_span) {
      const TmemAllocation* allocation = nullptr;
      if (!lookup_tmem_allocation(op.handle, &allocation)) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      op.transfer_col_base = allocation->payload_col_base;
      op.transfer_col_span = allocation->col_span;
    }

    uint32_t capacity = 0;
    if (!tmem_region_query(op.transfer_col_base, op.transfer_col_span, &capacity)) {
      op.completed = true;
      op.txn_initialized = true;
      return;
    }

    (void)tmem_.upsert_window(op.handle,
                              build_legacy_window_plan(op.window_id, op.tma_desc,
                                                       op.transfer_col_base, op.transfer_col_span));
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(op.tma_desc, op.window_id, &meta_window)) {
      meta_window.logical_col_base = op.tma_desc.meta_tmem_base;
      (void)tmem_.upsert_window(op.handle, meta_window);
    }

    op.payload_size_bytes = std::min<uint32_t>(tma_.payload_size_bytes(desc), capacity);
    uint32_t payload_packets = (op.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    op.remaining_base_cycles = kTmaStoreBaseLatency;
    op.remaining_tmem_read_packets = payload_packets;
  } break;
  case AsyncTensorOpType::TmemShift: {
    uint32_t packets = 0;
    TmemAllocation* allocation = nullptr;
    if (!lookup_tmem_allocation(op.handle, &allocation)) {
      op.completed = true;
      op.txn_initialized = true;
      return;
    }
    if (!tmem_.window_packet_count(op.handle, op.window_id, &packets)) {
      uint32_t size_bytes = 0;
      if (!tmem_region_query(allocation->payload_col_base, allocation->col_span, &size_bytes)) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      packets = (size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    op.remaining_tmem_read_packets = packets;
    op.remaining_tmem_write_packets = packets;
    if (op.refill_desc_id != 0) {
      const TmemWindowPlan* window = nullptr;
      if (!tmem_.lookup_window(op.handle, op.window_id, &window)
       || window->logical_col_span == 0) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      TmaDescriptor desc;
      if (!read_tma_descriptor(op.refill_desc_id, &desc)
       || desc.rows != 1) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      op.tma_desc = desc;
      op.refill_size_bytes = std::min<uint32_t>(tma_.payload_size_bytes(desc), window->logical_col_span);
      if (!tma_.load_payload(desc,
                             false,
                             window->logical_col_span,
                             &op.payload_buffer,
                             [this](void* data, uint64_t addr, uint32_t size) {
                               this->dcache_read(data, addr, size);
                             })) {
        op.completed = true;
        op.txn_initialized = true;
        return;
      }
      op.remaining_base_cycles = tma_.estimate_load_latency(desc, false);
      op.remaining_refill_write_packets = (window->logical_col_span + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
  } break;
  case AsyncTensorOpType::MmaLoad:
  case AsyncTensorOpType::MmaStore:
  case AsyncTensorOpType::Wmma:
    break;
  default:
    std::abort();
  }

  op.txn_initialized = true;
}

bool Core::has_pending_async_ops(uint32_t wid, bool committed_only) const {
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.wid != wid || op.completed) {
      continue;
    }
    if (committed_only && !op.committed) {
      continue;
    }
    if (!committed_only && op.committed) {
      continue;
    }
    return true;
  }
  return false;
}

bool Core::has_pending_local_tensor_ops(uint32_t wid) const {
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.wid != wid || op.completed) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::MmaLoad:
    case AsyncTensorOpType::MmaStore:
    case AsyncTensorOpType::Wmma:
      return true;
    default:
      break;
    }
  }
  return false;
}

void Core::resume_async_waiters(uint32_t async_id) {
  auto it = async_tensor_waiters_.find(async_id);
  if (it == async_tensor_waiters_.end()) {
    return;
  }
  auto waiters = it->second;
  async_tensor_waiters_.erase(it);
  for (uint32_t wid = 0; wid < arch_.num_warps(); ++wid) {
    if (waiters.test(wid)) {
      emulator_.resume(wid);
    }
  }
}

void Core::try_resume_fence_waiters() {
  for (uint32_t wid = 0; wid < fence_wait_states_.size(); ++wid) {
    const auto& fence_wait = fence_wait_states_.at(wid);
    if (!fence_wait.active) {
      continue;
    }
    bool ready = (fence_wait.mode == TcuFenceMode::After)
              ? !has_pending_async_ops(wid, true)
              : !has_pending_async_ops(wid, false);
    if (ready) {
      fence_wait_states_.at(wid).active = false;
      emulator_.resume(wid);
    }
  }
}

bool Core::tmem_region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const {
  return tmem_.region_query(col_base, col_span, size_bytes);
}

bool Core::tmem_query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const {
  return tmem_.query(handle, col_span, size_bytes);
}

bool Core::lookup_tmem_window(uint32_t handle, uint32_t window_id, const TmemWindowPlan** out) const {
  return tmem_.lookup_window(handle, window_id, out);
}

bool Core::tmem_window_epoch(uint32_t handle, uint32_t* epoch) const {
  return tmem_.window_epoch(handle, epoch);
}

bool Core::ensure_tmem_window_bound(uint32_t handle, uint32_t desc_id, TcuTarget target, uint32_t window_id) {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return false;
  }

  MmaDescriptor desc{};
  if (!read_mma_descriptor(desc_id, &desc)) {
    return false;
  }

  bool has_any_shape = (desc.a_rows != 0 && desc.a_cols != 0)
                    || (desc.b_rows != 0 && desc.b_cols != 0)
                    || (desc.c_rows != 0 && desc.c_cols != 0);
  if (!has_any_shape) {
    return true;
  }

  if (target == TcuTarget::None) {
    TmemWindowPlannerInput input{};
    input.a_shape = {desc.a_rows, desc.a_cols};
    input.b_shape = {desc.b_rows, desc.b_cols};
    input.c_shape = {desc.c_rows, desc.c_cols};
    input.fmt_a = desc.fmt_a;
    input.fmt_b = desc.fmt_b;
    input.fmt_c = desc.fmt_c;
    input.sparse_mode = desc.sparse_mode;
    input.allocation_col_span = allocation->col_span;
    TmemLayoutPlan plan{};
    if (!TmemWindowPlanner::build_dense_plan(input, &plan, nullptr)) {
      return false;
    }
    uint32_t epoch = 0;
    if (tmem_.window_epoch(handle, &epoch)) {
      plan.epoch = epoch;
    }
    return tmem_.bind_layout(handle, plan);
  }

  TmemWindowPlan window{};
  if (!build_single_target_window_plan(*allocation, desc, target, window_id, &window)) {
    return true;
  }
  if (!tmem_.upsert_window(handle, window)) {
    return false;
  }
  if (target == TcuTarget::A) {
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(desc, window_id, &meta_window)) {
      meta_window.logical_col_base = allocation->meta_col_base;
      meta_window.logical_line_base = 0;
      if (!tmem_.upsert_window(handle, meta_window)) {
        return false;
      }
    }
  }
  return true;
}

bool Core::tmem_region_copy_in(uint32_t col_base, uint32_t col_span, const uint8_t* data, uint32_t size_bytes) {
  return tmem_.region_copy_in(col_base, col_span, data, size_bytes);
}

bool Core::tmem_region_copy_out(uint32_t col_base, uint32_t col_span, uint8_t* data, uint32_t size_bytes) const {
  return tmem_.region_copy_out(col_base, col_span, data, size_bytes);
}

bool Core::tmem_copy_in(uint32_t handle, const uint8_t* data, uint32_t size_bytes) {
  return tmem_.copy_in(handle, data, size_bytes);
}

bool Core::tmem_copy_out(uint32_t handle, uint8_t* data, uint32_t size_bytes) const {
  return tmem_.copy_out(handle, data, size_bytes);
}

bool Core::tmem_region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes) {
  return tmem_.region_shift_down(col_base, col_span, row_bytes);
}

bool Core::tmem_handle_busy(uint32_t handle) const {
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.completed || op.handle != handle) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::TmaLoad:
    case AsyncTensorOpType::TmaStore:
    case AsyncTensorOpType::TmemShift:
      return true;
    default:
      break;
    }
  }
  return false;
}

bool Core::tmem_handle_ready_for_mma_load(uint32_t handle, TcuTarget target, uint32_t sparse_mode) const {
  return TmemHandleBlockReason::None == tmem_handle_load_block_reason(handle, target, sparse_mode);
}

bool Core::tmem_handle_ready_for_mma_store(uint32_t handle) const {
  return TmemHandleBlockReason::None == tmem_handle_store_block_reason(handle);
}

bool Core::has_inflight_tma_handle_activity() const {
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.completed) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::TmaLoad:
    case AsyncTensorOpType::TmaStore:
    case AsyncTensorOpType::TmemShift:
      return true;
    default:
      break;
    }
  }
  return false;
}

Core::TmemHandleBlockReason Core::tmem_handle_load_block_reason(uint32_t handle, TcuTarget target, uint32_t sparse_mode) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return TmemHandleBlockReason::Invalid;
  }
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.completed || op.handle != handle) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::TmaLoad:
      return TmemHandleBlockReason::BusyTmaLoad;
    case AsyncTensorOpType::TmaStore:
    case AsyncTensorOpType::TmemShift:
      return TmemHandleBlockReason::BusyTmaStoreOrShift;
    default:
      break;
    }
  }
  if (!allocation->payload_ready) {
    return TmemHandleBlockReason::PayloadNotReady;
  }
  bool needs_meta = (target == TcuTarget::A || target == TcuTarget::None)
                 && (sparse_mode != vortex::tensor::sparse_none);
  if (needs_meta && !allocation->meta_ready) {
    return TmemHandleBlockReason::MetaNotReady;
  }
  return TmemHandleBlockReason::None;
}

Core::TmemHandleBlockReason Core::tmem_handle_store_block_reason(uint32_t handle) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return TmemHandleBlockReason::Invalid;
  }
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.completed || op.handle != handle) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::TmaLoad:
      return TmemHandleBlockReason::BusyTmaLoad;
    case AsyncTensorOpType::TmaStore:
    case AsyncTensorOpType::TmemShift:
      return TmemHandleBlockReason::BusyTmaStoreOrShift;
    default:
      break;
    }
  }
  return TmemHandleBlockReason::None;
}

void Core::mark_mbarrier_phase_active(uint32_t barrier_id) {
  if (barrier_id >= mbarriers_.size()) {
    return;
  }
  auto& barrier = mbarriers_.at(barrier_id);
  if (!barrier.valid) {
    return;
  }
  barrier.phase_done = false;
}

void Core::try_complete_mbarrier(uint32_t barrier_id) {
  if (barrier_id >= mbarriers_.size()) {
    return;
  }
  auto& barrier = mbarriers_.at(barrier_id);
  if (!barrier.valid || barrier.pending_arrivals != 0 || barrier.pending_tx != 0) {
    return;
  }

  ++barrier.phase;
  barrier.phase_done = true;
  barrier.pending_arrivals = barrier.expected_arrivals;

  auto waiters = barrier.waiters_bitmap;
  barrier.waiters_bitmap.reset();
  for (uint32_t wid = 0; wid < arch_.num_warps(); ++wid) {
    if (waiters.test(wid)) {
      emulator_.resume(wid);
    }
  }
}

void Core::finalize_async_tensor_op(AsyncTensorOp& op) {
  if (!op.completed) {
    return;
  }
  resume_async_waiters(op.async_id);
  try_resume_fence_waiters();
  if (!op.committed || op.barrier_id >= mbarriers_.size()) {
    return;
  }
  auto& barrier = mbarriers_.at(op.barrier_id);
  if (!barrier.valid) {
    return;
  }
  if (barrier.pending_tx > 0) {
    --barrier.pending_tx;
  }
  try_complete_mbarrier(op.barrier_id);
}

void Core::process_async_tensor_op(AsyncTensorOp& op) {
  init_async_tensor_op_progress(op);
  if (op.completed) {
    finalize_async_tensor_op(op);
    return;
  }
  if (op.remaining_base_cycles != 0) {
    --op.remaining_base_cycles;
    return;
  }

  switch (op.type) {
  case AsyncTensorOpType::TmaLoad:
  case AsyncTensorOpType::TmaStore: {
    auto make_packet = [](const std::vector<uint8_t>& buffer, uint32_t packet_idx) {
      TmemPacket packet;
      auto offset = packet_idx * Tmem::kPacketBytes;
      if (offset < buffer.size()) {
        auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, buffer.size() - offset);
        std::copy_n(buffer.data() + offset, valid_bytes, packet.bytes.begin());
      }
      return packet;
    };

    if (op.type == AsyncTensorOpType::TmaLoad) {
      if (op.remaining_tmem_write_packets != 0) {
        auto payload_packets = (op.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
        if (op.payload_packet_cursor < payload_packets) {
          if (!try_acquire_tmem_window_linear_write_port(op.handle, op.window_id, op.payload_packet_cursor)) {
            ++perf_stats_.stall_tmem_write_port_busy;
            return;
          }
          auto packet = make_packet(op.payload_buffer, op.payload_packet_cursor);
          if (!tmem_.write_window_linear_packet(op.handle, op.window_id, op.payload_packet_cursor, packet)) {
            op.completed = true;
            finalize_async_tensor_op(op);
            return;
          }
          ++op.payload_packet_cursor;
          if (op.payload_packet_cursor == payload_packets) {
            if (op.use_meta_region) {
              tmem_.set_meta_ready(op.handle, true);
            } else {
              tmem_.set_payload_ready(op.handle, true);
              uint32_t row_bytes = op.tma_desc.stride_bytes
                                 ? op.tma_desc.stride_bytes
                                 : (op.tma_desc.cols * op.tma_desc.elem_bytes);
              if (row_bytes != 0) {
                (void)tmem_.set_row_bytes(op.handle, row_bytes);
              }
            }
          }
        } else {
          const TmemWindowPlan* meta_window = nullptr;
          bool use_meta_window = tmem_.lookup_window(op.handle, meta_shadow_window_id(op.window_id), &meta_window);
          bool acquired = use_meta_window
                        ? try_acquire_tmem_window_linear_write_port(op.handle, meta_shadow_window_id(op.window_id), op.meta_packet_cursor)
                        : try_acquire_tmem_region_write_port(op.meta_col_base, op.meta_col_span, op.meta_packet_cursor);
          if (!acquired) {
            ++perf_stats_.stall_tmem_write_port_busy;
            return;
          }
          auto packet = make_packet(op.meta_buffer, op.meta_packet_cursor);
          bool written = use_meta_window
                       ? tmem_.write_window_linear_packet(op.handle, meta_shadow_window_id(op.window_id), op.meta_packet_cursor, packet)
                       : tmem_.region_write_packet(op.meta_col_base, op.meta_col_span, op.meta_packet_cursor, packet);
          if (!written) {
            op.completed = true;
            finalize_async_tensor_op(op);
            return;
          }
          ++op.meta_packet_cursor;
          if ((op.meta_packet_cursor * Tmem::kPacketBytes) >= op.meta_size_bytes) {
            tmem_.set_meta_ready(op.handle, true);
          }
        }

        --op.remaining_tmem_write_packets;
        if (op.remaining_tmem_write_packets != 0) {
          return;
        }
        (void)tmem_.bump_window_epoch(op.handle);
      }
    } else {
      if (op.remaining_tmem_read_packets != 0) {
        if (!try_acquire_tmem_window_linear_read_port(op.handle, op.window_id, op.payload_packet_cursor)) {
          ++perf_stats_.stall_tmem_read_port_busy;
          return;
        }

        TmemPacket packet;
        if (!tmem_.read_window_linear_packet(op.handle, op.window_id, op.payload_packet_cursor, &packet)) {
          op.completed = true;
          finalize_async_tensor_op(op);
          return;
        }
        auto offset = op.payload_packet_cursor * Tmem::kPacketBytes;
        auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, op.payload_size_bytes - offset);
        tma_.store_payload(op.tma_desc,
                           packet.bytes.data(),
                           valid_bytes,
                           [offset, this](const void* data, uint64_t addr, uint32_t size) {
                             this->dcache_write(data, addr + offset, size);
                           });
        ++op.payload_packet_cursor;
        --op.remaining_tmem_read_packets;
        if (0 == op.remaining_tmem_read_packets) {
          if (op.use_meta_region) {
            tmem_.set_meta_ready(op.handle, true);
          } else {
            tmem_.set_payload_ready(op.handle, true);
          }
        } else {
          return;
        }
      }
    }

    uint64_t latency = perf_stats_.cycles >= op.issue_cycle
                     ? (perf_stats_.cycles - op.issue_cycle)
                     : 0;
    if (op.type == AsyncTensorOpType::TmaLoad) {
      perf_stats_.tma_load_latency_sum += latency;
    } else {
      perf_stats_.tma_store_latency_sum += latency;
    }
  } break;
  case AsyncTensorOpType::TmemShift: {
    TmemAllocation* allocation = nullptr;
    if (!lookup_tmem_allocation(op.handle, &allocation)) {
      op.completed = true;
      finalize_async_tensor_op(op);
      return;
    }
    bool use_window = (op.window_id != 0) || tmem_.lookup_window(op.handle, op.window_id, nullptr);
    if (!op.shift_window_applied) {
      auto packet_idx = op.payload_packet_cursor;
      if (op.remaining_tmem_read_packets != 0) {
        bool acquired = use_window
                      ? try_acquire_tmem_window_read_port(op.handle, op.window_id, packet_idx)
                      : try_acquire_tmem_region_read_port(allocation->payload_col_base, allocation->col_span, packet_idx);
        if (!acquired) {
          ++perf_stats_.stall_tmem_read_port_busy;
          return;
        }
      }
      if (op.remaining_tmem_write_packets != 0) {
        bool acquired = use_window
                      ? try_acquire_tmem_window_write_port(op.handle, op.window_id, packet_idx)
                      : try_acquire_tmem_region_write_port(allocation->payload_col_base, allocation->col_span, packet_idx);
        if (!acquired) {
          ++perf_stats_.stall_tmem_write_port_busy;
          if (op.remaining_tmem_read_packets != 0) {
            if (use_window) {
              tmem_.refund_window_read_packet(perf_stats_.cycles, op.handle, op.window_id, packet_idx);
            } else {
              refund_tmem_region_read_port(allocation->payload_col_base, allocation->col_span, packet_idx);
            }
          }
          return;
        }
      }
      if (op.remaining_tmem_read_packets != 0) {
        --op.remaining_tmem_read_packets;
      }
      if (op.remaining_tmem_write_packets != 0) {
        --op.remaining_tmem_write_packets;
      }
      ++op.payload_packet_cursor;
      if (op.remaining_tmem_read_packets != 0 || op.remaining_tmem_write_packets != 0) {
        return;
      }
      if (use_window) {
        (void)tmem_.shift_window_down(op.handle, op.window_id);
      } else {
        (void)tmem_region_shift_down(allocation->payload_col_base, allocation->col_span, allocation->row_bytes);
      }
      op.shift_window_applied = true;
      if (op.remaining_refill_write_packets == 0) {
        tmem_.set_payload_ready(op.handle, true);
        (void)tmem_.bump_window_epoch(op.handle);
      } else {
        return;
      }
    }

    if (op.remaining_refill_write_packets != 0) {
      if (!use_window) {
        op.completed = true;
        finalize_async_tensor_op(op);
        return;
      }
      auto chunk_idx = op.refill_packet_cursor;
      if (!tmem_.try_acquire_window_line_write_packet(perf_stats_.cycles, op.handle, op.window_id, 0, chunk_idx)) {
        ++perf_stats_.stall_tmem_write_port_busy;
        return;
      }
      TmemPacket packet;
      auto offset = chunk_idx * Tmem::kPacketBytes;
      if (offset < op.payload_buffer.size()) {
        auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, op.payload_buffer.size() - offset);
        std::copy_n(op.payload_buffer.data() + offset, valid_bytes, packet.bytes.begin());
      }
      if (!tmem_.write_window_line_chunk(op.handle, op.window_id, 0, chunk_idx, packet)) {
        op.completed = true;
        finalize_async_tensor_op(op);
        return;
      }
      ++perf_stats_.tmem_write_packets;
      ++op.refill_packet_cursor;
      --op.remaining_refill_write_packets;
      if (op.remaining_refill_write_packets != 0) {
        return;
      }
      tmem_.set_payload_ready(op.handle, true);
      (void)tmem_.bump_window_epoch(op.handle);
    }
  } break;
  case AsyncTensorOpType::MmaLoad:
  case AsyncTensorOpType::MmaStore:
  case AsyncTensorOpType::Wmma:
    break;
  default:
    std::abort();
  }

  op.completed = true;
  finalize_async_tensor_op(op);
}

void Core::advance_async_tensor_ops() {
  ensure_tmem_port_budgets();
  for (auto& entry : async_tensor_ops_) {
    auto& op = entry.second;
    if (!op.completed && perf_stats_.cycles >= op.ready_cycle) {
      process_async_tensor_op(op);
    }
  }
}

uint32_t Core::tmem_alloc(uint32_t col_span) {
  advance_async_tensor_ops();
  return tmem_.alloc(col_span);
}

bool Core::tmem_free(uint32_t handle) {
  advance_async_tensor_ops();
  return tmem_.free(handle);
}

void Core::tmem_rel_permit() {
  tmem_.seal_allocator();
}

uint32_t Core::tma_load(uint32_t wid, uint32_t handle, uint32_t desc_id, bool transpose_b, uint32_t window_id) {
  advance_async_tensor_ops();
  TmaDescriptor desc;
  if (!read_tma_descriptor(desc_id, &desc)) {
    return 0;
  }
  auto kind = static_cast<TcuPayloadKind>(desc.payload_kind);
  TmemAllocation* allocation = nullptr;
  if (lookup_tmem_allocation(handle, &allocation)) {
    tmem_.set_meta_region(handle, desc.meta_tmem_base, desc.meta_col_span);
    if (kind == TcuPayloadKind::SparseMeta) {
      tmem_.set_meta_ready(handle, false);
    } else {
      tmem_.set_payload_ready(handle, false);
    }
    if (kind != TcuPayloadKind::SparseMeta
     && desc.meta_addr != 0 && desc.meta_size_bytes != 0 && desc.meta_col_span != 0) {
      tmem_.set_meta_ready(handle, false);
    }
  }
  ++perf_stats_.tma_load_count;
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmaLoad;
  op.wid = wid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.window_id = window_id;
  op.issue_cycle = perf_stats_.cycles;
  op.ready_cycle = perf_stats_.cycles + 1;
  op.transpose_b = transpose_b;
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::tma_store(uint32_t wid, uint32_t handle, uint32_t desc_id, uint32_t window_id) {
  advance_async_tensor_ops();
  TmaDescriptor desc;
  if (!read_tma_descriptor(desc_id, &desc)) {
    return 0;
  }
  if (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta) {
    tmem_.set_meta_ready(handle, false);
  } else {
    tmem_.set_payload_ready(handle, false);
  }
  ++perf_stats_.tma_store_count;
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmaStore;
  op.wid = wid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.window_id = window_id;
  op.issue_cycle = perf_stats_.cycles;
  op.ready_cycle = perf_stats_.cycles + 1;
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::tmem_shift(uint32_t wid, uint32_t handle, uint32_t window_id, uint32_t refill_desc_id) {
  advance_async_tensor_ops();
  TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return 0;
  }
  tmem_.set_payload_ready(handle, false);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmemShift;
  op.wid = wid;
  op.handle = handle;
  op.window_id = window_id;
  op.refill_desc_id = refill_desc_id;
  op.issue_cycle = perf_stats_.cycles;
  op.ready_cycle = perf_stats_.cycles + 1;
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::mma_load_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_ops();
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::MmaLoad;
  op.wid = wid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.issue_cycle = perf_stats_.cycles;
  op.ready_cycle = std::numeric_limits<uint64_t>::max();
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::mma_store_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_ops();
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::MmaStore;
  op.wid = wid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.issue_cycle = perf_stats_.cycles;
  op.ready_cycle = std::numeric_limits<uint64_t>::max();
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::wmma_async_issue(uint32_t wid) {
  advance_async_tensor_ops();
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::Wmma;
  op.wid = wid;
  op.issue_cycle = perf_stats_.cycles;
  op.ready_cycle = std::numeric_limits<uint64_t>::max();
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

void Core::async_tensor_complete(uint32_t async_id) {
  auto it = async_tensor_ops_.find(async_id);
  if (it == async_tensor_ops_.end() || it->second.completed) {
    return;
  }
  it->second.completed = true;
  finalize_async_tensor_op(it->second);
  if (it->second.type == AsyncTensorOpType::Wmma) {
    async_tensor_ops_.erase(it);
  }
}

uint32_t Core::tc_commit(uint32_t wid, uint32_t barrier_id) {
  advance_async_tensor_ops();
  if (barrier_id >= mbarriers_.size()) {
    return 0;
  }
  auto& barrier = mbarriers_.at(barrier_id);
  if (!barrier.valid) {
    return 0;
  }

  uint32_t committed = 0;
  for (auto& entry : async_tensor_ops_) {
    auto& op = entry.second;
    if (op.wid != wid || op.completed || op.committed) {
      continue;
    }
    op.committed = true;
    op.barrier_id = barrier_id;
    ++committed;
  }

  if (committed != 0) {
    mark_mbarrier_phase_active(barrier_id);
    barrier.pending_tx += committed;
    try_complete_mbarrier(barrier_id);
  }
  try_resume_fence_waiters();
  return committed;
}

bool Core::tc_fence(uint32_t wid, TcuFenceMode mode) {
  advance_async_tensor_ops();
  if (mode == TcuFenceMode::Before) {
    // In-order warp issue already preserves relative ordering before a following thread sync.
    return true;
  }
  // AFTER acts as a completion fence for committed async tensor operations.
  if (!has_pending_async_ops(wid, true)) {
    return true;
  }
  auto& wait_state = fence_wait_states_.at(wid);
  wait_state.active = true;
  wait_state.mode = mode;
  ++perf_stats_.stall_wait_barrier;
  return false;
}

bool Core::tc_wait(uint32_t wid) {
  advance_async_tensor_ops();
  bool pending = false;
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.wid != wid || op.completed) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::MmaLoad:
    case AsyncTensorOpType::MmaStore:
    case AsyncTensorOpType::Wmma:
      async_tensor_waiters_[op.async_id].set(wid);
      pending = true;
      break;
    default:
      break;
    }
  }
  if (pending) {
    ++perf_stats_.stall_wait_barrier;
  }
  return !pending;
}

bool Core::mbarrier_init(uint32_t barrier_id, uint32_t count) {
  if (barrier_id >= mbarriers_.size()) {
    return false;
  }
  auto& barrier = mbarriers_.at(barrier_id);
  barrier = {};
  barrier.valid = true;
  barrier.expected_arrivals = count;
  barrier.pending_arrivals = count;
  barrier.phase_done = (count == 0);
  for (auto& wait_targets : mbarrier_wait_targets_) {
    wait_targets.erase(barrier_id);
  }
  return true;
}

void Core::mbarrier_arrive(uint32_t barrier_id) {
  if (barrier_id >= mbarriers_.size()) {
    return;
  }
  auto& barrier = mbarriers_.at(barrier_id);
  if (!barrier.valid) {
    return;
  }
  mark_mbarrier_phase_active(barrier_id);
  if (barrier.pending_arrivals > 0) {
    --barrier.pending_arrivals;
  }
  try_complete_mbarrier(barrier_id);
}

bool Core::mbarrier_wait(uint32_t wid, uint32_t barrier_id) {
  advance_async_tensor_ops();
  if (barrier_id >= mbarriers_.size()) {
    return true;
  }
  auto& barrier = mbarriers_.at(barrier_id);
  if (!barrier.valid) {
    return true;
  }

  auto& wait_targets = mbarrier_wait_targets_.at(wid);
  auto it = wait_targets.find(barrier_id);
  if (it != wait_targets.end()) {
    if (barrier.phase >= it->second) {
      wait_targets.erase(it);
      return true;
    }
    barrier.waiters_bitmap.set(wid);
    ++perf_stats_.stall_wait_barrier;
    return false;
  }

  if (barrier.phase_done) {
    return true;
  }

  wait_targets[barrier_id] = barrier.phase + 1;
  barrier.waiters_bitmap.set(wid);
  ++perf_stats_.stall_wait_barrier;
  return false;
}

bool Core::tma_wait(uint32_t wid, uint32_t async_id) {
  advance_async_tensor_ops();
  auto it = async_tensor_ops_.find(async_id);
  if (it == async_tensor_ops_.end()) {
    return true;
  }
  if (!it->second.completed) {
    async_tensor_waiters_[async_id].set(wid);
    ++perf_stats_.stall_wait_barrier;
    return false;
  }
  async_tensor_waiters_.erase(async_id);
  async_tensor_ops_.erase(it);
  return true;
}

bool Core::tmem_read_packet(uint32_t handle, uint32_t packet_idx, TmemPacket* out) {
  advance_async_tensor_ops();
  return tmem_.read_packet(handle, packet_idx, out);
}

bool Core::tmem_read_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) {
  advance_async_tensor_ops();
  return tmem_.read_window_packet(handle, window_id, packet_idx, out);
}

bool Core::tmem_read_meta_packet(uint32_t handle, uint32_t packet_idx, TmemPacket* out) {
  advance_async_tensor_ops();
  return tmem_.read_meta_packet(handle, packet_idx, out);
}

bool Core::tmem_write_packet(uint32_t handle, uint32_t packet_idx, const TmemPacket& in) {
  advance_async_tensor_ops();
  return tmem_.write_packet(handle, packet_idx, in);
}

bool Core::tmem_write_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in) {
  advance_async_tensor_ops();
  return tmem_.write_window_packet(handle, window_id, packet_idx, in);
}

#endif
