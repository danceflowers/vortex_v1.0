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
// Core owns the architectural pipeline plus the Core-side tensor background
// engine. The tensor background engine advances TMA load/store and TMEM shift
// transactions one cycle at a time before the later in-core pipeline stages
// observe TMEM-visible state in the same global cycle.

#include "core.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

namespace {

uint64_t env_u64_value(const char* name, uint64_t default_value) {
  auto value = std::getenv(name);
  if (nullptr == value || '\0' == value[0]) {
    return default_value;
  }
  return std::strtoull(value, nullptr, 0);
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
  , tmem_system_(TmemSystem::Create("tmem_system", this))
  , tma_(Tma::Create("tma", this))
  , tensor_async_op_completion_in_(this)
  , tma_async_op_completion_in_(this)
  , tmem_system_async_op_completion_in_(this)
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
  , mbarrier_wait_targets_(arch.num_warps())
  , fence_wait_states_(arch.num_warps())
  , next_async_id_(1)
#endif
{
  char sname[100];

#ifdef EXT_TCU_ENABLE
  tensor_unit_->TensorMemReqOut.bind(&tmem_system_->TensorExecuteReqIn);
  tmem_system_->TensorExecuteRspOut.bind(&tensor_unit_->TensorMemRspIn);
  tma_->TmemReqOut.bind(&tmem_system_->CoreTransferReqIn);
  tmem_system_->CoreTransferRspOut.bind(&tma_->TmemRspIn);
  tensor_unit_->TensorAsyncOpCompletionOut.bind(&tensor_async_op_completion_in_);
  tma_->AsyncOpCompletionOut.bind(&tma_async_op_completion_in_);
  tmem_system_->AsyncOpCompletionOut.bind(&tmem_system_async_op_completion_in_);
#endif

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
  tmem_system_->reset();
  tma_->reset();
  async_tensor_ops_.clear();
  pending_tcgen05_ldst_ops_.clear();
  async_tensor_waiters_.clear();
  tcgen05_ld_waiters_.reset();
  tcgen05_st_waiters_.reset();
  // mbarrier table is now an unordered_map keyed by LMEM address; reset = clear.
  mbarriers_.clear();
  for (auto& wait_targets : mbarrier_wait_targets_) {
    wait_targets.clear();
  }
  for (auto& fence_wait : fence_wait_states_) {
    fence_wait = {};
  }
  next_async_id_ = 1;
  last_tensor_completion_drain_cycle_ = std::numeric_limits<uint64_t>::max();
  last_tma_completion_drain_cycle_ = std::numeric_limits<uint64_t>::max();
  last_tcgen05_ldst_advance_cycle_ = std::numeric_limits<uint64_t>::max();
#endif

  perf_stats_ = PerfStats();
}

void Core::tick() {
#ifdef EXT_TCU_ENABLE
  execute_cycle_tmem_shift_reserved_handles_.clear();
  drain_tensor_execute_completion_notices();
  drain_tma_completion_notices();
  advance_tcgen05_ldst_async_ops();
  if (!tmem_system_async_op_completion_in_.empty()) {
    auto completion = tmem_system_async_op_completion_in_.front();
    async_tensor_complete(completion.async_id);
    tmem_system_async_op_completion_in_.pop();
  }
#endif
  this->commit();
  this->execute();
  this->issue();
  this->decode();
  this->fetch();
  this->schedule();
#ifdef EXT_TCU_ENABLE
  auto watchdog_cycle = env_u64_value("VORTEX_SIMX_TENSOR_WATCHDOG_CYCLES", 0);
  if (watchdog_cycle != 0 && perf_stats_.cycles >= watchdog_cycle) {
    this->dump_tensor_debug_state(std::cerr);
    std::abort();
  }
#endif

  ++perf_stats_.cycles;
  DPN(2, std::flush);
}

void Core::publish_visible_tensor_mem_state() {
  tmem_system_->publish_visible_state();
}

uint64_t Core::startup_arg() const {
  return emulator_.startup_arg();
}

void Core::dump_tensor_debug_state(std::ostream& os) const {
#ifdef EXT_TCU_ENABLE
  os << "==== tensor-debug-state begin ====\n";
  os << "cycle=" << perf_stats_.cycles << "\n";
  os << "emulator_running=" << emulator_.running()
     << " fetch_empty=" << fetch_latch_.empty()
     << " decode_empty=" << decode_latch_.empty()
     << " pending_instrs=" << pending_instrs_.size()
     << "\n";
  for (uint32_t wid = 0; wid < arch_.num_warps(); ++wid) {
    os << "  warp=" << wid
       << " active=" << emulator_.active(wid)
       << " stalled=" << emulator_.stalled(wid)
       << " tmask=" << emulator_.warp_tmask_count(wid)
       << " pc=0x" << std::hex << emulator_.warp_pc(wid) << std::dec
       << " stall_reason=" << static_cast<uint32_t>(emulator_.stall_reason(wid))
       << " ibuffer_empty=" << ibuffers_.at(wid).empty()
       << "\n";
    emulator_.dump_warp_front_state(os, wid);
  }
  os << "async_tensor_ops=" << async_tensor_ops_.size() << "\n";
  os << "async_tensor_waiters=" << async_tensor_waiters_.size() << "\n";
  for (const auto& entry : async_tensor_waiters_) {
    os << "  waiters async_id=" << entry.first
       << " mask=" << entry.second.to_string()
       << "\n";
  }
  for (auto* trace : pending_instrs_) {
    os << "  pending_trace " << *trace << "\n";
  }
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    os << "  async_id=" << op.async_id
       << " type=" << static_cast<uint32_t>(op.type)
       << " wid=" << op.wid
       << " handle=" << op.handle
       << " completed=" << op.completed
       << " committed=" << op.committed
       << " barrier=" << op.barrier_id
       << "\n";
  }
  tmem_system_->dump_debug_state(os);
  if (nullptr != tma_) {
    tma_->dump_debug_state(os);
  }
  if (nullptr != tensor_unit_) {
    tensor_unit_->dump_debug_state(os);
  }
  os << "==== tensor-debug-state end ====\n";
#else
  (void)os;
#endif
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

#ifdef EXT_TCU_ENABLE
    if (auto tcu_type = std::get_if<TcuType>(&trace->op_type)) {
      if (*tcu_type == TcuType::TCU_LD
       && !tcgen05_ld_trace_ready(reinterpret_cast<uint64_t>(trace))) {
        continue;
      }
    }
#endif

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

bool Core::lookup_tmem_allocation(uint32_t taddr, TmemAllocation** allocation) {
  return tmem_system_->lookup_allocation(taddr, allocation);
}

bool Core::lookup_tmem_allocation(uint32_t handle, const TmemAllocation** allocation) const {
  return tmem_system_->lookup_allocation(handle, allocation);
}

WarpMask Core::warpgroup_mask(uint32_t wgid) const {
  WarpMask mask;
  auto first = arch_.warpgroup_first_wid(wgid);
  for (uint32_t lane = 0; lane < arch_.warpgroup_size(); ++lane) {
    auto wid = first + lane;
    if (wid < arch_.num_warps()) {
      mask.set(wid);
    }
  }
  return mask;
}

bool Core::has_pending_async_ops(uint32_t wid, bool committed_only) const {
  auto wgid = arch_.warpgroup_id(wid);
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.wgid != wgid || op.completed) {
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
  auto wgid = arch_.warpgroup_id(wid);
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.wgid != wgid || op.completed) {
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
    auto wgid = arch_.warpgroup_id(wid);
    bool ready = (fence_wait.mode == TcuFenceMode::After)
              ? !has_pending_async_ops(wid, true)
              : !has_pending_async_ops(wid, false);
    if (ready) {
      auto mask = warpgroup_mask(wgid);
      for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
        if (mask.test(gwid)) {
          fence_wait_states_.at(gwid).active = false;
        }
      }
      for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
        if (mask.test(gwid)) {
          emulator_.resume(gwid);
        }
      }
    }
  }
}

bool Core::tmem_region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const {
  return tmem_system_->region_query(col_base, col_span, size_bytes);
}

bool Core::tmem_query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const {
  return tmem_system_->query(handle, col_span, size_bytes);
}

bool Core::tmem_lookup_allocation(uint32_t handle, const TmemAllocation** allocation) const {
  return lookup_tmem_allocation(handle, allocation);
}

bool Core::tmem_transfer_region(uint32_t handle, uint32_t* col_base, uint32_t* col_span) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation) || nullptr == allocation) {
    return false;
  }
  if (col_base) {
    *col_base = allocation->payload_col_base;
  }
  if (col_span) {
    *col_span = allocation->col_span;
  }
  return true;
}

uint64_t Core::enqueue_tmem_request(const TmemRequestDesc& desc) {
  return tmem_system_->enqueue_port_request(desc, desc.age);
}

bool Core::tmem_request_granted(uint64_t tag) const {
  return tmem_system_->request_granted(tag);
}

void Core::consume_tmem_request_grant(uint64_t tag) {
  tmem_system_->consume_request_grant(tag);
}

void Core::tmem_set_payload_ready(uint32_t handle, bool ready) {
  tmem_system_->set_payload_ready(handle, ready);
}

void Core::tmem_set_meta_ready(uint32_t handle, bool ready) {
  tmem_system_->set_meta_ready(handle, ready);
}

void Core::tmem_set_meta_region(uint32_t handle, uint32_t meta_col_base, uint32_t meta_col_span) {
  tmem_system_->set_meta_region(handle, meta_col_base, meta_col_span);
}

bool Core::tmem_set_row_bytes(uint32_t handle, uint32_t row_bytes) {
  return tmem_system_->set_row_bytes(handle, row_bytes);
}

bool Core::tmem_handle_busy(uint32_t handle) const {
  return tmem_system_->visible_shift_busy(handle);
}

bool Core::tmem_handle_ready_for_mma_load(uint32_t handle, TcuTarget target, uint32_t sparse_mode) const {
  return TmemHandleBlockReason::None == tmem_handle_load_block_reason(handle, target, sparse_mode);
}

bool Core::tmem_handle_ready_for_mma_store(uint32_t handle) const {
  return TmemHandleBlockReason::None == tmem_handle_store_block_reason(handle);
}

Core::TmemHandleBlockReason Core::tmem_handle_load_block_reason(uint32_t handle, TcuTarget target, uint32_t sparse_mode) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return TmemHandleBlockReason::Invalid;
  }
  if (execute_cycle_tmem_shift_reserved_handles_.count(handle) != 0) {
    return TmemHandleBlockReason::BusyTmemShift;
  }
  if (tmem_system_->visible_shift_busy(handle)) {
    return TmemHandleBlockReason::BusyTmemShift;
  }
  bool payload_ready = tmem_system_->visible_payload_ready(handle);
  if (!payload_ready) {
    return TmemHandleBlockReason::PayloadNotReady;
  }
  bool needs_meta = (target == TcuTarget::A || target == TcuTarget::None)
                 && (sparse_mode != vortex::tensor::sparse_none);
  bool meta_ready = tmem_system_->visible_meta_ready(handle);
  if (needs_meta && !meta_ready) {
    return TmemHandleBlockReason::MetaNotReady;
  }
  return TmemHandleBlockReason::None;
}

Core::TmemHandleBlockReason Core::tmem_handle_store_block_reason(uint32_t handle) const {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return TmemHandleBlockReason::Invalid;
  }
  if (execute_cycle_tmem_shift_reserved_handles_.count(handle) != 0) {
    return TmemHandleBlockReason::BusyTmemShift;
  }
  if (tmem_system_->visible_shift_busy(handle)) {
    return TmemHandleBlockReason::BusyTmemShift;
  }
  bool payload_ready = tmem_system_->visible_payload_ready(handle);
  if (!payload_ready) {
    return TmemHandleBlockReason::PayloadNotReady;
  }
  return TmemHandleBlockReason::None;
}

// ============================================================================
// LMEM mirror helper: encode the in-Core MBarrierEntry into the 8 B
// mbarrier_state_t bit layout and write it through to the kernel-visible
// LMEM location at mbar_addr. Called after every state transition.
// PTX §7.6.1 mbarrier object format: 64-bit packed value
//   bits[63:62]   phase parity (2-bit)
//   bits[61:32]   pending_arrival_count (30 bits)
//   bits[31:20]   pending_tx_count      (12 bits, byte units per §7.6.4)
//   bits[19:0]    expected_arrival_count (20 bits)
// ============================================================================
void Core::mirror_mbarrier_to_lmem(uint64_t mbar_addr, const MBarrierEntry& b) {
  uint64_t encoded = 0;
  encoded |= (uint64_t)(b.expected_arrival_count & 0xFFFFFu) << 0;
  encoded |= (uint64_t)(b.pending_tx_count       & 0xFFFu)   << 20;
  encoded |= (uint64_t)(b.pending_arrival_count  & 0x3FFFFFFFu) << 32;
  encoded |= (uint64_t)(b.phase                  & 0x3u)     << 62;
  this->lmem_write(&encoded, mbar_addr, sizeof(encoded));
}

// PTX §7.6.2 phase advance: when both arrival and tx targets are met, the
// phase parity bit toggles, expected_arrival_count is reloaded into
// pending_arrival_count, and the tx counters reset.
void Core::try_complete_mbarrier(uint64_t mbar_addr) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end()) {
    return;
  }
  auto& barrier = it->second;
  if (!barrier.valid
   || barrier.pending_arrival_count != 0
   || barrier.pending_tx_count < barrier.expected_tx_count) {
    return;
  }

  // Phase advance.
  barrier.phase = (barrier.phase + 1) & 0x3u;
  barrier.pending_arrival_count = barrier.expected_arrival_count;
  barrier.pending_tx_count = 0;
  barrier.expected_tx_count = 0;
  mirror_mbarrier_to_lmem(mbar_addr, barrier);

  // Wake up warps that were waiting on the prior phase.
  auto waiters = barrier.waiters_bitmap;
  barrier.waiters_bitmap.reset();
  for (uint32_t wid = 0; wid < arch_.num_warps(); ++wid) {
    if (waiters.test(wid)) {
      emulator_.resume(wid);
    }
  }
}

void Core::on_async_tensor_op_completed(AsyncTensorOp& op) {
  if (!op.completed) {
    return;
  }
  // GAP-2: cp.async.bulk.tensor.mbarrier::complete_tx::bytes wiring.
  // When the issue path (cpabulk_tensor_load) bound this op to a mbarrier
  // with the complete_tx::bytes flag set, the byte count completed by the
  // bulk transfer is added to that mbarrier's pending_tx_count. PTX §7.6.4.
  if (op.tx_bound_mbar != 0 && op.tx_bytes != 0) {
    mbarrier_complete_tx(op.tx_bound_mbar, op.tx_bytes);
    // Reset so a single op completion only fires complete_tx once.
    op.tx_bound_mbar = 0;
    op.tx_bytes = 0;
  }
  resume_async_waiters(op.async_id);
  try_resume_fence_waiters();
  if (!op.committed) {
    return;
  }
  // tcgen05.commit semantics (PTX §9.7.16.5.7): each prior tcgen05.async op
  // bound by tcgen05.commit signals an arrival on the bound mbarrier on
  // completion. NOT a tx_count update -- those go via mbarrier.complete_tx.
  // op.barrier_id was reinterpreted in tc_commit() as the LMEM mbar address.
  uint64_t mbar_addr = op.barrier_id;
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  auto& barrier = it->second;
  if (barrier.pending_arrival_count > 0) {
    --barrier.pending_arrival_count;
  }
  mirror_mbarrier_to_lmem(mbar_addr, barrier);
  try_complete_mbarrier(mbar_addr);
}

void Core::drain_tensor_execute_completion_notices() {
  if (tensor_async_op_completion_in_.empty()) {
    return;
  }
  auto cycle = SimPlatform::instance().cycles();
  if (last_tensor_completion_drain_cycle_ == cycle) {
    return;
  }
  auto completion = tensor_async_op_completion_in_.front();
  async_tensor_complete(completion.async_id);
  tensor_async_op_completion_in_.pop();
  last_tensor_completion_drain_cycle_ = cycle;
}

void Core::drain_tma_completion_notices() {
  if (tma_async_op_completion_in_.empty()) {
    return;
  }
  auto cycle = SimPlatform::instance().cycles();
  if (last_tma_completion_drain_cycle_ == cycle) {
    return;
  }
  auto completion = tma_async_op_completion_in_.front();
  async_tensor_complete(completion.async_id);
  tma_async_op_completion_in_.pop();
  last_tma_completion_drain_cycle_ = cycle;
}

void Core::advance_async_tensor_engine() {
  drain_tensor_execute_completion_notices();
  drain_tma_completion_notices();
  try_resume_fence_waiters();
}

uint32_t Core::launch_lmem_to_tmem_copy(uint32_t wid,
                                        uint32_t col_base,
                                        uint32_t col_span,
                                        uint64_t lmem_addr,
                                        uint32_t byte_offset,
                                        uint32_t total_bytes) {
  if (total_bytes == 0) {
    return 0;
  }
  uint32_t region_bytes = 0;
  if (!tmem_region_query(col_base, col_span, &region_bytes)
   || byte_offset >= region_bytes
   || total_bytes > region_bytes - byte_offset) {
    return 0;
  }

  auto wgid = arch_.warpgroup_id(wid);
  auto async_id = next_async_id_++;
  AsyncTensorOp async_op{};
  async_op.async_id = async_id;
  async_op.type = AsyncTensorOpType::TmemCopy;
  async_op.wid = wid;
  async_op.wgid = wgid;
  async_op.handle = col_base;
  async_op.issue_cycle = perf_stats_.cycles;
  async_op.payload_size_bytes = total_bytes;
  async_tensor_ops_[async_id] = async_op;

  tmem_system_->set_payload_ready(col_base, false, true);
  if (!tma_->issue_lmem_to_tmem_copy(async_id,
                                     wid,
                                     wgid,
                                     col_base,
                                     col_span,
                                     lmem_addr,
                                     byte_offset,
                                     total_bytes)) {
    async_tensor_ops_.erase(async_id);
    tmem_system_->set_payload_ready(col_base, true, true);
    return 0;
  }
  return async_id;
}

uint32_t Core::tmem_alloc(uint32_t col_span, uint32_t reserved_operand) {
  advance_async_tensor_engine();

  if (reserved_operand != kTmemAllocReservedOperand) {
    throw std::runtime_error(
      "TMEM_ALLOC rs2 is reserved in the PTX-only model; "
      "pass 0xffffffff and provide tcgen05.mma idesc through TCU_MMA rs1");
  }

  // col_span 必须是 {16, 32, 64} 之一 (16 × 2^n)
  if (col_span != 16 && col_span != 32 && col_span != 64) {
    throw std::runtime_error(
      "TMEM_ALLOC: col_span=" + std::to_string(col_span)
      + " invalid, must be 16, 32, or 64");
  }

  return tmem_system_->alloc(col_span);
}

bool Core::tmem_dealloc(uint32_t handle) {
  advance_async_tensor_engine();
  return tmem_system_->free(handle);
}

void Core::tmem_rel_permit() {
  tmem_system_->seal_allocator();
}

uint32_t Core::tmem_shift(uint32_t wid, uint32_t handle, uint32_t control_word) {
  advance_async_tensor_engine();
  if (control_word != kTcuTmemOpControlPtxOnly) {
    throw std::runtime_error(
      "TMEM_SHIFT non-zero control word is not part of the PTX-only model");
  }
  auto wgid = arch_.warpgroup_id(wid);
  TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return 0;
  }
  tmem_system_->set_payload_ready(handle, false);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmemShift;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.issue_cycle = perf_stats_.cycles;
  async_tensor_ops_[async_id] = std::move(op);
  execute_cycle_tmem_shift_reserved_handles_.insert(handle);
  (void)tmem_system_->issue_shift(async_id, wid, wgid, handle, perf_stats_.cycles);
  return async_id;
}

uint32_t Core::mma_load_async_issue(uint32_t wid, uint32_t handle, uint32_t idesc) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::MmaLoad;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.idesc = idesc;
  op.issue_cycle = perf_stats_.cycles;
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::mma_store_async_issue(uint32_t wid, uint32_t handle, uint32_t idesc) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::MmaStore;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.idesc = idesc;
  op.issue_cycle = perf_stats_.cycles;
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::wmma_async_issue(uint32_t wid) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::Wmma;
  op.wid = wid;
  op.wgid = wgid;
  op.issue_cycle = perf_stats_.cycles;
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

void Core::async_tensor_complete(uint32_t async_id) {
  auto it = async_tensor_ops_.find(async_id);
  if (it == async_tensor_ops_.end() || it->second.completed) {
    return;
  }
  it->second.completed = true;
  on_async_tensor_op_completed(it->second);
  if (it->second.type == AsyncTensorOpType::Wmma) {
    async_tensor_ops_.erase(it);
  }
}

void Core::issue_tcgen05_ld_async(uint32_t wid,
                                  uint64_t trace_id,
                                  const RegOpd& dst,
                                  const ThreadMask& tmask,
                                  const std::vector<uint32_t>& taddrs) {
  PendingTcgen05LdStOp op;
  op.trace_id = trace_id;
  op.wid = wid;
  op.is_store = false;
  op.dst = dst;
  op.tmask = tmask;
  op.issue_cycle = perf_stats_.cycles;
  op.taddrs = taddrs;
  op.values.assign(arch_.num_threads(), 0);
  op.accesses.reserve(arch_.num_threads());

  for (uint32_t t = 0; t < arch_.num_threads(); ++t) {
    if (!tmask.test(t)) {
      continue;
    }
    PendingTcgen05LdStAccess access{};
    access.thread = t;
    uint32_t taddr = taddrs.at(t);
    uint32_t lane_base = taddr & 0xFFFFu;
    uint32_t col_byte  = (taddr >> 16) & 0xFFFFu;
    uint32_t actual_lane = lane_base + t;
    uint32_t col_base = 0;
    const TmemAllocation* allocation = nullptr;
    if (tmem_find_allocation_by_lane(actual_lane, &col_base)
     && lookup_tmem_allocation(col_base, &allocation)
     && nullptr != allocation
     && allocation->valid) {
      uint32_t byte_offset = (actual_lane - col_base) * Tmem::kColBytes + col_byte;
      uint32_t region_bytes = allocation->col_span * Tmem::kColBytes;
      if (byte_offset < region_bytes) {
        access.valid = true;
        access.col_base = col_base;
        access.col_span = allocation->col_span;
        access.byte_offset = byte_offset;
        access.next_packet_idx = byte_offset / Tmem::kPacketBytes;
        access.last_packet_idx = std::min<uint32_t>(
            region_bytes - 1,
            byte_offset + static_cast<uint32_t>(sizeof(uint32_t)) - 1) / Tmem::kPacketBytes;
      }
    }
    op.accesses.push_back(access);
  }

  pending_tcgen05_ldst_ops_[trace_id] = std::move(op);
}

void Core::issue_tcgen05_st_async(uint32_t wid,
                                  uint64_t trace_id,
                                  const ThreadMask& tmask,
                                  const std::vector<uint32_t>& taddrs,
                                  const std::vector<uint32_t>& values) {
  PendingTcgen05LdStOp op;
  op.trace_id = trace_id;
  op.wid = wid;
  op.is_store = true;
  op.tmask = tmask;
  op.issue_cycle = perf_stats_.cycles;
  op.taddrs = taddrs;
  op.values.assign(arch_.num_threads(), 0);
  op.accesses.reserve(arch_.num_threads());

  for (uint32_t t = 0; t < arch_.num_threads(); ++t) {
    if (!tmask.test(t)) {
      continue;
    }
    op.values.at(t) = values.at(t);
    PendingTcgen05LdStAccess access{};
    access.thread = t;
    uint32_t taddr = taddrs.at(t);
    uint32_t lane_base = taddr & 0xFFFFu;
    uint32_t col_byte  = (taddr >> 16) & 0xFFFFu;
    uint32_t actual_lane = lane_base + t;
    uint32_t col_base = 0;
    const TmemAllocation* allocation = nullptr;
    if (tmem_find_allocation_by_lane(actual_lane, &col_base)
     && lookup_tmem_allocation(col_base, &allocation)
     && nullptr != allocation
     && allocation->valid) {
      uint32_t byte_offset = (actual_lane - col_base) * Tmem::kColBytes + col_byte;
      uint32_t region_bytes = allocation->col_span * Tmem::kColBytes;
      if (byte_offset < region_bytes) {
        access.valid = true;
        access.col_base = col_base;
        access.col_span = allocation->col_span;
        access.byte_offset = byte_offset;
        access.next_packet_idx = byte_offset / Tmem::kPacketBytes;
        access.last_packet_idx = std::min<uint32_t>(
            region_bytes - 1,
            byte_offset + static_cast<uint32_t>(sizeof(uint32_t)) - 1) / Tmem::kPacketBytes;
      }
    }
    op.accesses.push_back(access);
  }

  pending_tcgen05_ldst_ops_[trace_id] = std::move(op);
}

bool Core::tcgen05_ld_trace_ready(uint64_t trace_id) const {
  return pending_tcgen05_ldst_ops_.find(trace_id) == pending_tcgen05_ldst_ops_.end();
}

bool Core::has_pending_tcgen05_ldst(uint32_t wid, bool wait_store) const {
  for (const auto& entry : pending_tcgen05_ldst_ops_) {
    const auto& op = entry.second;
    if (op.wid == wid && op.is_store == wait_store) {
      return true;
    }
  }
  return false;
}

void Core::try_resume_tcgen05_ldst_waiters() {
  for (uint32_t wid = 0; wid < arch_.num_warps(); ++wid) {
    if (tcgen05_ld_waiters_.test(wid) && !has_pending_tcgen05_ldst(wid, false)) {
      tcgen05_ld_waiters_.reset(wid);
      emulator_.resume(wid);
    }
    if (tcgen05_st_waiters_.test(wid) && !has_pending_tcgen05_ldst(wid, true)) {
      tcgen05_st_waiters_.reset(wid);
      emulator_.resume(wid);
    }
  }
}

void Core::advance_tcgen05_ldst_async_ops() {
  if (pending_tcgen05_ldst_ops_.empty()) {
    return;
  }
  auto cycle = SimPlatform::instance().cycles();
  if (last_tcgen05_ldst_advance_cycle_ == cycle) {
    return;
  }
  last_tcgen05_ldst_advance_cycle_ = cycle;

  std::vector<uint64_t> completed_ops;
  for (auto& entry : pending_tcgen05_ldst_ops_) {
    auto& op = entry.second;
    if (perf_stats_.cycles <= op.issue_cycle) {
      continue;
    }

    while (op.next_access < op.accesses.size()) {
      auto& access = op.accesses.at(op.next_access);
      if (!access.valid) {
        ++op.next_access;
        continue;
      }

      if (access.request_tag == 0) {
        TmemRequestDesc request{};
        request.kind = op.is_store ? TmemRequestKind::RegionWrite
                                   : TmemRequestKind::RegionRead;
        request.age = (op.issue_cycle << 32)
                    | (static_cast<uint32_t>(op.trace_id)
                    +  static_cast<uint32_t>(op.next_access));
        request.col_base = access.col_base;
        request.col_span = access.col_span;
        request.packet_idx = access.next_packet_idx;
        access.request_tag = enqueue_tmem_request(request);
        break;
      }

      if (!tmem_request_granted(access.request_tag)) {
        if (op.is_store) {
          ++perf_stats_.stall_tmem_write_port_busy;
        } else {
          ++perf_stats_.stall_tmem_read_port_busy;
        }
        break;
      }

      consume_tmem_request_grant(access.request_tag);
      access.request_tag = 0;
      if (access.next_packet_idx < access.last_packet_idx) {
        ++access.next_packet_idx;
        break;
      }

      if (op.is_store) {
        uint32_t value = static_cast<uint32_t>(op.values.at(access.thread));
        uint8_t buf[4] = {
          static_cast<uint8_t>(value & 0xff),
          static_cast<uint8_t>((value >> 8) & 0xff),
          static_cast<uint8_t>((value >> 16) & 0xff),
          static_cast<uint8_t>((value >> 24) & 0xff),
        };
        (void)tmem_region_write_bytes(access.col_base,
                                      access.byte_offset,
                                      buf,
                                      sizeof(buf));
      } else {
        uint8_t buf[4] = {0, 0, 0, 0};
        (void)tmem_region_read_bytes(access.col_base,
                                     access.byte_offset,
                                     buf,
                                     sizeof(buf));
        op.values.at(access.thread) = static_cast<Word>(
            static_cast<uint32_t>(buf[0])
          | (static_cast<uint32_t>(buf[1]) << 8)
          | (static_cast<uint32_t>(buf[2]) << 16)
          | (static_cast<uint32_t>(buf[3]) << 24));
      }
      ++op.next_access;
      break;
    }

    if (op.next_access >= op.accesses.size()) {
      if (!op.is_store && op.dst.type == RegType::Integer && op.dst.idx != 0) {
        emulator_.write_ireg(op.wid, op.dst.idx, op.tmask, op.values);
      }
      completed_ops.push_back(entry.first);
    }
  }

  for (auto trace_id : completed_ops) {
    pending_tcgen05_ldst_ops_.erase(trace_id);
  }
  if (!completed_ops.empty()) {
    try_resume_tcgen05_ldst_waiters();
  }
}

// tcgen05.wait::ld/st only waits for prior tcgen05.ld/st instructions of the
// matching kind. MMA/cp/shift completion is synchronized via tcgen05.commit +
// mbarrier, not via wait::st.
bool Core::wait_for_inflight_tcgen05_ld_st(uint32_t wid, bool wait_store) {
  advance_async_tensor_engine();
  advance_tcgen05_ldst_async_ops();
  if (!has_pending_tcgen05_ldst(wid, wait_store)) {
    return true;
  }
  if (wait_store) {
    tcgen05_st_waiters_.set(wid);
  } else {
    tcgen05_ld_waiters_.set(wid);
  }
  ++perf_stats_.stall_wait_barrier;
  return false;
}

// PTX §7.6.3 mbarrier.init: initialize the 8 B mbarrier object in shared mem
// with `count` expected arrivals; phase parity starts at 0; tx counters at 0.
bool Core::mbarrier_init(uint64_t mbar_addr, uint32_t count) {
  auto& barrier = mbarriers_[mbar_addr];
  barrier = {};
  barrier.valid = true;
  barrier.expected_arrival_count = count;
  barrier.pending_arrival_count = count;
  barrier.phase = 0;
  // Drop any stale waiter targets for this address from prior generations.
  for (auto& wait_targets : mbarrier_wait_targets_) {
    wait_targets.erase(mbar_addr);
  }
  mirror_mbarrier_to_lmem(mbar_addr, barrier);
  return true;
}

// PTX §7.6.3 mbarrier.invalidate: tear down the object.
void Core::mbarrier_invalidate(uint64_t mbar_addr) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end()) {
    return;
  }
  // Wake any blocked waiters first (they will see the object invalid and exit).
  for (uint32_t wid = 0; wid < arch_.num_warps(); ++wid) {
    if (it->second.waiters_bitmap.test(wid)) {
      emulator_.resume(wid);
    }
    mbarrier_wait_targets_.at(wid).erase(mbar_addr);
  }
  mbarriers_.erase(it);
  // Clear the LMEM mirror so kernel observes invalidation.
  uint64_t zero = 0;
  this->lmem_write(&zero, mbar_addr, sizeof(zero));
}

// PTX §7.6.3 mbarrier.arrive: decrement pending_arrival_count by `decrement`.
// Returns the *current* phase token so caller can wait on this phase.
uint32_t Core::mbarrier_arrive(uint64_t mbar_addr, uint32_t decrement_count) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return 0;
  }
  auto& barrier = it->second;
  uint32_t observed_phase = barrier.phase;
  if (barrier.pending_arrival_count >= decrement_count) {
    barrier.pending_arrival_count -= decrement_count;
  } else {
    barrier.pending_arrival_count = 0;
  }
  mirror_mbarrier_to_lmem(mbar_addr, barrier);
  try_complete_mbarrier(mbar_addr);
  return observed_phase;
}

// PTX §7.6.3 mbarrier.arrive_drop: arrive AND decrement expected_arrival_count
// permanently (participant leaves the barrier).
void Core::mbarrier_arrive_drop(uint64_t mbar_addr) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  auto& barrier = it->second;
  if (barrier.expected_arrival_count > 0) {
    --barrier.expected_arrival_count;
  }
  if (barrier.pending_arrival_count > 0) {
    --barrier.pending_arrival_count;
  }
  mirror_mbarrier_to_lmem(mbar_addr, barrier);
  try_complete_mbarrier(mbar_addr);
}

// PTX §7.6.4 mbarrier.expect_tx: increase expected tx-byte target.
void Core::mbarrier_expect_tx(uint64_t mbar_addr, uint32_t tx_bytes) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  it->second.expected_tx_count += tx_bytes;
  mirror_mbarrier_to_lmem(mbar_addr, it->second);
}

// PTX §7.6.4 mbarrier.complete_tx: contribute completed tx bytes; if combined
// with arrival it can advance phase.
void Core::mbarrier_complete_tx(uint64_t mbar_addr, uint32_t tx_bytes) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  it->second.pending_tx_count += tx_bytes;
  mirror_mbarrier_to_lmem(mbar_addr, it->second);
  try_complete_mbarrier(mbar_addr);
}

// PTX §7.6.3 mbarrier.wait.parity: blocking wait until current phase parity
// differs from `phase_token`. Returns false to indicate stall (caller suspends
// the warp); returns true once the phase has advanced.
bool Core::mbarrier_wait(uint32_t wid, uint64_t mbar_addr, uint32_t phase_token) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto group_mask = warpgroup_mask(wgid);
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return true;
  }
  auto& barrier = it->second;
  // Phase parity comparison (PTX semantics): if current parity bit differs
  // from input token, the wait is satisfied.
  if ((barrier.phase & 0x1u) != (phase_token & 0x1u)) {
    for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
      if (group_mask.test(gwid)) {
        mbarrier_wait_targets_.at(gwid).erase(mbar_addr);
      }
    }
    return true;
  }
  for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
    if (group_mask.test(gwid)) {
      mbarrier_wait_targets_.at(gwid)[mbar_addr] = phase_token;
    }
  }
  barrier.waiters_bitmap |= group_mask;
  ++perf_stats_.stall_wait_barrier;
  return false;
}

// PTX §7.6.3 mbarrier.test_wait: non-blocking poll. Returns ready bit
// without suspending the warp.
bool Core::mbarrier_test_wait(uint64_t mbar_addr, uint32_t phase_token) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return true;
  }
  return (it->second.phase & 0x1u) != (phase_token & 0x1u);
}

// PTX §7.6.3 mbarrier.try_wait: bounded blocking wait. Behaves like wait()
// but returns false (timeout) after `1 << timeout_bucket` cycles. CModel
// approximation: behaves like wait() (no real timeout tracker), reports
// timeout=false on first poll if not ready, lets caller re-issue.
bool Core::mbarrier_try_wait(uint32_t wid, uint64_t mbar_addr,
                             uint32_t phase_token, uint32_t timeout_bucket) {
  (void)timeout_bucket;
  // For Phase-2 simx: equivalent to test_wait (non-blocking) -- the kernel
  // is responsible for re-issuing if the prior call returned not-ready.
  // Future work: schedule a wakeup at +(1 << timeout_bucket) cycles.
  (void)wid;
  return mbarrier_test_wait(mbar_addr, phase_token);
}

// ============================================================================
// Phase-2/3 Core API: cp.async.bulk.tensor / tcgen05.cp / tcgen05.{fence,commit,
// wait::ld, wait::st} thin wrappers.
//
// Phase-3.2 cpabulk_tensor_load/store follows the PTX-facing data plane:
//   - tensor_map_addr is a real DRAM pointer to a 128 B tensor_map_t
//   - args_lmem_ptr points at a 32 B cpabulk_transfer_args_t in LMEM
//   - cpbulk moves DRAM <-> LMEM; TMEM ingress is modeled separately by
//     tcgen05.cp from LMEM into TMEM.
// ============================================================================

// cp.async.bulk.tensor (DRAM -> shared) Phase-3.2 implementation.
uint32_t Core::cpabulk_tensor_load(uint32_t wid,
                                   uint64_t tensor_map_addr,
                                   uint64_t args_lmem_ptr,
                                   bool complete_tx) {
  if (!Tma::is_lmem_addr(args_lmem_ptr)) {
    throw std::runtime_error(
      "cp.async.bulk.tensor.load requires args_lmem_ptr to point to LMEM; "
      "raw window operands are not part of the PTX path");
  }

  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto result = tma_->cpabulk_tensor_load(tensor_map_addr,
                                          args_lmem_ptr,
                                          complete_tx);

  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmaLoad;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = 0;
  op.issue_cycle = perf_stats_.cycles;
  op.completed = true;        // direct copy completes synchronously for now
  op.payload_size_bytes = result.payload_size_bytes;
  op.tx_bound_mbar = result.tx_bound_mbar;
  op.tx_bytes = result.tx_bytes;
  async_tensor_ops_[async_id] = std::move(op);
  ++perf_stats_.tma_load_count;
  // Fire the completion hook now so mbarrier_complete_tx + waiter resume
  // both run (functional model; timing of the actual copy is approximated).
  auto& stored = async_tensor_ops_[async_id];
  on_async_tensor_op_completed(stored);
  return async_id;
}

// cp.async.bulk.tensor (shared -> DRAM) Phase-3.2 implementation.
uint32_t Core::cpabulk_tensor_store(uint32_t wid,
                                    uint64_t tensor_map_addr,
                                    uint64_t args_lmem_ptr) {
  if (!Tma::is_lmem_addr(args_lmem_ptr)) {
    throw std::runtime_error(
      "cp.async.bulk.tensor.store requires args_lmem_ptr to point to LMEM; "
      "raw window operands are not part of the PTX path");
  }

  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto result = tma_->cpabulk_tensor_store(tensor_map_addr, args_lmem_ptr);

  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmaStore;
  op.wid = wid;
  op.wgid = wgid;
  op.issue_cycle = perf_stats_.cycles;
  op.completed = true;
  op.payload_size_bytes = result.payload_size_bytes;
  async_tensor_ops_[async_id] = std::move(op);
  ++perf_stats_.tma_store_count;
  auto& stored = async_tensor_ops_[async_id];
  on_async_tensor_op_completed(stored);
  return async_id;
}

// Phase-3.3.1 GAP-4: TMEM byte-range R/W backing tcgen05.ld / tcgen05.st.
bool Core::tmem_region_read_bytes(uint32_t handle, uint32_t byte_offset, uint8_t* dst, uint32_t bytes) {
  return tmem_system_->handle_region_read_bytes(handle, byte_offset, dst, bytes);
}

bool Core::tmem_region_write_bytes(uint32_t handle, uint32_t byte_offset, const uint8_t* src, uint32_t bytes) {
  return tmem_system_->handle_region_write_bytes(handle, byte_offset, src, bytes);
}

bool Core::tmem_find_allocation_by_lane(uint32_t lane, uint32_t* col_base) const {
  return tmem_system_->find_allocation_by_lane(lane, col_base);
}

bool Core::tmem_cp(uint32_t wid,
	                   uint32_t taddr,
	                   uint64_t s_desc_lmem_ptr,
	                   uint32_t shape,
	                   uint32_t decompress) {
  advance_async_tensor_engine();
  // Read 64-bit s_desc from LMEM. PTX §9.7.16.2 shared-memory descriptor:
  //   bits[13:0]   start_address (in 16 B units, relative to shared-memory base)
  //   bits[16:14]  reserved
  //   bits[31:17]  leading dim stride, etc. (not used for the simple shape path)
  //   bits[53:32]  more layout fields
  //   bits[63:62]  swizzle mode
  uint64_t s_desc = 0;
  this->lmem_read(&s_desc, s_desc_lmem_ptr, sizeof(s_desc));
  uint32_t smem_offset = static_cast<uint32_t>(s_desc & 0x3FFFu) * 16u;
  uint64_t lmem_src = static_cast<uint64_t>(LMEM_BASE_ADDR) + smem_offset;

  // Per-shape byte count (PTX §9.7.16.5.3). All sizes follow rows × bits / 8.
  // Encoded values from TcuCpShape:
  //   0: 128x256b -> 128 * 32 = 4096 B
  //   1: 4x256b   ->   4 * 32 =  128 B
  //   2: 128x128b -> 128 * 16 = 2048 B
  //   3: 64x128b.warpx2  -> 64  * 16 = 1024 B
  //   4: 32x128b.warpx4  -> 32  * 16 =  512 B
  uint32_t bytes = 0;
  switch (shape) {
    case 0: bytes = 4096; break;
    case 1: bytes =  128; break;
    case 2: bytes = 2048; break;
    case 3: bytes = 1024; break;
    case 4: bytes =  512; break;
    default: bytes = 4096; break;
  }
  if (decompress != 0) {
    // GAP: decompression matrix (b6x16_p32 / b4x16_p64) not implemented yet;
    // the byte count would shrink for compressed sources. Continue with the
    // uncompressed footprint as a safe upper bound.
  }

  // Resolve PTX TADDR: [15:0]=lane base, [31:16]=column byte.
  uint32_t lane_base = taddr & 0xffffu;
  uint32_t col_byte  = (taddr >> 16) & 0xffffu;
  uint32_t col_base = 0;
  if (!tmem_find_allocation_by_lane(lane_base, &col_base)) {
    return false;
  }
  uint32_t col_span = 0;
  uint32_t size_bytes = 0;
  if (!tmem_query(col_base, &col_span, &size_bytes)) {
    return false;
  }
  if (lane_base < col_base) {
    return false;
  }
  uint32_t byte_offset = (lane_base - col_base) * Tmem::kColBytes + col_byte;
  if (byte_offset + bytes > size_bytes) {
    return false;
  }
  return launch_lmem_to_tmem_copy(wid,
                                  col_base,
                                  col_span,
                                  lmem_src,
                                  byte_offset,
                                  bytes) != 0;
}

// tcgen05.fence::{before,after}_thread_sync (PTX §9.7.16.5.1).
// Body inlined from former Core::tc_fence (now deleted).
bool Core::mbar_fence(uint32_t wid, TcuFenceMode mode) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  if (mode == TcuFenceMode::Before) {
    // In-order warp issue already preserves relative ordering before a
    // following thread sync.
    return true;
  }
  // AFTER acts as a completion fence for committed async tensor operations.
  if (!has_pending_async_ops(wid, true)) {
    return true;
  }
  auto mask = warpgroup_mask(wgid);
  for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
    if (!mask.test(gwid)) {
      continue;
    }
    auto& wait_state = fence_wait_states_.at(gwid);
    wait_state.active = true;
    wait_state.mode = mode;
  }
  ++perf_stats_.stall_wait_barrier;
  return false;
}

// tcgen05.commit (PTX §9.7.16.5.7): registers all currently inflight
// tcgen05.async ops in this warpgroup to the mbarrier at `mbar_addr`. Each
// registered op will trigger ONE mbarrier_arrive on completion (NOT a
// tx_count update). Body inlined from former Core::tc_commit (now deleted).
uint32_t Core::mbar_commit(uint32_t wid, uint64_t mbar_addr, uint32_t cta_mask) {
  (void)cta_mask; // Vortex single-cluster: cta_mask broadcast is no-op.
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return 0;
  }
  auto& barrier = it->second;

  uint32_t committed = 0;
  for (auto& entry : async_tensor_ops_) {
    auto& op = entry.second;
    if (op.wgid != wgid || op.completed || op.committed) {
      continue;
    }
    op.committed = true;
    // Reuse barrier_id slot in AsyncTensorOp to carry the LMEM mbar address
    // (the field is uint32; we cap to RV32 pointer width). On op completion,
    // on_async_tensor_op_completed() will call mbarrier_arrive on this address.
    op.barrier_id = static_cast<uint32_t>(mbar_addr);
    ++committed;
  }

  if (committed != 0) {
    barrier.expected_arrival_count += committed;
    barrier.pending_arrival_count  += committed;
    mirror_mbarrier_to_lmem(mbar_addr, barrier);
  }
  try_resume_fence_waiters();
  return committed;
}

bool Core::tcgen05_wait_ld(uint32_t wid) {
  return wait_for_inflight_tcgen05_ld_st(wid, false);
}

bool Core::tcgen05_wait_st(uint32_t wid) {
  return wait_for_inflight_tcgen05_ld_st(wid, true);
}

#endif
