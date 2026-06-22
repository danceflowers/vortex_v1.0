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
#include "tmem_module.h"
#include "tma_module.h"

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
  , tensor_async_op_completion_in_(this, 4)
  , mma_completion_in_(this, 4)
  , socket_(socket)
  , arch_(arch)
//#ifdef EXT_TCU_ENABLE
// #endif
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
//#ifdef EXT_TCU_ENABLE
  , next_async_id_(1)
// #endif
{
  char sname[100];

//#ifdef EXT_TCU_ENABLE
  // Old TMA/TmemSystem kept for TCU_LD/ST path only (no OTC MMA data goes here).
  // Per-warp mbarrier/fence bookkeeping is indexed by wid via .at(); size it to
  // num_warps so mbarrier_wait()/mbar_fence() don't index an empty vector.
  mbarrier_wait_targets_.resize(arch_.num_warps());
  fence_wait_states_.resize(arch_.num_warps());
//#endif

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

  // create lmem arbiter (NUM_LSU_BLOCKS LSU blocks + 2 OTC LMEM ports + 1 TMA
  // descriptor-read port). The TMA's transfer-args read is wired only on the
  // socket's owner core (core_idx 0) in bind_tensor_ports().
  snprintf(sname, 100, "%s-lmem_arb", this->name().c_str());
  lmem_arb_ = LsuArbiter::Create(sname, ArbiterType::RoundRobin,
                                 NUM_LSU_BLOCKS + 3, 1);

  // create lmem adapter
  snprintf(sname, 100, "%s-lsu_lmem_adapter", this->name().c_str());
  auto lsu_lmem_adapter = LsuMemAdapter::Create(sname, LSU_CHANNELS, 1);

  // connect lmem switch
  for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
    lmem_switch_.at(b)->ReqDC.bind(&mem_coalescers_.at(b)->ReqIn);
    lmem_switch_.at(b)->ReqLmem.bind(&lmem_arb_->ReqIn.at(b));

    mem_coalescers_.at(b)->RspIn.bind(&lmem_switch_.at(b)->RspDC);
    lmem_arb_->RspIn.at(b).bind(&lmem_switch_.at(b)->RspLmem);
  }

  // connect lmem arbiter
  lmem_arb_->ReqOut.at(0).bind(&lsu_lmem_adapter->ReqIn);
  lsu_lmem_adapter->RspIn.bind(&lmem_arb_->RspOut.at(0));

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
//#ifdef EXT_TCU_ENABLE
  // TMEM/TMA ops are warp-uniform (collective), like TCU_WMMA: dispatch the warp
  // once (num_lanes == NUM_THREADS) instead of once per thread, otherwise each op
  // is replicated NUM_THREADS times into the TMEM/TMA module.
  dispatchers_.at((int)FUType::TCU) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_TCU_BLOCKS, NUM_TCU_LANES);
  dispatchers_.at((int)FUType::TMEM) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_TCU_BLOCKS, NUM_TCU_LANES);
  dispatchers_.at((int)FUType::TMA) = SimPlatform::instance().create_object<Dispatcher>(this, 2, NUM_TCU_BLOCKS, NUM_TCU_LANES);
//#endif

  // initialize execute units
  func_units_.at((int)FUType::ALU) = SimPlatform::instance().create_object<AluUnit>(this);
  func_units_.at((int)FUType::FPU) = SimPlatform::instance().create_object<FpuUnit>(this);
  func_units_.at((int)FUType::LSU) = SimPlatform::instance().create_object<LsuUnit>(this);
  func_units_.at((int)FUType::SFU) = SimPlatform::instance().create_object<SfuUnit>(this);
#ifdef EXT_V_ENABLE
  func_units_.at((int)FUType::VPU) = SimPlatform::instance().create_object<VpuUnit>(this);
#endif
//#ifdef EXT_TCU_ENABLE
  func_units_.at((int)FUType::TCU) = SimPlatform::instance().create_object<TcuUnit>(this);
  func_units_.at((int)FUType::TMEM) = SimPlatform::instance().create_object<TmemUnit>(this);
  func_units_.at((int)FUType::TMA) = SimPlatform::instance().create_object<TmaUnit>(this);
//#endif

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

//#ifdef EXT_TCU_ENABLE
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
  last_tensor_completion_drain_cycle_ = std::numeric_limits<uint64_t>::max();
  last_tcgen05_ldst_advance_cycle_ = std::numeric_limits<uint64_t>::max();
//#endif

  perf_stats_ = PerfStats();
}

void Core::tick() {
//#ifdef EXT_TCU_ENABLE
  drain_tensor_execute_completion_notices();
  advance_tcgen05_ldst_async_ops();
  // OTC/TMEM/TMA ticked by TensorSocket, not here.
// #endif
  this->commit();
  this->execute();
  this->issue();
  this->decode();
  this->fetch();
  this->schedule();
//#ifdef EXT_TCU_ENABLE
  auto watchdog_cycle = env_u64_value("VORTEX_SIMX_TENSOR_WATCHDOG_CYCLES", 0);
  if (watchdog_cycle != 0 && perf_stats_.cycles >= watchdog_cycle) {
    this->dump_tensor_debug_state(std::cerr);
    std::abort();
  }
//#endif

  ++perf_stats_.cycles;
  DPN(2, std::flush);
}

void Core::publish_visible_tensor_mem_state() {
}

uint64_t Core::startup_arg() const {
  return emulator_.startup_arg();
}

void Core::dump_tensor_debug_state(std::ostream& os) const {
//#ifdef EXT_TCU_ENABLE
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
       << " taddr=" << op.taddr
       << " completed=" << op.completed
       << " committed=" << op.committed
       << " barrier=" << op.barrier_id
       << " tx_bound_mbar=0x" << std::hex << op.tx_bound_mbar << std::dec
       << " tx_bytes=" << op.tx_bytes
       << "\n";
  }
  os << "mbarriers=" << mbarriers_.size() << "\n";
  for (const auto& entry : mbarriers_) {
    const auto& mbar = entry.second;
    os << "  mbar_addr=0x" << std::hex << entry.first << std::dec
       << " valid=" << mbar.valid
       << " phase=" << mbar.phase
       << " expected_arrivals=" << mbar.expected_arrival_count
       << " pending_arrivals=" << mbar.pending_arrival_count
       << " expected_tx=" << mbar.expected_tx_count
       << " pending_tx=" << mbar.pending_tx_count
       << " waiters=" << mbar.waiters_bitmap.to_string()
       << "\n";
  }
  os << "==== tensor-debug-state end ====\n";
//#else
//  (void)os;
//#endif
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
        //#ifdef EXT_TCU_ENABLE
          case FUType::TCU:
          case FUType::TMEM:
          case FUType::TMA: ++perf_stats_.scrb_tcu; break;
        //#endif
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

//#ifdef EXT_TCU_ENABLE
    if (auto tcu_type = std::get_if<TcuType>(&trace->op_type)) {
      if (*tcu_type == TcuType::TCU_LD
       && !tcgen05_ld_trace_ready(reinterpret_cast<uint64_t>(trace))) {
        continue;
      }
    }
//#endif

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

void Core::bind_tensor_ports(TensorSocket* ts, uint32_t core_idx) {
  tensor_socket_   = ts;
  tensor_core_idx_ = core_idx;
  auto& otc = ts->tensor_core(core_idx);
  open_tensor_core_ = otc;  // for backward compat

  // Re-bind TcuUnit to this core's OpenTensorCore. The func unit Inputs were
  // self-bound (Inputs→Outputs pass-through) in the unit constructor; tear that
  // down before re-binding, otherwise the second bind() silently overwrites the
  // link (asserts under DEBUG; relies on NDEBUG in release).
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    func_units_.at((int)FUType::TCU)->Inputs.at(iw).unbind();
    func_units_.at((int)FUType::TCU)->Inputs.at(iw).bind(&otc->Inputs.at(iw));
    otc->Outputs.at(iw).bind(&func_units_.at((int)FUType::TCU)->Outputs.at(iw));
  }

  // Bind TmemUnit → TMEM's instruction ports.
  auto* tmem = ts->tmem();
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    func_units_.at((int)FUType::TMEM)->Inputs.at(iw).unbind();
    func_units_.at((int)FUType::TMEM)->Inputs.at(iw).bind(&tmem->Inputs.at(iw));
    tmem->Outputs.at(iw).bind(&func_units_.at((int)FUType::TMEM)->Outputs.at(iw));
  }

  // Bind TmaUnit → TMA's instruction ports.
  auto* tma = ts->tma();
  if (tma) {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      func_units_.at((int)FUType::TMA)->Inputs.at(iw).unbind();
      func_units_.at((int)FUType::TMA)->Inputs.at(iw).bind(&tma->Inputs.at(iw));
      tma->Outputs.at(iw).bind(&func_units_.at((int)FUType::TMA)->Outputs.at(iw));
    }
  }

  // Bind OTC LMEM read ports into the core's lmem_arb.
  //   ReqIn[NUM_LSU_BLOCKS + 0] → Stage2a B-matrix reads
  //   ReqIn[NUM_LSU_BLOCKS + 1] → Stage1 operand_block_t reads
  otc->LmemReadReq.bind(&lmem_arb_->ReqIn.at(NUM_LSU_BLOCKS));
  lmem_arb_->RspIn.at(NUM_LSU_BLOCKS).bind(&otc->LmemReadRsp);
  otc->DecodeLmemReadReq.bind(&lmem_arb_->ReqIn.at(NUM_LSU_BLOCKS + 1));
  lmem_arb_->RspIn.at(NUM_LSU_BLOCKS + 1).bind(&otc->DecodeLmemReadRsp);

  // MMA completion → this core's mbarrier engine (per-id arrive on tcgen05.commit).
  otc->AsyncCompletionOut.bind(&mma_completion_in_);

  // The TMA is shared by all cores in the socket but reads transfer-args from the
  // owner core's (core_idx 0) LMEM and reports complete_tx there. Wire its
  // descriptor-read path and completion port only on the owner core.
  // NOTE: single-socket-owner routing — multi-core complete_tx would need a
  // core index carried in TensorAsyncOpCompletion (future work).
  if (core_idx == 0 && tma) {
    tma->LmemDescReq.bind(&lmem_arb_->ReqIn.at(NUM_LSU_BLOCKS + 2));
    lmem_arb_->RspIn.at(NUM_LSU_BLOCKS + 2).bind(&tma->LmemDescRsp);
    tma->AsyncCompletionOut.bind(&tensor_async_op_completion_in_);
  }

  DT(2, "Core" << core_id_ << ": bound to TensorSocket tc_idx=" << core_idx);
}

bool Core::tmem_taddr_read_bytes(uint32_t taddr, uint32_t byte_offset,
                                 uint8_t* dst, uint32_t bytes) {
  if (tensor_socket_ == nullptr) return false;
  auto* tmem = tensor_socket_->tmem();
  if (tmem == nullptr) return false;
  return tmem->read_bytes(taddr, byte_offset, dst, bytes);
}

bool Core::tmem_taddr_write_bytes(uint32_t taddr, uint32_t byte_offset,
                                  const uint8_t* src, uint32_t bytes) {
  if (tensor_socket_ == nullptr) return false;
  auto* tmem = tensor_socket_->tmem();
  if (tmem == nullptr) return false;
  return tmem->write_bytes(taddr, byte_offset, src, bytes);
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

//#ifdef EXT_TCU_ENABLE

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

// ============================================================================
// LMEM mirror helper: encode the in-Core mbarrier_runtime_state_t into the 8 B
// mbarrier_state_t bit layout and write it through to the kernel-visible
// LMEM location at mbar_addr. Called after every state transition.
// PTX §7.6.1 mbarrier object format: 64-bit packed value
//   bits[63:62]   phase parity (2-bit)
//   bits[61:32]   pending_arrival_count (30 bits)
//   bits[31:20]   pending_tx_count      (12 bits, byte units per §7.6.4)
//   bits[19:0]    expected_arrival_count (20 bits)
// ============================================================================
void Core::mirror_mbarrier_to_lmem(uint64_t mbar_addr, const mbarrier_runtime_state_t& state) {
  uint64_t encoded = 0;
  encoded |= (uint64_t)(state.expected_arrival_count & 0xFFFFFu) << 0;
  encoded |= (uint64_t)(state.pending_tx_count       & 0xFFFu)   << 20;
  encoded |= (uint64_t)(state.pending_arrival_count  & 0x3FFFFFFFu) << 32;
  encoded |= (uint64_t)(state.phase                  & 0x3u)     << 62;
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
  auto& mbar = it->second;
  if (!mbar.valid
   || mbar.pending_arrival_count != 0
   || mbar.pending_tx_count < mbar.expected_tx_count) {
    return;
  }

  // Phase advance.
  mbar.phase = (mbar.phase + 1) & 0x3u;
  mbar.pending_arrival_count = mbar.expected_arrival_count;
  mbar.pending_tx_count = 0;
  mbar.expected_tx_count = 0;
  mirror_mbarrier_to_lmem(mbar_addr, mbar);

  // Wake up warps that were waiting on the prior phase.
  auto waiters = mbar.waiters_bitmap;
  mbar.waiters_bitmap.reset();
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
  // completion. NOT a tx_count update -- those go via mmbar.complete_tx.
  // op.barrier_id was reinterpreted in tc_commit() as the LMEM mbar address.
  uint64_t mbar_addr = op.barrier_id;
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  auto& mbar = it->second;
  if (mbar.pending_arrival_count > 0) {
    --mbar.pending_arrival_count;
  }
  mirror_mbarrier_to_lmem(mbar_addr, mbar);
  try_complete_mbarrier(mbar_addr);
}

// tcgen05.mma op registration at issue time. Each MMA gets a fresh async_id,
// tracked in async_tensor_ops_ (uncommitted/incomplete). tcgen05.commit later
// binds it to an mbarrier; OpenTensorCore echoes the async_id on completion so
// the drain below can match it per-id and trigger one mbarrier_arrive.
uint32_t Core::alloc_mma_async_op(uint32_t wid) {
  uint32_t aid = next_async_id_++;
  AsyncTensorOp op;
  op.async_id    = aid;
  op.type        = AsyncTensorOpType::Wmma;
  op.wid         = wid;
  op.issue_cycle = perf_stats_.cycles;
  async_tensor_ops_[aid] = op;
  return aid;
}

// Drain completion notices from the new tensor modules and feed them into the
// (per-async_id) mbarrier engine. At most one completion is dequeued per port
// per cycle (SimPort allows a single pop per port per tick). This is called from
// both tick() and the execute-stage mbarrier helpers, so a per-cycle guard
// prevents popping the same port twice within one cycle.
void Core::drain_tensor_execute_completion_notices() {
  auto cycle = SimPlatform::instance().cycles();
  if (last_tensor_completion_drain_cycle_ == cycle) {
    return;
  }
  last_tensor_completion_drain_cycle_ = cycle;

  // TMA: complete_tx bytes (PTX §7.6.4). Keyed by mbar address, order-independent.
  if (!tensor_async_op_completion_in_.empty()) {
    auto c = tensor_async_op_completion_in_.front();
    tensor_async_op_completion_in_.pop();
    if (c.tx_bound_mbar != 0 && c.tx_bytes != 0) {
      mbarrier_complete_tx(c.tx_bound_mbar, c.tx_bytes);
    }
  }
  // MMA: match each completion to its async_tensor_ops_ entry by async_id. If the
  // op was bound by tcgen05.commit, on_async_tensor_op_completed() issues one
  // mbarrier_arrive. Per-id matching makes out-of-order completion correct; an op
  // that completes before its commit is simply marked done and never bound.
  if (!mma_completion_in_.empty()) {
    auto c = mma_completion_in_.front();
    mma_completion_in_.pop();
    auto it = async_tensor_ops_.find(c.async_id);
    if (it != async_tensor_ops_.end()) {
      it->second.completed = true;
      on_async_tensor_op_completed(it->second);
      async_tensor_ops_.erase(it);
    }
  }
}

void Core::advance_async_tensor_engine() {
  drain_tensor_execute_completion_notices();
  try_resume_fence_waiters();
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
  // tcgen05 ld/st async tracking is handled by the OpenTensorCore LdstStage;
  // the legacy pending-ops map is never populated under the new architecture.
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

// PTX §7.6.3 mmbar.init: initialize the 8 B mbarrier object in shared mem
// with `count` expected arrivals; phase parity starts at 0; tx counters at 0.
bool Core::mbarrier_init(uint64_t mbar_addr, uint32_t count) {
  auto& mbar = mbarriers_[mbar_addr];
  mbar = {};
  mbar.valid = true;
  mbar.expected_arrival_count = count;
  mbar.pending_arrival_count = count;
  mbar.phase = 0;
  // Drop any stale waiter targets for this address from prior generations.
  for (auto& wait_targets : mbarrier_wait_targets_) {
    wait_targets.erase(mbar_addr);
  }
  mirror_mbarrier_to_lmem(mbar_addr, mbar);
  return true;
}

// PTX §7.6.3 mmbar.invalidate: tear down the object.
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

// PTX §7.6.3 mmbar.arrive: decrement pending_arrival_count by `decrement`.
// Returns the *current* phase token so caller can wait on this phase.
uint32_t Core::mbarrier_arrive(uint64_t mbar_addr, uint32_t decrement_count) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return 0;
  }
  auto& mbar = it->second;
  uint32_t observed_phase = mbar.phase;
  if (mbar.pending_arrival_count >= decrement_count) {
    mbar.pending_arrival_count -= decrement_count;
  } else {
    mbar.pending_arrival_count = 0;
  }
  mirror_mbarrier_to_lmem(mbar_addr, mbar);
  try_complete_mbarrier(mbar_addr);
  return observed_phase;
}

// PTX §7.6.3 mmbar.arrive_drop: arrive AND decrement expected_arrival_count
// permanently (participant leaves the barrier).
void Core::mbarrier_arrive_drop(uint64_t mbar_addr) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  auto& mbar = it->second;
  if (mbar.expected_arrival_count > 0) {
    --mbar.expected_arrival_count;
  }
  if (mbar.pending_arrival_count > 0) {
    --mbar.pending_arrival_count;
  }
  mirror_mbarrier_to_lmem(mbar_addr, mbar);
  try_complete_mbarrier(mbar_addr);
}

// PTX §7.6.4 mmbar.expect_tx: increase expected tx-byte target.
void Core::mbarrier_expect_tx(uint64_t mbar_addr, uint32_t tx_bytes) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return;
  }
  it->second.expected_tx_count += tx_bytes;
  mirror_mbarrier_to_lmem(mbar_addr, it->second);
}

// PTX §7.6.4 mmbar.complete_tx: contribute completed tx bytes; if combined
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

// PTX §7.6.3 mmbar.wait.parity: blocking wait until current phase parity
// differs from `phase_token`. Returns false to indicate stall (caller suspends
// the warp); returns true once the phase has advanced.
bool Core::mbarrier_wait(uint32_t wid, uint64_t mbar_addr, uint32_t phase_token) {
  advance_async_tensor_engine();
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return true;
  }
  auto& mbar = it->second;
  // Phase parity comparison (PTX semantics): if current parity bit differs
  // from input token, the wait is satisfied.
  if ((mbar.phase & 0x1u) != (phase_token & 0x1u)) {
    mbarrier_wait_targets_.at(wid).erase(mbar_addr);
    return true;
  }
  mbarrier_wait_targets_.at(wid)[mbar_addr] = phase_token;
  mbar.waiters_bitmap.set(wid);
  ++perf_stats_.stall_wait_barrier;
  return false;
}

// PTX §7.6.3 mmbar.test_wait: non-blocking poll. Returns ready bit
// without suspending the warp.
bool Core::mbarrier_test_wait(uint64_t mbar_addr, uint32_t phase_token) {
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return true;
  }
  return (it->second.phase & 0x1u) != (phase_token & 0x1u);
}

// PTX §7.6.3 mmbar.try_wait: bounded blocking wait. Behaves like wait()
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
// cp.async.bulk.tensor (shared -> DRAM) Phase-3.2 implementation.
// Phase-3.3.1 GAP-4: TMEM byte-range R/W backing tcgen05.ld / tcgen05.st.
// tcgen05.fence::{before,after}_thread_sync (PTX §9.7.16.5.1).
// Body inlined from former Core::tc_fence (now deleted).
bool Core::mbar_fence(uint32_t wid, TcuFenceMode mode) {
  advance_async_tensor_engine();
  if (mode == TcuFenceMode::Before) {
    // In-order warp issue already preserves relative ordering before a
    // following thread sync.
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

// tcgen05.commit (PTX §9.7.16.5.7): registers this warp's currently inflight
// tcgen05.async ops to the mbarrier at `mbar_addr`. Each registered op will
// trigger ONE mbarrier_arrive on completion (NOT a tx_count update).
uint32_t Core::mbar_commit(uint32_t wid, uint64_t mbar_addr, uint32_t cta_mask) {
  (void)cta_mask; // Vortex single-cluster: cta_mask broadcast is no-op.
  advance_async_tensor_engine();
  auto it = mbarriers_.find(mbar_addr);
  if (it == mbarriers_.end() || !it->second.valid) {
    return 0;
  }
  auto& mbar = it->second;

  uint32_t committed = 0;
  for (auto& entry : async_tensor_ops_) {
    auto& op = entry.second;
    if (op.wid != wid || op.completed || op.committed) {
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
    mbar.expected_arrival_count += committed;
    mbar.pending_arrival_count  += committed;
    mirror_mbarrier_to_lmem(mbar_addr, mbar);
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

//#endif
