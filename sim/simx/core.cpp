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

bool env_flag_enabled(const char* name, bool default_value) {
  auto value = std::getenv(name);
  if (nullptr == value || '\0' == value[0]) {
    return default_value;
  }
  return std::atoi(value) != 0;
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
  , mbarriers_(arch.num_barriers())
  , mbarrier_wait_targets_(arch.num_warps())
  , fence_wait_states_(arch.num_warps())
  , next_async_id_(1)
  , descriptor_tables_loaded_(false)
  , tmem_allocator_sealed_(false)
  , realistic_tma_load_(env_flag_enabled("VORTEX_SIMX_TMA_LOAD_REALISTIC", true))
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
  for (auto& bank : tmem_banks_) {
    bank.fill(0);
  }
  tmem_bank_allocs_.fill(false);
  tmem_allocations_.clear();
  tma_desc_table_.clear();
  mma_desc_table_.clear();
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
  descriptor_tables_loaded_ = false;
  tmem_allocator_sealed_ = false;
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

uint32_t Core::pack_tmem_handle(uint32_t start_bank, uint32_t bank_span) {
  return ((bank_span & 0xff) << 8) | (start_bank & 0xff);
}

bool Core::unpack_tmem_handle(uint32_t handle, uint32_t* start_bank, uint32_t* bank_span) {
  auto sb = handle & 0xff;
  auto span = (handle >> 8) & 0xff;
  if (span == 0 || sb >= kTmemNumBanks || (sb + span) > kTmemNumBanks) {
    return false;
  }
  *start_bank = sb;
  *bank_span = span;
  return true;
}

bool Core::lookup_tmem_allocation(uint32_t handle, TmemAllocation** allocation) {
  auto it = tmem_allocations_.find(handle);
  if (it == tmem_allocations_.end() || !it->second.valid) {
    return false;
  }
  if (allocation) {
    *allocation = &it->second;
  }
  return true;
}

bool Core::lookup_tmem_allocation(uint32_t handle, const TmemAllocation** allocation) const {
  auto it = tmem_allocations_.find(handle);
  if (it == tmem_allocations_.end() || !it->second.valid) {
    return false;
  }
  if (allocation) {
    *allocation = &it->second;
  }
  return true;
}

bool Core::ensure_descriptor_tables_loaded() {
  if (descriptor_tables_loaded_) {
    return true;
  }

  auto startup_arg = emulator_.startup_arg();
  if (0 == startup_arg) {
    return false;
  }

  vortex::tensor::descriptor_table_arg_t header;
  this->dcache_read(&header, startup_arg, sizeof(header));
  if (header.magic != vortex::tensor::descriptor_table_magic
   || header.version != vortex::tensor::descriptor_table_version) {
    return false;
  }

  tma_desc_table_.clear();
  mma_desc_table_.clear();
  if (header.tma_desc_count != 0) {
    tma_desc_table_.resize(header.tma_desc_count);
    this->dcache_read(tma_desc_table_.data(),
                      header.tma_desc_addr,
                      header.tma_desc_count * sizeof(TmaDescriptor));
  }
  if (header.mma_desc_count != 0) {
    mma_desc_table_.resize(header.mma_desc_count);
    this->dcache_read(mma_desc_table_.data(),
                      header.mma_desc_addr,
                      header.mma_desc_count * sizeof(MmaDescriptor));
  }
  descriptor_tables_loaded_ = true;
  return true;
}

bool Core::read_tma_descriptor(uint32_t desc_id, TmaDescriptor* out) {
  if (nullptr == out || !ensure_descriptor_tables_loaded() || desc_id >= tma_desc_table_.size()) {
    return false;
  }
  *out = tma_desc_table_.at(desc_id);
  return true;
}

bool Core::read_mma_descriptor(uint32_t desc_id, MmaDescriptor* out) {
  if (nullptr == out || !ensure_descriptor_tables_loaded() || desc_id >= mma_desc_table_.size()) {
    return false;
  }
  *out = mma_desc_table_.at(desc_id);
  return true;
}

uint32_t Core::estimate_tma_load_latency(const TmaDescriptor& desc, bool transpose_b) const {
  uint32_t matrix_bytes = desc.rows * desc.cols * desc.elem_bytes;
  uint32_t requested_size = desc.size_bytes ? desc.size_bytes : matrix_bytes;
  uint32_t size_bytes = requested_size ? requested_size : matrix_bytes;
  uint32_t transfer_cycles = std::max<uint32_t>(1, (size_bytes + kTmaLoadBytesPerCycle - 1) / kTmaLoadBytesPerCycle);
  uint32_t latency = kTmaLoadBaseLatency + transfer_cycles;
  if (transpose_b || ((desc.flags & 0x1) != 0)) {
    latency += kTmaTransposePenalty;
  }
  return latency;
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

bool Core::tmem_region_query(uint32_t start_bank, uint32_t bank_span, uint32_t* size_bytes) const {
  if (bank_span == 0 || start_bank >= kTmemNumBanks || (start_bank + bank_span) > kTmemNumBanks) {
    return false;
  }
  if (size_bytes) {
    *size_bytes = bank_span * tmem_banks_.front().size();
  }
  return true;
}

bool Core::tmem_query(uint32_t handle, uint32_t* bank_span, uint32_t* size_bytes) const {
  const TmemAllocation* allocation = nullptr;
  if (lookup_tmem_allocation(handle, &allocation)) {
    if (bank_span) {
      *bank_span = allocation->bank_span;
    }
    return tmem_region_query(allocation->start_bank, allocation->bank_span, size_bytes);
  }

  uint32_t start_bank = 0;
  uint32_t span = 0;
  if (!unpack_tmem_handle(handle, &start_bank, &span)) {
    return false;
  }
  if (bank_span) {
    *bank_span = span;
  }
  return tmem_region_query(start_bank, span, size_bytes);
}

bool Core::tmem_region_copy_in(uint32_t start_bank, uint32_t bank_span, const uint8_t* data, uint32_t size_bytes) {
  uint32_t capacity = 0;
  if (!tmem_region_query(start_bank, bank_span, &capacity)) {
    return false;
  }
  if (size_bytes > capacity) {
    return false;
  }

  uint32_t copied = 0;
  for (uint32_t bank = 0; bank < bank_span; ++bank) {
    auto& dst = tmem_banks_.at(start_bank + bank);
    uint32_t chunk = std::min<uint32_t>(dst.size(), size_bytes - copied);
    std::copy_n(data + copied, chunk, dst.begin());
    if (chunk < dst.size()) {
      std::fill(dst.begin() + chunk, dst.end(), 0);
    }
    copied += chunk;
    if (copied >= size_bytes) {
      break;
    }
  }
  return true;
}

bool Core::tmem_region_copy_out(uint32_t start_bank, uint32_t bank_span, uint8_t* data, uint32_t size_bytes) const {
  uint32_t capacity = 0;
  if (!tmem_region_query(start_bank, bank_span, &capacity)) {
    return false;
  }
  if (size_bytes > capacity) {
    return false;
  }
  uint32_t copied = 0;
  for (uint32_t bank = 0; bank < bank_span; ++bank) {
    const auto& src = tmem_banks_.at(start_bank + bank);
    uint32_t chunk = std::min<uint32_t>(src.size(), size_bytes - copied);
    std::copy_n(src.begin(), chunk, data + copied);
    copied += chunk;
    if (copied >= size_bytes) {
      break;
    }
  }
  return true;
}

bool Core::tmem_copy_in(uint32_t handle, const uint8_t* data, uint32_t size_bytes) {
  const TmemAllocation* allocation = nullptr;
  if (lookup_tmem_allocation(handle, &allocation)) {
    return tmem_region_copy_in(allocation->start_bank, allocation->bank_span, data, size_bytes);
  }

  uint32_t start_bank = 0;
  uint32_t span = 0;
  if (!unpack_tmem_handle(handle, &start_bank, &span)) {
    return false;
  }
  return tmem_region_copy_in(start_bank, span, data, size_bytes);
}

bool Core::tmem_copy_out(uint32_t handle, uint8_t* data, uint32_t size_bytes) const {
  const TmemAllocation* allocation = nullptr;
  if (lookup_tmem_allocation(handle, &allocation)) {
    return tmem_region_copy_out(allocation->start_bank, allocation->bank_span, data, size_bytes);
  }

  uint32_t start_bank = 0;
  uint32_t span = 0;
  if (!unpack_tmem_handle(handle, &start_bank, &span)) {
    return false;
  }
  return tmem_region_copy_out(start_bank, span, data, size_bytes);
}

bool Core::tmem_region_shift_down(uint32_t start_bank, uint32_t bank_span, uint32_t row_bytes) {
  uint32_t size_bytes = 0;
  if (!tmem_region_query(start_bank, bank_span, &size_bytes)) {
    return false;
  }
  if (row_bytes == 0 || row_bytes > size_bytes) {
    return false;
  }

  std::vector<uint8_t> tile(size_bytes, 0);
  if (!tmem_region_copy_out(start_bank, bank_span, tile.data(), size_bytes)) {
    return false;
  }

  std::copy_backward(tile.begin(), tile.end() - row_bytes, tile.end());
  std::fill(tile.begin(), tile.begin() + row_bytes, 0);
  return tmem_region_copy_in(start_bank, bank_span, tile.data(), size_bytes);
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
  switch (op.type) {
  case AsyncTensorOpType::TmaLoad:
  case AsyncTensorOpType::TmaStore: {
    TmaDescriptor desc;
    if (!read_tma_descriptor(op.descriptor_id, &desc)) {
      op.completed = true;
      finalize_async_tensor_op(op);
      return;
    }
    auto kind = static_cast<TcuPayloadKind>(desc.payload_kind);
    bool use_meta_region = (kind == TcuPayloadKind::SparseMeta);
    uint32_t start_bank = use_meta_region ? desc.meta_tmem_base : desc.tmem_base;
    uint32_t bank_span = use_meta_region ? desc.meta_bank_span : desc.bank_span;
    if (bank_span == 0) {
      const TmemAllocation* allocation = nullptr;
      if (!lookup_tmem_allocation(op.handle, &allocation)) {
        op.completed = true;
        finalize_async_tensor_op(op);
        return;
      }
      start_bank = allocation->start_bank;
      bank_span = allocation->bank_span;
    }

    uint32_t capacity = 0;
    if (!tmem_region_query(start_bank, bank_span, &capacity)) {
      op.completed = true;
      finalize_async_tensor_op(op);
      return;
    }

    uint32_t matrix_bytes = desc.rows * desc.cols * desc.elem_bytes;
    uint32_t requested_size = desc.size_bytes ? desc.size_bytes : matrix_bytes;
    uint32_t size_bytes = std::min<uint32_t>(requested_size ? requested_size : capacity, capacity);
    std::vector<uint8_t> buffer(size_bytes, 0);
    bool transpose_b = op.transpose_b || ((desc.flags & 0x1) != 0);

    if (op.type == AsyncTensorOpType::TmaLoad) {
      if (desc.rows > 0 && desc.cols > 0 && desc.elem_bytes > 0 && transpose_b) {
        std::vector<uint8_t> src(matrix_bytes, 0);
        this->dcache_read(src.data(), desc.addr, matrix_bytes);
        for (uint32_t r = 0; r < desc.rows; ++r) {
          for (uint32_t c = 0; c < desc.cols; ++c) {
            auto src_off = (r * desc.cols + c) * desc.elem_bytes;
            auto dst_off = (c * desc.rows + r) * desc.elem_bytes;
            if (dst_off + desc.elem_bytes <= buffer.size() && src_off + desc.elem_bytes <= src.size()) {
              std::copy_n(src.data() + src_off, desc.elem_bytes, buffer.data() + dst_off);
            }
          }
        }
      } else {
        this->dcache_read(buffer.data(), desc.addr, size_bytes);
      }
      (void)tmem_region_copy_in(start_bank, bank_span, buffer.data(), size_bytes);
    } else {
      if (tmem_region_copy_out(start_bank, bank_span, buffer.data(), size_bytes)) {
        this->dcache_write(buffer.data(), desc.addr, size_bytes);
      }
    }

    if (!use_meta_region) {
      TmemAllocation* allocation = nullptr;
      if (lookup_tmem_allocation(op.handle, &allocation)) {
        uint32_t row_bytes = desc.stride_bytes ? desc.stride_bytes : (desc.cols * desc.elem_bytes);
        if (row_bytes != 0) {
          allocation->row_bytes = std::min<uint32_t>(row_bytes, allocation->bank_span * kTmemBankSize);
        }
      }
    }
  } break;
  case AsyncTensorOpType::TmemShift: {
    const TmemAllocation* allocation = nullptr;
    if (!lookup_tmem_allocation(op.handle, &allocation)) {
      op.completed = true;
      finalize_async_tensor_op(op);
      return;
    }
    (void)tmem_region_shift_down(allocation->start_bank, allocation->bank_span, allocation->row_bytes);
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
  for (auto& entry : async_tensor_ops_) {
    auto& op = entry.second;
    if (!op.completed && perf_stats_.cycles >= op.ready_cycle) {
      process_async_tensor_op(op);
    }
  }
}

uint32_t Core::tmem_alloc(uint32_t bank_span) {
  advance_async_tensor_ops();
  if (tmem_allocator_sealed_ || bank_span == 0 || bank_span > tmem_banks_.size()) {
    return 0;
  }
  for (uint32_t start = 0; start + bank_span <= tmem_banks_.size(); ++start) {
    bool available = true;
    for (uint32_t i = 0; i < bank_span; ++i) {
      if (tmem_bank_allocs_.at(start + i)) {
        available = false;
        break;
      }
    }
    if (!available) {
      continue;
    }
    for (uint32_t i = 0; i < bank_span; ++i) {
      tmem_bank_allocs_.at(start + i) = true;
      tmem_banks_.at(start + i).fill(0);
    }
    auto handle = pack_tmem_handle(start, bank_span);
    tmem_allocations_[handle] = TmemAllocation{true, start, bank_span, 64};
    return handle;
  }
  return 0;
}

bool Core::tmem_free(uint32_t handle) {
  advance_async_tensor_ops();
  uint32_t start_bank = 0;
  uint32_t bank_span = 0;
  if (!unpack_tmem_handle(handle, &start_bank, &bank_span)) {
    return false;
  }
  for (uint32_t i = 0; i < bank_span; ++i) {
    tmem_bank_allocs_.at(start_bank + i) = false;
    tmem_banks_.at(start_bank + i).fill(0);
  }
  tmem_allocations_.erase(handle);
  return true;
}

void Core::tmem_rel_permit() {
  tmem_allocator_sealed_ = true;
}

uint32_t Core::tma_load(uint32_t wid, uint32_t handle, uint32_t desc_id, bool transpose_b) {
  advance_async_tensor_ops();
  TmaDescriptor desc;
  if (!read_tma_descriptor(desc_id, &desc)) {
    return 0;
  }
  uint32_t latency = realistic_tma_load_
                   ? estimate_tma_load_latency(desc, transpose_b)
                   : 1;
  auto async_id = next_async_id_++;
  async_tensor_ops_[async_id] = AsyncTensorOp{
    async_id,
    AsyncTensorOpType::TmaLoad,
    wid,
    handle,
    desc_id,
    perf_stats_.cycles + latency,
    transpose_b,
    false,
    false,
    0
  };
  return async_id;
}

uint32_t Core::tma_store(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_ops();
  TmaDescriptor desc;
  if (!read_tma_descriptor(desc_id, &desc)) {
    return 0;
  }
  auto async_id = next_async_id_++;
  async_tensor_ops_[async_id] = AsyncTensorOp{
    async_id,
    AsyncTensorOpType::TmaStore,
    wid,
    handle,
    desc_id,
    perf_stats_.cycles + 1,
    false,
    false,
    false,
    0
  };
  return async_id;
}

uint32_t Core::tmem_shift(uint32_t wid, uint32_t handle) {
  advance_async_tensor_ops();
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return 0;
  }
  __unused(allocation);
  auto async_id = next_async_id_++;
  async_tensor_ops_[async_id] = AsyncTensorOp{
    async_id,
    AsyncTensorOpType::TmemShift,
    wid,
    handle,
    0,
    perf_stats_.cycles + 1,
    false,
    false,
    false,
    0
  };
  return async_id;
}

uint32_t Core::mma_load_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_ops();
  auto async_id = next_async_id_++;
  async_tensor_ops_[async_id] = AsyncTensorOp{
    async_id,
    AsyncTensorOpType::MmaLoad,
    wid,
    handle,
    desc_id,
    std::numeric_limits<uint64_t>::max(),
    false,
    false,
    false,
    0
  };
  return async_id;
}

uint32_t Core::mma_store_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_ops();
  auto async_id = next_async_id_++;
  async_tensor_ops_[async_id] = AsyncTensorOp{
    async_id,
    AsyncTensorOpType::MmaStore,
    wid,
    handle,
    desc_id,
    std::numeric_limits<uint64_t>::max(),
    false,
    false,
    false,
    0
  };
  return async_id;
}

uint32_t Core::wmma_async_issue(uint32_t wid) {
  advance_async_tensor_ops();
  auto async_id = next_async_id_++;
  async_tensor_ops_[async_id] = AsyncTensorOp{
    async_id,
    AsyncTensorOpType::Wmma,
    wid,
    0,
    0,
    std::numeric_limits<uint64_t>::max(),
    false,
    false,
    false,
    0
  };
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
    return false;
  }

  if (barrier.phase_done) {
    return true;
  }

  wait_targets[barrier_id] = barrier.phase + 1;
  barrier.waiters_bitmap.set(wid);
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
    return false;
  }
  async_tensor_waiters_.erase(async_id);
  async_tensor_ops_.erase(it);
  return true;
}

bool Core::tmem_read_packet(uint32_t handle, uint32_t packet_idx, TmemPacket* out) {
  advance_async_tensor_ops();
  if (nullptr == out) {
    return false;
  }
  uint32_t size_bytes = 0;
  if (!tmem_query(handle, nullptr, &size_bytes)) {
    return false;
  }
  uint32_t offset = packet_idx * out->bytes.size();
  if (offset + out->bytes.size() > size_bytes) {
    return false;
  }
  std::vector<uint8_t> tile(size_bytes, 0);
  if (!tmem_copy_out(handle, tile.data(), size_bytes)) {
    return false;
  }
  std::copy_n(tile.data() + offset, out->bytes.size(), out->bytes.begin());
  return true;
}

bool Core::tmem_write_packet(uint32_t handle, uint32_t packet_idx, const TmemPacket& in) {
  advance_async_tensor_ops();
  uint32_t size_bytes = 0;
  if (!tmem_query(handle, nullptr, &size_bytes)) {
    return false;
  }
  uint32_t offset = packet_idx * in.bytes.size();
  if (offset + in.bytes.size() > size_bytes) {
    return false;
  }
  std::vector<uint8_t> tile(size_bytes, 0);
  if (!tmem_copy_out(handle, tile.data(), size_bytes)) {
    return false;
  }
  std::copy_n(in.bytes.begin(), in.bytes.size(), tile.begin() + offset);
  return tmem_copy_in(handle, tile.data(), size_bytes);
}

#endif
