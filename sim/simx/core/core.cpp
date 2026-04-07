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

// Core.cpp owns the front half of the tensor path:
// - descriptor interpretation and default window planning
// - Core-side asynchronous tensor transactions (TMA/TMEM_SHIFT)
// - the packet adapter between mathematical row-major windows and TMEM
//   window packets
//
// TensorUnit later consumes TMEM-visible state and advances the local operand
// memories and TensorCore pipeline.

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

uint64_t env_u64_value(const char* name, uint64_t default_value) {
  auto value = std::getenv(name);
  if (nullptr == value || '\0' == value[0]) {
    return default_value;
  }
  return std::strtoull(value, nullptr, 0);
}

TmemWindowTarget map_window_target(const TmaDescriptor& desc) {
  if (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta) {
    return TmemWindowTarget::Meta;
  }
  switch (desc.tile_role) {
  case 1:
    return TmemWindowTarget::A;
  case 2:
    return TmemWindowTarget::B;
  case 3:
    return TmemWindowTarget::C;
  case 4:
    return TmemWindowTarget::D;
  default:
    return TmemWindowTarget::A;
  }
}

bool uses_generic_math_window(const TmaDescriptor& desc) {
  return desc.tile_role == 0
      && static_cast<TcuPayloadKind>(desc.payload_kind) != TcuPayloadKind::SparseMeta;
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

bool build_generic_math_window(uint32_t window_id,
                               const TmaDescriptor& desc,
                               TmemWindowPlan* out) {
  if (nullptr == out || desc.rows == 0 || desc.cols == 0) {
    return false;
  }
  auto fmt = infer_window_fmt(desc);
  auto elem_bytes = fmt_elem_bytes(fmt);
  if (0 == elem_bytes) {
    return false;
  }
  TmemWindowPlan window{};
  window.window_id = window_id;
  window.target = map_window_target(desc);
  window.layout_kind = TmemWindowLayoutKind::MathRowMajor;
  window.elem_shape = {desc.rows, desc.cols};
  window.fmt = fmt;
  window.logical_col_base = 0;
  window.logical_line_base = 0;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, desc.rows), TmemWindowPlanner::kTileRows);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, desc.cols), TmemWindowPlanner::kTileCols);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.packet_cols = 8;
  window.packet_rows = std::max<uint32_t>(1, TmemWindowPlanner::kPacketBytes / (window.packet_cols * elem_bytes));
  window.packets_per_tile = ceil_div(TmemWindowPlanner::kTileRows, window.packet_rows)
                          * ceil_div(TmemWindowPlanner::kTileCols, window.packet_cols);
  window.logical_col_span = TmemWindowPlanner::kPacketBytes;
  window.logical_line_span = window.tile_count * window.packets_per_tile;
  window.logical_tile_col_span = TmemWindowPlanner::kPacketBytes;
  window.logical_tile_line_span = window.packets_per_tile;
  window.logical_packet_col_span = TmemWindowPlanner::kPacketBytes;
  window.logical_packet_line_span = 1;
  *out = window;
  return true;
}

uint32_t effective_fmt_d(const MmaDescriptor& desc) {
  return desc.fmt_d;
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

uint32_t fmt_for_target(const MmaDescriptor& desc, TcuTarget target, bool store_path) {
  switch (target) {
  case TcuTarget::A:
    return desc.fmt_a;
  case TcuTarget::B:
    return desc.fmt_b;
  case TcuTarget::C:
    return store_path ? effective_fmt_d(desc) : desc.fmt_c;
  default:
    return 0;
  }
}

TmemWindowTarget planner_target(TcuTarget target, bool store_path) {
  switch (target) {
  case TcuTarget::A:
    return TmemWindowTarget::A;
  case TcuTarget::B:
    return TmemWindowTarget::B;
  case TcuTarget::C:
    return store_path ? TmemWindowTarget::D : TmemWindowTarget::C;
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
                                     bool store_path,
                                     TmemWindowPlan* out) {
  (void)allocation;
  if (nullptr == out) {
    return false;
  }
  auto shape = shape_for_target(desc, target);
  auto fmt = fmt_for_target(desc, target, store_path);
  if (shape.empty() || fmt_elem_bytes(fmt) == 0) {
    return false;
  }
  return TmemWindowPlanner::build_single_dense_window(planner_target(target, store_path),
                                                      shape,
                                                      fmt,
                                                      desc.sparse_mode,
                                                      0, // col_span: auto
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
  window.layout_kind = TmemWindowLayoutKind::MathRowMajor;
  window.elem_shape = {static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.a_rows), 16u) * 4u),
                       static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.a_cols), 16u) * 16u)};
  window.fmt = vortex::tensor::uint8::id;
  window.sparse_mode = desc.sparse_mode;
  window.logical_line_base = 0;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, desc.a_rows), 16u);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, desc.a_cols), 16u);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.logical_col_span = Tmem::kPacketBytes;
  window.logical_line_span = window.tile_count;
  window.logical_tile_col_span = Tmem::kPacketBytes;
  window.logical_tile_line_span = 1;
  window.packet_cols = 16;
  window.packet_rows = 4;
  window.logical_packet_col_span = Tmem::kPacketBytes;
  window.logical_packet_line_span = 1;
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
   || desc.tile_role != 1) {
    return false;
  }
  TmemWindowPlan window{};
  window.window_id = meta_shadow_window_id(window_id);
  window.target = TmemWindowTarget::Meta;
  window.layout_kind = TmemWindowLayoutKind::MathRowMajor;
  window.elem_shape = {static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.rows), 16u) * 4u),
                       static_cast<uint16_t>(ceil_div(std::max<uint32_t>(1, desc.cols), 16u) * 16u)};
  window.fmt = vortex::tensor::uint8::id;
  window.logical_line_base = 0;
  window.tile_rows = ceil_div(std::max<uint32_t>(1, desc.rows), 16u);
  window.tile_cols = ceil_div(std::max<uint32_t>(1, desc.cols), 16u);
  window.tile_count = window.tile_rows * window.tile_cols;
  window.logical_col_span = Tmem::kPacketBytes;
  window.logical_line_span = window.tile_count;
  window.logical_tile_col_span = Tmem::kPacketBytes;
  window.logical_tile_line_span = 1;
  window.packet_cols = 16;
  window.packet_rows = 4;
  window.logical_packet_col_span = Tmem::kPacketBytes;
  window.logical_packet_line_span = 1;
  window.packets_per_tile = 1;
  *out = window;
  return true;
}

TmemWindowPlan build_legacy_window_plan(uint32_t window_id,
                                        const TmaDescriptor& desc,
                                        uint32_t transfer_region_col_base,
                                        uint32_t transfer_region_col_span) {
  (void)transfer_region_col_base;
  TmemWindowPlan window{};
  auto shape = TensorShape2D{desc.rows, desc.cols};
  auto fmt = infer_window_fmt(desc);
  bool built_from_shape = false;
  if (!shape.empty() && fmt_elem_bytes(fmt) != 0) {
    if (uses_generic_math_window(desc)) {
      (void)build_generic_math_window(window_id, desc, &window);
    } else {
      (void)TmemWindowPlanner::build_single_dense_window(map_window_target(desc),
                                                         shape,
                                                         fmt,
                                                         0,
                                                         0, // col_span: auto
                                                         window_id,
                                                         &window,
                                                         nullptr);
    }
    built_from_shape = true;
  } else {
    window.window_id = window_id;
    window.target = map_window_target(desc);
  }
  if (!built_from_shape) {
    auto packet_line_bytes = std::min<uint32_t>(Tmem::kPacketBytes, std::max<uint32_t>(1, transfer_region_col_span));
    window.logical_col_span = packet_line_bytes;
    window.logical_line_span = std::max<uint32_t>(1, ceil_div(std::max<uint32_t>(1, desc.size_bytes), packet_line_bytes));
    window.logical_tile_col_span = packet_line_bytes;
    window.logical_tile_line_span = window.logical_line_span;
    window.packet_cols = packet_line_bytes;
    window.packet_rows = 1;
    window.logical_packet_col_span = packet_line_bytes;
    window.logical_packet_line_span = 1;
    window.tile_rows = 1;
    window.tile_cols = 1;
    window.tile_count = 1;
    window.packets_per_tile = ceil_div(std::max<uint32_t>(1, desc.size_bytes), packet_line_bytes);
  }
  window.logical_col_base = 0;
  window.logical_line_base = 0;
  return window;
}

void adjust_legacy_subwindow_from_existing(const TmaDescriptor& desc,
                                           const TmemWindowPlan* existing,
                                           TmemWindowPlan* window) {
  if (nullptr == existing || nullptr == window) {
    return;
  }
  if ((desc.rows != 0 && desc.cols != 0) || desc.elem_bytes != 0) {
    return;
  }

  auto target = static_cast<TcuTarget>(desc.tile_role);
  switch (target) {
  case TcuTarget::C:
    if (existing->logical_col_span == window->logical_col_span
     && existing->logical_line_span >= window->logical_line_span) {
      window->logical_col_base = existing->logical_col_base;
      window->logical_line_base = existing->logical_line_base
                                + (existing->logical_line_span - window->logical_line_span);
    }
    break;
  case TcuTarget::A:
    window->logical_col_base = existing->logical_col_base;
    window->logical_line_base = existing->logical_line_base;
    break;
  default:
    break;
  }
}

bool preserve_existing_math_window(const TmemWindowPlan* existing,
                                   const TmaDescriptor& desc) {
  if (nullptr == existing || !TmemWindowPlanner::uses_math_packet_adapter(*existing)) {
    return false;
  }
  if (!uses_generic_math_window(desc) && existing->target != map_window_target(desc)) {
    return false;
  }
  auto inferred_fmt = infer_window_fmt(desc);
  if (fmt_elem_bytes(inferred_fmt) != 0 && existing->fmt != inferred_fmt) {
    return false;
  }
  if (desc.rows != 0 && existing->elem_shape.rows != desc.rows) {
    return false;
  }
  if (desc.cols != 0 && existing->elem_shape.cols != desc.cols) {
    return false;
  }
  return true;
}

uint32_t next_available_window_line(const TmemAllocation& allocation) {
  uint32_t next_line = 0;
  for (const auto& window : allocation.windows) {
    next_line = std::max<uint32_t>(next_line, window.logical_line_base + window.logical_line_span);
  }
  return next_line;
}

bool place_window_after_existing(const TmemAllocation& allocation,
                                 TmemWindowPlan* window) {
  if (nullptr == window) {
    return false;
  }
  window->logical_col_base = 0;
  window->logical_line_base = next_available_window_line(allocation);
  return window->logical_col_span <= allocation.col_span
      && (window->logical_line_base + window->logical_line_span) <= TmemWindowPlanner::kLogicalLines;
}

void encode_math_window_packet(const TmemWindowPlan& window,
                               const std::vector<uint8_t>& math_window_bytes,
                               uint32_t packet_idx,
                               TmemPacket* out) {
  if (nullptr == out) {
    return;
  }
  if (!TmemWindowPlanner::pack_math_packet(window, math_window_bytes, packet_idx, &out->bytes)) {
    out->bytes.fill(0);
  }
}

void decode_math_window_packet(const TmemWindowPlan& window,
                               uint32_t packet_idx,
                               const TmemPacket& packet,
                               std::vector<uint8_t>* math_window_bytes) {
  if (nullptr == math_window_bytes) {
    return;
  }
  (void)TmemWindowPlanner::unpack_math_packet(window, packet_idx, packet.bytes, math_window_bytes);
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
  , tma_frontend_(TmaFrontend::Create("tma_frontend", this, env_flag_enabled("VORTEX_SIMX_TMA_LOAD_REALISTIC", true)))
  , tmem_system_(TmemSystem::Create("tmem_system", this))
  , tensor_mem_req_in_(this)
  , tensor_mem_rsp_out_(this)
  , tensor_async_op_completion_in_(this)
  , tma_frontend_async_op_completion_in_(this)
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
  , tma_(env_flag_enabled("VORTEX_SIMX_TMA_LOAD_REALISTIC", true))
  , mbarriers_(arch.num_barriers())
  , mbarrier_wait_targets_(arch.num_warps())
  , fence_wait_states_(arch.num_warps())
  , next_async_id_(1)
#endif
{
  char sname[100];

#ifdef EXT_TCU_ENABLE
  tensor_unit_->TensorMemReqOut.bind(&tmem_system_->TensorExecuteReqIn);
  tmem_system_->TensorExecuteRspOut.bind(&tensor_unit_->TensorMemRspIn);
  tma_frontend_->TmemReqOut.bind(&tmem_system_->TmaFrontendReqIn);
  tmem_system_->TmaFrontendRspOut.bind(&tma_frontend_->TmemRspIn);
  tmem_system_->RefillChunkReqOut.bind(&tma_frontend_->RefillChunkReqIn);
  tma_frontend_->RefillChunkRspOut.bind(&tmem_system_->RefillChunkRspIn);
  tensor_unit_->TensorAsyncOpCompletionOut.bind(&tensor_async_op_completion_in_);
  tma_frontend_->AsyncOpCompletionOut.bind(&tma_frontend_async_op_completion_in_);
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
  tma_frontend_->reset();
  tmem_system_->reset();
  tmem_.reset();
  tma_.reset();
  async_tensor_ops_.clear();
  async_tmem_ops_fifo_.clear();
  pending_tensor_mem_reqs_.clear();
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
  visible_tma_load_busy_handles_.clear();
  visible_tma_store_or_shift_busy_handles_.clear();
#endif

  perf_stats_ = PerfStats();
}

void Core::tick() {
#ifdef EXT_TCU_ENABLE
  execute_cycle_tma_load_reserved_handles_.clear();
  execute_cycle_tma_store_or_shift_reserved_handles_.clear();
  drain_tensor_execute_completion_notices();
  if (!tma_frontend_async_op_completion_in_.empty()) {
    auto completion = tma_frontend_async_op_completion_in_.front();
    async_tensor_complete(completion.async_id);
    tma_frontend_async_op_completion_in_.pop();
  }
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
  visible_tma_load_busy_handles_.clear();
  visible_tma_store_or_shift_busy_handles_.clear();
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.completed) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::TmaLoad:
      visible_tma_load_busy_handles_.insert(op.handle);
      break;
    case AsyncTensorOpType::TmaStore:
    case AsyncTensorOpType::TmemShift:
      visible_tma_store_or_shift_busy_handles_.insert(op.handle);
      break;
    default:
      break;
    }
  }
  if (nullptr != tmem_system_) {
    tmem_system_->publish_visible_state();
    return;
  }
  tmem_.publish_visible_state();
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
       << " window=" << op.window_id
       << " completed=" << op.completed
       << " committed=" << op.committed
       << " barrier=" << op.barrier_id
       << "\n";
  }
  if (nullptr != tma_frontend_) {
    tma_frontend_->dump_debug_state(os);
  }
  if (nullptr != tmem_system_) {
    tmem_system_->dump_debug_state(os);
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
  if (nullptr != tmem_system_) {
    return tmem_system_->lookup_allocation(handle, allocation);
  }
  return tmem_.lookup_allocation(handle, allocation);
}

bool Core::lookup_tmem_allocation(uint32_t handle, const TmemAllocation** allocation) const {
  if (nullptr != tmem_system_) {
    return tmem_system_->lookup_allocation(handle, allocation);
  }
  return tmem_.lookup_allocation(handle, allocation);
}

bool Core::read_tma_descriptor(uint32_t desc_id, TmaDescriptor* out) {
  if (nullptr != tma_frontend_) {
    return tma_frontend_->read_tma_descriptor(desc_id, out);
  }
  return tma_.read_tma_descriptor(emulator_.startup_arg(),
                                  desc_id,
                                  out,
                                  [this](void* data, uint64_t addr, uint32_t size) {
                                    this->dcache_read(data, addr, size);
                                  });
}

bool Core::read_mma_descriptor(uint32_t desc_id, MmaDescriptor* out) {
  if (nullptr != tma_frontend_) {
    return tma_frontend_->read_mma_descriptor(desc_id, out);
  }
  return tma_.read_mma_descriptor(emulator_.startup_arg(),
                                  desc_id,
                                  out,
                                  [this](void* data, uint64_t addr, uint32_t size) {
                                    this->dcache_read(data, addr, size);
                                  });
}

void Core::reset_tmem_port_budgets() {
  if (nullptr != tmem_system_) {
    return;
  }
  tmem_.reset_port_budgets(perf_stats_.cycles);
}

void Core::ensure_tmem_port_budgets() {
  if (nullptr != tmem_system_) {
    return;
  }
  tmem_.ensure_port_budgets(perf_stats_.cycles);
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

void Core::initialize_async_tensor_transaction(AsyncTensorOp& op) {
  if (op.transaction_initialized) {
    return;
  }

  switch (op.type) {
  case AsyncTensorOpType::TmaLoad: {
    TmaDescriptor desc;
    if (!read_tma_descriptor(op.descriptor_id, &desc)) {
      op.completed = true;
      op.transaction_initialized = true;
      return;
    }
    op.tma_desc = desc;
    op.use_meta_region = (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta);
    op.transfer_region_col_base = op.use_meta_region ? desc.meta_tmem_base : desc.tmem_base;
    op.transfer_region_col_span = op.use_meta_region ? desc.meta_col_span : desc.bank_span;
    if (0 == op.transfer_region_col_span) {
      const TmemAllocation* allocation = nullptr;
      if (!lookup_tmem_allocation(op.handle, &allocation)) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      op.transfer_region_col_base = allocation->payload_col_base;
      op.transfer_region_col_span = allocation->col_span;
    }

    uint32_t capacity = 0;
    if (!tmem_region_query(op.transfer_region_col_base, op.transfer_region_col_span, &capacity)) {
      op.completed = true;
      op.transaction_initialized = true;
      return;
    }

    auto legacy_window = build_legacy_window_plan(op.window_id, op.tma_desc,
                                                  op.transfer_region_col_base, op.transfer_region_col_span);
    const TmemWindowPlan* existing_window = nullptr;
    bool preserve_payload_window = tmem_.lookup_window(op.handle, op.window_id, &existing_window)
                                && preserve_existing_math_window(existing_window, op.tma_desc);
    if (!preserve_payload_window) {
      if (op.tma_desc.tile_role != 0
       && fmt_elem_bytes(infer_window_fmt(op.tma_desc)) != 0
       && op.tma_desc.rows != 0
       && op.tma_desc.cols != 0) {
        (void)TmemWindowPlanner::build_single_dense_window(map_window_target(op.tma_desc),
                                                           {op.tma_desc.rows, op.tma_desc.cols},
                                                           infer_window_fmt(op.tma_desc),
                                                           0,
                                                           op.transfer_region_col_span,
                                                           op.window_id,
                                                           &legacy_window,
                                                           nullptr);
        const TmemAllocation* allocation = nullptr;
        if (lookup_tmem_allocation(op.handle, &allocation) && nullptr != allocation) {
          if (!place_window_after_existing(*allocation, &legacy_window)) {
            op.completed = true;
            op.transaction_initialized = true;
            return;
          }
        }
      } else if (nullptr != existing_window) {
        adjust_legacy_subwindow_from_existing(op.tma_desc, existing_window, &legacy_window);
      }
      (void)tmem_.upsert_window(op.handle, legacy_window);
    }
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(op.tma_desc, op.window_id, &meta_window)) {
      meta_window.logical_col_base = op.tma_desc.meta_tmem_base;
      const TmemWindowPlan* existing_meta_window = nullptr;
      bool preserve_meta_window = tmem_.lookup_window(op.handle,
                                                      meta_shadow_window_id(op.window_id),
                                                      &existing_meta_window)
                               && existing_meta_window->target == TmemWindowTarget::Meta
                               && TmemWindowPlanner::uses_math_packet_adapter(*existing_meta_window);
      if (!preserve_meta_window) {
        (void)tmem_.upsert_window(op.handle, meta_window);
      }
    }

    op.payload_size_bytes = std::min<uint32_t>(tma_.payload_size_bytes(desc), capacity);
    if (!tma_.load_payload(desc,
                           capacity,
                           &op.payload_staging_buffer,
                           [this](void* data, uint64_t addr, uint32_t size) {
                             this->dcache_read(data, addr, size);
                           })) {
      op.completed = true;
      op.transaction_initialized = true;
      return;
    }

    const TmemWindowPlan* payload_window = nullptr;
    bool use_math_payload_window = tmem_.lookup_window(op.handle, op.window_id, &payload_window)
                                && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);

    uint32_t meta_packets = 0;
    op.meta_region_col_base = desc.meta_tmem_base;
    op.meta_region_col_span = desc.meta_col_span;
    op.meta_size_bytes = 0;
    op.meta_staging_buffer.clear();
    if (!op.use_meta_region && desc.meta_addr != 0 && desc.meta_size_bytes != 0 && desc.meta_col_span != 0) {
      uint32_t meta_capacity = 0;
      if (!tmem_region_query(op.meta_region_col_base, op.meta_region_col_span, &meta_capacity)
       || !tma_.load_meta(desc,
                          meta_capacity,
                          &op.meta_staging_buffer,
                          [this](void* data, uint64_t addr, uint32_t size) {
                            this->dcache_read(data, addr, size);
                          })) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      op.meta_size_bytes = std::min<uint32_t>(desc.meta_size_bytes, meta_capacity);
      const TmemWindowPlan* meta_window = nullptr;
      if (tmem_.lookup_window(op.handle, meta_shadow_window_id(op.window_id), &meta_window)
       && TmemWindowPlanner::uses_math_packet_adapter(*meta_window)) {
        if (!tmem_.window_packet_count(op.handle, meta_shadow_window_id(op.window_id), &meta_packets)) {
          op.completed = true;
          op.transaction_initialized = true;
          return;
        }
      } else {
        meta_packets = (op.meta_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
      }
    }

    uint32_t payload_packets = 0;
    if (use_math_payload_window) {
      if (!tmem_.window_packet_count(op.handle, op.window_id, &payload_packets)) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
    } else {
      payload_packets = (op.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    op.remaining_launch_cycles = kTmaLoadBaseLatency;
    op.remaining_tmem_write_packets = payload_packets + meta_packets;
  } break;
  case AsyncTensorOpType::TmaStore: {
    TmaDescriptor desc;
    if (!read_tma_descriptor(op.descriptor_id, &desc)) {
      op.completed = true;
      op.transaction_initialized = true;
      return;
    }
    op.tma_desc = desc;
    op.use_meta_region = (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta);
    op.transfer_region_col_base = op.use_meta_region ? desc.meta_tmem_base : desc.tmem_base;
    op.transfer_region_col_span = op.use_meta_region ? desc.meta_col_span : desc.bank_span;
    if (0 == op.transfer_region_col_span) {
      const TmemAllocation* allocation = nullptr;
      if (!lookup_tmem_allocation(op.handle, &allocation)) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      op.transfer_region_col_base = allocation->payload_col_base;
      op.transfer_region_col_span = allocation->col_span;
    }

    uint32_t capacity = 0;
    if (!tmem_region_query(op.transfer_region_col_base, op.transfer_region_col_span, &capacity)) {
      op.completed = true;
      op.transaction_initialized = true;
      return;
    }

    auto legacy_window = build_legacy_window_plan(op.window_id, op.tma_desc,
                                                  op.transfer_region_col_base, op.transfer_region_col_span);
    const TmemWindowPlan* existing_window = nullptr;
    bool preserve_payload_window = tmem_.lookup_window(op.handle, op.window_id, &existing_window)
                                && preserve_existing_math_window(existing_window, op.tma_desc);
    if (!preserve_payload_window) {
      if (op.tma_desc.tile_role != 0
       && fmt_elem_bytes(infer_window_fmt(op.tma_desc)) != 0
       && op.tma_desc.rows != 0
       && op.tma_desc.cols != 0) {
        (void)TmemWindowPlanner::build_single_dense_window(map_window_target(op.tma_desc),
                                                           {op.tma_desc.rows, op.tma_desc.cols},
                                                           infer_window_fmt(op.tma_desc),
                                                           0,
                                                           op.transfer_region_col_span,
                                                           op.window_id,
                                                           &legacy_window,
                                                           nullptr);
        const TmemAllocation* allocation = nullptr;
        if (lookup_tmem_allocation(op.handle, &allocation) && nullptr != allocation) {
          if (!place_window_after_existing(*allocation, &legacy_window)) {
            op.completed = true;
            op.transaction_initialized = true;
            return;
          }
        }
      } else if (nullptr != existing_window) {
        adjust_legacy_subwindow_from_existing(op.tma_desc, existing_window, &legacy_window);
      }
      (void)tmem_.upsert_window(op.handle, legacy_window);
    }
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(op.tma_desc, op.window_id, &meta_window)) {
      meta_window.logical_col_base = op.tma_desc.meta_tmem_base;
      const TmemWindowPlan* existing_meta_window = nullptr;
      bool preserve_meta_window = tmem_.lookup_window(op.handle,
                                                      meta_shadow_window_id(op.window_id),
                                                      &existing_meta_window)
                               && existing_meta_window->target == TmemWindowTarget::Meta
                               && TmemWindowPlanner::uses_math_packet_adapter(*existing_meta_window);
      if (!preserve_meta_window) {
        (void)tmem_.upsert_window(op.handle, meta_window);
      }
    }

    op.payload_size_bytes = std::min<uint32_t>(tma_.payload_size_bytes(desc), capacity);
    const TmemWindowPlan* payload_window = nullptr;
    bool use_math_payload_window = tmem_.lookup_window(op.handle, op.window_id, &payload_window)
                                && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);
    if (use_math_payload_window) {
      op.payload_staging_buffer.assign(op.payload_size_bytes, 0);
    }
    uint32_t payload_packets = 0;
    if (use_math_payload_window) {
      if (!tmem_.window_packet_count(op.handle, op.window_id, &payload_packets)) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
    } else {
      payload_packets = (op.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    op.remaining_launch_cycles = kTmaStoreBaseLatency;
    op.remaining_tmem_read_packets = payload_packets;
  } break;
  case AsyncTensorOpType::TmemShift: {
    uint32_t packets = 0;
    TmemAllocation* allocation = nullptr;
    if (!lookup_tmem_allocation(op.handle, &allocation)) {
      op.completed = true;
      op.transaction_initialized = true;
      return;
    }
    if (!tmem_.window_packet_count(op.handle, op.window_id, &packets)) {
      uint32_t size_bytes = 0;
      if (!tmem_region_query(allocation->payload_col_base, allocation->col_span, &size_bytes)) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      packets = (size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
    op.remaining_tmem_read_packets = packets;
    op.remaining_tmem_write_packets = packets;
    if (op.refill_descriptor_id != 0) {
      const TmemWindowPlan* window = nullptr;
      uint32_t math_row_bytes = 0;
      if (!tmem_.lookup_window(op.handle, op.window_id, &window)
       || window->logical_col_span == 0) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      math_row_bytes = window->elem_shape.cols * fmt_elem_bytes(window->fmt);
      if (0 == math_row_bytes) {
        math_row_bytes = window->logical_col_span;
      }
      TmaDescriptor desc;
      if (!read_tma_descriptor(op.refill_descriptor_id, &desc)
       || desc.rows != 1) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      op.tma_desc = desc;
      op.refill_math_row_bytes = std::min<uint32_t>(tma_.payload_size_bytes(desc), math_row_bytes);
      if (!tma_.load_payload(desc,
                             math_row_bytes,
                             &op.payload_staging_buffer,
                             [this](void* data, uint64_t addr, uint32_t size) {
                               this->dcache_read(data, addr, size);
                             })) {
        op.completed = true;
        op.transaction_initialized = true;
        return;
      }
      op.remaining_launch_cycles = tma_.estimate_load_latency(desc);
      op.remaining_refill_line_packets = (op.refill_math_row_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
    }
  } break;
  case AsyncTensorOpType::MmaLoad:
  case AsyncTensorOpType::MmaStore:
  case AsyncTensorOpType::Wmma:
    break;
  default:
    std::abort();
  }

  op.transaction_initialized = true;
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
  if (nullptr != tmem_system_) {
    return tmem_system_->region_query(col_base, col_span, size_bytes);
  }
  return tmem_.region_query(col_base, col_span, size_bytes);
}

bool Core::tmem_query(uint32_t handle, uint32_t* col_span, uint32_t* size_bytes) const {
  if (nullptr != tmem_system_) {
    return tmem_system_->query(handle, col_span, size_bytes);
  }
  return tmem_.query(handle, col_span, size_bytes);
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
  if (nullptr != tmem_system_) {
    return tmem_system_->enqueue_port_request(desc, desc.age);
  }
  return tmem_.enqueue_port_request(perf_stats_.cycles, desc);
}

bool Core::tmem_request_granted(uint64_t tag) const {
  if (nullptr != tmem_system_) {
    return tmem_system_->request_granted(tag);
  }
  return tmem_.request_granted(tag);
}

void Core::consume_tmem_request_grant(uint64_t tag) {
  if (nullptr != tmem_system_) {
    tmem_system_->consume_request_grant(tag);
    return;
  }
  tmem_.consume_request_grant(tag);
}

bool Core::lookup_tmem_window(uint32_t handle, uint32_t window_id, const TmemWindowPlan** out) const {
  if (nullptr != tmem_system_) {
    return tmem_system_->lookup_window(handle, window_id, out);
  }
  return tmem_.lookup_window(handle, window_id, out);
}

bool Core::tmem_window_packet_count(uint32_t handle, uint32_t window_id, uint32_t* count) const {
  if (nullptr != tmem_system_) {
    return tmem_system_->window_packet_count(handle, window_id, count);
  }
  return tmem_.window_packet_count(handle, window_id, count);
}

bool Core::tmem_window_epoch(uint32_t handle, uint32_t* epoch) const {
  if (nullptr != tmem_system_) {
    return tmem_system_->window_epoch(handle, epoch);
  }
  return tmem_.window_epoch(handle, epoch);
}

bool Core::tmem_upsert_window(uint32_t handle, const TmemWindowPlan& window) {
  if (nullptr != tmem_system_) {
    return tmem_system_->upsert_window(handle, window);
  }
  return tmem_.upsert_window(handle, window);
}

void Core::tmem_set_payload_ready(uint32_t handle, bool ready) {
  if (nullptr != tmem_system_) {
    tmem_system_->set_payload_ready(handle, ready);
    return;
  }
  tmem_.set_payload_ready(handle, ready);
}

void Core::tmem_set_meta_ready(uint32_t handle, bool ready) {
  if (nullptr != tmem_system_) {
    tmem_system_->set_meta_ready(handle, ready);
    return;
  }
  tmem_.set_meta_ready(handle, ready);
}

void Core::tmem_set_meta_region(uint32_t handle, uint32_t meta_col_base, uint32_t meta_col_span) {
  if (nullptr != tmem_system_) {
    tmem_system_->set_meta_region(handle, meta_col_base, meta_col_span);
    return;
  }
  tmem_.set_meta_region(handle, meta_col_base, meta_col_span);
}

bool Core::tmem_set_row_bytes(uint32_t handle, uint32_t row_bytes) {
  if (nullptr != tmem_system_) {
    return tmem_system_->set_row_bytes(handle, row_bytes);
  }
  return tmem_.set_row_bytes(handle, row_bytes);
}

bool Core::tmem_bump_window_epoch(uint32_t handle) {
  if (nullptr != tmem_system_) {
    return tmem_system_->bump_window_epoch(handle);
  }
  return tmem_.bump_window_epoch(handle);
}

bool Core::ensure_tmem_window_bound(uint32_t handle, uint32_t desc_id, TcuTarget target, uint32_t window_id, bool store_path) {
  const TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return false;
  }

  auto lookup_bound_window = [&](uint32_t lookup_window_id, const TmemWindowPlan** out) -> bool {
    if (nullptr != tmem_system_) {
      return tmem_system_->lookup_window(handle, lookup_window_id, out);
    }
    return tmem_.lookup_window(handle, lookup_window_id, out);
  };

  auto read_window_epoch = [&](uint32_t* epoch) -> bool {
    if (nullptr != tmem_system_) {
      return tmem_system_->window_epoch(handle, epoch);
    }
    return tmem_.window_epoch(handle, epoch);
  };

  auto bind_layout_plan = [&](const TmemLayoutPlan& plan) -> bool {
    if (nullptr != tmem_system_) {
      return tmem_system_->bind_layout(handle, plan);
    }
    return tmem_.bind_layout(handle, plan);
  };

  auto upsert_bound_window = [&](const TmemWindowPlan& window) -> bool {
    if (nullptr != tmem_system_) {
      return tmem_system_->upsert_window(handle, window);
    }
    return tmem_.upsert_window(handle, window);
  };

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
    input.d_shape = {desc.c_rows, desc.c_cols};
    input.fmt_a = desc.fmt_a;
    input.fmt_b = desc.fmt_b;
    input.fmt_c = desc.fmt_c;
    input.fmt_d = effective_fmt_d(desc);
    input.sparse_mode = desc.sparse_mode;
    input.allocation_col_span = allocation->col_span;
    TmemLayoutPlan plan{};
    if (!TmemWindowPlanner::build_dense_plan(input, &plan, nullptr)) {
      return false;
    }
    uint32_t epoch = 0;
    if (read_window_epoch(&epoch)) {
      plan.epoch = epoch;
    }
    return bind_layout_plan(plan);
  }

  TmemWindowPlan window{};
  if (!build_single_target_window_plan(*allocation, desc, target, window_id, store_path, &window)) {
    return true;
  }
  const TmemWindowPlan* existing_window = nullptr;
  if (lookup_bound_window(window_id, &existing_window)) {
    return true;
  }
  if (!place_window_after_existing(*allocation, &window)) {
    return false;
  }
  if (!upsert_bound_window(window)) {
    return false;
  }
  if (target == TcuTarget::A) {
    TmemWindowPlan meta_window{};
    if (build_sparse_meta_window_plan(desc, window_id, &meta_window)) {
      meta_window.logical_col_base = allocation->meta_col_base;
      meta_window.logical_line_base = 0;
      if (!upsert_bound_window(meta_window)) {
        return false;
      }
    }
  }
  return true;
}

bool Core::tmem_region_copy_in(uint32_t col_base, uint32_t col_span, const uint8_t* data, uint32_t size_bytes) {
  if (nullptr != tmem_system_) {
    return false;
  }
  return tmem_.region_copy_in(col_base, col_span, data, size_bytes);
}

bool Core::tmem_region_copy_out(uint32_t col_base, uint32_t col_span, uint8_t* data, uint32_t size_bytes) const {
  if (nullptr != tmem_system_) {
    return false;
  }
  return tmem_.region_copy_out(col_base, col_span, data, size_bytes);
}

bool Core::tmem_copy_in(uint32_t handle, const uint8_t* data, uint32_t size_bytes) {
  if (nullptr != tmem_system_) {
    return false;
  }
  return tmem_.copy_in(handle, data, size_bytes);
}

bool Core::tmem_copy_out(uint32_t handle, uint8_t* data, uint32_t size_bytes) const {
  if (nullptr != tmem_system_) {
    return false;
  }
  return tmem_.copy_out(handle, data, size_bytes);
}

bool Core::tmem_region_shift_down(uint32_t col_base, uint32_t col_span, uint32_t row_bytes) {
  if (nullptr != tmem_system_) {
    return tmem_system_->region_shift_down(col_base, col_span, row_bytes);
  }
  return tmem_.region_shift_down(col_base, col_span, row_bytes);
}

bool Core::tmem_handle_busy(uint32_t handle) const {
  if (nullptr != tma_frontend_ && nullptr != tmem_system_) {
    return tma_frontend_->visible_load_busy(handle)
        || tma_frontend_->visible_store_busy(handle)
        || tmem_system_->visible_shift_busy(handle);
  }
  return visible_tma_load_busy_handles_.count(handle) != 0
      || visible_tma_store_or_shift_busy_handles_.count(handle) != 0;
}

bool Core::tmem_handle_ready_for_mma_load(uint32_t handle, TcuTarget target, uint32_t sparse_mode) const {
  return TmemHandleBlockReason::None == tmem_handle_load_block_reason(handle, target, sparse_mode);
}

bool Core::tmem_handle_ready_for_mma_store(uint32_t handle) const {
  return TmemHandleBlockReason::None == tmem_handle_store_block_reason(handle);
}

bool Core::has_inflight_tma_handle_activity() const {
  if (nullptr != tma_frontend_ && nullptr != tmem_system_) {
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
  if (execute_cycle_tma_load_reserved_handles_.count(handle) != 0) {
    return TmemHandleBlockReason::BusyTmaLoad;
  }
  if (execute_cycle_tma_store_or_shift_reserved_handles_.count(handle) != 0) {
    return TmemHandleBlockReason::BusyTmaStoreOrShift;
  }
  if (nullptr != tma_frontend_ && tma_frontend_->visible_load_busy(handle)) {
    return TmemHandleBlockReason::BusyTmaLoad;
  }
  if ((nullptr != tma_frontend_ && tma_frontend_->visible_store_busy(handle))
   || (nullptr != tmem_system_ && tmem_system_->visible_shift_busy(handle))) {
    return TmemHandleBlockReason::BusyTmaStoreOrShift;
  }
  bool payload_ready = (nullptr != tmem_system_) ? tmem_system_->visible_payload_ready(handle)
                                                 : allocation->visible_payload_ready;
  if (!payload_ready) {
    return TmemHandleBlockReason::PayloadNotReady;
  }
  bool needs_meta = (target == TcuTarget::A || target == TcuTarget::None)
                 && (sparse_mode != vortex::tensor::sparse_none);
  bool meta_ready = (nullptr != tmem_system_) ? tmem_system_->visible_meta_ready(handle)
                                              : allocation->visible_meta_ready;
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
  if (execute_cycle_tma_load_reserved_handles_.count(handle) != 0) {
    return TmemHandleBlockReason::BusyTmaLoad;
  }
  if (execute_cycle_tma_store_or_shift_reserved_handles_.count(handle) != 0) {
    return TmemHandleBlockReason::BusyTmaStoreOrShift;
  }
  if (nullptr != tma_frontend_ && tma_frontend_->visible_load_busy(handle)) {
    return TmemHandleBlockReason::BusyTmaLoad;
  }
  if ((nullptr != tma_frontend_ && tma_frontend_->visible_store_busy(handle))
   || (nullptr != tmem_system_ && tmem_system_->visible_shift_busy(handle))) {
    return TmemHandleBlockReason::BusyTmaStoreOrShift;
  }
  bool payload_ready = (nullptr != tmem_system_) ? tmem_system_->visible_payload_ready(handle)
                                                 : allocation->visible_payload_ready;
  if (!payload_ready) {
    return TmemHandleBlockReason::PayloadNotReady;
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

// Advance one already-issued Core-side async tensor transaction by one cycle.
//
// Hardware view:
// - consume fixed launch/setup latency first
// - then move at most one TMEM packet (or one refill line packet) per cycle
//   subject to TMEM ingress/egress and bank availability
// - mark TMEM-visible state ready only after the final required packet lands
bool Core::advance_one_async_tensor_transaction(AsyncTensorOp& op) {
  initialize_async_tensor_transaction(op);
  if (op.completed) {
    finalize_async_tensor_op(op);
    return true;
  }
  if (op.remaining_launch_cycles != 0) {
    --op.remaining_launch_cycles;
    return true;
  }

  switch (op.type) {
  case AsyncTensorOpType::TmaLoad:
  case AsyncTensorOpType::TmaStore: {
    // TMA sees a math-matrix view of the payload/meta image. The helper below
    // converts between that external row-major view and the TMEM packet view
    // chosen by the current window layout template.
    auto make_packet = [](const std::vector<uint8_t>& buffer, uint32_t packet_idx) {
      TmemPacket packet;
      auto offset = packet_idx * Tmem::kPacketBytes;
      if (offset < buffer.size()) {
        auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, buffer.size() - offset);
        std::copy_n(buffer.data() + offset, valid_bytes, packet.bytes.begin());
      }
      return packet;
    };
    const TmemWindowPlan* payload_window = nullptr;
    bool use_math_payload_window = tmem_.lookup_window(op.handle, op.window_id, &payload_window)
                                && TmemWindowPlanner::uses_math_packet_adapter(*payload_window);
    auto request_age = [&op](uint32_t ordinal) -> uint64_t {
      return (static_cast<uint64_t>(op.async_id) << 32) | ordinal;
    };

    if (op.type == AsyncTensorOpType::TmaLoad) {
      if (op.remaining_tmem_write_packets != 0) {
        // Load direction: staged payload/meta bytes already exist in the
        // external math view. This step emits at most one TMEM-visible packet
        // per cycle, subject to ingress and bank availability.
        uint32_t payload_packets = 0;
        if (use_math_payload_window) {
          if (!tmem_.window_packet_count(op.handle, op.window_id, &payload_packets)) {
            op.completed = true;
            finalize_async_tensor_op(op);
            return true;
          }
        } else {
          payload_packets = (op.payload_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
        }
        if (op.next_payload_packet_idx < payload_packets) {
          if (op.pending_tmem_request_tag == 0) {
            TmemRequestDesc request{};
            request.kind = use_math_payload_window
                         ? TmemRequestKind::WindowWrite
                         : TmemRequestKind::WindowLinearWrite;
            request.age = request_age(op.next_payload_packet_idx);
            request.handle = op.handle;
            request.window_id = op.window_id;
            request.packet_idx = op.next_payload_packet_idx;
            op.pending_tmem_request_tag = enqueue_tmem_request(request);
            return true;
          }
          if (!tmem_request_granted(op.pending_tmem_request_tag)) {
            ++perf_stats_.stall_tmem_write_port_busy;
            return true;
          }
          consume_tmem_request_grant(op.pending_tmem_request_tag);
          op.pending_tmem_request_tag = 0;
          ++perf_stats_.tmem_write_packets;
          TmemPacket packet;
          if (use_math_payload_window) {
            encode_math_window_packet(*payload_window, op.payload_staging_buffer, op.next_payload_packet_idx, &packet);
          } else {
            packet = make_packet(op.payload_staging_buffer, op.next_payload_packet_idx);
          }
          bool written = use_math_payload_window
                       ? tmem_.write_window_packet(op.handle, op.window_id, op.next_payload_packet_idx, packet)
                       : tmem_.write_window_linear_packet(op.handle, op.window_id, op.next_payload_packet_idx, packet);
          if (!written) {
            op.completed = true;
            finalize_async_tensor_op(op);
            return true;
          }
          ++op.next_payload_packet_idx;
          if (op.next_payload_packet_idx == payload_packets) {
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
          // Sparse meta follows the payload packets. It either lands in the
          // meta shadow window or in the legacy linear meta region, depending
          // on how the current handle/window was planned.
          const TmemWindowPlan* meta_window = nullptr;
          bool use_meta_window = tmem_.lookup_window(op.handle, meta_shadow_window_id(op.window_id), &meta_window);
          bool use_math_meta_window = use_meta_window && TmemWindowPlanner::uses_math_packet_adapter(*meta_window);
          if (op.pending_tmem_request_tag == 0) {
            TmemRequestDesc request{};
            request.kind = use_meta_window
                         ? (use_math_meta_window ? TmemRequestKind::WindowWrite
                                                 : TmemRequestKind::WindowLinearWrite)
                         : TmemRequestKind::RegionWrite;
            request.age = request_age(payload_packets + op.next_meta_packet_idx);
            request.handle = op.handle;
            request.window_id = use_meta_window ? meta_shadow_window_id(op.window_id) : 0;
            request.packet_idx = op.next_meta_packet_idx;
            request.col_base = op.meta_region_col_base;
            request.col_span = op.meta_region_col_span;
            op.pending_tmem_request_tag = enqueue_tmem_request(request);
            return true;
          }
          if (!tmem_request_granted(op.pending_tmem_request_tag)) {
            ++perf_stats_.stall_tmem_write_port_busy;
            return true;
          }
          consume_tmem_request_grant(op.pending_tmem_request_tag);
          op.pending_tmem_request_tag = 0;
          ++perf_stats_.tmem_write_packets;
          TmemPacket packet;
          if (use_math_meta_window) {
            encode_math_window_packet(*meta_window, op.meta_staging_buffer, op.next_meta_packet_idx, &packet);
          } else {
            packet = make_packet(op.meta_staging_buffer, op.next_meta_packet_idx);
          }
          bool written = use_meta_window
                       ? (use_math_meta_window
                          ? tmem_.write_window_packet(op.handle, meta_shadow_window_id(op.window_id), op.next_meta_packet_idx, packet)
                          : tmem_.write_window_linear_packet(op.handle, meta_shadow_window_id(op.window_id), op.next_meta_packet_idx, packet))
                       : tmem_.region_write_packet(op.meta_region_col_base, op.meta_region_col_span, op.next_meta_packet_idx, packet);
          if (!written) {
            op.completed = true;
            finalize_async_tensor_op(op);
            return true;
          }
          ++op.next_meta_packet_idx;
          if ((op.next_meta_packet_idx * Tmem::kPacketBytes) >= op.meta_size_bytes) {
            tmem_.set_meta_ready(op.handle, true);
          }
        }

        --op.remaining_tmem_write_packets;
        if (op.remaining_tmem_write_packets != 0) {
          return true;
        }
        (void)tmem_.bump_window_epoch(op.handle);
      }
    } else {
      if (op.remaining_tmem_read_packets != 0) {
        // Store direction: read one TMEM packet per cycle, then immediately
        // decode it back into the external math-matrix byte image.
        if (op.pending_tmem_request_tag == 0) {
          TmemRequestDesc request{};
          request.kind = use_math_payload_window
                       ? TmemRequestKind::WindowRead
                       : TmemRequestKind::WindowLinearRead;
          request.age = request_age(op.next_payload_packet_idx);
          request.handle = op.handle;
          request.window_id = op.window_id;
          request.packet_idx = op.next_payload_packet_idx;
          op.pending_tmem_request_tag = enqueue_tmem_request(request);
          return true;
        }
        if (!tmem_request_granted(op.pending_tmem_request_tag)) {
          ++perf_stats_.stall_tmem_read_port_busy;
          return true;
        }
        consume_tmem_request_grant(op.pending_tmem_request_tag);
        op.pending_tmem_request_tag = 0;
        ++perf_stats_.tmem_read_packets;

        TmemPacket packet;
        bool read_ok = use_math_payload_window
                    ? tmem_.read_window_packet(op.handle, op.window_id, op.next_payload_packet_idx, &packet)
                    : tmem_.read_window_linear_packet(op.handle, op.window_id, op.next_payload_packet_idx, &packet);
        if (!read_ok) {
          op.completed = true;
          finalize_async_tensor_op(op);
          return true;
        }
        if (use_math_payload_window) {
          decode_math_window_packet(*payload_window, op.next_payload_packet_idx, packet, &op.payload_staging_buffer);
        } else {
          auto offset = op.next_payload_packet_idx * Tmem::kPacketBytes;
          auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, op.payload_size_bytes - offset);
          tma_.store_payload(op.tma_desc,
                             packet.bytes.data(),
                             valid_bytes,
                             [offset, this](const void* data, uint64_t addr, uint32_t size) {
                               this->dcache_write(data, addr + offset, size);
                             });
        }
        ++op.next_payload_packet_idx;
        --op.remaining_tmem_read_packets;
        if (0 == op.remaining_tmem_read_packets) {
          if (use_math_payload_window && !op.payload_staging_buffer.empty()) {
            tma_.store_payload(op.tma_desc,
                               op.payload_staging_buffer.data(),
                               op.payload_size_bytes,
                               [this](const void* data, uint64_t addr, uint32_t size) {
                                 this->dcache_write(data, addr, size);
                               });
          }
          if (op.use_meta_region) {
            tmem_.set_meta_ready(op.handle, true);
          } else {
            tmem_.set_payload_ready(op.handle, true);
          }
        } else {
          return true;
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
      return true;
    }
    auto request_age = [&op](uint32_t ordinal) -> uint64_t {
      return (static_cast<uint64_t>(op.async_id) << 32) | ordinal;
    };
    bool use_window = (op.window_id != 0) || tmem_.lookup_window(op.handle, op.window_id, nullptr);
    if (!op.main_shift_body_complete) {
      // Phase 1: account for the existing window body that must be shifted
      // down by one math row. The Cmodel charges one TMEM read + one TMEM
      // write packet for each packet-sized chunk that participates.
      auto packet_idx = op.next_payload_packet_idx;
      if (op.remaining_tmem_read_packets != 0 && op.pending_tmem_request_tag == 0) {
        TmemRequestDesc request{};
        request.kind = use_window ? TmemRequestKind::WindowRead
                                  : TmemRequestKind::RegionRead;
        request.age = request_age(packet_idx * 2);
        request.handle = op.handle;
        request.window_id = op.window_id;
        request.packet_idx = packet_idx;
        request.col_base = allocation->payload_col_base;
        request.col_span = allocation->col_span;
        op.pending_tmem_request_tag = enqueue_tmem_request(request);
      }
      if (op.remaining_tmem_write_packets != 0 && op.pending_tmem_aux_request_tag == 0) {
        TmemRequestDesc request{};
        request.kind = use_window ? TmemRequestKind::WindowWrite
                                  : TmemRequestKind::RegionWrite;
        request.age = request_age(packet_idx * 2 + 1);
        request.handle = op.handle;
        request.window_id = op.window_id;
        request.packet_idx = packet_idx;
        request.col_base = allocation->payload_col_base;
        request.col_span = allocation->col_span;
        op.pending_tmem_aux_request_tag = enqueue_tmem_request(request);
      }
      if (op.pending_tmem_request_tag != 0 && !tmem_request_granted(op.pending_tmem_request_tag)) {
        ++perf_stats_.stall_tmem_read_port_busy;
        return true;
      }
      if (op.pending_tmem_aux_request_tag != 0 && !tmem_request_granted(op.pending_tmem_aux_request_tag)) {
        ++perf_stats_.stall_tmem_write_port_busy;
        return true;
      }
      if (op.pending_tmem_request_tag != 0) {
        consume_tmem_request_grant(op.pending_tmem_request_tag);
        op.pending_tmem_request_tag = 0;
        ++perf_stats_.tmem_read_packets;
        --op.remaining_tmem_read_packets;
      }
      if (op.pending_tmem_aux_request_tag != 0) {
        consume_tmem_request_grant(op.pending_tmem_aux_request_tag);
        op.pending_tmem_aux_request_tag = 0;
        ++perf_stats_.tmem_write_packets;
        --op.remaining_tmem_write_packets;
      }
      ++op.next_payload_packet_idx;
      if (op.remaining_tmem_read_packets != 0 || op.remaining_tmem_write_packets != 0) {
        return true;
      }
      op.main_shift_body_complete = true;
      if (op.remaining_refill_line_packets == 0) {
        bool shifted = use_window
                     ? tmem_.shift_window_math_rows_down(op.handle, op.window_id, nullptr, 0)
                     : tmem_region_shift_down(allocation->payload_col_base, allocation->col_span, allocation->row_bytes);
        if (!shifted) {
          op.completed = true;
          finalize_async_tensor_op(op);
          return true;
        }
        tmem_.set_payload_ready(op.handle, true);
        (void)tmem_.bump_window_epoch(op.handle);
      } else {
        return true;
      }
    }

    if (op.remaining_refill_line_packets != 0) {
      // Phase 2: optional refill of the new top math row. One packet-sized row
      // chunk is written per cycle through the normal TMEM ingress path.
      auto chunk_idx = op.next_refill_line_packet_idx;
      if (op.pending_tmem_request_tag == 0) {
        TmemRequestDesc request{};
        request.kind = use_window ? TmemRequestKind::WindowLineWrite
                                  : TmemRequestKind::RegionWrite;
        request.age = request_age(chunk_idx);
        request.handle = op.handle;
        request.window_id = op.window_id;
        request.packet_idx = chunk_idx;
        request.line_idx = 0;
        request.chunk_idx = chunk_idx;
        request.col_base = allocation->payload_col_base;
        request.col_span = allocation->col_span;
        op.pending_tmem_request_tag = enqueue_tmem_request(request);
        return true;
      }
      if (!tmem_request_granted(op.pending_tmem_request_tag)) {
        ++perf_stats_.stall_tmem_write_port_busy;
        return true;
      }
      consume_tmem_request_grant(op.pending_tmem_request_tag);
      op.pending_tmem_request_tag = 0;
      ++perf_stats_.tmem_write_packets;
      ++op.next_refill_line_packet_idx;
      --op.remaining_refill_line_packets;
      if (op.remaining_refill_line_packets != 0) {
        return true;
      }
      bool shifted = use_window
                   ? tmem_.shift_window_math_rows_down(op.handle, op.window_id,
                                                      op.payload_staging_buffer.data(),
                                                      op.refill_math_row_bytes)
                   : tmem_region_shift_down(allocation->payload_col_base, allocation->col_span, allocation->row_bytes);
      if (!shifted) {
        op.completed = true;
        finalize_async_tensor_op(op);
        return true;
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
  return true;
}

void Core::drain_tensor_execute_packet_requests() {
  // TensorExecuteSystem and Core/TMEM communicate through SimPort, so only one
  // request may be dequeued from this port per global cycle.
  if (tensor_mem_req_in_.empty()) {
    return;
  }
  auto request = tensor_mem_req_in_.front();
  PendingTensorMemReq pending{};
  pending.port_req = request;
  pending.tmem_request_tag = enqueue_tmem_request(request.port_request);
  pending_tensor_mem_reqs_[request.request_id] = pending;
  tensor_mem_req_in_.pop();
}

void Core::complete_granted_tensor_execute_packet_requests() {
  std::vector<uint64_t> completed_request_ids;
  completed_request_ids.reserve(pending_tensor_mem_reqs_.size());

  for (auto& entry : pending_tensor_mem_reqs_) {
    auto request_id = entry.first;
    auto& pending = entry.second;
    if (!tmem_request_granted(pending.tmem_request_tag)) {
      continue;
    }

    consume_tmem_request_grant(pending.tmem_request_tag);

    TensorMemPortRsp response{};
    response.request_id = request_id;
    response.access_type = pending.port_req.access_type;

    auto& request = pending.port_req.port_request;
    bool success = false;
    switch (request.kind) {
    case TmemRequestKind::WindowRead:
      success = tmem_read_window_packet(request.handle, request.window_id, request.packet_idx, &response.read_packet);
      break;
    case TmemRequestKind::WindowWrite:
      success = tmem_write_window_packet(request.handle, request.window_id, request.packet_idx, pending.port_req.write_packet);
      break;
    default:
      std::cerr << "Core error: tensor execute bridge received non-window TMEM request"
                << " request_id=" << request_id
                << " kind=" << static_cast<uint32_t>(request.kind)
                << " handle=" << request.handle
                << " window=" << request.window_id
                << " packet=" << request.packet_idx
                << std::endl;
      std::abort();
    }

    if (!success) {
      std::cerr << "Core error: TMEM port request completion failed"
                << " request_id=" << request_id
                << " kind=" << static_cast<uint32_t>(request.kind)
                << " handle=" << request.handle
                << " window=" << request.window_id
                << " packet=" << request.packet_idx
                << std::endl;
      std::abort();
    }

    // Delay-0 is sufficient here because TensorUnit ticks earlier than Core in
    // the platform object order, so the response becomes consumable next cycle.
    tensor_mem_rsp_out_.push(response, 0);
    completed_request_ids.push_back(request_id);
  }

  for (auto request_id : completed_request_ids) {
    pending_tensor_mem_reqs_.erase(request_id);
  }
}

void Core::drain_tensor_execute_completion_notices() {
  if (!tensor_async_op_completion_in_.empty()) {
    auto completion = tensor_async_op_completion_in_.front();
    async_tensor_complete(completion.async_id);
    tensor_async_op_completion_in_.pop();
  }
}

// Advance all Core-side asynchronous tensor transactions by one cycle.
// This models the TMA/TMEM background engine that updates TMEM-visible state
// before the later in-core pipeline stages observe it in the same global cycle.
void Core::advance_async_tensor_engine() {
  if (nullptr != tma_frontend_ && nullptr != tmem_system_) {
    return;
  }
  // The legacy Core-local async tensor engine is intentionally disabled once
  // the split TensorMemSystem/TensorExecuteSystem model is adopted. Keeping a
  // second executable timing path would make the model non-signoffable.
  std::abort();
}

void Core::compact_async_tmem_ops_fifo() {
  std::deque<uint32_t> compacted_fifo;
  for (auto async_id : async_tmem_ops_fifo_) {
    auto it = async_tensor_ops_.find(async_id);
    if (it == async_tensor_ops_.end() || it->second.completed) {
      continue;
    }
    compacted_fifo.push_back(async_id);
  }
  async_tmem_ops_fifo_.swap(compacted_fifo);
}

uint32_t Core::tmem_alloc(uint32_t col_span, uint32_t mma_desc_id) {
  advance_async_tensor_engine();

  // col_span 必须是 {16, 32, 64} 之一 (16 × 2^n)
  if (col_span != 16 && col_span != 32 && col_span != 64) {
    throw std::runtime_error(
      "TMEM_ALLOC: col_span=" + std::to_string(col_span)
      + " invalid, must be 16, 32, or 64");
  }

  // 如果绑定了 MMA descriptor，验证 col_span 并预构建 window layout plan
  MmaDescriptor mma_desc{};
  bool has_desc = (mma_desc_id != 0) && read_mma_descriptor(mma_desc_id, &mma_desc);

  if (has_desc) {
    // 构建 planner input — 每个 window 使用各自的精度
    TmemWindowPlannerInput planner_input{};
    planner_input.a_shape = {mma_desc.a_rows, mma_desc.a_cols};
    planner_input.b_shape = {mma_desc.b_rows, mma_desc.b_cols};
    planner_input.c_shape = {mma_desc.c_rows, mma_desc.c_cols};
    planner_input.d_shape = {mma_desc.c_rows, mma_desc.c_cols};
    planner_input.fmt_a = mma_desc.fmt_a;
    planner_input.fmt_b = mma_desc.fmt_b;
    planner_input.fmt_c = mma_desc.fmt_c;
    planner_input.fmt_d = mma_desc.fmt_d;
    planner_input.sparse_mode = mma_desc.sparse_mode;
    planner_input.allocation_col_span = col_span;

    auto min_col = TmemWindowPlanner::compute_min_col_span(planner_input);
    if (min_col == 0) {
      throw std::runtime_error(
        "TMEM_ALLOC: window shape + precision combination cannot fit in TMEM "
        "(exceeds 128 lines even at col_span=64)");
    }
    if (col_span < min_col) {
      throw std::runtime_error(
        "TMEM_ALLOC: col_span=" + std::to_string(col_span)
        + " too small, minimum " + std::to_string(min_col)
        + " required for this shape+precision");
    }
    planner_input.allocation_col_span = col_span;

    // 预构建 layout plan (A/B/C/D windows) 并持久化到 allocation 元数据
    TmemLayoutPlan layout{};
    std::string plan_reason;
    if (!TmemWindowPlanner::build_dense_plan(planner_input, &layout, &plan_reason)) {
      throw std::runtime_error(
        "TMEM_ALLOC: window plan failed — " + plan_reason);
    }

    // 分配 TMEM 列, 然后将 layout plan 存入 allocation 元数据
    uint32_t handle = 0;
    if (nullptr != tmem_system_) {
      handle = tmem_system_->alloc(col_span);
    } else {
      handle = tmem_.alloc(col_span);
    }
    if (handle != 0) {
      // 持久化 layout plan 和 mma_desc_id 到 allocation
      TmemAllocation* alloc = nullptr;
      if (lookup_tmem_allocation(handle, &alloc) && alloc != nullptr) {
        alloc->mma_desc_id = mma_desc_id;
        alloc->prebuilt_layout = layout;
      }
    }
    return handle;
  }

  // 无 MMA descriptor 绑定的后备路径
  uint32_t fallback_handle = 0;
  if (nullptr != tmem_system_) {
    fallback_handle = tmem_system_->alloc(col_span);
  } else {
    fallback_handle = tmem_.alloc(col_span);
  }
  if (fallback_handle == 0) {
    std::cerr << "TMEM alloc debug: col_span=" << col_span
              << " failed (payload_cols=" << kTmemPayloadCols << ")" << std::endl;
  }
  return fallback_handle;
}

bool Core::tmem_free(uint32_t handle) {
  advance_async_tensor_engine();
  if (nullptr != tmem_system_) {
    return tmem_system_->free(handle);
  }
  return tmem_.free(handle);
}

void Core::tmem_rel_permit() {
  if (nullptr != tmem_system_) {
    tmem_system_->seal_allocator();
    return;
  }
  tmem_.seal_allocator();
}

uint32_t Core::tma_load(uint32_t wid, uint32_t handle, uint32_t desc_id, uint32_t window_id) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  TmaDescriptor desc;
  if (!read_tma_descriptor(desc_id, &desc)) {
    return 0;
  }
  auto kind = static_cast<TcuPayloadKind>(desc.payload_kind);
  TmemAllocation* allocation = nullptr;
  if (lookup_tmem_allocation(handle, &allocation)) {
    if (nullptr != tmem_system_) {
      tmem_system_->set_meta_region(handle, desc.meta_tmem_base, desc.meta_col_span);
    } else {
      tmem_.set_meta_region(handle, desc.meta_tmem_base, desc.meta_col_span);
    }
      if (kind == TcuPayloadKind::SparseMeta) {
        if (nullptr != tmem_system_) {
          tmem_system_->set_meta_ready(handle, false);
        } else {
          tmem_.set_meta_ready(handle, false);
        }
      } else {
        if (nullptr != tmem_system_) {
          tmem_system_->set_payload_ready(handle, false);
        } else {
          tmem_.set_payload_ready(handle, false);
        }
    }
    if (kind != TcuPayloadKind::SparseMeta
     && desc.meta_addr != 0 && desc.meta_size_bytes != 0 && desc.meta_col_span != 0) {
      if (nullptr != tmem_system_) {
        tmem_system_->set_meta_ready(handle, false);
      } else {
        tmem_.set_meta_ready(handle, false);
      }
    }
  }
  ++perf_stats_.tma_load_count;
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmaLoad;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.window_id = window_id;
  op.issue_cycle = perf_stats_.cycles;
  op.first_service_cycle = perf_stats_.cycles + 1;
  async_tensor_ops_[async_id] = std::move(op);
  assert(nullptr != tma_frontend_);
  execute_cycle_tma_load_reserved_handles_.insert(handle);
  (void)tma_frontend_->issue_load(async_id, wid, wgid, handle, desc_id, window_id, perf_stats_.cycles);
  return async_id;
}

uint32_t Core::tma_store(uint32_t wid, uint32_t handle, uint32_t desc_id, uint32_t window_id) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  TmaDescriptor desc;
  if (!read_tma_descriptor(desc_id, &desc)) {
    return 0;
  }
  if (static_cast<TcuPayloadKind>(desc.payload_kind) == TcuPayloadKind::SparseMeta) {
    if (nullptr != tmem_system_) {
      tmem_system_->set_meta_ready(handle, false);
    } else {
      tmem_.set_meta_ready(handle, false);
    }
  } else {
    if (nullptr != tmem_system_) {
      tmem_system_->set_payload_ready(handle, false);
    } else {
      tmem_.set_payload_ready(handle, false);
    }
  }
  ++perf_stats_.tma_store_count;
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmaStore;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.window_id = window_id;
  op.issue_cycle = perf_stats_.cycles;
  op.first_service_cycle = perf_stats_.cycles + 1;
  async_tensor_ops_[async_id] = std::move(op);
  assert(nullptr != tma_frontend_);
  execute_cycle_tma_store_or_shift_reserved_handles_.insert(handle);
  (void)tma_frontend_->issue_store(async_id, wid, wgid, handle, desc_id, window_id, perf_stats_.cycles);
  return async_id;
}

uint32_t Core::tmem_shift(uint32_t wid, uint32_t handle, uint32_t window_id, uint32_t refill_descriptor_id) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  TmemAllocation* allocation = nullptr;
  if (!lookup_tmem_allocation(handle, &allocation)) {
    return 0;
  }
  if (nullptr != tmem_system_) {
    tmem_system_->set_payload_ready(handle, false);
  } else {
    tmem_.set_payload_ready(handle, false);
  }
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::TmemShift;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.window_id = window_id;
  op.refill_descriptor_id = refill_descriptor_id;
  op.issue_cycle = perf_stats_.cycles;
  op.first_service_cycle = perf_stats_.cycles + 1;
  async_tensor_ops_[async_id] = std::move(op);
  assert(nullptr != tmem_system_);
  execute_cycle_tma_store_or_shift_reserved_handles_.insert(handle);
  (void)tmem_system_->issue_shift(async_id, wid, wgid, handle, window_id, refill_descriptor_id, perf_stats_.cycles);
  return async_id;
}

uint32_t Core::mma_load_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::MmaLoad;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.issue_cycle = perf_stats_.cycles;
  op.first_service_cycle = std::numeric_limits<uint64_t>::max();
  async_tensor_ops_[async_id] = std::move(op);
  return async_id;
}

uint32_t Core::mma_store_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto async_id = next_async_id_++;
  AsyncTensorOp op{};
  op.async_id = async_id;
  op.type = AsyncTensorOpType::MmaStore;
  op.wid = wid;
  op.wgid = wgid;
  op.handle = handle;
  op.descriptor_id = desc_id;
  op.issue_cycle = perf_stats_.cycles;
  op.first_service_cycle = std::numeric_limits<uint64_t>::max();
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
  op.first_service_cycle = std::numeric_limits<uint64_t>::max();
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
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
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
    if (op.wgid != wgid || op.completed || op.committed) {
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
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  if (mode == TcuFenceMode::Before) {
    // In-order warp issue already preserves relative ordering before a following thread sync.
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

bool Core::tc_wait(uint32_t wid) {
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto group_mask = warpgroup_mask(wgid);
  bool pending = false;
  for (const auto& entry : async_tensor_ops_) {
    const auto& op = entry.second;
    if (op.wgid != wgid || op.completed) {
      continue;
    }
    switch (op.type) {
    case AsyncTensorOpType::MmaLoad:
    case AsyncTensorOpType::MmaStore:
    case AsyncTensorOpType::Wmma:
      async_tensor_waiters_[op.async_id] |= group_mask;
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
  advance_async_tensor_engine();
  auto wgid = arch_.warpgroup_id(wid);
  auto group_mask = warpgroup_mask(wgid);
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
      for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
        if (group_mask.test(gwid)) {
          mbarrier_wait_targets_.at(gwid).erase(barrier_id);
        }
      }
      return true;
    }
    barrier.waiters_bitmap |= group_mask;
    ++perf_stats_.stall_wait_barrier;
    return false;
  }

  if (barrier.phase_done) {
    return true;
  }

  for (uint32_t gwid = 0; gwid < arch_.num_warps(); ++gwid) {
    if (group_mask.test(gwid)) {
      mbarrier_wait_targets_.at(gwid)[barrier_id] = barrier.phase + 1;
    }
  }
  barrier.waiters_bitmap |= group_mask;
  ++perf_stats_.stall_wait_barrier;
  return false;
}

bool Core::tma_wait(uint32_t wid, uint32_t async_id) {
  advance_async_tensor_engine();
  auto group_mask = warpgroup_mask(arch_.warpgroup_id(wid));
  auto it = async_tensor_ops_.find(async_id);
  if (it == async_tensor_ops_.end()) {
    return true;
  }
  if (!it->second.completed) {
    async_tensor_waiters_[async_id] |= group_mask;
    ++perf_stats_.stall_wait_barrier;
    return false;
  }
  async_tensor_waiters_.erase(async_id);
  async_tensor_ops_.erase(it);
  return true;
}

bool Core::tmem_read_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, TmemPacket* out) {
  advance_async_tensor_engine();
  if (nullptr != tmem_system_) {
    return tmem_system_->read_window_packet(handle, window_id, packet_idx, out);
  }
  return tmem_.read_window_packet(handle, window_id, packet_idx, out);
}

bool Core::tmem_write_window_packet(uint32_t handle, uint32_t window_id, uint32_t packet_idx, const TmemPacket& in) {
  advance_async_tensor_engine();
  if (nullptr != tmem_system_) {
    return tmem_system_->write_window_packet(handle, window_id, packet_idx, in);
  }
  return tmem_.write_window_packet(handle, window_id, packet_idx, in);
}

#endif
