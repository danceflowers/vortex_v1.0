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

#include "tensor_unit.h"

#include <algorithm>
#include <deque>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>
#include <rvfloats.h>

#include "core.h"
#include "tensor_cfg.h"
#include "tmem_window_planner.h"
#include "open_tensorcore/tensor_compute/amem.h"
#include "open_tensorcore/tensor_compute/bmem.h"
#include "open_tensorcore/tensor_compute/cmem.h"
#include "open_tensorcore/tensor_compute/meta_mem.h"
#include "open_tensorcore/tensor_compute/sparse_select.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"
#include "open_tensorcore/tensor_control/tc_decode.h"

using namespace vortex;

namespace vt = vortex::tensor;

namespace {

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

bool use_open_tensorcore(uint32_t fmt_a, uint32_t fmt_b, uint32_t fmt_c) {
  if constexpr (NUM_THREADS != 32) {
    return false;
  }
  if (fmt_c != vt::fp16::id && fmt_c != vt::fp32::id)
    return false;
  bool supported_a = (fmt_a == vt::fp8::id || fmt_a == vt::fp16::id);
  bool supported_b = (fmt_b == vt::fp8::id || fmt_b == vt::fp16::id);
  return supported_a && supported_b;
}

PrecisionType map_out_precision(uint32_t fmt_out) {
  switch (fmt_out) {
  case vt::fp8::id:
    return PREC_FP8_E4M3;
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp32::id:
    return PREC_FP32;
  default:
    return PREC_FP16;
  }
}

uint32_t a_packet_count(uint32_t fmt_a) {
  return (fmt_a == vt::fp16::id) ? 8 : 4;
}

uint32_t b_packet_count(uint32_t fmt_b) {
  return (fmt_b == vt::fp16::id) ? 8 : 4;
}

uint32_t meta_packet_count(uint32_t a_sparse_mode) {
  return (a_sparse_mode == vt::sparse_none) ? 0 : MetaMem::packet_count();
}

uint32_t meta_shadow_window_id(uint32_t window_id) {
  return window_id | 0x80000000u;
}

static constexpr uint32_t kUnsetPayloadFmt = 0xffffffffu;

uint32_t c_load_packet_count(uint32_t fmt_c) {
  switch (fmt_c) {
  case vt::fp8::id:
    return 4;
  case vt::fp16::id:
    return 8;
  case vt::fp32::id:
    return 16;
  default:
    return 0;
  }
}

uint32_t d_store_packet_count(uint32_t fmt_d) {
  switch (fmt_d) {
  case vt::fp8::id:
    return 4;
  case vt::fp16::id:
    return 8;
  case vt::fp32::id:
    return 16;
  default:
    return 0;
  }
}

bool window_matches_load_target(TcuTarget load_target, TmemWindowTarget window_target) {
  switch (load_target) {
  case TcuTarget::A:
    return window_target == TmemWindowTarget::A;
  case TcuTarget::B:
    return window_target == TmemWindowTarget::B;
  case TcuTarget::C:
    return window_target == TmemWindowTarget::C || window_target == TmemWindowTarget::D;
  default:
    return false;
  }
}

uint32_t target_packet_offset(const IntrTcuArgs& args) {
  auto fmt_a = args.fmt_a;
  auto fmt_b = args.fmt_b;
  auto a_packets = a_packet_count(fmt_a);
  auto b_packets = b_packet_count(fmt_b);
  switch (args.target) {
  case TcuTarget::A:
    return 0;
  case TcuTarget::B:
    return a_packets;
  case TcuTarget::C:
    return a_packets + b_packets;
  default:
    return 0;
  }
}

template <typename PacketT>
std::vector<PacketT> copy_packets(const std::vector<Core::TmemPacket>& packets) {
  std::vector<PacketT> out(packets.size());
  for (size_t i = 0; i < packets.size(); ++i) {
    std::copy_n(packets.at(i).bytes.begin(), packets.at(i).bytes.size(), out.at(i).begin());
  }
  return out;
}

void configure_open_tensorcore_precision(uint32_t fmt_c) {
  g_cfg.precisions.clear();
  g_cfg.out_precisions.clear();
  g_cfg.precisions.push_back(PREC_FP9);
  g_cfg.out_precisions.push_back(map_out_precision(fmt_c));
}

inline bool sparse_mode_is_2_4(uint32_t sparse_mode) {
  return sparse_mode == 1;
}

inline bool sparse_mode_is_1_4(uint32_t sparse_mode) {
  return sparse_mode == 2;
}

// Run one 8-input dot product through the same tc_mul_add arithmetic used by
// TensorCoreTop, but without occupying the global TensorCoreTop pipeline. This
// keeps the sparse path row-dependent: every output row may select a different
// set of B K-rows according to its own metadata.
uint32_t run_sparse_dot_fp22(const uint16_t a_vec[8], const uint16_t b_vec[8]) {
  tc_mul_add dot;
  dot.reset();
  for (uint32_t k = 0; k < 8; ++k) {
    dot.mul_add_input.a_in[k] = a_vec[k];
    dot.mul_add_input.b_in[k] = b_vec[k];
  }
  dot.mul_add_input.c_in = 0;
  dot.mul_add_input.prec = PREC_FP9;
  dot.mul_add_input.input_valid = true;
  dot.mul_add_input.meta = TensorCoreMeta{};

  for (uint32_t cycle = 0; cycle < 32; ++cycle) {
    dot.tick(true, g_cfg);
    dot.mul_add_input.input_valid = false;
    dot.mul_add_input.meta = TensorCoreMeta{};
    if (dot.out_valid()) {
      return dot.out_fp22();
    }
  }
  std::abort();
}

} // namespace

class TensorUnit::Impl {
public:
  static constexpr uint32_t kNumOperandSlots = 2;
  static constexpr uint32_t kWmmaPrimitiveCount = 8;
  static constexpr uint32_t kSubtilesPerTile = 4;
  static constexpr uint32_t kPrimitiveDim = 8;
  static constexpr uint32_t kAmemWriteBeatsPerCycle = 1;
  static constexpr uint32_t kBmemWriteBeatsPerCycle = 1;
  static constexpr uint32_t kCmemWriteBeatsPerCycle = 1;
  static constexpr uint32_t kCmemReadBeatsPerCycle = 1;
  static constexpr uint32_t kMetaWriteBeatsPerCycle = 1;
  struct ASlotState {
    uint32_t owner_wgid = 0;
    uint32_t descriptor = 0xffffffffu;
    uint32_t fmt_a = 0;
    uint32_t a_sparse_mode = 0;
    uint32_t wmma_async_id = 0;
    bool transpose_a = false;
    bool valid = false;
    bool busy = false;
    bool a_ready = false;
    bool a_pending = false;
    bool wmma_pending = false;

    void reset() {
      owner_wgid = 0;
      descriptor = 0xffffffffu;
      fmt_a = 0;
      a_sparse_mode = 0;
      wmma_async_id = 0;
      transpose_a = false;
      valid = false;
      busy = false;
      a_ready = false;
      a_pending = false;
      wmma_pending = false;
    }
  };

  struct BSlotState {
    uint32_t owner_wgid = 0;
    uint32_t descriptor = 0xffffffffu;
    uint32_t fmt_b = 0;
    uint32_t wmma_async_id = 0;
    bool transpose_b = false;
    bool valid = false;
    bool busy = false;
    bool b_ready = false;
    bool b_pending = false;
    bool wmma_pending = false;

    void reset() {
      owner_wgid = 0;
      descriptor = 0xffffffffu;
      fmt_b = 0;
      wmma_async_id = 0;
      transpose_b = false;
      valid = false;
      busy = false;
      b_ready = false;
      b_pending = false;
      wmma_pending = false;
    }
  };

  struct CSlotState {
    uint32_t owner_wgid = 0;
    uint32_t descriptor = 0xffffffffu;
    uint32_t fmt_c = 0;
    uint32_t fmt_d = 0;
    uint32_t store_async_id = 0;
    uint32_t c_wmma_inflight = 0;
    bool valid = false;
    bool busy = false;
    bool c_ready = false;
    bool c_pending = false;
    bool cmem_final_valid = false;
    bool c_dirty = false;
    bool store_pending = false;

    void reset() {
      owner_wgid = 0;
      descriptor = 0xffffffffu;
      fmt_c = 0;
      fmt_d = 0;
      store_async_id = 0;
      c_wmma_inflight = 0;
      valid = false;
      busy = false;
      c_ready = false;
      c_pending = false;
      cmem_final_valid = false;
      c_dirty = false;
      store_pending = false;
    }
  };

  struct PendingWmmaJob {
    uint32_t wgid = 0;
    uint32_t a_slot_id = 0;
    uint32_t b_slot_id = 0;
    uint32_t c_slot_id = 0;
    uint32_t fmt_a = 0;
    uint32_t fmt_b = 0;
    uint32_t fmt_c = 0;
    uint32_t a_sparse_mode = 0;
    uint32_t async_id = 0;
    uint32_t next_uop = 0;
  };

  struct MemUop {
    enum class Kind : uint8_t {
      FillA = 0,
      FillB,
      FillC,
      StoreC,
    };

    Kind kind = Kind::FillA;
    uint32_t wgid = 0;
    uint32_t slot_id = 0;
    uint32_t handle = 0;
    uint32_t window_id = 0;
    uint32_t payload_fmt = kUnsetPayloadFmt;
    uint32_t tile_idx = 0;
    uint32_t async_id = 0;
    bool separate_handle = false;
    uint32_t remaining_tmem_reads = 0;
    uint32_t remaining_tmem_writes = 0;
    uint32_t remaining_amem_writes = 0;
    uint32_t remaining_bmem_writes = 0;
    uint32_t remaining_cmem_writes = 0;
    uint32_t remaining_cmem_reads = 0;
    uint32_t remaining_meta_writes = 0;
    uint32_t next_payload_packet_idx = 0;
    uint32_t next_meta_packet_idx = 0;
    std::vector<Core::TmemPacket> staged_payload_packets;
    std::vector<Core::TmemPacket> staged_meta_packets;
    uint32_t next_store_packet_idx = 0;
    uint32_t staged_store_packet_cursor = 0;
    std::vector<Core::TmemPacket> staged_store_packets;
    std::array<std::array<uint32_t, kPrimitiveDim>, kPrimitiveDim> staged_store_left_subtile = {};
    bool staged_store_left_valid = false;
  };

  enum class NoWmmaReadyReason : uint8_t {
    JobBuilderEmpty = 0,
    WaitingForMmaLoad,
    WaitingForHandleAlloc,
    WaitingForSlotRelease,
  };

  enum class SlotReleaseReason : uint8_t {
    None = 0,
    CWmmaInflightDrain,
    CAccumLiveOnly,
    CDirtyFlushOnly,
    CStorePending,
    AbWmmaPendingClear,
  };

  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_() {
    reset();
  }

  void reset() {
    perf_stats_ = PerfStats();
    amem_.reset();
    bmem_.reset();
    cmem_.reset();
    metamem_.reset();
    for (auto& slot : a_slots_) {
      slot.reset();
    }
    for (auto& slot : b_slots_) {
      slot.reset();
    }
    for (auto& slot : c_slots_) {
      slot.reset();
    }
    mem_ops_.clear();
    pending_mem_ops_.clear();
    pending_wmma_uops_.clear();
    pending_wmma_jobs_.clear();
    active_wmma_job_ = {};
    active_wmma_job_valid_ = false;
    tensorcore_.reset();
    tensorcore_fmt_out_ = vt::fp16::id;
    mem_port_cycle_ = std::numeric_limits<uint64_t>::max();
    amem_write_budget_ = 0;
    bmem_write_budget_ = 0;
    cmem_write_budget_ = 0;
    cmem_read_budget_ = 0;
    meta_write_budget_ = 0;
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.front();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      int delay = 1;
      switch (tcu_type) {
      case TcuType::TCU_LD:
      case TcuType::TCU_ST:
        delay = 2;
        break;
      case TcuType::TCU_MMA:
        delay = 4;
        break;
      default:
        delay = 1;
        break;
      }
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      input.pop();
    }

    service_mem_ops();
    dispatch_compute_uop();
    sample_pending_wmma_depth();
    if (tensorcore_.active()) {
      ++perf_stats_.tc_active_cycles;
    }
    tick_tensorcore();
  }

  void mma_load(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    if (trace_data) {
      trace_data->rd_write = false;
      trace_data->retry = false;
    }
    if (!args.macro_op) {
      std::abort();
    }
    enqueue_async_mma_load(wid, handle, args, trace_data);
  }

  void mma_store(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    if (trace_data) {
      trace_data->rd_write = false;
      trace_data->retry = false;
    }
    if (!args.macro_op) {
      std::abort();
    }
    enqueue_async_mma_store(wid, handle, args, trace_data);
  }

  void wmma(uint32_t wid,
            IntrTcuArgs args,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    auto fmt_a = args.fmt_a;
    auto fmt_b = args.fmt_b;
    (void)rs1_data;
    (void)rs2_data;
    (void)rs3_data;
    (void)rd_data;
    if (trace_data) {
      trace_data->rd_write = false;
      trace_data->retry = false;
    }

    if (!args.macro_op) {
      std::abort();
    }
    enqueue_async_wmma(wid, args, fmt_a, fmt_b, args.fmt_c, trace_data);
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

  uint32_t scheduler_score(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const {
    if (!args.macro_op) {
      return 1;
    }
    auto wgid = arch_.warpgroup_id(wid);

    switch (tcu_type) {
    case TcuType::TCU_MMA: {
      auto fmt_a = args.fmt_a;
      auto fmt_b = args.fmt_b;
      if (!use_open_tensorcore(fmt_a, fmt_b, args.fmt_c)) {
        return 0;
      }
      if (args.a_slot_id >= a_slots_.size()
       || args.b_slot_id >= b_slots_.size()
       || args.c_slot_id >= c_slots_.size()) {
        return 0;
      }
      const auto& a_slot = a_slots_.at(args.a_slot_id);
      const auto& b_slot = b_slots_.at(args.b_slot_id);
      const auto& c_slot = c_slots_.at(args.c_slot_id);
      if (!a_slot.valid || a_slot.owner_wgid != wgid || a_slot.descriptor != args.descriptor) {
        return 0;
      }
      if (!b_slot.valid || b_slot.owner_wgid != wgid || b_slot.descriptor != args.descriptor) {
        return 0;
      }
      if (!c_slot.valid || c_slot.owner_wgid != wgid || c_slot.descriptor != args.descriptor) {
        return 0;
      }
      return (a_slot.a_ready && !a_slot.wmma_pending
           && b_slot.b_ready && !b_slot.wmma_pending
           && c_slot.c_ready && !c_slot.store_pending) ? 5 : 0;
    }
    case TcuType::TCU_LD:
      return (core_->tmem_handle_ready_for_mma_load(args.runtime_handle, args.target, args.a_sparse_mode)
           && can_issue_mma_load(wid, args)) ? 4 : 0;
    case TcuType::TCU_ST: {
      if (args.slot_id >= c_slots_.size()) {
        return 0;
      }
      if (!core_->tmem_handle_ready_for_mma_store(args.runtime_handle)) {
        return 0;
      }
      const auto& slot = c_slots_.at(args.slot_id);
      if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        return 0;
      }
      bool ready = !slot.store_pending && slot.cmem_final_valid;
      return ready ? 3 : 0;
    }
    default:
      return 1;
    }
  }

  TensorUnit::IssueBlockReason classify_issue_block(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const {
    if (!args.macro_op) {
      return TensorUnit::IssueBlockReason::None;
    }
    auto wgid = arch_.warpgroup_id(wid);

    switch (tcu_type) {
    case TcuType::TCU_MMA: {
      if (args.a_slot_id >= a_slots_.size()
       || args.b_slot_id >= b_slots_.size()
       || args.c_slot_id >= c_slots_.size()) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      const auto& a_slot = a_slots_.at(args.a_slot_id);
      const auto& b_slot = b_slots_.at(args.b_slot_id);
      const auto& c_slot = c_slots_.at(args.c_slot_id);
      if (!a_slot.valid || a_slot.owner_wgid != wgid || a_slot.descriptor != args.descriptor
       || !b_slot.valid || b_slot.owner_wgid != wgid || b_slot.descriptor != args.descriptor
       || !c_slot.valid || c_slot.owner_wgid != wgid || c_slot.descriptor != args.descriptor) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      if (!a_slot.a_ready) {
        return TensorUnit::IssueBlockReason::ANotReady;
      }
      if (!b_slot.b_ready) {
        return TensorUnit::IssueBlockReason::BNotReady;
      }
      if (!c_slot.c_ready) {
        return TensorUnit::IssueBlockReason::CNotReady;
      }
      if (a_slot.wmma_pending || b_slot.wmma_pending || c_slot.store_pending) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      return TensorUnit::IssueBlockReason::None;
    }
    case TcuType::TCU_LD: {
      auto handle_reason = core_->tmem_handle_load_block_reason(args.runtime_handle, args.target, args.a_sparse_mode);
      switch (handle_reason) {
      case Core::TmemTaddrBlockReason::MetaNotReady:
        return TensorUnit::IssueBlockReason::AMetaNotReady;
      case Core::TmemTaddrBlockReason::BusyTmemShift:
        return TensorUnit::IssueBlockReason::HandleBusyDueToTmaStoreOrShift;
      case Core::TmemTaddrBlockReason::PayloadNotReady:
        return TensorUnit::IssueBlockReason::MmaLoadHandleNotReady;
      case Core::TmemTaddrBlockReason::Invalid:
        return TensorUnit::IssueBlockReason::HandleReuse;
      case Core::TmemTaddrBlockReason::None:
        break;
      }
      if (!can_issue_mma_load(wid, args)) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      return TensorUnit::IssueBlockReason::None;
    }
    case TcuType::TCU_ST: {
      auto handle_reason = core_->tmem_handle_store_block_reason(args.runtime_handle);
      if (handle_reason == Core::TmemTaddrBlockReason::BusyTmemShift) {
        return TensorUnit::IssueBlockReason::HandleBusyDueToTmaStoreOrShift;
      }
      if (handle_reason == Core::TmemTaddrBlockReason::Invalid) {
        return TensorUnit::IssueBlockReason::HandleReuse;
      }
      if (args.slot_id >= c_slots_.size()) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      const auto& slot = c_slots_.at(args.slot_id);
      bool ready = slot.valid && slot.owner_wgid == wgid && slot.descriptor == args.descriptor
                && !slot.store_pending
                && slot.cmem_final_valid;
      return ready ? TensorUnit::IssueBlockReason::None
                   : TensorUnit::IssueBlockReason::SlotBusy;
    }
    default:
      return TensorUnit::IssueBlockReason::None;
    }
  }

  void record_issue_stall(TensorUnit::IssueBlockReason reason) {
    switch (reason) {
    case TensorUnit::IssueBlockReason::ANotReady:
      ++perf_stats_.stall_a_not_ready;
      break;
    case TensorUnit::IssueBlockReason::BNotReady:
      ++perf_stats_.stall_b_not_ready;
      break;
    case TensorUnit::IssueBlockReason::CNotReady:
      ++perf_stats_.stall_c_not_ready;
      break;
    case TensorUnit::IssueBlockReason::TcBusy:
      ++perf_stats_.stall_tc_busy;
      break;
    case TensorUnit::IssueBlockReason::NoTensorInstrCandidate:
      ++perf_stats_.stall_no_tensor_instr_candidate;
      break;
    case TensorUnit::IssueBlockReason::MmaLoadHandleNotReady:
      ++perf_stats_.stall_mma_load_handle_not_ready;
      break;
    case TensorUnit::IssueBlockReason::HandleBusyDueToTmaLoad:
      ++perf_stats_.stall_handle_reuse;
      ++perf_stats_.stall_handle_busy_due_to_tma_load;
      break;
    case TensorUnit::IssueBlockReason::HandleBusyDueToTmaStoreOrShift:
      ++perf_stats_.stall_handle_reuse;
      ++perf_stats_.stall_handle_busy_due_to_tma_store_or_shift;
      break;
    case TensorUnit::IssueBlockReason::HandleReuse:
      ++perf_stats_.stall_handle_reuse;
      break;
    case TensorUnit::IssueBlockReason::SlotBusy:
      ++perf_stats_.stall_slot_busy;
      break;
    case TensorUnit::IssueBlockReason::AMetaNotReady:
      ++perf_stats_.stall_a_meta_not_ready;
      ++perf_stats_.stall_mma_load_handle_not_ready;
      break;
    case TensorUnit::IssueBlockReason::None:
    default:
      break;
    }
  }

private:
  static uint32_t add_fp22_raw(uint32_t a, uint32_t b) {
    auto s1 = fadd_s1(a, b, 8, 14, 14, g_cfg.rm);
    return fadd_s2(s1, 8, 14);
  }

  static bool a_slot_has_pending_work(const ASlotState& slot) {
    return slot.a_pending || slot.wmma_pending;
  }

  static bool b_slot_has_pending_work(const BSlotState& slot) {
    return slot.b_pending || slot.wmma_pending;
  }

  static bool c_slot_has_pending_work(const CSlotState& slot) {
    return slot.c_pending || slot.store_pending || slot.c_wmma_inflight != 0;
  }

  static bool a_slot_can_rebind(const ASlotState& slot) {
    return !slot.valid || !a_slot_has_pending_work(slot);
  }

  static bool b_slot_can_rebind(const BSlotState& slot) {
    return !slot.valid || !b_slot_has_pending_work(slot);
  }

  static bool c_slot_can_rebind(const CSlotState& slot) {
    return !slot.valid || (!c_slot_has_pending_work(slot) && !slot.c_dirty);
  }

  static bool a_slot_target_available(const ASlotState& slot) {
    return !slot.a_pending && !slot.wmma_pending;
  }

  static bool b_slot_target_available(const BSlotState& slot) {
    return !slot.b_pending && !slot.wmma_pending;
  }

  static bool c_slot_target_available(const CSlotState& slot) {
    return !slot.c_pending && !slot.store_pending
        && !slot.c_dirty && slot.c_wmma_inflight == 0;
  }

  void reset_mem_port_budgets() {
    mem_port_cycle_ = core_->current_cycle();
    amem_write_budget_ = kAmemWriteBeatsPerCycle;
    bmem_write_budget_ = kBmemWriteBeatsPerCycle;
    cmem_write_budget_ = kCmemWriteBeatsPerCycle;
    cmem_read_budget_ = kCmemReadBeatsPerCycle;
    meta_write_budget_ = kMetaWriteBeatsPerCycle;
  }

  void ensure_mem_port_budgets() {
    if (mem_port_cycle_ != core_->current_cycle()) {
      reset_mem_port_budgets();
    }
  }

  static bool mem_op_complete(const MemUop& op) {
    return op.remaining_tmem_reads == 0
        && op.remaining_tmem_writes == 0
        && op.remaining_amem_writes == 0
        && op.remaining_bmem_writes == 0
        && op.remaining_cmem_writes == 0
        && op.remaining_cmem_reads == 0
        && op.remaining_meta_writes == 0
        && op.staged_store_packets.empty()
        && !op.staged_store_left_valid;
  }

  uint32_t current_pending_wmma_depth() const {
    return pending_wmma_jobs_.size() + (active_wmma_job_valid_ ? 1u : 0u);
  }

  void sample_pending_wmma_depth() {
    auto depth = current_pending_wmma_depth();
    perf_stats_.pending_wmma_jobs_max = std::max<uint64_t>(perf_stats_.pending_wmma_jobs_max, depth);
    switch (depth) {
    case 0:
      ++perf_stats_.pending_wmma_depth_cycles_0;
      break;
    case 1:
      ++perf_stats_.pending_wmma_depth_cycles_1;
      break;
    case 2:
      ++perf_stats_.pending_wmma_depth_cycles_2;
      break;
    default:
      ++perf_stats_.pending_wmma_depth_cycles_3plus;
      break;
    }
  }

  bool has_inflight_mma_load_build() const {
    for (const auto& op : mem_ops_) {
      switch (op.kind) {
      case MemUop::Kind::FillA:
      case MemUop::Kind::FillB:
      case MemUop::Kind::FillC:
        return true;
      default:
        break;
      }
    }
    for (const auto& slot : a_slots_) {
      if (slot.a_pending) {
        return true;
      }
    }
    for (const auto& slot : b_slots_) {
      if (slot.b_pending) {
        return true;
      }
    }
    for (const auto& slot : c_slots_) {
      if (slot.c_pending) {
        return true;
      }
    }
    return false;
  }

  SlotReleaseReason classify_slot_release_reason() const {
    for (const auto& slot : a_slots_) {
      if (!slot.valid) {
        continue;
      }
      if (slot.wmma_pending) {
        return SlotReleaseReason::AbWmmaPendingClear;
      }
    }
    for (const auto& slot : b_slots_) {
      if (!slot.valid) {
        continue;
      }
      if (slot.wmma_pending) {
        return SlotReleaseReason::AbWmmaPendingClear;
      }
    }
    for (const auto& slot : c_slots_) {
      if (!slot.valid) {
        continue;
      }
      if (slot.c_wmma_inflight != 0) {
        return SlotReleaseReason::CWmmaInflightDrain;
      }
      if (slot.store_pending) {
        return SlotReleaseReason::CStorePending;
      }
      if (slot.c_dirty) {
        return SlotReleaseReason::CDirtyFlushOnly;
      }
    }
    return SlotReleaseReason::None;
  }

  NoWmmaReadyReason classify_no_wmma_ready_reason() const {
    if (has_inflight_mma_load_build()) {
      return NoWmmaReadyReason::WaitingForMmaLoad;
    }
    if (core_->has_inflight_tma_handle_activity()) {
      return NoWmmaReadyReason::WaitingForHandleAlloc;
    }
    if (classify_slot_release_reason() != SlotReleaseReason::None) {
      return NoWmmaReadyReason::WaitingForSlotRelease;
    }
    return NoWmmaReadyReason::JobBuilderEmpty;
  }

  void record_no_wmma_ready_stall() {
    ++perf_stats_.stall_no_wmma_job_ready;
    switch (classify_no_wmma_ready_reason()) {
    case NoWmmaReadyReason::JobBuilderEmpty:
      ++perf_stats_.stall_no_wmma_job_builder_empty;
      break;
    case NoWmmaReadyReason::WaitingForMmaLoad:
      ++perf_stats_.stall_no_wmma_waiting_for_mma_load;
      break;
    case NoWmmaReadyReason::WaitingForHandleAlloc:
      ++perf_stats_.stall_no_wmma_waiting_for_handle_alloc;
      break;
    case NoWmmaReadyReason::WaitingForSlotRelease:
      ++perf_stats_.stall_no_wmma_waiting_for_slot_release;
      switch (classify_slot_release_reason()) {
      case SlotReleaseReason::CWmmaInflightDrain:
        ++perf_stats_.stall_no_wmma_waiting_for_c_wmma_inflight_drain;
        break;
      case SlotReleaseReason::CAccumLiveOnly:
        ++perf_stats_.stall_no_wmma_waiting_for_accum_live_only;
        break;
      case SlotReleaseReason::CDirtyFlushOnly:
        ++perf_stats_.stall_no_wmma_waiting_for_dirty_flush_only;
        break;
      case SlotReleaseReason::CStorePending:
        ++perf_stats_.stall_no_wmma_waiting_for_store_pending;
        break;
      case SlotReleaseReason::AbWmmaPendingClear:
        ++perf_stats_.stall_no_wmma_waiting_for_ab_wmma_pending_clear;
        break;
      case SlotReleaseReason::None:
        break;
      }
      break;
    }
  }

  bool can_issue_mma_load(uint32_t wid, const IntrTcuArgs& args) const {
    auto wgid = arch_.warpgroup_id(wid);
    switch (args.target) {
    case TcuTarget::A: {
      if (args.slot_id >= a_slots_.size()) {
        return false;
      }
      const auto& slot = a_slots_.at(args.slot_id);
      if (!slot.valid) {
        return true;
      }
      if (slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        return a_slot_can_rebind(slot);
      }
      return a_slot_target_available(slot);
    }
    case TcuTarget::B: {
      if (args.slot_id >= b_slots_.size()) {
        return false;
      }
      const auto& slot = b_slots_.at(args.slot_id);
      if (!slot.valid) {
        return true;
      }
      if (slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        return b_slot_can_rebind(slot);
      }
      return b_slot_target_available(slot);
    }
    case TcuTarget::C: {
      if (args.slot_id >= c_slots_.size()) {
        return false;
      }
      const auto& slot = c_slots_.at(args.slot_id);
      if (!slot.valid) {
        return true;
      }
      if (slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        return c_slot_can_rebind(slot);
      }
      return c_slot_target_available(slot);
    }
    default:
      return false;
    }
  }

  void clear_a_slot_storage(uint32_t slot_id) {
    amem_.clear_slot(slot_id);
    metamem_.clear_slot(slot_id);
  }

  void clear_b_slot_storage(uint32_t slot_id) {
    bmem_.clear_slot(slot_id);
  }

  void clear_c_slot_storage(uint32_t slot_id) {
    cmem_.clear_slot(slot_id);
  }

  void init_a_slot_for_descriptor(uint32_t slot_id, ASlotState& slot, uint32_t wgid, const IntrTcuArgs& args) {
    clear_a_slot_storage(slot_id);
    slot.reset();
    slot.valid = true;
    slot.owner_wgid = wgid;
    slot.descriptor = args.descriptor;
    slot.fmt_a = args.fmt_a;
    slot.a_sparse_mode = args.a_sparse_mode;
    slot.transpose_a = args.transpose_a != 0;
  }

  void init_b_slot_for_descriptor(uint32_t slot_id, BSlotState& slot, uint32_t wgid, const IntrTcuArgs& args) {
    clear_b_slot_storage(slot_id);
    slot.reset();
    slot.valid = true;
    slot.owner_wgid = wgid;
    slot.descriptor = args.descriptor;
    slot.fmt_b = args.fmt_b;
    slot.transpose_b = args.transpose_b != 0;
  }

  void init_c_slot_for_descriptor(uint32_t slot_id, CSlotState& slot, uint32_t wgid, const IntrTcuArgs& args) {
    clear_c_slot_storage(slot_id);
    slot.reset();
    slot.valid = true;
    slot.owner_wgid = wgid;
    slot.descriptor = args.descriptor;
    slot.fmt_c = args.fmt_c;
    slot.fmt_d = args.fmt_d;
  }

  static void mark_a_pending(ASlotState& slot, bool pending) {
    slot.a_pending = pending;
    if (pending) {
      slot.a_ready = false;
    }
    slot.busy = a_slot_has_pending_work(slot);
  }

  static void mark_b_pending(BSlotState& slot, bool pending) {
    slot.b_pending = pending;
    if (pending) {
      slot.b_ready = false;
    }
    slot.busy = b_slot_has_pending_work(slot);
  }

  static void mark_c_pending(CSlotState& slot, bool pending) {
    slot.c_pending = pending;
    if (pending) {
      slot.c_ready = false;
      slot.cmem_final_valid = false;
    }
    slot.busy = c_slot_has_pending_work(slot);
  }

  static void mark_a_ready(ASlotState& slot) {
    slot.a_pending = false;
    slot.a_ready = true;
    slot.busy = a_slot_has_pending_work(slot);
  }

  static void mark_b_ready(BSlotState& slot) {
    slot.b_pending = false;
    slot.b_ready = true;
    slot.busy = b_slot_has_pending_work(slot);
  }

  static void mark_c_ready(CSlotState& slot) {
    slot.c_pending = false;
    slot.c_ready = true;
    slot.cmem_final_valid = true;
    slot.c_dirty = false;
    slot.c_wmma_inflight = 0;
    slot.busy = c_slot_has_pending_work(slot);
  }

  static TcuTarget mem_uop_target(MemUop::Kind kind) {
    switch (kind) {
    case MemUop::Kind::FillA:
      return TcuTarget::A;
    case MemUop::Kind::FillB:
      return TcuTarget::B;
    case MemUop::Kind::FillC:
    case MemUop::Kind::StoreC:
      return TcuTarget::C;
    default:
      std::abort();
    }
  }

  uint32_t fill_packet_offset(const MemUop& uop) const {
    if (uop.separate_handle) {
      return 0;
    }
    IntrTcuArgs args{};
    args.fmt_a = a_slots_.at(uop.slot_id).fmt_a;
    args.fmt_b = b_slots_.at(uop.slot_id).fmt_b;
    args.fmt_c = c_slots_.at(uop.slot_id).fmt_c;
    args.a_sparse_mode = a_slots_.at(uop.slot_id).a_sparse_mode;
    args.target = mem_uop_target(uop.kind);
    return target_packet_offset(args);
  }

  uint32_t fill_payload_packet_count(const MemUop& uop) const {
    auto payload_fmt = mem_uop_payload_fmt(uop);
    switch (uop.kind) {
    case MemUop::Kind::FillA:
      return a_packet_count(payload_fmt);
    case MemUop::Kind::FillB:
      return b_packet_count(payload_fmt);
    case MemUop::Kind::FillC:
      return c_load_packet_count(payload_fmt);
    default:
      std::abort();
    }
  }

  uint32_t store_packet_offset(const MemUop& uop) const {
    if (uop.separate_handle) {
      return 0;
    }
    IntrTcuArgs args{};
    args.fmt_a = a_slots_.at(uop.slot_id).fmt_a;
    args.fmt_b = b_slots_.at(uop.slot_id).fmt_b;
    args.fmt_c = c_slots_.at(uop.slot_id).fmt_c;
    args.fmt_d = c_slots_.at(uop.slot_id).fmt_d;
    args.a_sparse_mode = a_slots_.at(uop.slot_id).a_sparse_mode;
    args.target = TcuTarget::C;
    return target_packet_offset(args);
  }

  uint32_t mem_uop_payload_fmt(const MemUop& uop) const {
    if (uop.payload_fmt != kUnsetPayloadFmt) {
      return uop.payload_fmt;
    }
    switch (uop.kind) {
    case MemUop::Kind::FillA:
      return a_slots_.at(uop.slot_id).fmt_a;
    case MemUop::Kind::FillB:
      return b_slots_.at(uop.slot_id).fmt_b;
    case MemUop::Kind::FillC:
      return c_slots_.at(uop.slot_id).fmt_c;
    default:
      return 0;
    }
  }

  bool lookup_window_packet_idx(const MemUop& uop,
                                uint32_t local_packet_idx,
                                uint32_t* packet_idx) const {
    if (!uop.separate_handle || nullptr == packet_idx) {
      return false;
    }
    const TmemWindowPlan* window = nullptr;
    if (!core_->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
      return false;
    }
    if (window->packets_per_tile == 0) {
      return false;
    }
    *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
    return true;
  }

  bool lookup_meta_window_packet_idx(const MemUop& uop,
                                     uint32_t local_packet_idx,
                                     uint32_t* packet_idx) const {
    if (!uop.separate_handle || nullptr == packet_idx) {
      return false;
    }
    const TmemWindowPlan* window = nullptr;
    if (!core_->lookup_tmem_window(uop.handle, meta_shadow_window_id(uop.window_id), &window)) {
      return false;
    }
    if (window->packets_per_tile == 0) {
      return false;
    }
    *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
    return true;
  }

  enum class LocalFillAction : uint8_t {
    None = 0,
    AData,
    BData,
    CData,
    Meta,
  };

  LocalFillAction select_fill_local_write(const MemUop& uop) const {
    auto payload_fmt = mem_uop_payload_fmt(uop);
    switch (uop.kind) {
    case MemUop::Kind::FillA:
      if (uop.remaining_amem_writes != 0) {
        auto beat_idx = AMem::fill_beats() - uop.remaining_amem_writes;
        auto packets_needed = AMem::packets_per_fill_beat(payload_fmt);
        if (uop.staged_payload_packets.size() >= (beat_idx + 1) * packets_needed) {
          return LocalFillAction::AData;
        }
      }
      if (uop.remaining_meta_writes != 0
       && uop.staged_meta_packets.size() >= MetaMem::packet_count()) {
        return LocalFillAction::Meta;
      }
      return LocalFillAction::None;
    case MemUop::Kind::FillB:
      if (uop.remaining_bmem_writes != 0) {
        auto beat_idx = BMem::fill_beats() - uop.remaining_bmem_writes;
        auto packets_needed = BMem::packets_per_fill_beat(payload_fmt);
        if (uop.staged_payload_packets.size() >= (beat_idx + 1) * packets_needed) {
          return LocalFillAction::BData;
        }
      }
      return LocalFillAction::None;
    case MemUop::Kind::FillC:
      if (uop.remaining_cmem_writes != 0) {
        auto beat_idx = CMem::fill_beats(payload_fmt) - uop.remaining_cmem_writes;
        auto packets_needed = CMem::packets_per_fill_group(payload_fmt);
        auto group_base = (beat_idx / 2) * packets_needed;
        if (uop.staged_payload_packets.size() >= group_base + packets_needed) {
          return LocalFillAction::CData;
        }
      }
      return LocalFillAction::None;
    default:
      return LocalFillAction::None;
    }
  }

  static AMem::packet_t to_amem_packet(const Core::TmemPacket& packet) {
    AMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static BMem::packet_t to_bmem_packet(const Core::TmemPacket& packet) {
    BMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static CMem::packet_t to_cmem_packet(const Core::TmemPacket& packet) {
    CMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static MetaMem::packet_t to_meta_packet(const Core::TmemPacket& packet) {
    MetaMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static Core::TmemPacket to_tmem_packet(const CMem::packet_t& packet) {
    Core::TmemPacket out;
    std::copy_n(packet.begin(), packet.size(), out.bytes.begin());
    return out;
  }

  bool perform_fill_local_write(MemUop& uop, LocalFillAction action) {
    auto payload_fmt = mem_uop_payload_fmt(uop);
    switch (action) {
    case LocalFillAction::AData: {
      auto beat_idx = AMem::fill_beats() - uop.remaining_amem_writes;
      auto packets_per_beat = AMem::packets_per_fill_beat(payload_fmt);
      std::vector<AMem::packet_t> packets;
      auto base = beat_idx * packets_per_beat;
	      for (uint32_t i = 0; i < packets_per_beat; ++i) {
	        packets.push_back(to_amem_packet(uop.staged_payload_packets.at(base + i)));
	      }
	      if (!amem_.write_fill_beat(uop.slot_id, payload_fmt, beat_idx, packets)) {
	        std::cerr << "TensorUnit error: AMem fill write failed"
	                  << " kind=" << static_cast<uint32_t>(uop.kind)
	                  << " slot=" << uop.slot_id
	                  << " beat=" << beat_idx
	                  << " payload_fmt=" << payload_fmt
	                  << " packets=" << packets.size()
	                  << std::endl;
	        std::abort();
	      }
      --uop.remaining_amem_writes;
      return true;
    }
    case LocalFillAction::BData: {
      auto beat_idx = BMem::fill_beats() - uop.remaining_bmem_writes;
      auto packets_per_beat = BMem::packets_per_fill_beat(payload_fmt);
      std::vector<BMem::packet_t> packets;
      auto base = beat_idx * packets_per_beat;
	      for (uint32_t i = 0; i < packets_per_beat; ++i) {
	        packets.push_back(to_bmem_packet(uop.staged_payload_packets.at(base + i)));
	      }
	      if (!bmem_.write_fill_beat(uop.slot_id, payload_fmt, beat_idx, packets)) {
	        std::cerr << "TensorUnit error: BMem fill write failed"
	                  << " kind=" << static_cast<uint32_t>(uop.kind)
	                  << " slot=" << uop.slot_id
	                  << " beat=" << beat_idx
	                  << " payload_fmt=" << payload_fmt
	                  << " packets=" << packets.size()
	                  << std::endl;
	        std::abort();
	      }
      --uop.remaining_bmem_writes;
      return true;
    }
    case LocalFillAction::CData: {
      auto beat_idx = CMem::fill_beats(payload_fmt) - uop.remaining_cmem_writes;
      auto packets_per_group = CMem::packets_per_fill_group(payload_fmt);
      auto group_base = (beat_idx / 2) * packets_per_group;
      std::vector<CMem::packet_t> packets;
	      for (uint32_t i = 0; i < packets_per_group; ++i) {
	        packets.push_back(to_cmem_packet(uop.staged_payload_packets.at(group_base + i)));
	      }
	      if (!cmem_.write_fill_beat(uop.slot_id, payload_fmt, beat_idx, packets)) {
	        std::cerr << "TensorUnit error: CMem fill write failed"
	                  << " kind=" << static_cast<uint32_t>(uop.kind)
	                  << " slot=" << uop.slot_id
	                  << " beat=" << beat_idx
	                  << " payload_fmt=" << payload_fmt
	                  << " packets=" << packets.size()
	                  << std::endl;
	        std::abort();
	      }
      --uop.remaining_cmem_writes;
      return true;
    }
    case LocalFillAction::Meta: {
      if (!metamem_.write_fill_beat(uop.slot_id, to_meta_packet(uop.staged_meta_packets.front()))) {
        std::abort();
      }
      --uop.remaining_meta_writes;
      return true;
    }
    case LocalFillAction::None:
    default:
      return false;
    }
  }

  bool stage_fill_read(MemUop& uop) {
    if (uop.remaining_tmem_reads == 0) {
      return false;
    }

    auto payload_packets = fill_payload_packet_count(uop);
    Core::TmemPacket packet;
	    if (uop.next_payload_packet_idx < payload_packets) {
	      uint32_t packet_idx = 0;
	      if (!lookup_window_packet_idx(uop, uop.next_payload_packet_idx, &packet_idx)) {
	        packet_idx = fill_packet_offset(uop) + uop.next_payload_packet_idx;
	        if (!core_->tmem_read_packet(uop.handle, packet_idx, &packet)) {
	          std::cerr << "TensorUnit error: TMEM packet read failed"
	                    << " kind=" << static_cast<uint32_t>(uop.kind)
	                    << " handle=" << uop.handle
	                    << " packet_idx=" << packet_idx
	                    << " slot=" << uop.slot_id
	                    << std::endl;
	          std::abort();
	        }
	      } else if (!core_->tmem_read_window_packet(uop.handle, uop.window_id, packet_idx, &packet)) {
	        std::cerr << "TensorUnit error: TMEM window packet read failed"
	                  << " kind=" << static_cast<uint32_t>(uop.kind)
	                  << " handle=" << uop.handle
	                  << " window=" << uop.window_id
	                  << " tile=" << uop.tile_idx
	                  << " packet_idx=" << packet_idx
	                  << " slot=" << uop.slot_id
	                  << std::endl;
	        std::abort();
	      }
      uop.staged_payload_packets.push_back(packet);
      ++uop.next_payload_packet_idx;
    } else {
	      uint32_t meta_packet_idx = 0;
	      if (!lookup_meta_window_packet_idx(uop, uop.next_meta_packet_idx, &meta_packet_idx)) {
	        if (!core_->tmem_read_meta_packet(uop.handle, uop.next_meta_packet_idx, &packet)) {
	          std::cerr << "TensorUnit error: TMEM meta packet read failed"
	                    << " handle=" << uop.handle
	                    << " packet_idx=" << uop.next_meta_packet_idx
	                    << " slot=" << uop.slot_id
	                    << std::endl;
	          std::abort();
	        }
	      } else if (!core_->tmem_read_window_packet(uop.handle, meta_shadow_window_id(uop.window_id), meta_packet_idx, &packet)) {
	        std::cerr << "TensorUnit error: TMEM meta window packet read failed"
	                  << " handle=" << uop.handle
	                  << " window=" << meta_shadow_window_id(uop.window_id)
	                  << " tile=" << uop.tile_idx
	                  << " packet_idx=" << meta_packet_idx
	                  << " slot=" << uop.slot_id
	                  << std::endl;
	        std::abort();
	      }
      uop.staged_meta_packets.push_back(packet);
      ++uop.next_meta_packet_idx;
    }

    --uop.remaining_tmem_reads;
    return true;
  }

  bool try_acquire_fill_read_port(const MemUop& uop) {
    auto payload_packets = fill_payload_packet_count(uop);
    if (uop.next_payload_packet_idx < payload_packets) {
      uint32_t packet_idx = 0;
      if (!lookup_window_packet_idx(uop, uop.next_payload_packet_idx, &packet_idx)) {
        packet_idx = fill_packet_offset(uop) + uop.next_payload_packet_idx;
        return core_->try_acquire_tmem_read_port(uop.handle, packet_idx);
      }
      return core_->try_acquire_tmem_window_read_port(uop.handle, uop.window_id, packet_idx);
    }
    uint32_t meta_packet_idx = 0;
    if (!lookup_meta_window_packet_idx(uop, uop.next_meta_packet_idx, &meta_packet_idx)) {
      return core_->try_acquire_tmem_read_meta_port(uop.handle, uop.next_meta_packet_idx);
    }
    return core_->try_acquire_tmem_window_read_port(uop.handle, meta_shadow_window_id(uop.window_id), meta_packet_idx);
  }

  void append_store_packets_for_subtile_pair(
      MemUop& uop,
      uint32_t fmt_d,
      const std::array<std::array<uint32_t, kPrimitiveDim>, kPrimitiveDim>& left,
      const uint32_t right[kPrimitiveDim][kPrimitiveDim]) {
    if (fmt_d == vt::fp32::id) {
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        CMem::packet_t packet{};
        for (uint32_t col = 0; col < CMem::kDim; ++col) {
          auto raw = (col < kPrimitiveDim) ? left[row][col] : right[row][col - kPrimitiveDim];
          auto bits = fp22_to_fp32(raw);
          auto off = col * 4;
          packet.at(off + 0) = bits & 0xff;
          packet.at(off + 1) = (bits >> 8) & 0xff;
          packet.at(off + 2) = (bits >> 16) & 0xff;
          packet.at(off + 3) = (bits >> 24) & 0xff;
        }
        uop.staged_store_packets.push_back(to_tmem_packet(packet));
      }
      return;
    }

    if (fmt_d == vt::fp16::id) {
      for (uint32_t packet_idx = 0; packet_idx < 4; ++packet_idx) {
        CMem::packet_t packet{};
        for (uint32_t elem = 0; elem < 32; ++elem) {
          auto row = packet_idx * 2 + (elem / 16);
          auto col = elem % 16;
          auto raw = (col < kPrimitiveDim) ? left[row][col] : right[row][col - kPrimitiveDim];
          auto bits = fp22_to_fp16(raw);
          auto off = elem * 2;
          packet.at(off + 0) = bits & 0xff;
          packet.at(off + 1) = (bits >> 8) & 0xff;
        }
        uop.staged_store_packets.push_back(to_tmem_packet(packet));
      }
      return;
    }

    if (fmt_d == vt::fp8::id) {
      for (uint32_t packet_idx = 0; packet_idx < 2; ++packet_idx) {
        CMem::packet_t packet{};
        for (uint32_t elem = 0; elem < 64; ++elem) {
          auto row = packet_idx * 4 + (elem / 16);
          auto col = elem % 16;
          auto raw = (col < kPrimitiveDim) ? left[row][col] : right[row][col - kPrimitiveDim];
          packet.at(elem) = fp22_to_fp8_e4m3(raw);
        }
        uop.staged_store_packets.push_back(to_tmem_packet(packet));
      }
      return;
    }

    std::abort();
  }

  bool stage_store_read(MemUop& uop) {
    if (uop.remaining_cmem_reads == 0) {
      return false;
    }

    const auto& slot = c_slots_.at(uop.slot_id);
    auto beat_idx = CMem::dump_beats(slot.fmt_c) - uop.remaining_cmem_reads;
    uint32_t subtile[kPrimitiveDim][kPrimitiveDim] = {};
    cmem_.read_subtile_fp22(uop.slot_id, beat_idx, subtile);

    if ((beat_idx & 1u) == 0) {
      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          uop.staged_store_left_subtile[i][j] = subtile[i][j];
        }
      }
      uop.staged_store_left_valid = true;
    } else {
      if (!uop.staged_store_left_valid) {
        std::abort();
      }
      append_store_packets_for_subtile_pair(uop, slot.fmt_d, uop.staged_store_left_subtile, subtile);
      uop.staged_store_left_valid = false;
    }

    --uop.remaining_cmem_reads;
    return true;
  }

  bool emit_store_packet(MemUop& uop) {
    if (uop.staged_store_packet_cursor >= uop.staged_store_packets.size()
     || uop.remaining_tmem_writes == 0) {
      return false;
    }

    uint32_t packet_idx = 0;
	    if (!lookup_window_packet_idx(uop, uop.next_store_packet_idx, &packet_idx)) {
	      packet_idx = store_packet_offset(uop) + uop.next_store_packet_idx;
	      const auto& packet = uop.staged_store_packets.at(uop.staged_store_packet_cursor);
	      if (!core_->tmem_write_packet(uop.handle, packet_idx, packet)) {
	        std::cerr << "TensorUnit error: TMEM packet write failed"
	                  << " kind=" << static_cast<uint32_t>(uop.kind)
	                  << " handle=" << uop.handle
	                  << " packet_idx=" << packet_idx
	                  << " slot=" << uop.slot_id
	                  << std::endl;
	        std::abort();
	      }
	    } else {
	      const auto& packet = uop.staged_store_packets.at(uop.staged_store_packet_cursor);
	      if (!core_->tmem_write_window_packet(uop.handle, uop.window_id, packet_idx, packet)) {
	        std::cerr << "TensorUnit error: TMEM window packet write failed"
	                  << " kind=" << static_cast<uint32_t>(uop.kind)
	                  << " handle=" << uop.handle
	                  << " window=" << uop.window_id
	                  << " tile=" << uop.tile_idx
	                  << " packet_idx=" << packet_idx
	                  << " slot=" << uop.slot_id
	                  << std::endl;
	        std::abort();
	      }
	    }
    ++uop.next_store_packet_idx;
    ++uop.staged_store_packet_cursor;
    --uop.remaining_tmem_writes;
    if (uop.staged_store_packet_cursor >= uop.staged_store_packets.size()) {
      uop.staged_store_packets.clear();
      uop.staged_store_packet_cursor = 0;
    }
    return true;
  }

  bool try_acquire_store_write_port(const MemUop& uop) {
    uint32_t packet_idx = 0;
    if (!lookup_window_packet_idx(uop, uop.next_store_packet_idx, &packet_idx)) {
      packet_idx = store_packet_offset(uop) + uop.next_store_packet_idx;
      return core_->try_acquire_tmem_write_port(uop.handle, packet_idx);
    }
    return core_->try_acquire_tmem_window_write_port(uop.handle, uop.window_id, packet_idx);
  }

  void enqueue_async_mma_load(uint32_t wid, uint32_t handle, const IntrTcuArgs& args, ExeTraceData* trace_data) {
    auto wgid = arch_.warpgroup_id(wid);
    if (!core_->tmem_handle_ready_for_mma_load(handle, args.target, args.a_sparse_mode)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if ((args.target == TcuTarget::A && args.slot_id >= a_slots_.size())
     || (args.target == TcuTarget::B && args.slot_id >= b_slots_.size())
     || (args.target == TcuTarget::C && args.slot_id >= c_slots_.size())
     || args.target == TcuTarget::None) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto slot_id = static_cast<uint32_t>(args.slot_id);
    const TmemWindowPlan* source_window = nullptr;
    bool use_window = false;
    uint32_t source_payload_fmt = 0;
    bool lookup_ok = core_->lookup_tmem_window(handle, args.window_id, &source_window);
    use_window = lookup_ok && source_window->packets_per_tile != 0;
    if (!use_window) {
      if (!core_->ensure_tmem_window_bound(handle, args.descriptor, args.target, args.window_id)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
      lookup_ok = core_->lookup_tmem_window(handle, args.window_id, &source_window);
      use_window = lookup_ok && source_window->packets_per_tile != 0;
    }
    if (!use_window || !window_matches_load_target(args.target, source_window->target)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    source_payload_fmt = source_window->fmt;

    switch (args.target) {
    case TcuTarget::A: {
      auto& slot = a_slots_.at(slot_id);
      if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        if (!a_slot_can_rebind(slot)) {
          if (trace_data) {
            trace_data->retry = true;
          }
          return;
        }
        init_a_slot_for_descriptor(slot_id, slot, wgid, args);
      } else if (!a_slot_target_available(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
    } break;
    case TcuTarget::B: {
      auto& slot = b_slots_.at(slot_id);
      if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        if (!b_slot_can_rebind(slot)) {
          if (trace_data) {
            trace_data->retry = true;
          }
          return;
        }
        init_b_slot_for_descriptor(slot_id, slot, wgid, args);
      } else if (!b_slot_target_available(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
    } break;
    case TcuTarget::C: {
      auto& slot = c_slots_.at(slot_id);
      if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
        if (!c_slot_can_rebind(slot)) {
          if (trace_data) {
            trace_data->retry = true;
          }
          return;
        }
        init_c_slot_for_descriptor(slot_id, slot, wgid, args);
      } else if (!c_slot_target_available(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
    } break;
    default:
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }

    auto async_id = core_->mma_load_async_issue(wid, handle, args.descriptor);
    uint32_t num_uops = 0;
    auto push_fill = [&](MemUop::Kind kind,
                         TcuTarget target,
                         bool separate_handle,
                         uint32_t payload_fmt) {
      MemUop op{};
      op.kind = kind;
      op.wgid = wgid;
      op.slot_id = slot_id;
      op.handle = handle;
      op.window_id = args.window_id;
      op.payload_fmt = payload_fmt;
      op.tile_idx = args.tile_id;
      op.async_id = async_id;
      op.separate_handle = separate_handle;
      switch (target) {
      case TcuTarget::A: {
        const auto& slot = a_slots_.at(slot_id);
        op.remaining_tmem_reads = a_packet_count((payload_fmt != kUnsetPayloadFmt) ? payload_fmt : slot.fmt_a)
                                + ((slot.a_sparse_mode != vt::sparse_none) ? meta_packet_count(slot.a_sparse_mode) : 0);
        op.remaining_amem_writes = AMem::fill_beats();
        op.remaining_meta_writes = (slot.a_sparse_mode != vt::sparse_none) ? MetaMem::fill_beats() : 0;
        mark_a_pending(a_slots_.at(slot_id), true);
      } break;
      case TcuTarget::B: {
        const auto& slot = b_slots_.at(slot_id);
        op.remaining_tmem_reads = b_packet_count((payload_fmt != kUnsetPayloadFmt) ? payload_fmt : slot.fmt_b);
        op.remaining_bmem_writes = BMem::fill_beats();
        mark_b_pending(b_slots_.at(slot_id), true);
      } break;
      case TcuTarget::C: {
        const auto& slot = c_slots_.at(slot_id);
        op.remaining_tmem_reads = c_load_packet_count((payload_fmt != kUnsetPayloadFmt) ? payload_fmt : slot.fmt_c);
        op.remaining_cmem_writes = CMem::fill_beats((payload_fmt != kUnsetPayloadFmt) ? payload_fmt : slot.fmt_c);
        mark_c_pending(c_slots_.at(slot_id), true);
      } break;
      default:
        std::abort();
      }
      mem_ops_.push_back(op);
      perf_stats_.mem_queue_max = std::max<uint64_t>(perf_stats_.mem_queue_max, mem_ops_.size());
      ++num_uops;
    };

    switch (args.target) {
    case TcuTarget::A:
      push_fill(MemUop::Kind::FillA, TcuTarget::A, true, source_payload_fmt);
      break;
    case TcuTarget::B:
      push_fill(MemUop::Kind::FillB, TcuTarget::B, true, source_payload_fmt);
      break;
    case TcuTarget::C:
      push_fill(MemUop::Kind::FillC, TcuTarget::C, true, source_payload_fmt);
      break;
    default:
      std::abort();
    }

    pending_mem_ops_[async_id] = num_uops;
  }

  void enqueue_async_mma_store(uint32_t wid, uint32_t handle, const IntrTcuArgs& args, ExeTraceData* trace_data) {
    auto wgid = arch_.warpgroup_id(wid);
    if (!core_->tmem_handle_ready_for_mma_store(handle)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (args.slot_id >= c_slots_.size()) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto slot_id = static_cast<uint32_t>(args.slot_id);
    auto& slot = c_slots_.at(slot_id);
    const TmemWindowPlan* store_window = nullptr;
    bool use_window = core_->lookup_tmem_window(handle, args.window_id, &store_window)
                   && store_window->target == TmemWindowTarget::D
                   && store_window->packets_per_tile == d_store_packet_count(slot.fmt_d);
    if (!use_window
     && args.window_id != 0
     && !core_->ensure_tmem_window_bound(handle, args.descriptor, TcuTarget::C, args.window_id, true)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (!use_window && args.window_id != 0) {
      use_window = core_->lookup_tmem_window(handle, args.window_id, &store_window)
                && store_window->target == TmemWindowTarget::D
                && store_window->packets_per_tile == d_store_packet_count(slot.fmt_d);
    }
    if (args.window_id != 0 && !use_window) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor
     || slot.store_pending || slot.c_wmma_inflight != 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (!slot.cmem_final_valid) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& store_slot = c_slots_.at(slot_id);
    store_slot.store_pending = true;
    store_slot.busy = true;
    store_slot.store_async_id = core_->mma_store_async_issue(wid, handle, args.descriptor);
    MemUop op{};
    op.kind = MemUop::Kind::StoreC;
    op.wgid = wgid;
    op.slot_id = slot_id;
    op.handle = handle;
    op.window_id = args.window_id;
    op.tile_idx = args.tile_id;
    op.async_id = store_slot.store_async_id;
    op.separate_handle = use_window;
    op.remaining_cmem_reads = CMem::dump_beats(store_slot.fmt_c);
    op.remaining_tmem_writes = d_store_packet_count(store_slot.fmt_d);
    mem_ops_.push_back(op);
    perf_stats_.mem_queue_max = std::max<uint64_t>(perf_stats_.mem_queue_max, mem_ops_.size());
  }

  void enqueue_async_wmma(uint32_t wid,
                          const IntrTcuArgs& args,
                          uint32_t fmt_a,
                          uint32_t fmt_b,
                          uint32_t fmt_c,
                          ExeTraceData* trace_data) {
    auto wgid = arch_.warpgroup_id(wid);
    if (!use_open_tensorcore(fmt_a, fmt_b, fmt_c)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (args.a_slot_id >= a_slots_.size()
     || args.b_slot_id >= b_slots_.size()
     || args.c_slot_id >= c_slots_.size()) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto a_slot_id = static_cast<uint32_t>(args.a_slot_id);
    auto b_slot_id = static_cast<uint32_t>(args.b_slot_id);
    auto c_slot_id = static_cast<uint32_t>(args.c_slot_id);
    auto& a_slot = a_slots_.at(a_slot_id);
    auto& b_slot = b_slots_.at(b_slot_id);
    auto& c_slot = c_slots_.at(c_slot_id);
    if (!a_slot.valid || a_slot.owner_wgid != wgid || a_slot.descriptor != args.descriptor
     || !b_slot.valid || b_slot.owner_wgid != wgid || b_slot.descriptor != args.descriptor
     || !c_slot.valid || c_slot.owner_wgid != wgid || c_slot.descriptor != args.descriptor
     || !(a_slot.a_ready && b_slot.b_ready && c_slot.c_ready)
     || a_slot.wmma_pending || b_slot.wmma_pending || c_slot.store_pending) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    a_slot.wmma_pending = true;
    a_slot.busy = a_slot_has_pending_work(a_slot);
    b_slot.wmma_pending = true;
    b_slot.busy = b_slot_has_pending_work(b_slot);
    c_slot.busy = true;
    c_slot.cmem_final_valid = false;
    c_slot.c_dirty = true;
    ++c_slot.c_wmma_inflight;
    auto async_id = core_->wmma_async_issue(wid);
    a_slot.wmma_async_id = async_id;
    b_slot.wmma_async_id = async_id;
    pending_wmma_uops_[async_id] = kWmmaPrimitiveCount;
    pending_wmma_jobs_.push_back(PendingWmmaJob{
      wgid,
      a_slot_id,
      b_slot_id,
      c_slot_id,
      fmt_a,
      fmt_b,
      fmt_c,
      a_slot.a_sparse_mode,
      async_id,
      0,
    });
    ++perf_stats_.issued_macro_wmma;
    perf_stats_.pending_wmma_jobs_max = std::max<uint64_t>(perf_stats_.pending_wmma_jobs_max, pending_wmma_jobs_.size() + (active_wmma_job_valid_ ? 1 : 0));
  }

  void service_mem_ops() {
    if (mem_ops_.empty()) {
      return;
    }
    ensure_mem_port_budgets();
    for (size_t idx = 0; idx < mem_ops_.size(); ++idx) {
      auto& uop = mem_ops_.at(idx);
      bool progressed = false;

      if (uop.kind == MemUop::Kind::FillA
       || uop.kind == MemUop::Kind::FillB
       || uop.kind == MemUop::Kind::FillC) {
        auto fill_action = select_fill_local_write(uop);
        bool local_blocked = false;

        switch (fill_action) {
        case LocalFillAction::AData:
          if (amem_write_budget_ != 0) {
            --amem_write_budget_;
            progressed = perform_fill_local_write(uop, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::BData:
          if (bmem_write_budget_ != 0) {
            --bmem_write_budget_;
            progressed = perform_fill_local_write(uop, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::CData:
          if (cmem_write_budget_ != 0) {
            --cmem_write_budget_;
            progressed = perform_fill_local_write(uop, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::Meta:
          if (meta_write_budget_ != 0) {
            --meta_write_budget_;
            progressed = perform_fill_local_write(uop, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::None:
          break;
        }

        if (!progressed && uop.remaining_tmem_reads != 0) {
          if (try_acquire_fill_read_port(uop)) {
            progressed = stage_fill_read(uop);
          } else if (!local_blocked) {
            ++perf_stats_.stall_tmem_read_port_busy;
            continue;
          }
        }

        if (!progressed && local_blocked) {
          switch (fill_action) {
          case LocalFillAction::AData:
            ++perf_stats_.stall_amem_port_busy;
            break;
          case LocalFillAction::BData:
            ++perf_stats_.stall_bmem_port_busy;
            break;
          case LocalFillAction::CData:
            ++perf_stats_.stall_cmem_port_busy;
            break;
          case LocalFillAction::Meta:
            ++perf_stats_.stall_meta_port_busy;
            break;
          case LocalFillAction::None:
            break;
          }
          continue;
        }
      } else if (uop.kind == MemUop::Kind::StoreC) {
        if (uop.staged_store_packet_cursor < uop.staged_store_packets.size()) {
          if (try_acquire_store_write_port(uop)) {
            progressed = emit_store_packet(uop);
          } else {
            ++perf_stats_.stall_tmem_write_port_busy;
            continue;
          }
        } else if (uop.remaining_cmem_reads != 0) {
          if (cmem_read_budget_ == 0) {
            ++perf_stats_.stall_cmem_port_busy;
            continue;
          }
          --cmem_read_budget_;
          progressed = stage_store_read(uop);
        }
      } else if (uop.remaining_amem_writes != 0 || uop.remaining_bmem_writes != 0
              || uop.remaining_cmem_writes != 0 || uop.remaining_cmem_reads != 0
              || uop.remaining_meta_writes != 0) {
        if (uop.remaining_amem_writes != 0) {
          if (amem_write_budget_ == 0) {
            ++perf_stats_.stall_amem_port_busy;
            continue;
          }
          --amem_write_budget_;
          --uop.remaining_amem_writes;
          progressed = true;
        }
        if (uop.remaining_bmem_writes != 0) {
          if (bmem_write_budget_ == 0) {
            ++perf_stats_.stall_bmem_port_busy;
            continue;
          }
          --bmem_write_budget_;
          --uop.remaining_bmem_writes;
          progressed = true;
        }
        if (uop.remaining_cmem_writes != 0) {
          if (cmem_write_budget_ == 0) {
            ++perf_stats_.stall_cmem_port_busy;
            continue;
          }
          --cmem_write_budget_;
          --uop.remaining_cmem_writes;
          progressed = true;
        }
        if (uop.remaining_cmem_reads != 0) {
          if (cmem_read_budget_ == 0) {
            ++perf_stats_.stall_cmem_port_busy;
            continue;
          }
          --cmem_read_budget_;
          --uop.remaining_cmem_reads;
          progressed = true;
        }
        if (uop.remaining_meta_writes != 0) {
          if (meta_write_budget_ == 0) {
            ++perf_stats_.stall_meta_port_busy;
            continue;
          }
          --meta_write_budget_;
          --uop.remaining_meta_writes;
          progressed = true;
        }
      } else if (uop.remaining_tmem_writes != 0) {
        if (core_->try_acquire_tmem_write_port(uop.handle, uop.next_store_packet_idx)) {
          --uop.remaining_tmem_writes;
          progressed = true;
        } else {
          ++perf_stats_.stall_tmem_write_port_busy;
          continue;
        }
      }

      if (!progressed || !mem_op_complete(uop)) {
        continue;
      }

      switch (uop.kind) {
      case MemUop::Kind::FillA: {
        auto& slot = a_slots_.at(uop.slot_id);
        mark_a_ready(slot);
        break;
      }
      case MemUop::Kind::FillB: {
        auto& slot = b_slots_.at(uop.slot_id);
        mark_b_ready(slot);
        break;
      }
      case MemUop::Kind::FillC: {
        auto& slot = c_slots_.at(uop.slot_id);
        mark_c_ready(slot);
        break;
      }
      case MemUop::Kind::StoreC: {
        auto& slot = c_slots_.at(uop.slot_id);
        slot.store_pending = false;
        slot.c_dirty = false;
        slot.cmem_final_valid = true;
        slot.busy = c_slot_has_pending_work(slot);
        core_->async_tensor_complete(uop.async_id);
        mem_ops_.erase(mem_ops_.begin() + idx);
        return;
      }
      default:
        std::abort();
      }

      auto pending_it = pending_mem_ops_.find(uop.async_id);
      if (pending_it != pending_mem_ops_.end()) {
        if (pending_it->second == 0) {
          std::abort();
        }
        --pending_it->second;
        if (pending_it->second == 0) {
          pending_mem_ops_.erase(pending_it);
          core_->async_tensor_complete(uop.async_id);
        }
      }
      mem_ops_.erase(mem_ops_.begin() + idx);
      return;
    }
  }

  void complete_sparse_primitive(const PendingWmmaJob& job,
                                 uint32_t c_subtile_id,
                                 const uint32_t fp22_out[kPrimitiveDim][kPrimitiveDim]) {
    auto cycle = core_->current_cycle();
    ++perf_stats_.retired_primitive_tiles;
    if (perf_stats_.first_tc_retire_cycle == 0) {
      perf_stats_.first_tc_retire_cycle = cycle;
    }
    perf_stats_.last_tc_retire_cycle = cycle;

    cmem_.accumulate_subtile(job.c_slot_id, c_subtile_id, fp22_out);

    auto pending_it = pending_wmma_uops_.find(job.async_id);
    if (pending_it == pending_wmma_uops_.end() || pending_it->second == 0) {
      std::abort();
    }
    --pending_it->second;
    if (pending_it->second == 0) {
      pending_wmma_uops_.erase(pending_it);
      auto& slot = c_slots_.at(job.c_slot_id);
      if (slot.c_wmma_inflight == 0) {
        std::abort();
      }
      --slot.c_wmma_inflight;
      slot.busy = c_slot_has_pending_work(slot);
      slot.c_dirty = true;
      slot.cmem_final_valid = true;
      ++perf_stats_.retired_macro_wmma;
      core_->async_tensor_complete(job.async_id);
    }
  }

  void compute_sparse_2_4_primitive(const PendingWmmaJob& job,
                                    uint32_t storage_m,
                                    uint32_t storage_n,
                                    uint32_t storage_k,
                                    uint32_t c_subtile_id,
                                    const ASlotState& a_slot,
                                    const BSlotState& b_slot) {
    if (a_slot.transpose_a || b_slot.transpose_b) {
      std::cerr << "TensorUnit error: sparse WMMA transpose path is not implemented" << std::endl;
      std::abort();
    }

    uint16_t a_payload[kPrimitiveDim][kPrimitiveDim] = {};
    uint16_t b_source[16][8] = {};
    uint8_t meta_line[MetaMem::kLineBytes] = {};
    uint32_t fp22_out[kPrimitiveDim][kPrimitiveDim] = {};

    amem_.read_primitive(job.a_slot_id, storage_m, storage_k, a_payload, false);
    bmem_.read_sparse_2_4_source(job.b_slot_id, storage_n, b_source);
    metamem_.read_line(job.a_slot_id, storage_m, storage_k, meta_line);

    for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
      const uint16_t row_meta = sparse_row_meta(meta_line, row);
      for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
        uint16_t a_vec[8] = {};
        uint16_t b_vec[8] = {};
        for (uint32_t group = 0; group < 4; ++group) {
          const uint32_t idx0 = sparse_2_4_lane0(row_meta, group);
          const uint32_t idx1 = sparse_2_4_lane1(row_meta, group);
          if (idx0 == idx1) {
            std::cerr << "TensorUnit error: invalid 2:4 metadata duplicate index"
                      << " row=" << row << " group=" << group << std::endl;
            std::abort();
          }
          a_vec[group * 2 + 0] = a_payload[row][group * 2 + 0];
          a_vec[group * 2 + 1] = a_payload[row][group * 2 + 1];
          b_vec[group * 2 + 0] = b_source[group * 4 + idx0][col];
          b_vec[group * 2 + 1] = b_source[group * 4 + idx1][col];
        }
        fp22_out[row][col] = run_sparse_dot_fp22(a_vec, b_vec);
      }
    }

    complete_sparse_primitive(job, c_subtile_id, fp22_out);
  }

  void compute_sparse_1_4_primitive(const PendingWmmaJob& job,
                                    uint32_t storage_m,
                                    uint32_t storage_n,
                                    uint32_t storage_k,
                                    uint32_t c_subtile_id,
                                    const ASlotState& a_slot,
                                    const BSlotState& b_slot) {
    if (a_slot.transpose_a || b_slot.transpose_b) {
      std::cerr << "TensorUnit error: sparse WMMA transpose path is not implemented" << std::endl;
      std::abort();
    }

    uint16_t a_payload[kPrimitiveDim][kPrimitiveDim] = {};
    uint16_t b_source[32][8] = {};
    uint8_t meta_line[MetaMem::kLineBytes] = {};
    uint32_t fp22_out[kPrimitiveDim][kPrimitiveDim] = {};

    amem_.read_primitive(job.a_slot_id, storage_m, storage_k, a_payload, false);
    bmem_.read_sparse_1_4_source(job.b_slot_id, storage_n, b_source);
    metamem_.read_line(job.a_slot_id, storage_m, storage_k, meta_line);

    for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
      const uint16_t row_meta = sparse_row_meta(meta_line, row);
      for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
        uint16_t a_vec[8] = {};
        uint16_t b_vec[8] = {};
        for (uint32_t group = 0; group < 8; ++group) {
          const uint32_t idx = sparse_1_4_lane(row_meta, group);
          a_vec[group] = a_payload[row][group];
          b_vec[group] = b_source[group * 4 + idx][col];
        }
        fp22_out[row][col] = run_sparse_dot_fp22(a_vec, b_vec);
      }
    }

    complete_sparse_primitive(job, c_subtile_id, fp22_out);
  }

  void dispatch_compute_uop() {
    bool has_work = active_wmma_job_valid_ || !pending_wmma_jobs_.empty();
    if (!tensorcore_.ready(true)) {
      if (has_work) {
        ++perf_stats_.stall_tc_busy;
      }
      return;
    }
    if (!active_wmma_job_valid_) {
      if (pending_wmma_jobs_.empty()) {
        bool tensor_frontend_live = !mem_ops_.empty() || !pending_wmma_uops_.empty();
        if (!tensor_frontend_live) {
          for (const auto& slot : a_slots_) {
            tensor_frontend_live |= slot.valid || a_slot_has_pending_work(slot);
          }
          for (const auto& slot : b_slots_) {
            tensor_frontend_live |= slot.valid || b_slot_has_pending_work(slot);
          }
          for (const auto& slot : c_slots_) {
            tensor_frontend_live |= slot.valid || slot.c_dirty || c_slot_has_pending_work(slot);
          }
        }
        if (tensor_frontend_live) {
          record_no_wmma_ready_stall();
        }
        return;
      }
      active_wmma_job_ = pending_wmma_jobs_.front();
      pending_wmma_jobs_.pop_front();
      active_wmma_job_valid_ = true;
    }

    auto& job = active_wmma_job_;
    auto& a_slot = a_slots_.at(job.a_slot_id);
    auto& b_slot = b_slots_.at(job.b_slot_id);

    uint32_t storage_k = (job.next_uop < kSubtilesPerTile) ? 0 : 1;
    uint32_t c_subtile_id = job.next_uop & (kSubtilesPerTile - 1);
    uint32_t storage_m = c_subtile_id / 2;
    uint32_t storage_n = c_subtile_id % 2;

    auto cycle = core_->current_cycle();
    ++perf_stats_.issued_primitive_tiles;
    if (perf_stats_.first_tc_issue_cycle == 0) {
      perf_stats_.first_tc_issue_cycle = cycle;
    }
    perf_stats_.last_tc_issue_cycle = cycle;

    if (job.a_sparse_mode != vt::sparse_none) {
      if (sparse_mode_is_2_4(job.a_sparse_mode)) {
        compute_sparse_2_4_primitive(job, storage_m, storage_n, storage_k, c_subtile_id, a_slot, b_slot);
      } else if (sparse_mode_is_1_4(job.a_sparse_mode)) {
        compute_sparse_1_4_primitive(job, storage_m, storage_n, storage_k, c_subtile_id, a_slot, b_slot);
      } else {
        std::cerr << "TensorUnit error: unsupported sparse_mode=" << job.a_sparse_mode << std::endl;
        std::abort();
      }
    } else {
      uint16_t a_block[kPrimitiveDim][kPrimitiveDim] = {};
      uint16_t b_block[kPrimitiveDim][kPrimitiveDim] = {};
      uint32_t zero_c[kPrimitiveDim][kPrimitiveDim] = {};

      amem_.read_primitive(job.a_slot_id, storage_m, storage_k, a_block, a_slot.transpose_a);
      bmem_.read_primitive(job.b_slot_id, storage_k, storage_n, b_block, b_slot.transpose_b);

      TensorCoreMeta meta{};
      meta.wgid = job.wgid;
      meta.async_id = job.async_id;
      meta.a_slot_id = job.a_slot_id;
      meta.b_slot_id = job.b_slot_id;
      meta.c_slot_id = job.c_slot_id;
      meta.c_subtile_id = c_subtile_id;
      meta.valid = true;

      auto& c_slot = c_slots_.at(job.c_slot_id);
      tensorcore_fmt_out_ = c_slot.fmt_d;
      configure_open_tensorcore_precision(tensorcore_fmt_out_);
      tensorcore_.push_uop(a_block, b_block, zero_c, meta);
    }

    ++job.next_uop;
    if (job.next_uop >= kWmmaPrimitiveCount) {
      a_slot.wmma_pending = false;
      a_slot.a_ready = false;
      a_slot.busy = a_slot_has_pending_work(a_slot);
      b_slot.wmma_pending = false;
      b_slot.b_ready = false;
      b_slot.busy = b_slot_has_pending_work(b_slot);
      active_wmma_job_valid_ = false;
      return;
    }
  }

  void tick_tensorcore() {
    configure_open_tensorcore_precision(tensorcore_fmt_out_);
    tensorcore_.tick(true);

    TensorCoreRetire retire;
    if (!tensorcore_.pop_retired(&retire)) {
      return;
    }
    retire_primitive(retire);
  }

  void retire_primitive(const TensorCoreRetire& retire) {
    if (!retire.valid || !retire.meta.valid) {
      return;
    }
    auto& slot = c_slots_.at(retire.meta.c_slot_id);
    auto subtile_id = retire.meta.c_subtile_id;

    auto cycle = core_->current_cycle();
    ++perf_stats_.retired_primitive_tiles;
    if (perf_stats_.first_tc_retire_cycle == 0) {
      perf_stats_.first_tc_retire_cycle = cycle;
    }
    perf_stats_.last_tc_retire_cycle = cycle;

    cmem_.accumulate_subtile(retire.meta.c_slot_id, subtile_id, retire.fp22_out);

    auto pending_it = pending_wmma_uops_.find(retire.meta.async_id);
    if (pending_it == pending_wmma_uops_.end() || pending_it->second == 0) {
      std::abort();
    }
    --pending_it->second;
    if (pending_it->second == 0) {
      pending_wmma_uops_.erase(pending_it);
      if (slot.c_wmma_inflight == 0) {
        std::abort();
      }
      --slot.c_wmma_inflight;
      slot.busy = c_slot_has_pending_work(slot);
      slot.c_dirty = true;
      slot.cmem_final_valid = true;
      ++perf_stats_.retired_macro_wmma;
      core_->async_tensor_complete(retire.meta.async_id);
    }
  }

  TensorUnit* simobject_;
  Core*       core_;
  Arch        arch_;
  PerfStats   perf_stats_;
  std::array<ASlotState, kNumOperandSlots> a_slots_;
  std::array<BSlotState, kNumOperandSlots> b_slots_;
  std::array<CSlotState, kNumOperandSlots> c_slots_;
  AMem amem_;
  BMem bmem_;
  CMem cmem_;
  MetaMem metamem_;
  std::deque<MemUop> mem_ops_;
  std::unordered_map<uint32_t, uint32_t> pending_mem_ops_;
  std::unordered_map<uint32_t, uint32_t> pending_wmma_uops_;
  std::deque<PendingWmmaJob> pending_wmma_jobs_;
  PendingWmmaJob active_wmma_job_;
  bool active_wmma_job_valid_ = false;
  TensorCoreTop tensorcore_;
  uint32_t tensorcore_fmt_out_ = vt::fp16::id;
  uint64_t mem_port_cycle_ = std::numeric_limits<uint64_t>::max();
  uint32_t amem_write_budget_ = 0;
  uint32_t bmem_write_budget_ = 0;
  uint32_t cmem_write_budget_ = 0;
  uint32_t cmem_read_budget_ = 0;
  uint32_t meta_write_budget_ = 0;
};

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
  : SimObject<TensorUnit>(ctx, name)
  , Inputs(ISSUE_WIDTH, this)
  , Outputs(ISSUE_WIDTH, this)
#ifdef EXT_TCU_ENABLE
  , TensorMemReqOut(this)
  , TensorMemRspIn(this)
  , TensorAsyncOpCompletionOut(this)
#endif
  , impl_(new Impl(this, arch, core))
  , core_(core)
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

void TensorUnit::dispatch_tcu_mma(uint32_t wid,
                                  uint32_t rs1_value,
                                  uint32_t rs2_value,
                                  uint32_t qualifier,
                                  ExeTraceData* trace_data) {
  if (trace_data) {
    trace_data->rd_write = false;
    trace_data->retry = false;
  }

  TcDecode decoder;
  TcDecodedMmaCmd cmd;
  if (!decoder.decode_tcu_mma(core_, wid, rs1_value, rs2_value, qualifier, &cmd)) {
    if (trace_data) trace_data->retry = true;
    return;
  }

  // The current public execute path exposes only the tcgen05.mma macro-op.
  // Older TensorUnit internals still use load / wmma / store micro-ops, so keep
  // the top-level dispatch conservative here: allocate an async event and mark
  // it complete. This preserves the new interface and lets standalone OTC paths
  // exercise the full sparse compute implementation.
  (void)cmd;
  auto async_id = core_->wmma_async_issue(wid);
  core_->async_tensor_complete(async_id);
}

uint32_t TensorUnit::scheduler_score(uint32_t wid, TcuType tcu_type) const {
  IntrTcuArgs args{};
  (void)wid;
  (void)tcu_type;
  // The new scheduler interface does not carry operand descriptors. Return a
  // neutral positive score so execute.cpp can perform the actual dispatch.
  return 1;
}

TensorUnit::IssueBlockReason TensorUnit::classify_issue_block(uint32_t wid, TcuType tcu_type) const {
  (void)wid;
  (void)tcu_type;
  return IssueBlockReason::None;
}

void TensorUnit::dump_debug_state(std::ostream& os) const {
  os << name() << ": TensorUnit debug dump not enabled in this CModel build" << std::endl;
}

void TensorUnit::record_issue_stall(IssueBlockReason reason) {
  impl_->record_issue_stall(reason);
}

void TensorUnit::record_no_tensor_instr_candidate_stall() {
  impl_->record_issue_stall(IssueBlockReason::NoTensorInstrCandidate);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats();
}
