// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "tensor_unit.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "core.h"
#include "tensor_cfg.h"
#include "open_tensorcore/tensor_compute/amem.h"
#include "open_tensorcore/tensor_compute/bmem.h"
#include "open_tensorcore/tensor_compute/cmem.h"
#include "open_tensorcore/tensor_compute/dmem.h"
#include "open_tensorcore/tensor_compute/sparse_select.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"
#include "open_tensorcore/tensor_control/tc_decode.h"

using namespace vortex;

namespace {

void reset_trace_for_async_dispatch(TensorUnit::ExeTraceData* trace_data) {
  if (trace_data) {
    trace_data->rd_write = false;
    trace_data->retry = false;
  }
}

// Map descriptor format IDs into the compute pipeline precision enum.
PrecisionType fmt_to_precision(uint32_t fmt) {
  namespace vt = vortex::tensor;
  switch (fmt) {
  case vt::fp8::id:  return PREC_FP8_E4M3;
  case vt::fp16::id: return PREC_FP16;
  case vt::fp32::id: return PREC_FP32;
  default:           return PREC_FP9;
  }
}

// Decode the shared-memory descriptor used by tcgen05.mma B operands.
uint64_t shared_desc_to_lmem_addr(uint64_t sdesc) {
  uint32_t smem_offset = static_cast<uint32_t>(sdesc & 0x3fffu) * 16u;
  return static_cast<uint64_t>(LMEM_BASE_ADDR) + smem_offset;
}

// Element width used when packing/unpacking TMEM result packets.
uint32_t fmt_element_bytes(uint32_t fmt) {
  namespace vt = vortex::tensor;
  switch (fmt) {
  case vt::fp8::id:  return 1;
  case vt::fp16::id: return 2;
  case vt::fp32::id: return 4;
  default:           return 0;
  }
}

// DMem dumps one 8x8 subtile as a format-dependent sequence of 64B packets.
uint32_t dmem_rows_per_packet(uint32_t fmt) {
  constexpr uint32_t kDmemPrimitiveDim = 8;
  auto elem_bytes = fmt_element_bytes(fmt);
  return elem_bytes == 0 ? 0 : Tmem::kPacketBytes / (kDmemPrimitiveDim * elem_bytes);
}

} // namespace

class TensorUnit::Impl {
public:
  static constexpr uint32_t kMaxQueuedMma = 4;
  static constexpr uint32_t kTileM = 16;
  static constexpr uint32_t kTileN = 16;
  static constexpr uint32_t kTileK = 16;
  static constexpr uint32_t kPrimitiveDim = 8;
  static constexpr uint32_t kSubtiles = 4;
  static constexpr uint32_t kDenseAccumPhases = 4;
  static constexpr uint32_t kSparseAccumPhases = 2;
  static constexpr uint32_t kSparseMetaPacketBytes = Tmem::kPacketBytes;
  static constexpr uint32_t kSparseMetaPacketBaseByteOffset = 512;

  enum class MmaStage : uint8_t {
    FillA,
    FillMeta,
    FillB,
    FillC,
    Compute,
    StoreD,
    Complete,
  };

  struct TaddrRegion {
    uint32_t col_base = 0;
    uint32_t col_span = 0;
    uint32_t byte_offset = 0;
    uint32_t size_bytes = 0;
  };

  struct MmaOp {
    TcDecodedMmaCmd cmd = {};
    uint32_t wid = 0;
    uint32_t async_id = 0;
    bool initialized = false;
    MmaStage stage = MmaStage::FillA;

    TaddrRegion a_region = {};
    TaddrRegion d_region = {};
    uint32_t a_packet_base = 0;
    uint32_t d_packet_base = 0;

    std::vector<TmemPacket> a_packets;
    std::vector<TmemPacket> meta_packets;
    std::vector<BMem::packet_t> b_packets;
    std::vector<TmemPacket> c_packets;
    std::vector<TmemPacket> d_store_packets;

    uint32_t next_a_packet = 0;
    uint32_t meta_packet_base = 0;
    uint32_t next_b_packet = 0;
    uint32_t next_c_packet = 0;
    uint32_t next_d_packet = 0;
    uint64_t pending_request_id = 0;
    uint32_t pending_packet_index = 0;

    uint32_t issue_subtile = 0;
    uint32_t issue_accum_phase = 0;
    uint32_t final_retired_subtiles = 0;
  };

  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch) {
    reset();
  }

  void reset() {
    perf_stats_ = PerfStats();
    tc_decode_.reset();
    pending_mma_.clear();
    completed_tensor_mem_responses_.clear();
    next_tensor_mem_request_id_ = 1;
    amem_.reset();
    bmem_.reset();
    cmem_.reset();
    dmem_.reset();
    tensorcore_.reset();
  }

  void tick() {
    ++perf_stats_.latency;
    latch_tensor_instruction_pipe();
    receive_tensor_mem_responses();

    if (tensorcore_.active()) {
      ++perf_stats_.tc_active_cycles;
    }

    if (!pending_mma_.empty()) {
      advance_mma(pending_mma_.front());
      if (!pending_mma_.empty()
       && pending_mma_.front().stage == MmaStage::Complete) {
        complete_front_mma();
      }
    }
  }

  bool enqueue_tcu_mma(uint32_t wid,
                       uint32_t rs1_value,
                       uint32_t rs2_value,
                       uint32_t qualifier,
                       ExeTraceData* trace_data) {
    if (pending_mma_.size() >= kMaxQueuedMma) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return false;
    }

    TcDecodedMmaCmd cmd;
    if (!tc_decode_.decode_tcu_mma(core_, wid, rs1_value, rs2_value,
                                   qualifier, &cmd)) {
      std::cerr << "TensorUnit error: tcgen05.mma tcdecode failed"
                << " rs1_idesc=0x" << std::hex << rs1_value
                << " rs2_opblock=0x" << rs2_value
                << " qualifier=0x" << qualifier << std::dec
                << std::endl;
      std::abort();
    }
    if (!validate_mma(cmd)) {
      std::cerr << "TensorUnit error: unsupported tcgen05.mma idesc"
                << " fmt_a=" << cmd.fmt_a
                << " fmt_b=" << cmd.fmt_b
                << " fmt_d=" << cmd.fmt_d
                << " shape_m=" << cmd.shape_m
                << " shape_n=" << cmd.shape_n
                << " enable_input_d=" << static_cast<uint32_t>(cmd.enable_input_d)
                << " ws=" << static_cast<uint32_t>(cmd.ws)
                << " sp=" << static_cast<uint32_t>(cmd.sp)
                << std::endl;
      std::abort();
    }

    MmaOp op{};
    op.cmd = cmd;
    op.wid = wid;

    if (!resolve_taddr(cmd.a_taddr, &op.a_region)
     || !resolve_taddr(cmd.d_taddr, &op.d_region)) {
      std::cerr << "TensorUnit error: tcgen05.mma TADDR does not map to TMEM allocation"
                << " a_taddr=0x" << std::hex << cmd.a_taddr
                << " d_taddr=0x" << cmd.d_taddr << std::dec << std::endl;
      std::abort();
    }

    auto a_block_reason = core_->tmem_taddr_load_block_reason(op.a_region.col_base,
                                                               TcuTarget::A,
                                                               cmd.sparsity_kind);
    if (a_block_reason != TmemTaddrBlockReason::None) {
      if (trace_data) {
        trace_data->retry = true;
      }
      record_issue_stall(IssueBlockReason::MmaLoadTaddrNotReady);
      return false;
    }
    auto d_block_reason = (cmd.enable_input_d != 0)
        ? core_->tmem_taddr_load_block_reason(op.d_region.col_base,
                                               TcuTarget::C,
                                               tensor::sparse_none)
        : core_->tmem_taddr_store_block_reason(op.d_region.col_base);
    if (d_block_reason != TmemTaddrBlockReason::None) {
      if (trace_data) {
        trace_data->retry = true;
      }
      record_issue_stall(IssueBlockReason::TaddrBusyDueToTmaStoreOrShift);
      return false;
    }

    op.async_id = core_->wmma_async_issue(wid);
    if ((op.a_region.byte_offset % Tmem::kPacketBytes) != 0
     || (op.d_region.byte_offset % Tmem::kPacketBytes) != 0) {
      std::cerr << "TensorUnit error: tcgen05.mma requires 64B-aligned A/D TADDR offsets"
                << std::endl;
      std::abort();
    }

    auto a_bytes = AMem::packet_count(cmd.fmt_a) * Tmem::kPacketBytes;
    if (op.a_region.byte_offset + a_bytes > op.a_region.size_bytes) {
      std::cerr << "TensorUnit error: A TMEM tile exceeds allocation" << std::endl;
      std::abort();
    }
    op.a_packet_base = op.a_region.byte_offset / Tmem::kPacketBytes;
    op.d_packet_base = op.d_region.byte_offset / Tmem::kPacketBytes;
    op.a_packets.assign(AMem::packet_count(cmd.fmt_a), TmemPacket{});
    if (cmd.sparsity_kind != tensor::sparse_none) {
      uint32_t meta_byte_offset = op.a_region.byte_offset
                                + kSparseMetaPacketBaseByteOffset
                                + uint32_t(cmd.sparsity_meta_sel) * kSparseMetaPacketBytes;
      if (meta_byte_offset + kSparseMetaPacketBytes > op.a_region.size_bytes) {
        std::cerr << "TensorUnit error: sparse metadata packet exceeds A allocation"
                  << std::endl;
        std::abort();
      }
      op.meta_packet_base = meta_byte_offset / Tmem::kPacketBytes;
      op.meta_packets.assign(1, TmemPacket{});
    }
    op.b_packets.assign(BMem::packet_count(cmd.fmt_b, cmd.sparsity_kind),
                        BMem::packet_t{});
    if (cmd.enable_input_d != 0) {
      auto c_bytes = CMem::packet_count(cmd.fmt_c) * Tmem::kPacketBytes;
      if (op.d_region.byte_offset + c_bytes > op.d_region.size_bytes) {
        std::cerr << "TensorUnit error: input D/C TMEM tile exceeds allocation" << std::endl;
        std::abort();
      }
      op.c_packets.assign(CMem::packet_count(cmd.fmt_c), TmemPacket{});
    }

    pending_mma_.push_back(std::move(op));
    ++perf_stats_.issued_macro_wmma;
    perf_stats_.pending_wmma_jobs_max =
        std::max<uint64_t>(perf_stats_.pending_wmma_jobs_max, pending_mma_.size());
    return true;
  }

  uint32_t scheduler_score(uint32_t, TcuType tcu_type) const {
    if (tcu_type == TcuType::TCU_MMA && pending_mma_.size() >= kMaxQueuedMma) {
      return 0;
    }
    return 1;
  }

  IssueBlockReason classify_issue_block(uint32_t, TcuType tcu_type) const {
    if (tcu_type == TcuType::TCU_MMA && pending_mma_.size() >= kMaxQueuedMma) {
      return IssueBlockReason::TcBusy;
    }
    return IssueBlockReason::None;
  }

  void record_issue_stall(IssueBlockReason reason) {
    switch (reason) {
    case IssueBlockReason::ANotReady: ++perf_stats_.stall_a_not_ready; break;
    case IssueBlockReason::BNotReady: ++perf_stats_.stall_b_not_ready; break;
    case IssueBlockReason::CNotReady: ++perf_stats_.stall_c_not_ready; break;
    case IssueBlockReason::TcBusy: ++perf_stats_.stall_tc_busy; break;
    case IssueBlockReason::NoTensorInstrCandidate:
      ++perf_stats_.stall_no_tensor_instr_candidate;
      break;
    case IssueBlockReason::MmaLoadTaddrNotReady:
      ++perf_stats_.stall_mma_load_taddr_not_ready;
      break;
    case IssueBlockReason::TaddrReuse:
      ++perf_stats_.stall_taddr_reuse;
      break;
    case IssueBlockReason::TaddrBusyDueToTmaLoad:
      ++perf_stats_.stall_taddr_busy_due_to_tma_load;
      break;
    case IssueBlockReason::TaddrBusyDueToTmaStoreOrShift:
      ++perf_stats_.stall_taddr_busy_due_to_tma_store_or_shift;
      break;
    case IssueBlockReason::SlotBusy:
      ++perf_stats_.stall_slot_busy;
      break;
    case IssueBlockReason::AMetaNotReady:
      ++perf_stats_.stall_a_meta_not_ready;
      break;
    case IssueBlockReason::None:
      break;
    }
  }

  void dump_debug_state(std::ostream& os) const {
    os << "[TensorUnit] pending_mma=" << pending_mma_.size()
       << " completed_tmem_rsp=" << completed_tensor_mem_responses_.size()
       << " tensorcore_active=" << tensorcore_.active()
       << "\n";
    if (!pending_mma_.empty()) {
      const auto& op = pending_mma_.front();
      os << "  front async=" << op.async_id
         << " stage=" << static_cast<uint32_t>(op.stage)
         << " a_pkt=" << op.next_a_packet << "/" << op.a_packets.size()
         << " b_pkt=" << op.next_b_packet << "/" << op.b_packets.size()
         << " c_pkt=" << op.next_c_packet << "/" << op.c_packets.size()
         << " d_pkt=" << op.next_d_packet << "/" << op.d_store_packets.size()
         << " subtile=" << op.issue_subtile
         << " accum_phase=" << op.issue_accum_phase
         << " final_retired=" << op.final_retired_subtiles
         << "\n";
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:
  // Accept only the subset of tcgen05.mma currently modeled by this C model.
  bool validate_mma(const TcDecodedMmaCmd& cmd) const {
    namespace vt = vortex::tensor;
    uint32_t shape_m = cmd.shape_m == 0 ? kTileM : cmd.shape_m;
    uint32_t shape_n = cmd.shape_n == 0 ? kTileN : cmd.shape_n;
    if (shape_m != kTileM || shape_n != kTileN) {
      return false;
    }
    auto supported_input = [](uint32_t fmt) {
      return fmt == vt::fp16::id || fmt == vt::fp8::id;
    };
    auto supported_accum_output = [](uint32_t fmt) {
      return fmt == vt::fp32::id || fmt == vt::fp16::id || fmt == vt::fp8::id;
    };
    if (!supported_input(cmd.fmt_a) || !supported_input(cmd.fmt_b)
     || !supported_accum_output(cmd.fmt_c) || !supported_accum_output(cmd.fmt_d)) {
      return false;
    }
    if (cmd.ws != 0) {
      return false;
    }
    if (cmd.sparsity_kind == vt::sparse_none) {
      if (cmd.sp != 0) {
        return false;
      }
    } else {
      if (cmd.sp == 0 || cmd.fmt_a != vt::fp8::id || cmd.fmt_b != vt::fp8::id) {
        return false;
      }
    }
    if (cmd.transpose_a != 0 || cmd.transpose_b != 0
     || cmd.saturate != 0 || cmd.output_negate != 0) {
      return false;
    }
    return true;
  }

  // Resolve a logical TADDR into its backing TMEM allocation and byte offset.
  bool resolve_taddr(uint32_t taddr, TaddrRegion* out) const {
    if (out == nullptr) {
      return false;
    }
    uint32_t lane_base = taddr & 0xffffu;
    uint32_t col_byte = (taddr >> 16) & 0xffffu;
    uint32_t col_base = 0;
    if (!core_->tmem_find_allocation_by_lane(lane_base, &col_base)) {
      return false;
    }
    uint32_t col_span = 0;
    uint32_t size_bytes = 0;
    if (!core_->tmem_query(col_base, &col_span, &size_bytes)) {
      return false;
    }
    if (lane_base < col_base) {
      return false;
    }
    uint32_t byte_offset = (lane_base - col_base) * Tmem::kColBytes + col_byte;
    if (byte_offset >= size_bytes) {
      return false;
    }
    out->col_base = col_base;
    out->col_span = col_span;
    out->byte_offset = byte_offset;
    out->size_bytes = size_bytes;
    return true;
  }

  // Keep the original instruction trace pipeline moving independently of the
  // asynchronous tensor macro operation.
  void latch_tensor_instruction_pipe() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty()) {
        continue;
      }
      auto trace = input.front();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      uint32_t delay = (tcu_type == TcuType::TCU_MMA) ? 4 : 1;
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      input.pop();
    }
  }

  // Collect packet-level TMEM responses so the active macro state machine can
  // consume them when its outstanding request completes.
  void receive_tensor_mem_responses() {
    if (simobject_->TensorMemRspIn.empty()) {
      return;
    }
    auto rsp = simobject_->TensorMemRspIn.front();
    completed_tensor_mem_responses_[rsp.request_id] = rsp;
    simobject_->TensorMemRspIn.pop();
  }

  // Reset local operand/result SRAMs before starting a new macro MMA.
  void initialize_op(MmaOp& op) {
    if (op.initialized) {
      return;
    }
    amem_.clear();
    bmem_.clear();
    cmem_.clear();
    dmem_.clear();
    tensorcore_.reset();
    op.initialized = true;
  }

  // Step the front macro MMA through fill, compute, and store phases.
  void advance_mma(MmaOp& op) {
    initialize_op(op);
    switch (op.stage) {
    case MmaStage::FillA:
      advance_fill_a(op);
      break;
    case MmaStage::FillMeta:
      advance_fill_meta(op);
      break;
    case MmaStage::FillB:
      advance_fill_b(op);
      break;
    case MmaStage::FillC:
      advance_fill_c(op);
      break;
    case MmaStage::Compute:
      advance_compute(op);
      break;
    case MmaStage::StoreD:
      advance_store_d(op);
      break;
    case MmaStage::Complete:
      break;
    }
  }

  // Read A packets from TMEM and fill AMem once the complete tile is present.
  void advance_fill_a(MmaOp& op) {
    if (op.pending_request_id != 0) {
      auto rsp_it = completed_tensor_mem_responses_.find(op.pending_request_id);
      if (rsp_it == completed_tensor_mem_responses_.end()) {
        ++perf_stats_.stall_tmem_read_port_busy;
        return;
      }
      op.a_packets.at(op.pending_packet_index) = rsp_it->second.read_packet;
      completed_tensor_mem_responses_.erase(rsp_it);
      op.pending_request_id = 0;
      ++op.next_a_packet;
    }

    if (op.next_a_packet >= op.a_packets.size()) {
      write_amem_lines(op);
      op.stage = (op.cmd.sparsity_kind != tensor::sparse_none)
          ? MmaStage::FillMeta
          : MmaStage::FillB;
      return;
    }

    TensorMemPortReq req{};
    req.request_id = next_tensor_mem_request_id_++;
    req.arbitration_age = (uint64_t(op.async_id) << 32) | op.next_a_packet;
    req.access_type = TensorMemPortReq::AccessType::Read;
    req.port_request.kind = Tmem::PortRequestKind::RegionRead;
    req.port_request.col_base = op.a_region.col_base;
    req.port_request.col_span = op.a_region.col_span;
    req.port_request.packet_idx = op.a_packet_base + op.next_a_packet;
    simobject_->TensorMemReqOut.push(req, 0);
    op.pending_request_id = req.request_id;
    op.pending_packet_index = op.next_a_packet;
  }

  // Read the optional sparse metadata packet associated with A.
  void advance_fill_meta(MmaOp& op) {
    if (op.pending_request_id != 0) {
      auto rsp_it = completed_tensor_mem_responses_.find(op.pending_request_id);
      if (rsp_it == completed_tensor_mem_responses_.end()) {
        return;
      }
      op.meta_packets.at(0) = rsp_it->second.read_packet;
      completed_tensor_mem_responses_.erase(rsp_it);
      op.pending_request_id = 0;
      op.stage = MmaStage::FillB;
      return;
    }

    TensorMemPortReq req{};
    req.request_id = next_tensor_mem_request_id_++;
    req.arbitration_age = (uint64_t(op.async_id) << 32) | 0x10000ull;
    req.access_type = TensorMemPortReq::AccessType::Read;
    req.port_request.kind = Tmem::PortRequestKind::RegionRead;
    req.port_request.col_base = op.a_region.col_base;
    req.port_request.col_span = op.a_region.col_span;
    req.port_request.packet_idx = op.meta_packet_base;
    simobject_->TensorMemReqOut.push(req, 0);
    op.pending_request_id = req.request_id;
    op.pending_packet_index = 0;
  }

  // Convert the fetched A packet sequence into AMem fill lines.
  void write_amem_lines(MmaOp& op) {
    uint32_t packets_per_line = AMem::packets_per_fill_line(op.cmd.fmt_a);
    for (uint32_t line = 0; line < AMem::kDepth; ++line) {
      std::vector<AMem::packet_t> line_packets;
      line_packets.reserve(packets_per_line);
      for (uint32_t p = 0; p < packets_per_line; ++p) {
        line_packets.push_back(op.a_packets.at(line * packets_per_line + p).bytes);
      }
      if (!amem_.write_fill_line(op.cmd.fmt_a, line, line_packets)) {
        std::cerr << "TensorUnit error: AMem fill failed" << std::endl;
        std::abort();
      }
    }
  }

  // Load B packets from shared local memory and fill BMem.
  void advance_fill_b(MmaOp& op) {
    if (op.next_b_packet < op.b_packets.size()) {
      uint64_t b_lmem_addr = shared_desc_to_lmem_addr(op.cmd.b_sdesc)
                           + uint64_t(op.next_b_packet) * Tmem::kPacketBytes;
      core_->lmem_read(op.b_packets.at(op.next_b_packet).data(),
                       b_lmem_addr,
                       Tmem::kPacketBytes);
      ++op.next_b_packet;
      return;
    }

    uint32_t packets_per_line = BMem::packets_per_fill_line(op.cmd.fmt_b,
                                                            op.cmd.sparsity_kind);
    for (uint32_t line = 0; line < BMem::kDepth; ++line) {
      std::vector<BMem::packet_t> line_packets;
      line_packets.reserve(packets_per_line);
      for (uint32_t p = 0; p < packets_per_line; ++p) {
        line_packets.push_back(op.b_packets.at(line * packets_per_line + p));
      }
      if (!bmem_.write_fill_line(op.cmd.fmt_b, line, line_packets,
                                 op.cmd.sparsity_kind)) {
        std::cerr << "TensorUnit error: BMem fill failed" << std::endl;
        std::abort();
      }
    }
    op.stage = (op.cmd.enable_input_d != 0) ? MmaStage::FillC : MmaStage::Compute;
  }

  // Read the optional input-D/C tile from TMEM before compute starts.
  void advance_fill_c(MmaOp& op) {
    if (op.pending_request_id != 0) {
      auto rsp_it = completed_tensor_mem_responses_.find(op.pending_request_id);
      if (rsp_it == completed_tensor_mem_responses_.end()) {
        ++perf_stats_.stall_tmem_read_port_busy;
        return;
      }
      op.c_packets.at(op.pending_packet_index) = rsp_it->second.read_packet;
      completed_tensor_mem_responses_.erase(rsp_it);
      op.pending_request_id = 0;
      ++op.next_c_packet;
    }

    if (op.next_c_packet >= op.c_packets.size()) {
      write_input_d_to_cmem(op);
      op.stage = MmaStage::Compute;
      return;
    }

    TensorMemPortReq req{};
    req.request_id = next_tensor_mem_request_id_++;
    req.arbitration_age = (uint64_t(op.async_id) << 32) | op.next_c_packet;
    req.access_type = TensorMemPortReq::AccessType::Read;
    req.port_request.kind = Tmem::PortRequestKind::RegionRead;
    req.port_request.col_base = op.d_region.col_base;
    req.port_request.col_span = op.d_region.col_span;
    req.port_request.packet_idx = op.d_packet_base + op.next_c_packet;
    simobject_->TensorMemReqOut.push(req, 0);
    op.pending_request_id = req.request_id;
    op.pending_packet_index = op.next_c_packet;
  }

  // Convert input-D packets into CMem subtiles for the first accumulation.
  void write_input_d_to_cmem(MmaOp& op) {
    uint32_t packets_per_subtile = CMem::packets_per_subtile(op.cmd.fmt_c);
    for (uint32_t subtile = 0; subtile < CMem::kDepth; ++subtile) {
      std::vector<CMem::packet_t> packets;
      packets.reserve(packets_per_subtile);
      for (uint32_t p = 0; p < packets_per_subtile; ++p) {
        packets.push_back(op.c_packets.at(subtile * packets_per_subtile + p).bytes);
      }
      if (!cmem_.write_fill_subtile(op.cmd.fmt_c, subtile, packets)) {
        std::cerr << "TensorUnit error: input D/C fill failed" << std::endl;
        std::abort();
      }
    }
  }

  // Check whether the next 8x8 primitive can be accepted by TensorCore.
  static uint32_t accum_phase_count(uint32_t sparsity_kind) {
    if (sparsity_kind == tensor::sparse_none) {
      return kDenseAccumPhases;
    }
    if (sparsity_kind == tensor::sparse_2_4
     || sparsity_kind == tensor::sparse_1_4) {
      return kSparseAccumPhases;
    }
    std::abort();
  }

  static uint32_t mem_k_phase_for_accum_phase(uint32_t sparsity_kind,
                                              uint32_t accum_phase) {
    return (sparsity_kind == tensor::sparse_none) ? (accum_phase / 2)
                                                  : accum_phase;
  }

  bool current_compute_ready(const MmaOp& op) const {
    if (op.issue_subtile >= kSubtiles) {
      return false;
    }
    if (op.issue_accum_phase == 0 && op.cmd.enable_input_d != 0
     && !cmem_.subtile_valid(op.issue_subtile)) {
      return false;
    }
    return tensorcore_.ready(true);
  }

  // Build one primitive from AMem/BMem/CMem and push it into TensorCore.
  void issue_current_primitive(MmaOp& op) {
    uint32_t subtile = op.issue_subtile;
    uint32_t accum_phase = op.issue_accum_phase;
    uint32_t mem_k_phase = mem_k_phase_for_accum_phase(op.cmd.sparsity_kind,
                                                       accum_phase);
    uint32_t storage_m = subtile / 2;
    uint32_t storage_n = subtile % 2;

    uint16_t a[kPrimitiveDim][kPrimitiveDim] = {};
    uint16_t b[kPrimitiveDim][kPrimitiveDim] = {};
    uint32_t c[kPrimitiveDim][kPrimitiveDim] = {};
    amem_.read_primitive(mem_k_phase * 2 + storage_m, a, false);
    bmem_.read_primitive(mem_k_phase * 2 + storage_n, b, false);
    uint16_t sparse_row_meta[kPrimitiveDim] = {};
    if (op.cmd.sparsity_kind != tensor::sparse_none) {
      if (op.meta_packets.empty()) {
        std::cerr << "TensorUnit error: sparse metadata packet not loaded"
                  << std::endl;
        std::abort();
      }
      uint32_t meta_line = storage_m * 2 + mem_k_phase;
      const uint8_t* meta_base =
          op.meta_packets.front().bytes.data() + meta_line * 16;
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        sparse_row_meta[row] = vortex::sparse::row_meta_bits(meta_base, row);
      }
    }
    if (accum_phase == 0 && op.cmd.enable_input_d != 0) {
      cmem_.read_subtile_fp22(subtile, c);
    }

    TensorCoreMeta meta{};
    meta.wid = op.wid;
    meta.async_id = op.async_id;
    meta.c_subtile_id = subtile;
    meta.accum_phase_id = accum_phase;
    meta.valid = true;

    tensorcore_.push_uop(a, b, c, meta, op.cmd.sparsity_kind, sparse_row_meta);
    ++perf_stats_.issued_primitive_tiles;
    if (perf_stats_.setup_end_cycle == 0) {
      perf_stats_.setup_end_cycle = perf_stats_.latency;
    }

    if (op.issue_accum_phase + 1 < accum_phase_count(op.cmd.sparsity_kind)) {
      ++op.issue_accum_phase;
    } else {
      op.issue_accum_phase = 0;
      ++op.issue_subtile;
    }
  }

  // Run TensorCore, retire completed primitives, and detect macro completion.
  void advance_compute(MmaOp& op) {
    if (current_compute_ready(op)) {
      issue_current_primitive(op);
    } else if (op.issue_subtile < kSubtiles) {
      ++perf_stats_.stall_tc_busy;
    }

    tensorcore_.tick(true);
    TensorCoreRetire retired{};
    if (tensorcore_.pop_retired(&retired)) {
      uint32_t subtile = retired.meta.c_subtile_id;
      if (retired.meta.accum_phase_id == 0) {
        dmem_.write_subtile_fp22(subtile, retired.fp22_out);
      } else {
        dmem_.accumulate_subtile_fp22(subtile, retired.fp22_out);
      }
      ++perf_stats_.retired_primitive_tiles;
      perf_stats_.epilogue_begin_cycle = perf_stats_.latency;
      if (retired.meta.accum_phase_id
          == (accum_phase_count(op.cmd.sparsity_kind) - 1)) {
        ++op.final_retired_subtiles;
      }
    }

    if (op.final_retired_subtiles == kSubtiles) {
      build_d_store_packets(op);
      op.stage = MmaStage::StoreD;
    }
  }

  // Repack the 16x16 D tile from DMem's subtile layout into TMEM packet order.
  void build_d_store_packets(MmaOp& op) {
    if (op.d_region.col_span < 32) {
      std::cerr << "TensorUnit error: D allocation must cover at least 32 lanes for warp tcgen05.ld"
                << std::endl;
      std::abort();
    }
    uint32_t elem_bytes = fmt_element_bytes(op.cmd.fmt_d);
    if (elem_bytes == 0) {
      std::cerr << "TensorUnit error: unsupported D output format"
                << " fmt_d=" << op.cmd.fmt_d << std::endl;
      std::abort();
    }
    std::vector<uint8_t> row_major(kTileM * kTileN * elem_bytes, 0);
    for (uint32_t subtile = 0; subtile < kSubtiles; ++subtile) {
      std::vector<DMem::packet_t> packets;
      if (!dmem_.dump_subtile_packets(op.cmd.fmt_d, subtile, &packets)) {
        std::cerr << "TensorUnit error: DMem dump failed" << std::endl;
        std::abort();
      }
      uint32_t storage_m = subtile / 2;
      uint32_t storage_n = subtile % 2;
      uint32_t rows_per_packet = dmem_rows_per_packet(op.cmd.fmt_d);
      for (uint32_t segment = 0; segment < packets.size(); ++segment) {
        const auto& packet = packets.at(segment);
        for (uint32_t row_in_packet = 0; row_in_packet < rows_per_packet; ++row_in_packet) {
          uint32_t local_row = segment * rows_per_packet + row_in_packet;
          for (uint32_t col = 0; col < kPrimitiveDim; ++col) {
            uint32_t global_row = storage_m * kPrimitiveDim + local_row;
            uint32_t global_col = storage_n * kPrimitiveDim + col;
            uint32_t src = (row_in_packet * kPrimitiveDim + col) * elem_bytes;
            uint32_t dst = (global_row * kTileN + global_col) * elem_bytes;
            std::memcpy(row_major.data() + dst, packet.data() + src, elem_bytes);
          }
        }
      }
    }

    uint32_t packet_count = op.d_region.col_span * Tmem::kPacketsPerCol;
    op.d_store_packets.assign(packet_count, TmemPacket{});
    for (uint32_t elem = 0; elem < kTileM * kTileN; ++elem) {
      uint32_t lane = elem % 32;
      uint32_t chunk = elem / 32;
      uint32_t byte_offset = lane * Tmem::kColBytes + chunk * elem_bytes;
      uint32_t packet_idx = byte_offset / Tmem::kPacketBytes;
      uint32_t packet_off = byte_offset % Tmem::kPacketBytes;
      std::memcpy(op.d_store_packets.at(packet_idx).bytes.data() + packet_off,
                  row_major.data() + elem * elem_bytes,
                  elem_bytes);
    }
  }

  // Write result packets back to TMEM and wait for each packet response.
  void advance_store_d(MmaOp& op) {
    if (op.pending_request_id != 0) {
      auto rsp_it = completed_tensor_mem_responses_.find(op.pending_request_id);
      if (rsp_it == completed_tensor_mem_responses_.end()) {
        ++perf_stats_.stall_tmem_write_port_busy;
        return;
      }
      completed_tensor_mem_responses_.erase(rsp_it);
      op.pending_request_id = 0;
      ++op.next_d_packet;
    }

    if (op.next_d_packet >= op.d_store_packets.size()) {
      op.stage = MmaStage::Complete;
      return;
    }

    TensorMemPortReq req{};
    req.request_id = next_tensor_mem_request_id_++;
    req.arbitration_age = (uint64_t(op.async_id) << 32) | op.next_d_packet;
    req.access_type = TensorMemPortReq::AccessType::Write;
    req.port_request.kind = Tmem::PortRequestKind::RegionWrite;
    req.port_request.col_base = op.d_region.col_base;
    req.port_request.col_span = op.d_region.col_span;
    req.port_request.packet_idx = op.d_packet_base + op.next_d_packet;
    req.write_packet = op.d_store_packets.at(op.next_d_packet);
    simobject_->TensorMemReqOut.push(req, 0);
    op.pending_request_id = req.request_id;
    op.pending_packet_index = op.next_d_packet;
  }

  // Notify Core that the macro MMA async_id has completed.
  void complete_front_mma() {
    TensorAsyncOpCompletion completion{};
    completion.async_id = pending_mma_.front().async_id;
    simobject_->TensorAsyncOpCompletionOut.push(completion, 0);
    ++perf_stats_.retired_macro_wmma;
    pending_mma_.pop_front();
  }

  TensorUnit* simobject_;
  Core* core_;
  const Arch& arch_;
  PerfStats perf_stats_;
  AMem amem_;
  BMem bmem_;
  CMem cmem_;
  DMem dmem_;
  TensorCoreTop tensorcore_;
  TcDecode tc_decode_;
  std::deque<MmaOp> pending_mma_;
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

void TensorUnit::dispatch_tcu_mma(uint32_t wid,
                                  uint32_t rs1_value,
                                  uint32_t rs2_value,
                                  uint32_t qualifier,
                                  ExeTraceData* trace_data) {
  reset_trace_for_async_dispatch(trace_data);
  impl_->enqueue_tcu_mma(wid, rs1_value, rs2_value, qualifier, trace_data);
}

uint32_t TensorUnit::scheduler_score(uint32_t wid, TcuType tcu_type) const {
  return impl_->scheduler_score(wid, tcu_type);
}

TensorUnit::IssueBlockReason TensorUnit::classify_issue_block(uint32_t wid, TcuType tcu_type) const {
  return impl_->classify_issue_block(wid, tcu_type);
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

void TensorUnit::dump_debug_state(std::ostream& os) const {
  impl_->dump_debug_state(os);
}
