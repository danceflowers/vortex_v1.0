// ============================================================================
// TensorMemPipeline 实现 -- TMEM ↔ SRAM 传输引擎（多 pipeline 友好版）
// ============================================================================
//
// 设计目标：
//   1. 保留原始 cycle 语义：多个 MemUop 可在同一 tick 推进本地阶段；
//      TMEM issue 端口仍然是共享瓶颈。
//   2. 把主调度函数拆成 fill/store/complete 三类 helper，减少大函数中的重复分支。
//   3. 把 FillA/FillB/FillC 的公共收尾逻辑收敛到统一 helper，降低后续扩展成本。
// ============================================================================

#include "open_tensorcore/local_memory/tensor_mem_pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "core.h"
#include "open_tensorcore/local_memory/tensor_local_mem_arbiter.h"
#include "open_tensorcore/tensor_control/tensor_mem_manager.h"
#include "open_tensorcore/tensor_helper/tensor_debug_utils.h"

namespace vortex {

namespace {

using LocalFillAction = TensorMemPipeline::LocalFillAction;

enum class StepResult : uint8_t {
  NoProgress = 0,
  Progress,
  Blocked,
  IssuedTmemRequest,
};

struct TensorMemPipelineCtx {
  Core& core;
  SimPort<TensorMemPortReq>& req_out;
  SimPort<TensorAsyncOpCompletion>& completion_out;
 // TensorLocalMemArbiter& mem_arbiter;

  AMem& amem;
  BMem& bmem;
  CMem& cmem;
  DMem& dmem;
  MetaMem& metamem;

  std::deque<tud::MemUop>& mem_ops;
  std::unordered_map<uint32_t, uint32_t>& pending_mem_ops;
  std::unordered_map<uint64_t, TensorMemPortRsp>& completed_tensor_mem_responses;

  uint64_t& next_tensor_mem_request_id;

  tud::AMemState& amem_state;
  tud::BMemState& bmem_state;
  tud::CMemState& cmem_state;
  tud::DMemState& dmem_state;

  TensorUnit::PerfStats& perf_stats;
  uint32_t& rr_index;

  // 多个 tensor memory pipeline 可并行推进本地阶段。
  // 这里仅建模共享 TMEM issue 端口：每次 advance 只允许发出一个 TMEM 请求。
  bool tmem_bus_used = false;
};

static StepResult progress_if(bool progressed) {
  return progressed ? StepResult::Progress : StepResult::NoProgress;
}

static bool is_fill_uop(const tud::MemUop& uop) {
  using Kind = tud::MemUop::Kind;
  return uop.kind == Kind::FillA || uop.kind == Kind::FillB || uop.kind == Kind::FillC;
}

static bool is_store_uop(const tud::MemUop& uop) {
  return uop.kind == tud::MemUop::Kind::StoreC;
}

static MetaMem::packet_t to_meta_packet(const Core::TmemPacket& packet) {
  MetaMem::packet_t out{};
  std::copy_n(packet.bytes.begin(), out.size(), out.begin());
  return out;
}

template <typename PacketT>
static Core::TmemPacket to_tmem_packet(const PacketT& packet) {
  Core::TmemPacket out;
  std::copy_n(packet.begin(), packet.size(), out.bytes.begin());
  return out;
}

static uint64_t mem_uop_request_age(const tud::MemUop& uop, uint32_t ordinal) {
  return (static_cast<uint64_t>(uop.async_id) << 32) | ordinal;
}

static bool lookup_window_packet_idx(const tud::MemUop& uop,
                                     uint32_t local_packet_idx,
                                     Core* core,
                                     uint32_t* packet_idx) {
  if (!uop.separate_handle || nullptr == core || nullptr == packet_idx) {
    return false;
  }
  const TmemWindowPlan* window = nullptr;
  if (!core->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
    return false;
  }
  if (window->packets_per_tile == 0
   || local_packet_idx >= window->packets_per_tile
   || uop.tile_idx >= window->tile_count) {
    return false;
  }
  *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
  return true;
}

static bool lookup_meta_window_packet_idx(const tud::MemUop& uop,
                                          uint32_t local_packet_idx,
                                          Core* core,
                                          uint32_t* packet_idx) {
  if (!uop.separate_handle || nullptr == core || nullptr == packet_idx) {
    return false;
  }
  const TmemWindowPlan* window = nullptr;
  auto shadow_window_id = tud::meta_shadow_window_id(uop.window_id);
  if (!core->lookup_tmem_window(uop.handle, shadow_window_id, &window)) {
    return false;
  }
  if (window->packets_per_tile == 0
   || local_packet_idx >= window->packets_per_tile
   || uop.tile_idx >= window->tile_count) {
    return false;
  }
  *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
  return true;
}

static void fill_stage_step(tud::MemUop* uop) {
  if (nullptr == uop) {
    return;
  }
  switch(uop->fill_stage){
    case tud::FillPipelineStage::LocalWrite :
  if(uop->fill_memwrite_valid){
    uop->fill_stage = tud::FillPipelineStage::Done;
    return;
  }break;

    case tud::FillPipelineStage::Conversion :
  if (uop->fill_convert_valid) {
    uop->fill_stage = tud::FillPipelineStage::LocalWrite;
    return;
  } break;

  case tud::FillPipelineStage::Ingress :
  if (uop->fill_input_valid) {
    uop->fill_stage = tud::FillPipelineStage::Conversion;
    return;
  } break;

  case tud::FillPipelineStage::WaitTmemRsp :
  if  (uop->pending_tmem_request_tag != 0
   || uop->remaining_tmem_read_packets != 0
   || uop->fill_input_packet_count != 0) {
    uop->fill_stage = tud::FillPipelineStage::WaitTmemRsp;
    return;
  }else if (uop->fill_ingress_valid) {
    uop->fill_stage = tud::FillPipelineStage::Ingress;
    return;
  } break;

  default : break;

  
}
}

static void refresh_store_stage(tud::MemUop* uop) {
  if (nullptr == uop) {
    return;
  }
  if (uop->store_conv_valid) {
    uop->store_stage = (uop->staged_store_count < tud::kOutputPacketBufferDepth)
        ? tud::StorePipelineStage::StageBuffer
        : tud::StorePipelineStage::Emit;
    return;
  }
  if (uop->store_raw_subtile_valid) {
    uop->store_stage = (uop->staged_store_count >= tud::kOutputPacketBufferDepth)
        ? tud::StorePipelineStage::Emit
        : tud::StorePipelineStage::Conversion;
    return;
  }
  if (uop->pending_tmem_request_tag != 0 || uop->staged_store_count != 0) {
    uop->store_stage = tud::StorePipelineStage::Emit;
    return;
  }
  if (uop->remaining_cmem_dump_subtiles != 0) {
    uop->store_stage = tud::StorePipelineStage::Read;
    return;
  }
  uop->store_stage = tud::StorePipelineStage::Done;
}

static bool fill_read_backpressured(const tud::MemUop& uop) {
  return uop.fill_ingress_valid
      || uop.fill_input_valid
      || uop.fill_convert_valid
      || (uop.fill_input_packets_needed != 0
       && uop.fill_input_packet_count >= uop.fill_input_packets_needed);
}

static void reset_fill_input(tud::MemUop& uop) {
  uop.fill_input_valid = false;
  uop.fill_input_kind = tud::FillConvertKind::None;
  uop.fill_input_packet_count = 0;
  uop.fill_input_packets_needed = 0;
}

static void finish_fill_conversion(tud::MemUop& uop) {
  uop.fill_convert_kind = uop.fill_input_kind;
  uop.fill_convert_valid = true;
  uop.fill_convert_index = uop.fill_input_index;
  reset_fill_input(uop);
  fill_stage_step(&uop);
}

static void clear_fill_conversion(tud::MemUop& uop) {
  uop.fill_convert_valid = false;
  uop.fill_convert_kind = tud::FillConvertKind::None;
}

static bool begin_fill_input(tud::MemUop& uop, uint32_t payload_fmt) {
  if (uop.fill_input_packet_count != 0) {
    return true;
  }

  switch (uop.kind) {
  case tud::MemUop::Kind::FillA:
    if (uop.remaining_amem_fill_lines == 0) {
      return false;
    }
    uop.fill_input_kind = tud::FillConvertKind::ALine;
    uop.fill_input_index = uop.line_idx;
    uop.fill_input_packets_needed = AMem::packets_per_fill_line(payload_fmt);
    break;
  case tud::MemUop::Kind::FillB:
    if (uop.remaining_bmem_fill_lines == 0) {
      return false;
    }
    uop.fill_input_kind = tud::FillConvertKind::BLine;
    uop.fill_input_index = uop.line_idx;
    uop.fill_input_packets_needed = BMem::packets_per_fill_line(payload_fmt);
    break;
  case tud::MemUop::Kind::FillC:
    if (uop.remaining_cmem_fill_subtiles == 0) {
      return false;
    }
    uop.fill_input_kind = tud::FillConvertKind::CSubtile;
    uop.fill_input_index = uop.line_idx;
    uop.fill_input_packets_needed = CMem::packets_per_subtile(payload_fmt);
    break;
  default:
    return false;
  }

  if (uop.fill_input_packets_needed == 0
   || uop.fill_input_packets_needed > uop.fill_input_packets.size()) {
    std::abort();
  }
  return true;
}

static std::vector<Core::TmemPacket> make_fill_packet_slice(const tud::MemUop& uop) {
  std::vector<Core::TmemPacket> packet_slice;
  packet_slice.reserve(uop.fill_input_packet_count);
  for (uint32_t packet = 0; packet < uop.fill_input_packet_count; ++packet) {
    packet_slice.push_back(uop.fill_input_packets.at(packet));
  }
  return packet_slice;
}

static StepResult step_fill_local_write(TensorMemPipelineCtx& ctx, tud::MemUop& uop) {
  const auto action = TensorMemPipeline::select_fill_local_write(uop);
  if (action == LocalFillAction::None) {
    return StepResult::NoProgress;
  }
  return progress_if(TensorMemPipeline::perform_fill_local_write(
      &uop, action, &ctx.amem, &ctx.bmem, &ctx.cmem, &ctx.metamem));
}

static StepResult step_fill_tmem(TensorMemPipelineCtx& ctx, tud::MemUop& uop) {
  if (uop.remaining_tmem_read_packets == 0 || fill_read_backpressured(uop)) {
    return StepResult::NoProgress;
  }

  if (uop.pending_tmem_request_tag != 0) {
    TensorMemPortRsp response{};
    if (!TensorMemPipeline::take_tensor_mem_response(
            uop.pending_tmem_request_tag,
            &ctx.completed_tensor_mem_responses,
            &response)) {
      ++ctx.perf_stats.stall_tmem_read_port_busy;
      return StepResult::Blocked;
    }

    uop.pending_tmem_request_tag = 0;
    return progress_if(TensorMemPipeline::accept_fill_read_response(
        &uop,
        response,
        tud::fill_payload_packet_count(uop, ctx.amem_state, ctx.bmem_state, ctx.cmem_state)));
  }

  if (ctx.tmem_bus_used) {
    return StepResult::Blocked;
  }

  uop.pending_tmem_request_tag = TensorMemPipeline::issue_fill_read_request(
      uop,
      &ctx.core,
      &ctx.req_out,
      &ctx.next_tensor_mem_request_id,
      ctx.amem_state,
      ctx.bmem_state,
      ctx.cmem_state);
  ctx.tmem_bus_used = true;
  return StepResult::IssuedTmemRequest;
}

static StepResult step_fill_uop(TensorMemPipelineCtx& ctx, tud::MemUop& uop) {
  fill_stage_step(&uop);
  const auto payload_fmt = tud::mem_uop_payload_fmt(
      uop, ctx.amem_state, ctx.bmem_state, ctx.cmem_state);

  switch (uop.fill_stage) {
  case tud::FillPipelineStage::LocalWrite:
    return step_fill_local_write(ctx, uop);
  case tud::FillPipelineStage::Conversion:
    return progress_if(TensorMemPipeline::step_fill_conversion(&uop, payload_fmt));
  case tud::FillPipelineStage::Ingress:
    return progress_if(TensorMemPipeline::stage_fill_conversion_input(&uop, payload_fmt));
  case tud::FillPipelineStage::WaitTmemRsp:
    return step_fill_tmem(ctx, uop);
  case tud::FillPipelineStage::Done:
    return StepResult::NoProgress;
  }

  return StepResult::NoProgress;
}

static StepResult step_store_emit(TensorMemPipelineCtx& ctx, tud::MemUop& uop) {
  if (uop.pending_tmem_request_tag != 0) {
    TensorMemPortRsp response{};
    if (!TensorMemPipeline::take_tensor_mem_response(
            uop.pending_tmem_request_tag,
            &ctx.completed_tensor_mem_responses,
            &response)) {
      ++ctx.perf_stats.stall_tmem_write_port_busy;
      return StepResult::Blocked;
    }

    uop.pending_tmem_request_tag = 0;
    refresh_store_stage(&uop);
    return progress_if(TensorMemPipeline::emit_store_packet(&uop));
  }

  if (uop.staged_store_count == 0) {
    return StepResult::NoProgress;
  }

  if (ctx.tmem_bus_used) {
    return StepResult::Blocked;
  }

  uop.pending_tmem_request_tag = TensorMemPipeline::issue_store_write_request(
      uop,
      &ctx.core,
      &ctx.req_out,
      &ctx.next_tensor_mem_request_id);
  ctx.tmem_bus_used = true;
  return StepResult::IssuedTmemRequest;
}

static StepResult step_store_uop(TensorMemPipelineCtx& ctx, tud::MemUop& uop) {
  refresh_store_stage(&uop);

  switch (uop.store_stage) {
  case tud::StorePipelineStage::Emit:
    return step_store_emit(ctx, uop);
  case tud::StorePipelineStage::StageBuffer:
    return progress_if(TensorMemPipeline::stage_store_converted_packet(&uop));
  case tud::StorePipelineStage::Conversion:
    return progress_if(TensorMemPipeline::advance_store_conversion(&uop, ctx.dmem_state));
  case tud::StorePipelineStage::Read:
    return progress_if(TensorMemPipeline::stage_store_read(
        &uop, ctx.cmem_state, ctx.dmem_state, &ctx.cmem, &ctx.dmem));
  case tud::StorePipelineStage::Done:
    return StepResult::NoProgress;
  }

  return StepResult::NoProgress;
}

static StepResult step_legacy_uop(tud::MemUop& uop) {
  bool progressed = false;

  if (uop.remaining_amem_fill_lines != 0
   || uop.remaining_bmem_fill_lines != 0
   || uop.remaining_cmem_fill_subtiles != 0
   || uop.remaining_cmem_dump_subtiles != 0
   || uop.remaining_metamem_fill_packets != 0) {
    if (uop.remaining_amem_fill_lines != 0) {
      --uop.remaining_amem_fill_lines;
      progressed = true;
    }
    if (uop.remaining_bmem_fill_lines != 0) {
      --uop.remaining_bmem_fill_lines;
      progressed = true;
    }
    if (uop.remaining_cmem_fill_subtiles != 0) {
      --uop.remaining_cmem_fill_subtiles;
      progressed = true;
    }
    if (uop.remaining_cmem_dump_subtiles != 0) {
      --uop.remaining_cmem_dump_subtiles;
      progressed = true;
    }
    if (uop.remaining_metamem_fill_packets != 0) {
      --uop.remaining_metamem_fill_packets;
      progressed = true;
    }
    return progress_if(progressed);
  }

  if (uop.remaining_tmem_write_packets != 0) {
    std::cerr << "TensorUnit error: reached legacy tensor writeback path"
              << " kind=" << static_cast<uint32_t>(uop.kind)
              << " handle=" << uop.handle
              << " window=" << uop.window_id
              << " tile=" << uop.tile_idx
              << std::endl;
    std::abort();
  }

  return StepResult::NoProgress;
}

static void mark_fill_mem_valid(tud::MemUop::Kind kind,
                                tud::AMemState* amem_state,
                                tud::BMemState* bmem_state,
                                tud::CMemState* cmem_state) {
  switch (kind) {
  case tud::MemUop::Kind::FillA:
    TensorMemManager::mark_a_valid(amem_state);
    break;
  case tud::MemUop::Kind::FillB:
    TensorMemManager::mark_b_valid(bmem_state);
    break;
  case tud::MemUop::Kind::FillC:
    TensorMemManager::mark_c_valid(cmem_state);
    break;
  default:
    std::abort();
  }
}

static void complete_mem_uop(TensorMemPipelineCtx& ctx, size_t idx) {
  auto& uop = ctx.mem_ops.at(idx);

  switch (uop.kind) {
  case tud::MemUop::Kind::FillA:
  case tud::MemUop::Kind::FillB:
  case tud::MemUop::Kind::FillC: {
    const auto async_id = uop.async_id;
    const auto kind = uop.kind;

    auto pending_it = ctx.pending_mem_ops.find(async_id);
    if (pending_it == ctx.pending_mem_ops.end() || pending_it->second == 0) {
      std::abort();
    }

    --pending_it->second;
    if (pending_it->second == 0) {
      ctx.pending_mem_ops.erase(pending_it);
      mark_fill_mem_valid(kind, &ctx.amem_state, &ctx.bmem_state, &ctx.cmem_state);
      ctx.completion_out.push({async_id}, 1);
    }

    ctx.mem_ops.erase(ctx.mem_ops.begin() + idx);
    return;
  }

  case tud::MemUop::Kind::StoreC: {
    const auto async_id = uop.async_id;
    ctx.dmem_state.d_store_pending = false;
    ctx.dmem_state.d_valid = ctx.dmem.valid();
    ctx.completion_out.push({async_id}, 1);
    ctx.mem_ops.erase(ctx.mem_ops.begin() + idx);
    return;
  }

  default:
    std::abort();
  }
}

} // namespace

// ============================================================================
// MemUop 完成检测
// ============================================================================

bool TensorMemPipeline::mem_op_complete(const tud::MemUop& op) {
  return op.remaining_tmem_read_packets == 0
      && op.remaining_tmem_write_packets == 0
      && op.remaining_amem_fill_lines == 0
      && op.remaining_bmem_fill_lines == 0
      && op.remaining_cmem_fill_subtiles == 0
      && op.remaining_cmem_dump_subtiles == 0
      && op.remaining_metamem_fill_packets == 0
      && !op.fill_ingress_valid
      && !op.fill_input_valid
      && op.fill_input_packet_count == 0
      && !op.fill_convert_valid
      && !op.store_raw_subtile_valid
      && !op.store_conv_valid
      && op.staged_store_count == 0;
}

void TensorMemPipeline::receive_one_tmem_response(
    SimPort<TensorMemPortRsp>* rsp_in,
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses) {
  if (nullptr == rsp_in || nullptr == completed_tensor_mem_responses || rsp_in->empty()) {
    return;
  }

  auto response = rsp_in->front();
  (*completed_tensor_mem_responses)[response.request_id] = response;
  rsp_in->pop();
}

bool TensorMemPipeline::take_tensor_mem_response(
    uint64_t request_id,
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses,
    TensorMemPortRsp* out) {
  if (nullptr == completed_tensor_mem_responses) {
    return false;
  }

  auto it = completed_tensor_mem_responses->find(request_id);
  if (it == completed_tensor_mem_responses->end()) {
    return false;
  }

  if (out != nullptr) {
    *out = it->second;
  }
  completed_tensor_mem_responses->erase(it);
  return true;
}

TensorMemPipeline::LocalFillAction TensorMemPipeline::select_fill_local_write(const tud::MemUop& uop) {
  if (uop.fill_convert_valid) {
    switch (uop.fill_convert_kind) {
    case tud::FillConvertKind::ALine:
      return LocalFillAction::AData;
    case tud::FillConvertKind::BLine:
      return LocalFillAction::BData;
    case tud::FillConvertKind::CSubtile:
      return LocalFillAction::CData;
    case tud::FillConvertKind::None:
    default:
      break;
    }
  }

  if (uop.kind == tud::MemUop::Kind::FillA
   && uop.remaining_metamem_fill_packets != 0
   && !uop.staged_meta_packets.empty()) {
    return LocalFillAction::Meta;
  }

  return LocalFillAction::None;
}

// ============================================================================
// Fill 阶段函数
// ============================================================================

bool TensorMemPipeline::stage_fill_conversion_input(tud::MemUop* uop, uint32_t payload_fmt) {
  if (nullptr == uop) {
    return false;
  }
  if (!uop->fill_ingress_valid || uop->fill_input_valid || uop->fill_convert_valid) {
    return false;
  }
  if (!begin_fill_input(*uop, payload_fmt)) {
    return false;
  }
  if (uop->fill_input_packet_count >= uop->fill_input_packets_needed) {
    std::abort();
  }

  uop->fill_input_packets.at(uop->fill_input_packet_count++) = uop->fill_ingress_packet;
  uop->fill_ingress_valid = false;
  uop->fill_input_valid = (uop->fill_input_packet_count == uop->fill_input_packets_needed);

  fill_stage_step(uop);
  return true;
}

bool TensorMemPipeline::step_fill_conversion(tud::MemUop* uop, uint32_t payload_fmt) {
  if (nullptr == uop || uop->fill_convert_valid || !uop->fill_input_valid) {
    return false;
  }

  auto packet_slice = make_fill_packet_slice(*uop);
  bool converted = false;

  switch (uop->fill_input_kind) {
  case tud::FillConvertKind::ALine:
    converted = AMem::convert_fill_packets(
        payload_fmt,
        tud::copy_packets<AMem::packet_t>(packet_slice),
        uop->fill_convert_fp9);
    break;
  case tud::FillConvertKind::BLine:
    converted = BMem::convert_fill_packets(
        payload_fmt,
        tud::copy_packets<BMem::packet_t>(packet_slice),
        uop->fill_convert_fp9);
    break;
  case tud::FillConvertKind::CSubtile:
    converted = CMem::convert_fill_packets(
        payload_fmt,
        tud::copy_packets<CMem::packet_t>(packet_slice),
        uop->fill_convert_fp22);
    break;
  case tud::FillConvertKind::None:
  default:
    return false;
  }

  if (!converted) {
    std::abort();
  }

  finish_fill_conversion(*uop);
  return true;
}

bool TensorMemPipeline::perform_fill_local_write(
    tud::MemUop* uop,
    LocalFillAction action,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    MetaMem* metamem) {
  if (nullptr == uop || nullptr == amem || nullptr == bmem || nullptr == cmem || nullptr == metamem) {
    return false;
  }

  auto finish_data_write = [&]() {
    clear_fill_conversion(*uop);
    fill_stage_step(uop);
    return true;
  };

  switch (action) {
  case LocalFillAction::AData: {
    const auto line_idx = uop->fill_convert_index;
    if (!amem->write_converted_line(line_idx, uop->fill_convert_fp9)) {
      std::cerr << "TensorUnit error: AMem fill write failed"
                << " line=" << line_idx
                << std::endl;
      std::abort();
    }
    --uop->remaining_amem_fill_lines;
    return finish_data_write();
  }

  case LocalFillAction::BData: {
    const auto line_idx = uop->fill_convert_index;
    if (!bmem->write_converted_line(line_idx, uop->fill_convert_fp9)) {
      std::cerr << "TensorUnit error: BMem fill write failed"
                << " line=" << line_idx
                << std::endl;
      std::abort();
    }
    --uop->remaining_bmem_fill_lines;
    return finish_data_write();
  }

  case LocalFillAction::CData: {
    const auto subtile_idx = uop->fill_convert_index;
    if (!cmem->write_converted_subtile(subtile_idx, uop->fill_convert_fp22)) {
      std::cerr << "TensorUnit error: CMem fill write failed"
                << " subtile=" << subtile_idx
                << std::endl;
      std::abort();
    }
    --uop->remaining_cmem_fill_subtiles;
    return finish_data_write();
  }

  case LocalFillAction::Meta: {
    if (uop->staged_meta_packets.empty()) {
      std::abort();
    }

    const auto packet = to_meta_packet(uop->staged_meta_packets.front());
    if (!metamem->write_fill_packet(packet)) {
      std::abort();
    }

    // staged_meta_packets 如果是 vector，这里仍然是 O(n)。
    // 真正的性能优化应当在 MemUop 内把它改成 ring buffer/deque。
    uop->staged_meta_packets.erase(uop->staged_meta_packets.begin());
    --uop->remaining_metamem_fill_packets;
    fill_stage_step(uop);
    return true;
  }

  case LocalFillAction::None:
  default:
    return false;
  }
}

bool TensorMemPipeline::accept_fill_read_response(
    tud::MemUop* uop,
    const TensorMemPortRsp& response,
    uint32_t payload_packets) {
  if (nullptr == uop || uop->remaining_tmem_read_packets == 0) {
    return false;
  }

  if (uop->next_payload_packet_idx < payload_packets) {
    if (uop->fill_ingress_valid) {
      std::cerr << "TensorUnit error: payload ingress buffer overflow"
                << " kind=" << static_cast<uint32_t>(uop->kind)
                << " line_idx=" << uop->line_idx
                << " next_payload=" << uop->next_payload_packet_idx
                << std::endl;
      std::abort();
    }

    uop->fill_ingress_packet = response.read_packet;
    uop->fill_ingress_valid = true;
    ++uop->next_payload_packet_idx;
  } else {
    uop->staged_meta_packets.push_back(response.read_packet);
    ++uop->next_meta_packet_idx;
  }

  --uop->remaining_tmem_read_packets;
  fill_stage_step(uop);
  return true;
}

uint64_t TensorMemPipeline::issue_fill_read_request(
    const tud::MemUop& uop,
    Core* core,
    SimPort<TensorMemPortReq>* req_out,
    uint64_t* next_tensor_mem_request_id,
    const tud::AMemState& amem_state,
    const tud::BMemState& bmem_state,
    const tud::CMemState& cmem_state) {
  if (nullptr == core || nullptr == req_out || nullptr == next_tensor_mem_request_id) {
    return 0;
  }

  TensorMemPortReq request{};
  request.request_id = (*next_tensor_mem_request_id)++;
  request.arbitration_age = mem_uop_request_age(
      uop, uop.next_payload_packet_idx + uop.next_meta_packet_idx);
  request.access_type = TensorMemPortReq::AccessType::Read;
  request.port_request.age = request.arbitration_age;

  const auto payload_packets = tud::fill_payload_packet_count(
      uop, amem_state, bmem_state, cmem_state);

  if (uop.next_payload_packet_idx < payload_packets) {
    uint32_t packet_idx = 0;
    if (!lookup_window_packet_idx(uop, uop.next_payload_packet_idx, core, &packet_idx)) {
      std::cerr << "TensorUnit error: missing TMEM window mapping for tensor payload read"
                << " handle=" << uop.handle
                << " window=" << uop.window_id
                << " tile=" << uop.tile_idx
                << " local_packet=" << uop.next_payload_packet_idx
                << " kind=" << static_cast<uint32_t>(uop.kind)
                << " payload_fmt=" << tud::mem_uop_payload_fmt(uop, amem_state, bmem_state, cmem_state)
                << std::endl;

      const TmemWindowPlan* window = nullptr;
      if (core->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
        log_window_plan_summary(
            "TensorUnit detail: payload window", uop.handle, uop.window_id, uop.tile_idx, window);
      }
      std::abort();
    }

    request.port_request.kind = Core::TmemRequestKind::WindowRead;
    request.port_request.handle = uop.handle;
    request.port_request.window_id = uop.window_id;
    request.port_request.packet_idx = packet_idx;
  } else {
    uint32_t meta_packet_idx = 0;
    const auto shadow_window_id = tud::meta_shadow_window_id(uop.window_id);
    if (!lookup_meta_window_packet_idx(uop, uop.next_meta_packet_idx, core, &meta_packet_idx)) {
      std::cerr << "TensorUnit error: missing TMEM shadow window mapping for tensor meta read"
                << " handle=" << uop.handle
                << " window=" << shadow_window_id
                << " tile=" << uop.tile_idx
                << " local_packet=" << uop.next_meta_packet_idx
                << " kind=" << static_cast<uint32_t>(uop.kind)
                << std::endl;

      const TmemWindowPlan* window = nullptr;
      if (core->lookup_tmem_window(uop.handle, shadow_window_id, &window)) {
        log_window_plan_summary(
            "TensorUnit detail: meta shadow window", uop.handle, shadow_window_id, uop.tile_idx, window);
      }
      std::abort();
    }

    request.port_request.kind = Core::TmemRequestKind::WindowRead;
    request.port_request.handle = uop.handle;
    request.port_request.window_id = shadow_window_id;
    request.port_request.packet_idx = meta_packet_idx;
  }

  req_out->push(request);
  return request.request_id;
}

// ============================================================================
// Store 阶段函数（从 DMem 读出，精度转换，写回 TMEM）
// ============================================================================

bool TensorMemPipeline::stage_store_read(
    tud::MemUop* uop,
    const tud::CMemState& c_state,
    const tud::DMemState& d_state,
    CMem* cmem,
    DMem* dmem) {
  (void)c_state;
  (void)cmem;

  if (nullptr == uop || nullptr == dmem) {
    return false;
  }
  if (uop->remaining_cmem_dump_subtiles == 0 || uop->store_raw_subtile_valid) {
    return false;
  }

  const auto subtile_idx = DMem::dump_subtiles(d_state.fmt_d) - uop->remaining_cmem_dump_subtiles;
  if (!dmem->valid()) {
    std::cerr << "TensorUnit error: DMem dump failed"
              << " subtile=" << subtile_idx
              << " fmt_d=" << d_state.fmt_d
              << std::endl;
    std::abort();
  }

  dmem->read_subtile_fp22(subtile_idx, uop->store_raw_subtile);
  uop->store_raw_subtile_valid = true;
  uop->store_raw_next_segment = 0;
  --uop->remaining_cmem_dump_subtiles;

  refresh_store_stage(uop);
  return true;
}

bool TensorMemPipeline::advance_store_conversion(tud::MemUop* uop, const tud::DMemState& d_state) {
  if (nullptr == uop) {
    return false;
  }
  if (uop->store_conv_valid || !uop->store_raw_subtile_valid) {
    return false;
  }

  const auto packets_per_subtile = DMem::packets_per_subtile(d_state.fmt_d);
  if (packets_per_subtile == 0) {
    std::abort();
  }

  DMem::packet_t packet{};
  if (!DMem::build_dump_packet(
          d_state.fmt_d,
          uop->store_raw_subtile,
          uop->store_raw_next_segment,
          &packet)) {
    std::abort();
  }

  uop->store_conv_packet = to_tmem_packet(packet);
  uop->store_conv_valid = true;
  ++uop->store_raw_next_segment;

  if (uop->store_raw_next_segment >= packets_per_subtile) {
    uop->store_raw_subtile_valid = false;
    uop->store_raw_next_segment = 0;
  }

  refresh_store_stage(uop);
  return true;
}

bool TensorMemPipeline::stage_store_converted_packet(tud::MemUop* uop) {
  if (nullptr == uop) {
    return false;
  }
  if (!uop->store_conv_valid || uop->staged_store_count >= tud::kOutputPacketBufferDepth) {
    return false;
  }

  uop->staged_store_packets.at(uop->staged_store_tail) = uop->store_conv_packet;
  uop->staged_store_tail = (uop->staged_store_tail + 1) % tud::kOutputPacketBufferDepth;
  ++uop->staged_store_count;
  uop->store_conv_valid = false;

  refresh_store_stage(uop);
  return true;
}

bool TensorMemPipeline::emit_store_packet(tud::MemUop* uop) {
  if (nullptr == uop) {
    return false;
  }
  if (uop->staged_store_count == 0 || uop->remaining_tmem_write_packets == 0) {
    return false;
  }

  ++uop->next_store_packet_idx;
  uop->staged_store_head = (uop->staged_store_head + 1) % tud::kOutputPacketBufferDepth;
  --uop->staged_store_count;
  --uop->remaining_tmem_write_packets;

  refresh_store_stage(uop);
  return true;
}

uint64_t TensorMemPipeline::issue_store_write_request(
    const tud::MemUop& uop,
    Core* core,
    SimPort<TensorMemPortReq>* req_out,
    uint64_t* next_tensor_mem_request_id) {
  if (nullptr == core || nullptr == req_out || nullptr == next_tensor_mem_request_id) {
    return 0;
  }
  if (uop.staged_store_count == 0) {
    std::abort();
  }

  TensorMemPortReq request{};
  request.request_id = (*next_tensor_mem_request_id)++;
  request.arbitration_age = mem_uop_request_age(uop, uop.next_store_packet_idx);
  request.access_type = TensorMemPortReq::AccessType::Write;
  request.port_request.age = request.arbitration_age;
  request.write_packet = uop.staged_store_packets.at(uop.staged_store_head);

  uint32_t packet_idx = 0;
  if (!lookup_window_packet_idx(uop, uop.next_store_packet_idx, core, &packet_idx)) {
    std::cerr << "TensorUnit error: missing TMEM window mapping for tensor store write"
              << " handle=" << uop.handle
              << " window=" << uop.window_id
              << " tile=" << uop.tile_idx
              << " local_packet=" << uop.next_store_packet_idx
              << std::endl;
    std::abort();
  }

  request.port_request.kind = Core::TmemRequestKind::WindowWrite;
  request.port_request.handle = uop.handle;
  request.port_request.window_id = uop.window_id;
  request.port_request.packet_idx = packet_idx;
  req_out->push(request);
  return request.request_id;
}

// ============================================================================
// 主调度入口
// ============================================================================

void TensorMemPipeline::advance_mem_pipeline(
    Core* core,
    SimPort<TensorMemPortReq>* req_out,
    SimPort<TensorAsyncOpCompletion>* completion_out,
    TensorLocalMemArbiter* mem_arbiter,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    DMem* dmem,
    MetaMem* metamem,
    std::deque<tud::MemUop>* mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses,
    uint64_t* next_tensor_mem_request_id,
    tud::AMemState* amem_state,
    tud::BMemState* bmem_state,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    TensorUnit::PerfStats* perf_stats,
    uint32_t* rr_index) {
  if (nullptr == core || nullptr == req_out || nullptr == completion_out || nullptr == mem_arbiter
   || nullptr == amem || nullptr == bmem || nullptr == cmem || nullptr == dmem || nullptr == metamem
   || nullptr == mem_ops || nullptr == pending_mem_ops || nullptr == completed_tensor_mem_responses
   || nullptr == next_tensor_mem_request_id || nullptr == amem_state || nullptr == bmem_state
   || nullptr == cmem_state || nullptr == dmem_state || nullptr == perf_stats || nullptr == rr_index) {
    return;
  }

  if (mem_ops->empty()) {
    return;
  }

  const auto n = mem_ops->size();
  if (*rr_index >= n) {
    *rr_index = 0;
  }

  TensorMemPipelineCtx ctx{
      *core,
      *req_out,
      *completion_out,
      *mem_arbiter,
      *amem,
      *bmem,
      *cmem,
      *dmem,
      *metamem,
      *mem_ops,
      *pending_mem_ops,
      *completed_tensor_mem_responses,
      *next_tensor_mem_request_id,
      *amem_state,
      *bmem_state,
      *cmem_state,
      *dmem_state,
      *perf_stats,
      *rr_index,
      false};

  for (size_t round = 0; round < n; ++round) {
    const size_t idx = (ctx.rr_index + round) % n;
    auto& uop = ctx.mem_ops.at(idx);

    StepResult step = StepResult::NoProgress;
    if (is_fill_uop(uop)) {
      step = step_fill_uop(ctx, uop);
    } else if (is_store_uop(uop)) {
      step = step_store_uop(ctx, uop);
    } else {
      step = step_legacy_uop(uop);
    }

    if (step == StepResult::IssuedTmemRequest) {
      ctx.rr_index = (idx + 1) % n;
      continue;
    }

    if (step != StepResult::Progress || !mem_op_complete(uop)) {
      continue;
    }

    complete_mem_uop(ctx, idx);
    return;
  }
}

} // namespace vortex
