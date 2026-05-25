// tensor_async_frontend.cpp
//
// Async tensor-operation dispatch frontend implementation.

#include "open_tensorcore/tensor_control/tensor_async_frontend.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "core.h"
#include "tensor_cfg.h"
#include "open_tensorcore/tensor_control/tensor_mem_manager.h"
#include "open_tensorcore/tensor_helper/tensor_debug_utils.h"

namespace vortex {

namespace vt = vortex::tensor;

namespace {

bool use_open_tensorcore(uint32_t fmt_a, uint32_t fmt_b, uint32_t fmt_c) {
  if constexpr (NUM_THREADS != 32) {
    return false;
  }
  if (fmt_c != vt::fp8::id && fmt_c != vt::fp16::id && fmt_c != vt::fp32::id) {
    return false;
  }
  bool supported_a = (fmt_a == vt::fp8::id || fmt_a == vt::fp16::id);
  bool supported_b = (fmt_b == vt::fp8::id || fmt_b == vt::fp16::id);
  return supported_a && supported_b;
}

} // namespace

// ============================================================================
// enqueue_async_mma_load
// ============================================================================
//
// MMA_LOAD(A/B) creates four MemUops, one per AMem/BMem line.
// MMA_LOAD(C) creates four MemUops, one per CMem subtile.
// The weight-stationary B fast path reuses locked BMem data and skips fill.
// ============================================================================
void tcissue_unit::issue_mma_load(
    Core* core,
    const Arch& arch,
    uint32_t wid,
    uint32_t taddr,
    const IntrTcuArgs& args,
    tud::AMemState* amem_state,
    tud::BMemState* bmem_state,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    tud::AMemState* pub_amem_state,
    tud::BMemState* pub_bmem_state,
    tud::CMemState* pub_cmem_state,
    tud::DMemState* pub_dmem_state,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    DMem* dmem,
    MetaMem* metamem,
    std::deque<tud::MemUop>* mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
    TensorUnit::PerfStats* perf_stats,
    TensorUnit::ExeTraceData* trace_data) {
  if (nullptr == core || nullptr == amem_state || nullptr == bmem_state || nullptr == cmem_state
   || nullptr == dmem_state || nullptr == amem || nullptr == bmem || nullptr == cmem
   || nullptr == dmem || nullptr == metamem || nullptr == mem_ops
   || nullptr == pending_mem_ops || nullptr == perf_stats) {
    return;
  }
  (void)pub_amem_state; (void)pub_bmem_state; (void)pub_cmem_state; (void)pub_dmem_state;
  (void)arch; (void)wid;

  
  // // Check whether the TMEM TADDR is ready for MMA load.
  // if (!core->tmem_taddr_ready_for_mma_load(taddr, args.target, args.a_sparse_mode)) {
  //   if (trace_data) { trace_data->retry = true; }
  //   return;
  // }

  if (args.target == TcuTarget::None) {
    if (trace_data) { trace_data->retry = true; }
    return;
  }

  

  // Check local-memory state and bind the descriptor.
  switch (args.target) {
  case TcuTarget::A: {
    if (!TensorMemManager::is_a_ready(*amem_state)) {
      if (trace_data) { trace_data->retry = true; }
      return;
    }
    TensorMemManager::bind_a_descriptor(amem_state, args, amem, metamem);
    TensorMemManager::mark_a_pending(amem_state);
  } break;
  case TcuTarget::B: {
    // Weight-stationary fast path: reuse locked BMem without a new fill.
    if (bmem_state->ws && (bmem_state->use || bmem_state->lastuse)) {
      // Data is already resident in BMem; issue an async ID with no MemUops.
      auto async_id = core->mma_load_async_issue(wid, taddr, args.descriptor);
      (*pending_mem_ops)[async_id] = 0;  // No pending uops.
      // Completion is intentionally elided in this legacy path.
      return;
    }
    if (!TensorMemManager::is_b_ready(*bmem_state)) {
      if (trace_data) { trace_data->retry = true; }
      return;
    }
    TensorMemManager::bind_b_descriptor(bmem_state, args, bmem);
    TensorMemManager::mark_b_pending(bmem_state);
  } break;
  case TcuTarget::C: {
    if (!TensorMemManager::is_c_ready(*cmem_state)) {
      if (trace_data) { trace_data->retry = true; }
      return;
    }
    TensorMemManager::bind_c_descriptor(cmem_state, args, cmem);
    TensorMemManager::mark_c_pending(cmem_state);
    // C fill also binds DMem's descriptor, though D has no data yet.
    TensorMemManager::bind_d_descriptor(dmem_state, args, dmem);
  } break;
  default:
    if (trace_data) { trace_data->retry = true; }
    return;
  }

  // Create MemUops and enqueue them.
  auto async_id = core->mma_load_async_issue(wid, taddr, args.descriptor);
  uint32_t num_uops = 0;

  switch (args.target) {
  case TcuTarget::A: {
    auto fmt = (source_payload_fmt != tud::kUnsetPayloadFmt) ? source_payload_fmt : amem_state->fmt_a;
    for (uint32_t line = 0; line < AMem::fill_lines(); ++line) {
      tud::MemUop op{};
      op.kind = tud::MemUop::Kind::FillA;
      op.line_idx = line;
      op.taddr = taddr;
      op.window_id = args.window_id;
      op.payload_fmt = source_payload_fmt;
      op.tile_idx = args.tile_id;
      op.async_id = async_id;
      op.separate_taddr = true;
      op.remaining_tmem_read_packets = AMem::packets_per_fill_line(fmt);
      op.remaining_amem_fill_lines = 1;
      // Sparse metadata is attached only to line 0's MemUop.
      if (line == 0 && amem_state->a_sparse_mode != vt::sparse_none) {
        op.remaining_tmem_read_packets += tud::meta_packet_count(amem_state->a_sparse_mode);
        op.remaining_metamem_fill_packets = MetaMem::fill_packets();
      }
      mem_ops->push_back(op);
      ++num_uops;
    }
  } break;
  case TcuTarget::B: {
    auto fmt = (source_payload_fmt != tud::kUnsetPayloadFmt) ? source_payload_fmt : bmem_state->fmt_b;
    for (uint32_t line = 0; line < BMem::fill_lines(); ++line) {
      tud::MemUop op{};
      op.kind = tud::MemUop::Kind::FillB;
      op.line_idx = line;
      op.taddr = taddr;
      op.window_id = args.window_id;
      op.payload_fmt = source_payload_fmt;
      op.tile_idx = args.tile_id;
      op.async_id = async_id;
      op.separate_taddr = true;
      op.remaining_tmem_read_packets = BMem::packets_per_fill_line(fmt);
      op.remaining_bmem_fill_lines = 1;
      mem_ops->push_back(op);
      ++num_uops;
    }
  } break;
  case TcuTarget::C: {
    auto fmt = (source_payload_fmt != tud::kUnsetPayloadFmt) ? source_payload_fmt : cmem_state->fmt_c;
    auto packets_per_sub = CMem::packets_per_subtile(fmt);
    for (uint32_t sub = 0; sub < CMem::fill_subtiles(fmt); ++sub) {
      tud::MemUop op{};
      op.kind = tud::MemUop::Kind::FillC;
      op.line_idx = sub;
      op.taddr = taddr;
      op.window_id = args.window_id;
      op.payload_fmt = source_payload_fmt;
      op.tile_idx = args.tile_id;
      op.async_id = async_id;
      op.separate_taddr = true;
      op.remaining_tmem_read_packets = packets_per_sub;
      op.remaining_cmem_fill_subtiles = 1;
      mem_ops->push_back(op);
      ++num_uops;
    }
  } break;
  default:
    std::abort();
  }

  (*pending_mem_ops)[async_id] = num_uops;
}

// ============================================================================
// enqueue_async_mma_store
// ============================================================================
//
// MMA_STORE dumps DMem, converts to the target precision, and writes TMEM.
// It can issue only when DMem is valid and no store/WMMA owns it.
// ============================================================================
void tcissue_unit::issue_mma_store(
    Core* core,
    const Arch& arch,
    uint32_t wid,
    uint32_t taddr,
    const IntrTcuArgs& args,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    tud::CMemState* pub_cmem_state,
    tud::DMemState* pub_dmem_state,
    CMem* cmem,
    DMem* dmem,
    std::deque<tud::MemUop>* mem_ops,
    TensorUnit::PerfStats* perf_stats,
    TensorUnit::ExeTraceData* trace_data) {
  (void)cmem; (void)dmem; (void)arch; (void)wid;
  (void)pub_cmem_state; (void)pub_dmem_state; (void)cmem_state;
  if (nullptr == core || nullptr == dmem_state || nullptr == mem_ops || nullptr == perf_stats) {
    return;
  }

  if (!core->tmem_taddr_ready_for_mma_store(taddr)) {
    if (trace_data) { trace_data->retry = true; }
    return;
  }

  // Validate DMem state before scheduling the store.
  if (dmem_state->descriptor != args.descriptor
   || dmem_state->d_store_pending
   || dmem_state->d_wmma_inflight != 0) {
    if (trace_data) { trace_data->retry = true; }
    return;
  }
  if (!dmem_state->d_valid) {
    if (trace_data) { trace_data->retry = true; }
    return;
  }

  // Create the StoreC MemUop.
  dmem_state->d_store_pending = true;
  dmem_state->store_async_id = core->mma_store_async_issue(wid, taddr, args.descriptor);

  tud::MemUop op{};
  op.kind = tud::MemUop::Kind::StoreC;
  op.line_idx = 0;
  op.taddr = taddr;
  op.window_id = args.window_id;
  op.tile_idx = args.tile_id;
  op.async_id = dmem_state->store_async_id;
  op.separate_taddr = use_window;
  op.store_from_dmem = true;
  op.remaining_cmem_dump_subtiles = DMem::dump_subtiles(dmem_state->fmt_d);
  op.remaining_tmem_write_packets = tud::d_store_packet_count(dmem_state->fmt_d);
  mem_ops->push_back(op);
}

// ============================================================================
// enqueue_async_wmma
// ============================================================================
//
// Legacy WMMA issue gate:
//   - AMem and BMem must both be valid.
//   - The first accumulation step also requires valid CMem data.
//
// ============================================================================
// void TensorAsyncFrontend::enqueue_async_wmma(
//     Core* core,
//     const Arch& arch,
//     uint32_t wid,
//     const IntrTcuArgs& args,
//     uint32_t fmt_a,
//     uint32_t fmt_b,
//     uint32_t fmt_c,
//     tud::AMemState* amem_state,
//     tud::BMemState* bmem_state,
//     tud::CMemState* cmem_state,
//     tud::DMemState* dmem_state,
//     tud::AMemState* pub_amem_state,
//     tud::BMemState* pub_bmem_state,
//     tud::CMemState* pub_cmem_state,
//     tud::DMemState* pub_dmem_state,
//     std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
//     std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
//     bool* prev_or,
//     bool active_wmma_job_valid,
//     TensorUnit::PerfStats* perf_stats,
//     TensorUnit::ExeTraceData* trace_data) {
//   (void)pub_amem_state; (void)pub_bmem_state; (void)pub_cmem_state; (void)pub_dmem_state;
//   if (nullptr == core || nullptr == amem_state || nullptr == bmem_state
//    || nullptr == cmem_state || nullptr == dmem_state
//    || nullptr == pending_wmma_uops || nullptr == pending_wmma_jobs
//    || nullptr == prev_or || nullptr == perf_stats) {
//     return;
//   }
//   auto issue_wid = wid;

//   if (!use_open_tensorcore(fmt_a, fmt_b, fmt_c)) {
//     if (trace_data) { trace_data->retry = true; }
//     return;
//   }

//   // Identify the first step in an accumulation sequence.
//   bool or_bit = (args.output_resident != 0);
//   bool is_first = !or_bit || !(*prev_or);

//   // Issue gating.
//   if (amem_state->descriptor != args.descriptor || !amem_state->a_valid || amem_state->a_wmma_pending) {
//     if (trace_data) { trace_data->retry = true; }
//     return;
//   }
//   bool b_ok = (bmem_state->b_valid && !bmem_state->b_wmma_pending
//                && (bmem_state->descriptor == args.descriptor
//                    || (bmem_state->b_ws_locked && bmem_state->ws_descriptor == args.descriptor)));
//   if (!b_ok) {
//     if (trace_data) { trace_data->retry = true; }
//     return;
//   }
//   if (is_first) {
//     if (cmem_state->descriptor != args.descriptor || !cmem_state->c_valid) {
//       if (trace_data) { trace_data->retry = true; }
//       return;
//     }
//   } else {
//     // Middle/tail steps need the previous DMem result.
//     if (!dmem_state->d_valid) {
//       if (trace_data) { trace_data->retry = true; }
//       return;
//     }
//   }

//   // Mark local-memory ownership state.
//   amem_state->a_wmma_pending = true;
//   bmem_state->b_wmma_pending = true;
//   if (is_first) {
//     ++cmem_state->c_wmma_inflight;
//   }
//   ++dmem_state->d_wmma_inflight;

//   auto async_id = core->wmma_async_issue(wid);
//   amem_state->wmma_async_id = async_id;
//   bmem_state->wmma_async_id = async_id;

//   (*pending_wmma_uops)[async_id] = tud::kWmmaPrimitiveCount;

//   bool ws = (args.ws != 0);
//   tud::PendingWmmaJob wmma_job{};
//   wmma_job.wid = issue_wid;
//   wmma_job.fmt_a = fmt_a;
//   wmma_job.fmt_b = fmt_b;
//   wmma_job.fmt_c = fmt_c;
//   wmma_job.a_sparse_mode = amem_state->a_sparse_mode;
//   wmma_job.async_id = async_id;
//   wmma_job.output_resident = or_bit;
//   wmma_job.ws = ws;
//   wmma_job.is_first_in_accum_seq = is_first;
//   wmma_job.next_uop = 0;
//   pending_wmma_jobs->push_back(wmma_job);

//   // Update prev_or so the next WMMA can identify middle/tail steps.
//   *prev_or = or_bit;

//   ++perf_stats->issued_macro_wmma;
//   perf_stats->pending_wmma_jobs_max = std::max<uint64_t>(perf_stats->pending_wmma_jobs_max,
//                                                          pending_wmma_jobs->size() + (active_wmma_job_valid ? 1u : 0u));
// }

// ============================================================================
// enqueue_async_tcu_mma — single-instruction tcgen05.mma fan-out
// ============================================================================
//
// PTX semantics: tcgen05.mma { [d_taddr], a, b_sdesc, idesc, enable_input_d }
// is one async op that internally issues fill→compute→drain. CModel mirrors
// that by calling the existing per-stage helpers in a fixed order.
//
// Phase-2 implementation strategy: each stage is dispatched with a synthetic
// IntrTcuArgs whose fields match the per-stage requirements. The frontend
// helpers verify operand readiness and queue MemUops / WmmaJobs themselves.
// On any retry we propagate the retry flag and abort fan-out (caller will
// reissue the whole tcu_mma; the partial state in mem_ops/pending queues is
// idempotent because state objects only flip on success).
//
// Note: currently the helpers track separate async_ids per stage. A future
// pass will collapse them into one outer async_id (so mbarrier_arrive bound
// via tcgen05.commit fires exactly once per tcu_mma).
// ============================================================================
void tcissue_unit::issue_tcu_mma(
    Core* core,
    const Arch& arch,
    uint32_t wid,
    const IntrTcuArgs& args,
    uint32_t fmt_a, uint32_t fmt_b, uint32_t fmt_c, uint32_t fmt_d,
    uint32_t a_handle, uint32_t a_tile_id,
    uint32_t b_handle, uint32_t b_tile_id,
    uint32_t c_handle, uint32_t c_tile_id,
    uint32_t d_handle, uint32_t d_tile_id,
    tud::AMemState* amem_state,
    tud::BMemState* bmem_state,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    tud::AMemState* pub_amem_state,
    tud::BMemState* pub_bmem_state,
    tud::CMemState* pub_cmem_state,
    tud::DMemState* pub_dmem_state,
    AMem* amem, BMem* bmem, CMem* cmem, DMem* dmem,
    MetaMem* metamem,
    std::deque<tud::MemUop>* mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
    std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
    bool* prev_or,
    bool active_wmma_job_valid,
    TensorUnit::PerfStats* perf_stats,
    TensorUnit::ExeTraceData* trace_data) {

  // PTX enable_input_d (idesc bit) maps to Vortex output_resident OR-bit
  // semantics: we read C from CMem on the *first* mma in an accumulate
  // sequence, and read D from DMem on subsequent mmas. The first/middle/tail
  // detection uses the existing prev_or upper-edge heuristic.
  bool or_bit = (args.output_resident != 0);
  bool is_first = !or_bit || !(*prev_or);

  // Helper to clone the args + override per-stage fields the lower-level
  // enqueue_* helpers consume.
  auto stage_args = [&](TcuTarget target, uint32_t tile_id) -> IntrTcuArgs {
    IntrTcuArgs a = args;
    a.target = target;
    a.tile_id = tile_id;
    a.fmt_a = fmt_a;
    a.fmt_b = fmt_b;
    a.fmt_c = fmt_c;
    a.fmt_d = fmt_d;
    return a;
  };

  // ---- Stage 1: fill A (TMEM -> AMem) ----
  {
    auto a_args = stage_args(TcuTarget::A, a_tile_id);
    enqueue_async_mma_load(core, arch, wid, a_handle, a_args,
                           amem_state, bmem_state, cmem_state, dmem_state,
                           pub_amem_state, pub_bmem_state, pub_cmem_state, pub_dmem_state,
                           amem, bmem, cmem, dmem, metamem,
                           mem_ops, pending_mem_ops, perf_stats, trace_data);
    if (trace_data && trace_data->retry) return;
  }

  // ---- Stage 2: fill B (TMEM -> BMem) ----
  // WS fast path: when bmem already locked with same descriptor, the helper
  // skips queueing fills (data reuse). That logic is preserved.
  {
    auto b_args = stage_args(TcuTarget::B, b_tile_id);
    enqueue_async_mma_load(core, arch, wid, b_handle, b_args,
                           amem_state, bmem_state, cmem_state, dmem_state,
                           pub_amem_state, pub_bmem_state, pub_cmem_state, pub_dmem_state,
                           amem, bmem, cmem, dmem, metamem,
                           mem_ops, pending_mem_ops, perf_stats, trace_data);
    if (trace_data && trace_data->retry) return;
  }

  // ---- Stage 3 (conditional): fill C (TMEM -> CMem) ----
  // Only on first-in-accum-seq when enable_input_d=1, OR on enable_input_d=0
  // single-shot mode (which still consumes C for the bias/accum input).
  if (is_first) {
    auto c_args = stage_args(TcuTarget::C, c_tile_id);
    enqueue_async_mma_load(core, arch, wid, c_handle, c_args,
                           amem_state, bmem_state, cmem_state, dmem_state,
                           pub_amem_state, pub_bmem_state, pub_cmem_state, pub_dmem_state,
                           amem, bmem, cmem, dmem, metamem,
                           mem_ops, pending_mem_ops, perf_stats, trace_data);
    if (trace_data && trace_data->retry) return;
  }

  // ---- Stage 4: WMMA primitive sequence (8 primitives) ----
  {
    auto wmma_args = args;
    wmma_args.fmt_a = fmt_a;
    wmma_args.fmt_b = fmt_b;
    wmma_args.fmt_c = fmt_c;
    wmma_args.fmt_d = fmt_d;
    enqueue_async_wmma(core, arch, wid, wmma_args,
                       fmt_a, fmt_b, fmt_c,
                       amem_state, bmem_state, cmem_state, dmem_state,
                       pub_amem_state, pub_bmem_state, pub_cmem_state, pub_dmem_state,
                       pending_wmma_uops, pending_wmma_jobs,
                       prev_or, active_wmma_job_valid,
                       perf_stats, trace_data);
    if (trace_data && trace_data->retry) return;
  }

  // ---- Stage 5: drain D (DMem -> TMEM at d_taddr) ----
  // tcgen05.mma writes D to the TMEM region pointed by d_taddr; the existing
  // mma_store helper handles the DMem->TMEM transfer.
  {
    auto d_args = stage_args(TcuTarget::None, d_tile_id);
    enqueue_async_mma_store(core, arch, wid, d_handle, d_args,
                            cmem_state, dmem_state, pub_cmem_state, pub_dmem_state,
                            cmem, dmem, mem_ops, perf_stats, trace_data);
    if (trace_data && trace_data->retry) return;
  }
}

//=============================================================================
//issue uop
//=============================================================================

void tcissue_unit::issue_tcu_mma_uop(
    Core* core,
    const Arch& arch,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    DMem* dmem,
    TensorCoreTop* tensorcore,
    tud::AMemState* amem_state,
    tud::BMemState* bmem_state,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    const std::deque<tud::MemUop>& mem_ops,
   // std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
    PendingWmmaJob* active_wmma_job,
    TensorUnit::PerfStats* perf_stats) {
  (void)arch;
  if (nullptr == core || nullptr == mem_arbiter || nullptr == amem || nullptr == bmem || nullptr == cmem
   || nullptr == dmem || nullptr == tensorcore
   || nullptr == amem_state || nullptr == bmem_state || nullptr == cmem_state || nullptr == dmem_state
   || nullptr == pending_wmma_uops 
   || nullptr == perf_stats) {
    return;
  }

  // TensorCore pipeline readiness.

  if (!tensorcore->ready(true)) {
    //if (has_work) {
      ++perf_stats->stall_tc_busy;
    //}
    return;
  }

  // Compute the current primitive coordinates.
  uint32_t k_phase = active_wmma_job.next_uop / tud::kPrimitivesPerKPhase;
  uint32_t c_subtile_id = active_wmma_job.next_uop % tud::kPrimitivesPerKPhase;
  uint32_t storage_m = c_subtile_id / 2;
  uint32_t storage_n = c_subtile_id % 2;

  // Read local SRAM operands.
  uint16_t a_block[tud::kPrimitiveDim][tud::kPrimitiveDim] = {0};
  uint16_t b_block[tud::kPrimitiveDim][tud::kPrimitiveDim] = {0};
  uint32_t c_bypass[tud::kPrimitiveDim][tud::kPrimitiveDim] = {0};

  auto a_line_idx = k_phase * 2 + storage_m;
  auto b_line_idx = k_phase * 2 + storage_n;
  amem->read_primitive(a_line_idx, a_block, amem_state->transpose_a);
  bmem->read_primitive(b_line_idx, b_block, bmem_state->transpose_b);


  if (active_wmma_job->enable_input_d) {
    cmem->read_subtile_fp22(c_subtile_id, c_bypass);//D = A*B+C
  }else{
    c_bypass[tud::kPrimitiveDim][tud::kPrimitiveDim] = {0};//D=A*B
  }


  // Build metadata and issue to TensorCore.
  TensorCoreMeta meta{};
  meta.wid = active_wmma_job->wid;
  meta.async_id = job.async_id;
  meta.c_subtile_id = c_subtile_id;
  // meta.valid = true;


  tensorcore->push_uop(a_block, b_block, c_bypass, meta);

  auto cycle = core->current_cycle();
  ++perf_stats->issued_primitive_tiles;
  if (perf_stats->setup_end_cycle == 0) {
    perf_stats->setup_end_cycle = cycle;
  }

  // Advance the active WMMA job state.
  ++job.next_uop;


  if (job.next_uop == tud::kPrimitivesPerKPhase && job.is_first_in_accum_seq)

  if (job.next_uop >= tud::kWmmaPrimitiveCount) {

    TensorMemManager::mark_b_ready(bmem_state);
    TensorMemManager::mark_a_ready(amem_state);

  }
}

//=================================================================================
//check if tcissue unit can issue mma_load instructions
//=================================================================================

bool tcissue_unit::can_issue_mma_load(
    uint32_t wid,
    const IntrTcuArgs& args,
    const Arch& arch,
    const tud::AMemState& pub_amem_state,
    const tud::BMemState& pub_bmem_state,
    const tud::CMemState& pub_cmem_state,
    //const tud::DMemState& pub_dmem_state
    ) {
  (void)wid; (void)arch; (void)pub_dmem_state;
  // switch (args.target) {
  // case TcuTarget::A:

  bool ws_a_locked = !args.ws && (pub_amem_state.use || pub_amem_state.lastuse);
     
    return TensorMemManager::is_a_ready(pub_amem_state);
  // case TcuTarget::B:
  bool ws_b_locked = args.ws && (pub_bmem_state.use || pub_bmem_state.lastuse);


  if(args.enable_input_d){
    if(ws_a_locked){
      return TensorMemManager::is_b_ready(pub_bmem_state) && ws_a_locked && TensorMemManager::is_c_ready(pub_cmem_state); 
    }else if(ws_b_locked){
      return ws_b_locked && TensorMemManager::is_a_ready(pub_amem_state) && TensorMemManager::is_c_ready(pub_cmem_state); 
    }else{
      return TensorMemManager::is_b_ready(pub_bmem_state) && is_a_ready(pub_amem_state) && TensorMemManager::is_c_ready(pub_cmem_state);
    }
  }else{
    if(ws_a_locked){
      return TensorMemManager::is_b_ready(pub_bmem_state) && ws_a_locked; 
    }else if(ws_b_locked){
      return ws_b_locked && TensorMemManager::is_a_ready(pub_amem_state); 
    }else{
      return TensorMemManager::is_b_ready(pub_bmem_state) && is_a_ready(pub_amem_state);
    }
  }


}

} // namespace vortex
