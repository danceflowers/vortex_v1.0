// Stage1 TcDecodeStage implementation.
//
// MMA path uses an asynchronous LMEM read for operand_block_t (32 B):
//   1. Partially decode idesc + qualifier → all DecodedMmaJob fields except
//      the four op_block-sourced fields (a_taddr, d_taddr, b_sdesc, lanes_off).
//   2. Issue LsuReq via LmemReadReq for cycle-accurate LMEM bank-conflict timing.
//   3. On LsuRsp: functional core_->lmem_read() to get the actual data.
//   4. Fill remaining fields and push to Output with decode_latency.
//
// This replaces the prior instantaneous functional read (the TODO at the old
// line 131) and is the same 时序功能分离 pattern used by OperandFetchStage.

#include "stage1_tcdecode.h"

#include <cstring>

#include "core.h"
#include "debug.h"
#include "idescriptor.h"
#include "opentensorcore.h"
#include "tensor_cfg.h"

namespace vortex {

namespace vt = vortex::tensor;

namespace {

/// Map atype/btype/ctype/dtype 2-bit fields from i_descriptor_t to Vortex fmt ids.
/// atype, btype: 00=fp8, 01=fp16
/// ctype, dtype: 00=fp8, 01=fp16, 10=fp32
bool idesc_to_fmts(uint32_t atype, uint32_t btype,
                   uint32_t ctype, uint32_t dtype,
                   uint32_t* fmt_a, uint32_t* fmt_b,
                   uint32_t* fmt_c, uint32_t* fmt_d) {
  auto type_to_fmt = [](uint32_t code, bool is_cd) -> uint32_t {
    switch (code & 0x3u) {
    case 0: return vt::fp8::id;                         // 00 = FP8
    case 1: return vt::fp16::id;                        // 01 = FP16
    case 2: return is_cd ? vt::fp32::id : vt::fp16::id; // 10 = FP32 (C/D only)
    default: return 0xFFFFFFFFu;                        // invalid
    }
  };
  *fmt_a = type_to_fmt(atype, false);
  *fmt_b = type_to_fmt(btype, false);
  *fmt_c = type_to_fmt(ctype, true);
  *fmt_d = type_to_fmt(dtype, true);
  return *fmt_a != 0xFFFFFFFFu && *fmt_b != 0xFFFFFFFFu
      && *fmt_c != 0xFFFFFFFFu && *fmt_d != 0xFFFFFFFFu;
}

// Convert sparsity kind + .sp qualifier -> Vortex sparse_mode.
uint8_t sparsity_to_mode(uint32_t kind_bit, bool sp_qualifier) {
  if (!sp_qualifier) return vt::sparse_none;
  return (kind_bit == 0) ? vt::sparse_2_4 : vt::sparse_1_4;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor / reset
// ---------------------------------------------------------------------------

TcDecodeStage::TcDecodeStage(const SimContext& ctx, const char* name, Core* core,
                             OpenTensorCore* otc,
                             const TcDecodeConfig& config)
  : SimObject<TcDecodeStage>(ctx, name)
  , Input(this, config.port_depth)
  , Output(this, config.port_depth)
  , OutputLdSt(this, config.port_depth)
  , LmemReadReq(this, config.lmem_read_depth)
  , LmemReadRsp(this, config.lmem_rsp_depth)
  , core_(core)
  , otc_(otc)
  , config_(config) {
  this->reset();
}

void TcDecodeStage::reset() {
  pending_.valid = false;
  pending_.pending_req_id = 0;
  pending_.op_block_ready = false;
  completed_lmem_rsp_.clear();
  next_request_id_ = 1;
}

// ---------------------------------------------------------------------------
// tick — main entry point
// ---------------------------------------------------------------------------

void TcDecodeStage::tick() {
  // ---- 1. Drain LMEM responses ----
  if (!LmemReadRsp.empty()) {
    auto rsp = LmemReadRsp.front();
    LmemReadRsp.pop();
    completed_lmem_rsp_[rsp.tag] = rsp;
  }

  // ---- 2. Pending MMA decode ----
  if (pending_.valid) {
    // Phase 2: LMEM response already received, waiting for collector buffer.
    if (pending_.op_block_ready) {
      if (otc_ && !otc_->collector_ready(pending_.job.wid,
                                          pending_.job.collector_buffer,
                                          pending_.job.ws,
                                          pending_.job.bbuf_idx)) {
        return;  // stall: collector buffer not ready, retry next tick
      }
      // Collector buffer ready — push to Output.
      DT(3, "TcDecodeStage: job a_taddr=0x" << std::hex << pending_.job.a_taddr
         << " d_taddr=0x" << pending_.job.d_taddr
         << " shape=" << std::dec << pending_.job.shape_m << "x" << pending_.job.shape_n
         << " fmt_a=" << pending_.job.fmt_a << " fmt_b=" << pending_.job.fmt_b
         << " fmt_c=" << pending_.job.fmt_c << " fmt_d=" << pending_.job.fmt_d
         << " enable_input_d=" << static_cast<uint32_t>(pending_.job.enable_input_d)
         << " sp=" << static_cast<uint32_t>(pending_.job.sp)
         << " collector_buffer=" << static_cast<uint32_t>(pending_.job.collector_buffer)
         << " ws=" << static_cast<uint32_t>(pending_.job.ws)
         << " bbuf_idx=" << static_cast<int>(pending_.job.bbuf_idx));

      Output.push(pending_.job, config_.decode_latency);
      pending_.valid = false;
      pending_.op_block_ready = false;
      return;
    }

    // Phase 1: Waiting for LMEM response.
    auto it = completed_lmem_rsp_.find(pending_.pending_req_id);
    if (it != completed_lmem_rsp_.end()) {
      // Response received — functional read of operand_block_t (32 B).
      operand_block_t op_block;
      static_assert(sizeof(op_block) == 32);
      core_->lmem_read(&op_block, pending_.op_block_lmem_ptr, sizeof(op_block));
      completed_lmem_rsp_.erase(it);

      // Complete the decode: fill op_block-sourced fields.
      auto& job = pending_.job;
      job.a_taddr   = op_block.a_taddr;
      job.d_taddr   = op_block.d_taddr;
      job.b_sdesc   = (uint64_t(op_block.b_sdesc_hi) << 32)
                    | uint64_t(op_block.b_sdesc_lo);
      job.lanes_off = op_block.lanes_off;

      // Parse bbuf_idx from operand_block_t.reserved0 when ws=1.
      if (job.ws == 1) {
        job.bbuf_idx = static_cast<int8_t>(op_block.reserved0 & 0x3);
      } else {
        job.bbuf_idx = -1;
      }

      pending_.op_block_ready = true;
      // Fall through: collector check happens next tick.
    }
    // Stall: waiting for LMEM response or collector buffer.
    return;
  }







  // ---- 3. No pending decode — accept new input ----
  if (Input.empty()) return;
  auto trace = Input.front();

  // ---- LD/ST path: minimal decode → LdStJob ----
  auto* ldst = dynamic_cast<TcuLdStTraceData*>(trace->data.get());
  if (ldst != nullptr) {
    if (OutputLdSt.full()) return;
    Input.pop();

    LdStJob job;
    job.is_load = ldst->is_load;
    job.fmt     = ldst->fmt;
    job.rd      = ldst->rd;
    job.trace   = trace;
    job.st_values = ldst->values;

    DT(3, "TcDecode: " << (job.is_load ? "LD" : "ST")
       << " trace=#" << trace->uuid << " fmt=" << job.fmt);
    OutputLdSt.push(job, config_.ldst_latency);
    return;
  }

  // ---- MMA path: partial decode + issue LsuReq for operand_block_t ----
  if (Output.full()) return;

  // Read the TcuTraceData filled by the execute stage.
  auto* tcu_data = dynamic_cast<TcuTraceData*>(trace->data.get());
  if (tcu_data == nullptr) {
    DT(1, "TcDecodeStage: trace #" << trace->uuid
       << " missing TcuTraceData, skipping");
    return;
  }

  DT(3, "TcDecodeStage: wid=" << trace->wid
     << " cycle=" << this->current_cycle()
     << " idesc=0x" << std::hex << tcu_data->idesc
     << " opblock=0x" << tcu_data->operand_block_lmem_ptr
     << " qual=0x" << tcu_data->qualifier << std::dec);

  // Decode 32-bit i_descriptor_t from rs1.
  i_descriptor_t idesc;
  static_assert(sizeof(idesc) == sizeof(uint32_t));
  std::memcpy(&idesc, &tcu_data->idesc, sizeof(idesc));

  uint32_t qualifier = tcu_data->qualifier;

  // Parse qualifier modifier bits (7-bit funct7 field).
  uint8_t enable_input_d    = (qualifier >> 0) & 1;
  uint8_t ws                = (qualifier >> 1) & 1;
  uint8_t sp                = (qualifier >> 2) & 1;
  uint8_t cta_group         = (qualifier >> 3) & 1;
  uint8_t collector_buffer = (qualifier >> 4) & 0x3;
  uint8_t multicast         = (qualifier >> 6) & 1;

  // Extract precision from idesc atype/btype/ctype/dtype fields.
  uint32_t fmt_a = 0, fmt_b = 0, fmt_c = 0, fmt_d = 0;
  if (!idesc_to_fmts(idesc.atype, idesc.btype, idesc.ctype, idesc.dtype,
                     &fmt_a, &fmt_b, &fmt_c, &fmt_d)) {
    DT(1, "TcDecodeStage: unsupported idesc precision"
       << " atype=" << idesc.atype << " btype=" << idesc.btype
       << " ctype=" << idesc.ctype << " dtype=" << idesc.dtype);
    return;
  }

  // Assemble partial DecodedMmaJob (everything except op_block fields).
  DecodedMmaJob job;
  job.uuid              = trace->uuid;
  job.wid               = trace->wid;
  job.cid               = trace->cid;
  job.fmt_a             = fmt_a;
  job.fmt_b             = fmt_b;
  job.fmt_c             = fmt_c;
  job.fmt_d             = fmt_d;
  job.shape_m           = static_cast<uint32_t>(idesc.shape_m) << 4;
  job.shape_n           = static_cast<uint32_t>(idesc.shape_n) << 4;
  job.sparsity_kind     = sparsity_to_mode(idesc.sparsity_kind, sp != 0);
  job.sparsity_meta_sel = idesc.sparsity_meta_sel;
  job.saturate          = idesc.saturate;
  job.transpose_a       = idesc.transpose_a;
  job.transpose_b       = idesc.transpose_b;
  job.enable_input_d    = enable_input_d;
  job.ws                = ws;
  job.sp                = sp;
  job.cta_group         = cta_group;
  job.collector_buffer = collector_buffer;
  job.multicast         = multicast;
  // a_taddr, d_taddr, b_sdesc, lanes_off filled after LMEM response.

  // Issue LsuReq for cycle-accurate LMEM bank-conflict timing.
  // The functional data read happens when the response arrives (step 2).
  if (LmemReadReq.full()) return;  // port busy, retry next tick

  Input.pop();  // Consume the trace — we have everything we need from it.

  uint64_t op_block_lmem_ptr = static_cast<uint64_t>(tcu_data->operand_block_lmem_ptr);
  LsuReq req(LSU_CHANNELS);
  req.write = false;
  req.tag   = static_cast<uint32_t>(next_request_id_++);
  req.addrs.at(0) = op_block_lmem_ptr;
  req.mask.set(0);
  LmemReadReq.push(req, 0);

  DT(4, "TcDecodeStage: lmem read req_id=" << req.tag
     << " addr=0x" << std::hex << op_block_lmem_ptr << std::dec
     << " trace=#" << trace->uuid);

  // Save pending state for completion in a future tick.
  pending_.valid             = true;
  pending_.pending_req_id    = req.tag;
  pending_.trace             = trace;
  pending_.job               = job;
  pending_.op_block_lmem_ptr = op_block_lmem_ptr;
}

}  // namespace vortex
