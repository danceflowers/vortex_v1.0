// Stage1: TcDecodeStage — second-level decode for TCU_WMMA and TCU_LD/ST.
//
// MMA path: idesc + qualifier + operand_block → DecodedMmaJob (Output)
//   The operand_block_t (32 B) read from LMEM is modelled with cycle-accurate
//   timing via LmemReadReq/Rsp SimPorts (时序功能分离):
//     1. Partially decode idesc + qualifier → DecodedMmaJob fields
//     2. Issue LsuReq for timing (bank conflict + crossbar latency)
//     3. On LsuRsp: functional core_->lmem_read() for actual data
//     4. Fill remaining fields (a_taddr, d_taddr, b_sdesc, lanes_off)
//     5. Push to Output with decode_latency
//   While waiting for the LMEM response no new input is accepted.
//
// LD/ST path: minimal decode → LdStJob (OutputLdSt)

#pragma once

#include <simobject.h>
#include <unordered_map>
#include "instr_trace.h"
#include "types.h"
#include "otc_types.h"

namespace vortex {

class Core;
class OpenTensorCore;

struct TcDecodeConfig {
  uint32_t port_depth       = 1;
  uint32_t decode_latency   = 2;   // MMA decode delay (after LMEM response)
  uint32_t ldst_latency     = 1;   // LD/ST decode delay (no LMEM read)
  uint32_t lmem_read_depth  = 1;   // LmemReadReq port depth
  uint32_t lmem_rsp_depth   = 1;   // LmemReadRsp port depth
};

class TcDecodeStage : public SimObject<TcDecodeStage> {
public:
  SimPort<instr_trace_t*> Input;
  SimPort<DecodedMmaJob>  Output;
  SimPort<LdStJob>        OutputLdSt;

  // LMEM read port for operand_block_t (32 B) — cycle-accurate timing.
  SimPort<LsuReq> LmemReadReq;
  SimPort<LsuRsp> LmemReadRsp;

  TcDecodeStage(const SimContext& ctx, const char* name, Core* core,
                OpenTensorCore* otc = nullptr,
                const TcDecodeConfig& config = TcDecodeConfig{});

  void reset();
  void tick();

private:
  // Partially-decoded MMA job waiting for LMEM response.
  struct PendingDecode {
    bool            valid = false;
    uint64_t        pending_req_id = 0;
    instr_trace_t*  trace = nullptr;
    DecodedMmaJob   job;              // all fields filled except op_block ones
    uint64_t        op_block_lmem_ptr = 0;
    bool            op_block_ready = false;  // true after LMEM response processed
  };

  Core*               core_;
  OpenTensorCore*     otc_ = nullptr;  // for collector_ready() queries
  TcDecodeConfig      config_;
  PendingDecode       pending_;
  std::unordered_map<uint64_t, LsuRsp> completed_lmem_rsp_;
  uint64_t            next_request_id_ = 1;
};

}  // namespace vortex
