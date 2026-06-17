// OpenTensorCore — top-level SIMT TensorCore pipeline.
//
// Contains three internal stages connected by SimPort bindings:
//   Stage1 (TcDecodeStage)    — semantic decode of TCU_WMMA idesc + qualifier
//   Stage2 (OperandFetchStage) — TMEM A/C + LMEM B reads, precision conversion
//   Stage3 (ComputePipeline)  — TensorCoreTop backpressure compute + D writeback
//
// External ports:
//   Inputs[ISSUE_WIDTH]  — receive instr_trace_t* from Vortex execute stage
//   Outputs[ISSUE_WIDTH] — emit completed instr_trace_t* to Vortex commit stage
//   TmemReadReq/Rsp      — 512-bit TMEM read port (A/C)
//   TmemWriteReq/Rsp     — 512-bit TMEM write port (D)
//   LmemReadReq/Rsp      — LSU-style LMEM read port (B)
//
// Pending traces are tracked in a uuid→{trace, lane} map so the correct
// Output lane is used on completion.

#pragma once

#include <simobject.h>
#include <memory>
#include <unordered_map>
#include "types.h"
#include "otc_types.h"
#include "tensor_mem_port_types.h"
#include "stage1_tcdecode.h"
#include "stage2_operandfetch.h"
#include "TCU_LDST_Stage.h"
#include "stage3_compute.h"

namespace vortex {

class Core;

struct OpenTensorCoreConfig {
  TcDecodeConfig     stage1;
  OperandFetchConfig stage2;
  ComputeConfig      stage3;
  uint32_t           max_inflight = 16;  // max traces in-flight in the OTC pipeline
};

class OpenTensorCore : public SimObject<OpenTensorCore> {
public:
  // Vortex pipeline ports.
  std::vector<SimPort<instr_trace_t*>> Inputs;
  std::vector<SimPort<instr_trace_t*>> Outputs;

  // TMEM read port (A/C, from Stage2).
  SimPort<TensorMemPortReq> TmemReadReq;
  SimPort<TensorMemPortRsp> TmemReadRsp;

  // TMEM write port (D, from Stage3).
  SimPort<TensorMemPortReq> TmemWriteReq;
  SimPort<TensorMemPortRsp> TmemWriteRsp;

  // LMEM read ports — bound to the core's lmem_arb in Core::bind_tensor_ports().
  // Stage2a: B-matrix reads.  Stage1: operand_block_t reads (32 B per MMA).
  SimPort<LsuReq> LmemReadReq;
  SimPort<LsuRsp> LmemReadRsp;
  SimPort<LsuReq> DecodeLmemReadReq;
  SimPort<LsuRsp> DecodeLmemReadRsp;

  OpenTensorCore(const SimContext& ctx, const char* name, Core* core,
                 const OpenTensorCoreConfig& config = OpenTensorCoreConfig{});

  void reset();
  void tick();

  /// Check whether the collector buffer is ready for a new MMA from warp `wid`.
  bool collector_ready(uint32_t wid, uint8_t collector_buffer,
                       uint8_t ws, int8_t bbuf_idx) const;

  DMem* dmem() { return &dmem_; }

private:
  struct PendingTrace { instr_trace_t* trace = nullptr; uint32_t lane = 0; };

  void feed_mma_traces();

  Core*               core_;
  OpenTensorCoreConfig config_;
  DMem                dmem_;

  std::unique_ptr<TcDecodeStage>     stage1_;
  std::unique_ptr<OperandFetchStage> stage2a_;
  std::unique_ptr<TCU_LDST_Stage>    stage2b_;
  std::unique_ptr<ComputePipeline>   stage3_;

  std::unordered_map<uint64_t, PendingTrace> pending_traces_;
};

}  // namespace vortex
