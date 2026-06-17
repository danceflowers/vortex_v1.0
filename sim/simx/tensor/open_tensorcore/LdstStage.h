// Stage2b: DMemAccessStage — TCU_LD/ST DMem ↔ register transfer.
//
// Receives LdStJob from Stage1 (TcDecode) via SimPort. Operates on the shared
// DMem (owned by OpenTensorCore). Timing is counter-based:
//
//   IDLE → (receive job) → DELAY(latency--) → EXECUTE → DONE → push trace
//
// Functional work (DMem read/write, precision conversion, register writeback)
// happens in the EXECUTE tick. The latency counter models the aggregate delay.

#pragma once

#include <simobject.h>
#include <array>
#include <cstdint>
#include "types.h"
#include "otc_types.h"
#include "open_tensorcore/tensor_compute/dmem.h"

namespace vortex {

class Core;

class LdstStage : public SimObject<LdstStage> {
public:
  SimPort<LdStJob> Input;
  SimPort<instr_trace_t*> Output;

  LdstStage(const SimContext& ctx, const char* name, Core* core, DMem* dmem,
                  const LdStConfig& config = LdStConfig{});

  void reset();
  void tick();

private:
  enum class State : uint8_t { IDLE, EXECUTE, DONE };

  struct Job {
    LdStJob desc;
    instr_trace_t* trace = nullptr;
    State state = State::IDLE;
    uint32_t latency_remaining = 0;

    // Intermediate buffers.
    uint32_t fp22_buf[4][8][8] = {};   // 4 subtiles × 8×8 FP22
    uint32_t out_buf[256] = {};        // converted output (LD only)
  };

  void execute_ld(Job& job);
  void execute_st(Job& job);
  void complete(Job& job);

  Core* core_;
  DMem* dmem_;
  LdStConfig config_;
  Job active_;
};

}  // namespace vortex
