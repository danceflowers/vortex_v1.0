// OpenTensorCore top-level implementation.

#include "opentensorcore.h"
#include "core.h"
#include "debug.h"

namespace vortex {

OpenTensorCore::OpenTensorCore(const SimContext& ctx, const char* name,
                               Core* core, const OpenTensorCoreConfig& config)
  : SimObject<OpenTensorCore>(ctx, name)
  , Inputs(ISSUE_WIDTH, this)
  , Outputs(ISSUE_WIDTH, this)
  // All external ports are capacity-0 relays: the OTC forwards between its
  // internal stages and the TensorSocket arbiters / core lmem_arb. A bind()
  // source must be a virtual (capacity-0) port (simobject.h:91).
  , TmemReadReq(this, 0)
  , TmemReadRsp(this, 0)
  , TmemWriteReq(this, 0)
  , TmemWriteRsp(this, 0)
  , LmemReadReq(this, 0)
  , LmemReadRsp(this, 0)
  , DecodeLmemReadReq(this, 0)
  , DecodeLmemReadRsp(this, 0)
  , AsyncCompletionOut(this, 0)
  , core_(core)
  , config_(config) {

  char sname[128];
  snprintf(sname, sizeof(sname), "%s_stage1", name);
  stage1_ = std::make_unique<TcDecodeStage>(ctx, sname, core, this, config.stage1);
  snprintf(sname, sizeof(sname), "%s_stage2a", name);
  stage2a_ = std::make_unique<OperandFetchStage>(ctx, sname, core, config.stage2);
  snprintf(sname, sizeof(sname), "%s_stage2b", name);
  stage2b_ = std::make_unique<LdstStage>(ctx, sname, core, &dmem_);
  snprintf(sname, sizeof(sname), "%s_stage3", name);
  stage3_ = std::make_unique<ComputePipeline>(ctx, sname, core, &dmem_, config.stage3);

  // MMA path: Stage1 → Stage2a → Stage3.
  stage1_->Output.bind(&stage2a_->Input);
  stage2a_->Output.bind(&stage3_->Input);

  // LD/ST path: Stage1 → Stage2b.
  stage1_->OutputLdSt.bind(&stage2b_->Input);

  // TMEM ports.
  stage2a_->TmemReadReq.bind(&this->TmemReadReq);
  this->TmemReadRsp.bind(&stage2a_->TmemReadRsp);
  stage3_->TmemWriteReq.bind(&this->TmemWriteReq);
  this->TmemWriteRsp.bind(&stage3_->TmemWriteRsp);

  // LMEM ports (Stage2a B reads + Stage1 operand_block reads).
  stage2a_->LmemReadReq.bind(&this->LmemReadReq);
  this->LmemReadRsp.bind(&stage2a_->LmemReadRsp);
  stage1_->LmemReadReq.bind(&this->DecodeLmemReadReq);
  this->DecodeLmemReadRsp.bind(&stage1_->LmemReadRsp);

  this->reset();
}

void OpenTensorCore::reset() {
  stage1_->reset();
  stage2a_->reset();
  stage2b_->reset();
  stage3_->reset();
  pending_traces_.clear();
}

void OpenTensorCore::tick() {
  
feed_mma_traces();
  stage3_->tick();
  stage2b_->tick();
  stage2a_->tick();
  stage1_->tick();

  if (!stage3_->Output.empty()) {
    auto done = stage3_->Output.front(); stage3_->Output.pop();
    auto it = pending_traces_.find(done.uuid);
    if (it != pending_traces_.end()) {
      auto trace = it->second.trace;
      uint32_t lane = it->second.lane;
      // Echo the Core-assigned async id so Core can match this MMA completion to
      // its async_tensor_ops_ entry per-id (out-of-order completion safe).
      uint32_t aid = 0;
      if (auto* td = dynamic_cast<TcuTraceData*>(trace->data.get())) aid = td->async_id;
      pending_traces_.erase(it);
      Outputs.at(lane).push(trace, 0);
      TensorAsyncOpCompletion c;
      c.async_id = aid;
      AsyncCompletionOut.push(c, 1);
    } else {
      DT(1, "OpenTensorCore: completed uuid=#" << done.uuid << " not found");
    }
  }

  // LD/ST completion (Stage2b) → commit the original trace back to the core.
  if (!stage2b_->Output.empty()) {
    auto trace = stage2b_->Output.front(); stage2b_->Output.pop();
    auto it = pending_traces_.find(trace->uuid);
    if (it != pending_traces_.end()) {
      uint32_t lane = it->second.lane;
      pending_traces_.erase(it);
      Outputs.at(lane).push(trace, 0);
    } else {
      DT(1, "OpenTensorCore: LD/ST completed uuid=#" << trace->uuid << " not found");
    }
  }
}

void OpenTensorCore::feed_mma_traces() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    if (Inputs.at(iw).empty()) continue;
    auto trace = Inputs.at(iw).front();

    // Both MMA (TcuTraceData) and LD/ST (TcuLdStTraceData) traces go through
    // Stage1: MMA → Stage2a → Stage3, LD/ST → Stage2b. Anything else passes
    // straight through to commit.
    bool is_ldst = dynamic_cast<TcuLdStTraceData*>(trace->data.get()) != nullptr;
    bool is_mma  = dynamic_cast<TcuTraceData*>(trace->data.get()) != nullptr;
    if (!is_ldst && !is_mma) {
      Inputs.at(iw).pop();
      Outputs.at(iw).push(trace, 3);
      continue;
    }
    if (stage1_->Input.full()) continue;
    if (pending_traces_.size() >= config_.max_inflight) {
      DT(3, "OpenTensorCore: pending_traces full (" << pending_traces_.size()
         << "/" << config_.max_inflight << "), backpressuring Input[" << iw << "]");
      continue;
    }
    Inputs.at(iw).pop();
    pending_traces_[trace->uuid] = {trace, iw};
    stage1_->Input.push(trace, 0);
  }
}

bool OpenTensorCore::collector_ready(uint32_t wid, uint8_t collector_buffer,
                                     uint8_t ws, int8_t bbuf_idx) const {
  if (ws == 0) {
    return stage3_->abuf_ready(wid, collector_buffer);
  } else {
    return stage3_->bbuf_ready(bbuf_idx, wid, collector_buffer);
  }
}

}  // namespace vortex
