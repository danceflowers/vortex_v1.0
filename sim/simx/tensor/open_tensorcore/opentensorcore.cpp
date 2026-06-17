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
  , TmemReadReq(this, 1)
  , TmemReadRsp(this, 1)
  , TmemWriteReq(this, 1)
  , TmemWriteRsp(this, 1)
  , LmemReadReq(this, 1)
  , LmemReadRsp(this, 1)
  , DecodeLmemReadReq(this, 1)
  , DecodeLmemReadRsp(this, 1)
  , core_(core)
  , config_(config) {

  char sname[128];
  snprintf(sname, sizeof(sname), "%s_stage1", name);
  stage1_ = std::make_unique<TcDecodeStage>(ctx, sname, core, this, config.stage1);
  snprintf(sname, sizeof(sname), "%s_stage2a", name);
  stage2a_ = std::make_unique<OperandFetchStage>(ctx, sname, core, config.stage2);
  snprintf(sname, sizeof(sname), "%s_stage2b", name);
  stage2b_ = std::make_unique<TCU_LDST_Stage>(ctx, sname, core, &dmem_);
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

  while (!stage3_->Output.empty()) {
    auto done = stage3_->Output.front(); stage3_->Output.pop();
    auto it = pending_traces_.find(done.uuid);
    if (it == pending_traces_.end()) {
      DT(1, "OpenTensorCore: completed uuid=#" << done.uuid << " not found");
      continue;
    }
    auto trace = it->second.trace;
    uint32_t lane = it->second.lane;
    pending_traces_.erase(it);
    Outputs.at(lane).push(trace, 0);
  }
}

void OpenTensorCore::feed_mma_traces() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    if (Inputs.at(iw).empty()) continue;
    auto trace = Inputs.at(iw).front();

    // Skip LD/ST traces (handled by Stage1 → Stage2b path).
    if (dynamic_cast<TcuLdStTraceData*>(trace->data.get())) continue;

    auto* td = dynamic_cast<TcuTraceData*>(trace->data.get());
    if (td == nullptr) {
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
