// Stage2b LdstStage implementation.

#include "LdstStage.h"
#include "core.h"
#include "debug.h"
#include "open_tensorcore/tensor_compute/fp_types.h"
#include "open_tensorcore/tensor_compute/fp22_to_fp16.h"

namespace vortex {

LdstStage::LdstStage(const SimContext& ctx, const char* name,
                                 Core* core, DMem* dmem, const LdStConfig& config)
  : SimObject<LdstStage>(ctx, name)
  , Input(this, config.port_depth)
  , Output(this, config.port_depth)
  , core_(core)
  , dmem_(dmem)
  , config_(config) {
  this->reset();
}

void LdstStage::reset() {
  active_ = Job{};
}

void LdstStage::tick() {
  // Accept new job if idle.
  if (active_.state == State::IDLE && !Input.empty()) {
    auto job = Input.front(); Input.pop();
    active_.desc = job;
    active_.trace = job.trace;
    active_.latency_remaining = config_.latency;
    active_.state = State::EXECUTE;
    DT(3, "DMemAccess: " << (job.is_load ? "LD" : "ST")
       << " fmt=" << job.fmt << " latency=" << config_.latency);
    return;
  }

  if (active_.state == State::IDLE) return;

  switch (active_.state) {
  case State::EXECUTE:
    if (active_.latency_remaining > 0) { --active_.latency_remaining; break; }
    if (active_.desc.is_load) execute_ld(active_);
    else execute_st(active_);
    active_.state = State::DONE;
    break;
  case State::DONE:
    complete(active_);
    active_ = Job{};
    break;
  default: break;
  }
}

void LdstStage::execute_ld(Job& job) {
  // Read 4 subtiles from DMem into fp22_buf.
  for (uint32_t s = 0; s < 4; ++s)
    dmem_->read_subtile_fp22(s, job.fp22_buf[s]);

  // Convert FP22 → output format, write out_buf[row-major].
  for (uint32_t s = 0; s < 4; ++s) {
    uint32_t sm = s / 2, sn = s % 2;
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        uint32_t idx = (sm * 8 + i) * 16 + (sn * 8 + j);
        switch (job.desc.fmt) {
        case tensor::fp32::id: job.out_buf[idx] = fp22_to_fp32(job.fp22_buf[s][i][j]); break;
        case tensor::fp16::id: job.out_buf[idx] = fp22_to_fp16(job.fp22_buf[s][i][j]); break;
        case tensor::fp8::id:  job.out_buf[idx] = fp22_to_fp8_e4m3(job.fp22_buf[s][i][j]); break;
        default: break;
        }
      }
    }
  }

  DT(3, "DMemAccess: LD done, " << job.desc.rd << ".." << (job.desc.rd + 7));
}

void LdstStage::execute_st(Job& job) {
  // Convert st_values → FP22, write to DMem subtiles.
  for (uint32_t s = 0; s < 4; ++s) {
    uint32_t sm = s / 2, sn = s % 2;
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        uint32_t idx = (sm * 8 + i) * 16 + (sn * 8 + j);
        switch (job.desc.fmt) {
        case tensor::fp32::id: job.fp22_buf[s][i][j] = fp32_to_fp22(job.desc.st_values[idx]); break;
        case tensor::fp16::id: job.fp22_buf[s][i][j] = fp16_to_fp22(job.desc.st_values[idx]); break;
        case tensor::fp8::id:  job.fp22_buf[s][i][j] = fp9_to_fp22(job.desc.st_values[idx] & 0xFF); break;
        default: break;
        }
      }
    }
    dmem_->write_subtile_fp22(s, job.fp22_buf[s]);
  }

  DT(3, "DMemAccess: ST done");
}

void LdstStage::complete(Job& job) {
  if (job.desc.is_load && job.desc.rd != 0 && job.desc.trace) {
    auto trace = job.desc.trace;
    ThreadMask tmask(trace->arch.num_threads());
    for (uint32_t t = 0; t < trace->arch.num_threads(); ++t)
      if (trace->tmask.test(t)) tmask.set(t);
    for (uint32_t r = 0; r < 8; ++r) {
      std::vector<Word> vals(trace->arch.num_threads());
      for (uint32_t t = 0; t < trace->arch.num_threads(); ++t)
        vals[t] = job.out_buf[t * 8 + r];
      core_->write_ireg(trace->wid, job.desc.rd + r, tmask, vals);
    }
  }
  DT(3, "DMemAccess: complete trace=#" << job.desc.trace->uuid);
  Output.push(job.desc.trace, 0);
}

}  // namespace vortex
