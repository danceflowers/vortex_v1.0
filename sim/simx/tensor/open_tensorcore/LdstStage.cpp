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
  // Scalar tcgen05.ld: thread t reads the 32-bit TMEM cell at
  // (lane = taddr.lane + t, col_byte). TMEM is the raw architectural backing
  // store — no precision conversion (the MMA store path already converted D to
  // the external fmt before writing TMEM).
  auto trace = job.desc.trace;
  uint32_t nthreads = trace ? trace->arch.num_threads() : 0;
  for (uint32_t t = 0; t < nthreads; ++t) {
    if (trace && !trace->tmask.test(t)) continue;
    uint32_t taddr_t = job.desc.taddr + t;   // lane += t (lane is taddr[15:0])
    uint32_t val = 0;
    core_->tmem_taddr_read_bytes(taddr_t, 0,
                                 reinterpret_cast<uint8_t*>(&val), 4);
    job.out_buf[t] = val;
  }
  DT(3, "DMemAccess: LD done, rd=" << job.desc.rd
     << " taddr=0x" << std::hex << job.desc.taddr << std::dec);
}

void LdstStage::execute_st(Job& job) {
  // Scalar tcgen05.st: thread t writes its 32-bit value to the TMEM cell at
  // (lane = taddr.lane + t, col_byte).
  auto trace = job.desc.trace;
  uint32_t nthreads = trace ? trace->arch.num_threads() : 0;
  for (uint32_t t = 0; t < nthreads; ++t) {
    if (trace && !trace->tmask.test(t)) continue;
    uint32_t taddr_t = job.desc.taddr + t;   // lane += t
    uint32_t val = job.desc.st_values[t];
    core_->tmem_taddr_write_bytes(taddr_t, 0,
                                  reinterpret_cast<const uint8_t*>(&val), 4);
  }
  DT(3, "DMemAccess: ST done taddr=0x" << std::hex << job.desc.taddr << std::dec);
}

void LdstStage::complete(Job& job) {
  if (job.desc.is_load && job.desc.rd != 0 && job.desc.trace) {
    auto trace = job.desc.trace;
    ThreadMask tmask(trace->arch.num_threads());
    for (uint32_t t = 0; t < trace->arch.num_threads(); ++t)
      if (trace->tmask.test(t)) tmask.set(t);
    // One destination register per thread (scalar load).
    std::vector<Word> vals(trace->arch.num_threads());
    for (uint32_t t = 0; t < trace->arch.num_threads(); ++t)
      vals[t] = job.out_buf[t];
    core_->write_ireg(trace->wid, job.desc.rd, tmask, vals);
  }
  DT(3, "DMemAccess: complete trace=#" << job.desc.trace->uuid);
  Output.push(job.desc.trace, 0);
}

}  // namespace vortex
