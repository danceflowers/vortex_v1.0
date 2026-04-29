// ============================================================================
// tensor_runtime_utils.cpp —— TensorUnit 运行时辅助函数实现（单实例简化版）
// ============================================================================

#include "open_tensorcore/tensor_top/tensor_runtime_utils.h"

#include "core.h"
#include "open_tensorcore/local_memory/tensor_mem_pipeline.h"

namespace vortex {

void reset_all_tensor_unit_state(
    TensorUnit::PerfStats* perf_stats,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    DMem* dmem,
    MetaMem* metamem,
    tud::AMemState* amem_state,
    tud::BMemState* bmem_state,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    tud::AMemState* pub_amem_state,
    tud::BMemState* pub_bmem_state,
    tud::CMemState* pub_cmem_state,
    tud::DMemState* pub_dmem_state,
    std::deque<tud::MemUop>* mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
    std::deque<TensorCoreRetire>* pending_tensorcore_retires,
    std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
    tud::PendingWmmaJob* active_wmma_job,
    bool* active_wmma_job_valid,
    bool* prev_or,
    TensorCoreTop* tensorcore,
    TensorLocalMemArbiter* mem_arbiter,
    uint64_t* next_tensor_mem_request_id,
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses) {
  if (nullptr == perf_stats || nullptr == amem || nullptr == bmem || nullptr == cmem || nullptr == dmem
   || nullptr == metamem || nullptr == amem_state || nullptr == bmem_state || nullptr == cmem_state
   || nullptr == dmem_state || nullptr == pub_amem_state || nullptr == pub_bmem_state
   || nullptr == pub_cmem_state || nullptr == pub_dmem_state
   || nullptr == mem_ops || nullptr == pending_mem_ops || nullptr == pending_wmma_uops
   || nullptr == pending_tensorcore_retires || nullptr == pending_wmma_jobs
   || nullptr == active_wmma_job || nullptr == active_wmma_job_valid || nullptr == prev_or
   || nullptr == tensorcore || nullptr == mem_arbiter
   || nullptr == next_tensor_mem_request_id || nullptr == completed_tensor_mem_responses) {
    return;
  }

  *perf_stats = TensorUnit::PerfStats();
  amem->reset();
  bmem->reset();
  cmem->reset();
  dmem->reset();
  metamem->reset();
  amem_state->reset();
  bmem_state->reset();
  cmem_state->reset();
  dmem_state->reset();
  *pub_amem_state = *amem_state;
  *pub_bmem_state = *bmem_state;
  *pub_cmem_state = *cmem_state;
  *pub_dmem_state = *dmem_state;
  mem_ops->clear();
  pending_mem_ops->clear();
  pending_wmma_uops->clear();
  pending_tensorcore_retires->clear();
  pending_wmma_jobs->clear();
  *active_wmma_job = {};
  *active_wmma_job_valid = false;
  *prev_or = false;
  tensorcore->reset();
  mem_arbiter->reset();
  *next_tensor_mem_request_id = 1;
  completed_tensor_mem_responses->clear();
}

void tick_tmem_to_local_sram_transfer_pipeline(
    Core* core,
    SimPort<TensorMemPortReq>* req_out,
    SimPort<TensorAsyncOpCompletion>* completion_out,
    TensorLocalMemArbiter* mem_arbiter,
    uint32_t amem_write_lines_per_cycle,
    uint32_t bmem_write_lines_per_cycle,
    uint32_t cmem_write_subtiles_per_cycle,
    uint32_t cmem_read_subtiles_per_cycle,
    uint32_t meta_write_packets_per_cycle,
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
   || nullptr == cmem_state || nullptr == dmem_state || nullptr == perf_stats) {
    return;
  }

  if (mem_arbiter->current_cycle() != core->current_cycle()) {
    mem_arbiter->reset_for_cycle(core->current_cycle(),
                                 amem_write_lines_per_cycle,
                                 bmem_write_lines_per_cycle,
                                 cmem_write_subtiles_per_cycle,
                                 cmem_read_subtiles_per_cycle,
                                 meta_write_packets_per_cycle);
  }

  TensorMemPipeline::advance_tensor_memory_pipeline(core,
                                                     req_out,
                                                     completion_out,
                                                     mem_arbiter,
                                                     amem,
                                                     bmem,
                                                     cmem,
                                                     dmem,
                                                     metamem,
                                                     mem_ops,
                                                     pending_mem_ops,
                                                     completed_tensor_mem_responses,
                                                     next_tensor_mem_request_id,
                                                     amem_state,
                                                     bmem_state,
                                                     cmem_state,
                                                     dmem_state,
                                                     perf_stats,
                                                     rr_index);
}

} // namespace vortex
