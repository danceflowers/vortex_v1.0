#pragma once

// ============================================================================
// tensor_runtime_utils.h —— TensorUnit 运行时辅助函数（单实例简化版）
// ============================================================================

#include <cstdint>
#include <deque>
#include <unordered_map>

#include "tensor_mem_port_types.h"
#include "open_tensorcore/tensor_top/tensor_unit.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/dmem.h"
#include "open_tensorcore/local_memory/meta_mem.h"
#include "open_tensorcore/local_memory/tensor_local_mem_arbiter.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"

namespace vortex {

class Core;

namespace tud = tensor_unit_detail;

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
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses);

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
    uint32_t* rr_index);

} // namespace vortex
