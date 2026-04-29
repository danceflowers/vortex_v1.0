// tensor_wmma_retire_unit.cpp
//
// WMMA 原语退休单元实现（单实例简化版）。

#include "open_tensorcore/tensor_control/tensor_wmma_retire_unit.h"

#include "core.h"
#include "open_tensorcore/tensor_control/tensor_mem_manager.h"

namespace vortex {

void TensorWmmaRetireUnit::advance_tensorcore_pipeline(
    Core* core,
    TensorCoreTop* tensorcore,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
    std::deque<TensorCoreRetire>* pending_tensorcore_retires,
    CMem* cmem,
    DMem* dmem,
    TensorUnit::PerfStats* perf_stats,
    SimPort<TensorAsyncOpCompletion>* async_completion_out) {
  if (nullptr == tensorcore || nullptr == pending_tensorcore_retires) {
    return;
  }
  tensorcore->tick(true);

  //这个pending_tensorcore_retires是deque，相当于FIFO
  TensorCoreRetire retire;
  if (tensorcore->pop_retired(&retire)) {
    pending_tensorcore_retires->push_back(retire);
  }

  if (!pending_tensorcore_retires->empty()) {
    if (retire_primitive(core,
                         pending_tensorcore_retires->front(),
                         //mem_arbiter,
                         cmem_state,
                         dmem_state,
                         pending_wmma_uops,
                         cmem,
                         dmem,
                         perf_stats,
                         async_completion_out)) {
      pending_tensorcore_retires->pop_front();
    }
  }
}

//处理已完成的uop，核心逻辑是完成一个primitive，对应pending_wmma_uops就要减计算，pending_wmma_uops是unordered_map，所以对应硬件是一个寄存器表
bool TensorWmmaRetireUnit::retire_primitive(
    Core* core,
    const TensorCoreRetire& retire,
  //  TensorLocalMemArbiter* mem_arbiter,
    tud::CMemState* cmem_state,
    tud::DMemState* dmem_state,
    std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
    CMem* cmem,
    DMem* dmem,
    TensorUnit::PerfStats* perf_stats,
    SimPort<TensorAsyncOpCompletion>* async_completion_out) {
  (void)cmem_state; (void)cmem; (void)mem_arbiter;
  //做一遍检查
  if (!retire.valid) {
    return true;
  }
  if (nullptr == core || nullptr == dmem_state || nullptr == pending_wmma_uops
   || nullptr == dmem || nullptr == perf_stats || nullptr == async_completion_out) {
    return false;
  }

  auto subtile_id = retire.meta.c_subtile_id;
  auto cycle = core->current_cycle();
  //性能计数器，统计完成的primitive tile数量
  ++perf_stats->retired_primitive_tiles;
  if (perf_stats->first_tc_retire_cycle == 0) {
    perf_stats->first_tc_retire_cycle = cycle;
  }
  perf_stats->last_tc_retire_cycle = cycle;

  // 所有退休结果统一写入 DMem (原位累加 / 最终结果)
  dmem->write_subtile_fp22(subtile_id, retire.fp22_out);

  auto pending_it = pending_wmma_uops->find(retire.meta.async_id);
  if (pending_it == pending_wmma_uops->end() || pending_it->second == 0) {
    std::abort();
  }
  --pending_it->second;

  // 最后一个 primitive 退休: WMMA 宏操作完成
  if (pending_it->second == 0) {
    pending_wmma_uops->erase(pending_it);

    // if (dmem_state->d_wmma_inflight == 0) {
    //   std::abort();
    // }
    //--dmem_state->d_wmma_inflight;
    dmem_state->d_valid = dmem->valid();

    ++perf_stats->retired_macro_wmma;

    async_completion_out->push({retire.meta.async_id}, 1);
  }
  return true;
}

} // namespace vortex
