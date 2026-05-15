// tensor_wmma_retire_unit.cpp
//
// Legacy single-instance WMMA primitive retire implementation.

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

  // pending_tensorcore_retires is used as a FIFO between pop and retire.
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

// Retire one completed primitive and decrement the async_id's pending count.
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
  // Validate retire input and required state pointers.
  if (!retire.valid) {
    return true;
  }
  if (nullptr == core || nullptr == dmem_state || nullptr == pending_wmma_uops
   || nullptr == dmem || nullptr == perf_stats || nullptr == async_completion_out) {
    return false;
  }

  auto subtile_id = retire.meta.c_subtile_id;
  auto cycle = core->current_cycle();
  // Performance counters track primitive-tile retirement cycles.
  ++perf_stats->retired_primitive_tiles;
  perf_stats->epilogue_begin_cycle = cycle;

  // All primitive results are materialized in DMem.
  dmem->write_subtile_fp22(subtile_id, retire.fp22_out);

  auto pending_it = pending_wmma_uops->find(retire.meta.async_id);
  if (pending_it == pending_wmma_uops->end() || pending_it->second == 0) {
    std::abort();
  }
  --pending_it->second;

  // Last primitive retired: the macro WMMA operation is complete.
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
