// tensor_wmma_retire_unit.h
//
// Legacy single-instance WMMA primitive retire unit.
//
// Retired primitive results are written to DMem[c_subtile_id]. When all
// primitives for an async_id retire, the unit updates DMem visibility and emits
// an async completion notification.

#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>

#include "simobject.h"
#include "tensor_mem_port_types.h"
#include "open_tensorcore/tensor_top/tensor_unit.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/local_memory/tensor_local_mem_arbiter.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/dmem.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"

namespace vortex {

namespace tud = tensor_unit_detail;

class TensorWmmaRetireUnit {
public:
  static void advance_tensorcore_pipeline(
      Core* core,
      TensorCoreTop* tensorcore,
      TensorLocalMemArbiter* mem_arbiter,
      tud::CMemState* cmem_state,
      tud::DMemState* dmem_state,
      std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
      std::deque<TensorCoreRetire>* pending_tensorcore_retires,
      CMem* cmem,
      DMem* dmem,
      TensorUnit::PerfStats* perf_stats,
      SimPort<TensorAsyncOpCompletion>* async_completion_out);

  static bool retire_primitive(
      Core* core,
      const TensorCoreRetire& retire,
      TensorLocalMemArbiter* mem_arbiter,
      tud::CMemState* cmem_state,
      tud::DMemState* dmem_state,
      std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
      CMem* cmem,
      DMem* dmem,
      TensorUnit::PerfStats* perf_stats,
      SimPort<TensorAsyncOpCompletion>* async_completion_out);
};

} // namespace vortex
