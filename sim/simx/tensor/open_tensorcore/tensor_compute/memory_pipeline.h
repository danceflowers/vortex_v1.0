#pragma once

// ============================================================================
// TensorMemPipeline -- TMEM ↔ SRAM 传输引擎（多 pipeline 友好版）
// ============================================================================
//
// 负责 A/B/CMem 的 fill（TMEM → SRAM）和 DMem 的 store（DMem → TMEM）搬运。
// 该类本身保持无状态 static API，便于外层实例化多个 tensor memory pipeline。
// 本实现只在主调度入口中保留 TMEM issue 端口的共享限制；其他本地阶段默认可由
// 多个 pipeline 并行推进。
// ============================================================================

#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>

#include "simobject.h"
#include "tensor_mem_port_types.h"
#include "open_tensorcore/tensor_top/tensor_unit.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/dmem.h"
#include "open_tensorcore/local_memory/meta_mem.h"

namespace vortex {

class Core;
class TensorLocalMemArbiter;

namespace tud = tensor_unit_detail;

class TensorMemPipeline {
public:
  enum class LocalFillAction : uint8_t {
    None = 0,
    AData,
    BData,
    CData,
    Meta,
  };

  static bool mem_op_complete(const tud::MemUop& op);

  static void receive_one_tmem_response(
      SimPort<TensorMemPortRsp>* rsp_in,
      std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses);

  static bool take_tensor_mem_response(
      uint64_t request_id,
      std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses,
      TensorMemPortRsp* out);

  static LocalFillAction select_fill_local_write(const tud::MemUop& uop);

  static bool stage_fill_conversion_input(tud::MemUop* uop, uint32_t payload_fmt);

  static bool advance_fill_conversion(tud::MemUop* uop, uint32_t payload_fmt);

  static bool perform_fill_local_write(
      tud::MemUop* uop,
      LocalFillAction action,
      AMem* amem,
      BMem* bmem,
      CMem* cmem,
      MetaMem* metamem);

  static bool accept_fill_read_response(
      tud::MemUop* uop,
      const TensorMemPortRsp& response,
      uint32_t payload_packets);

  static uint64_t issue_fill_read_request(
      const tud::MemUop& uop,
      Core* core,
      SimPort<TensorMemPortReq>* req_out,
      uint64_t* next_tensor_mem_request_id,
      const tud::AMemState& amem_state,
      const tud::BMemState& bmem_state,
      const tud::CMemState& cmem_state);

  static bool stage_store_read(
      tud::MemUop* uop,
      const tud::CMemState& c_state,
      const tud::DMemState& d_state,
      CMem* cmem,
      DMem* dmem);

  static bool advance_store_conversion(
      tud::MemUop* uop,
      const tud::DMemState& d_state);

  static bool stage_store_converted_packet(tud::MemUop* uop);

  static bool emit_store_packet(tud::MemUop* uop);

  static uint64_t issue_store_write_request(
      const tud::MemUop& uop,
      Core* core,
      SimPort<TensorMemPortReq>* req_out,
      uint64_t* next_tensor_mem_request_id);

  // 主调度入口。保持原有签名，便于直接替换旧实现。
  static void advance_tensor_memory_pipeline(
      Core* core,
      SimPort<TensorMemPortReq>* req_out,
      SimPort<TensorAsyncOpCompletion>* completion_out,
      TensorLocalMemArbiter* mem_arbiter,
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
};

} // namespace vortex
