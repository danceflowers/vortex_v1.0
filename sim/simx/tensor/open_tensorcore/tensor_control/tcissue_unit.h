// tensor_async_frontend.h
//
// 张量异步操作分发前端（单实例简化版）
//
// 支持三种操作类型：
//   - mma_load  : 从 TMEM 加载数据到本地 SRAM（AMem/BMem/CMem）
//                 MMA_LOAD(A/B/C) 各自展开为 4 个 MemUop（每 line/subtile 一个）
//   - mma_store : 将 DMem 中的最终结果写回 TMEM
//   - wmma      : 发起矩阵乘累加，创建 PendingWmmaJob 入队
//
// 本模块不是周期精确的：所有验证和入队在一个周期内完成，失败则通过
// trace_data->retry = true 让核心下周期重试。

#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>

#include "open_tensorcore/tensor_top/tensor_unit.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/dmem.h"
#include "open_tensorcore/local_memory/meta_mem.h"

namespace vortex {

namespace tud = tensor_unit_detail;

class TensorAsyncFrontend {
public:
  static void enqueue_async_mma_load(
      Core* core,
      const Arch& arch,
      uint32_t wid,
      uint32_t handle,
      const IntrTcuArgs& args,
      tud::AMemState* amem_state,
      tud::BMemState* bmem_state,
      tud::CMemState* cmem_state,
      tud::DMemState* dmem_state,
      tud::AMemState* pub_amem_state,
      tud::BMemState* pub_bmem_state,
      tud::CMemState* pub_cmem_state,
      tud::DMemState* pub_dmem_state,
      AMem* amem,
      BMem* bmem,
      CMem* cmem,
      DMem* dmem,
      MetaMem* metamem,
      std::deque<tud::MemUop>* mem_ops,
      std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
      TensorUnit::PerfStats* perf_stats,
      TensorUnit::ExeTraceData* trace_data);

  static void enqueue_async_mma_store(
      Core* core,
      const Arch& arch,
      uint32_t wid,
      uint32_t taddr,
      const IntrTcuArgs& args,
      tud::CMemState* cmem_state,
      tud::DMemState* dmem_state,
      tud::CMemState* pub_cmem_state,
      tud::DMemState* pub_dmem_state,
      CMem* cmem,
      DMem* dmem,
      std::deque<tud::MemUop>* mem_ops,
      TensorUnit::PerfStats* perf_stats,
      TensorUnit::ExeTraceData* trace_data);

  static void enqueue_async_wmma(
      Core* core,
      const Arch& arch,
      uint32_t wid,
      const IntrTcuArgs& args,
      uint32_t fmt_a,
      uint32_t fmt_b,
      uint32_t fmt_c,
      tud::AMemState* amem_state,
      tud::BMemState* bmem_state,
      tud::CMemState* cmem_state,
      tud::DMemState* dmem_state,
      tud::AMemState* pub_amem_state,
      tud::BMemState* pub_bmem_state,
      tud::CMemState* pub_cmem_state,
      tud::DMemState* pub_dmem_state,
      std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
      std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
      bool* prev_or,
      bool active_wmma_job_valid,
      TensorUnit::PerfStats* perf_stats,
      TensorUnit::ExeTraceData* trace_data);

  // ==========================================================================
  // enqueue_async_tcu_mma — single public entry point for tcgen05.mma.
  //
  // PTX semantics: one tcgen05.mma instruction internally issues
  //   (1) fill matrix A from TMEM (or shared via b_sdesc-style operand) -> AMem
  //   (2) fill matrix B (always sourced from shared) -> BMem
  //   (3) fill matrix C from TMEM if enable_input_d (idesc.input_d=1) -> CMem
  //   (4) WMMA primitive sequence (8 primitives for m16n16k16)
  //   (5) drain D (DMem -> TMEM at d_taddr)
  // all bound to a single async_id. Vortex TensorUnit issues these as MemUops
  // + PendingWmmaJob in the existing queues; the caller (TensorUnit::dispatch_
  // tcu_mma) provides the decoded fmt/handles/tile_ids from the idesc +
  // operand_block_t pair.
  //
  // The first_in_accum_seq detection (prev_or upper edge) is preserved.
  // ==========================================================================
  static void enqueue_async_tcu_mma(
      Core* core,
      const Arch& arch,
      uint32_t wid,
      const IntrTcuArgs& args,
      uint32_t fmt_a, uint32_t fmt_b, uint32_t fmt_c, uint32_t fmt_d,
      //uint32_t a_handle, 
      //uint32_t b_handle, 
      //uint32_t c_handle, 
      //uint32_t d_handle, 
      tud::AMemState* amem_state,
      tud::BMemState* bmem_state,
      tud::CMemState* cmem_state,
      tud::DMemState* dmem_state,
      tud::AMemState* pub_amem_state,
      tud::BMemState* pub_bmem_state,
      tud::CMemState* pub_cmem_state,
      tud::DMemState* pub_dmem_state,
      AMem* amem, BMem* bmem, CMem* cmem, DMem* dmem,
      MetaMem* metamem,
      std::deque<tud::MemUop>* mem_ops,
      std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
      std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
      std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
      bool* prev_or,
      bool active_wmma_job_valid,
      TensorUnit::PerfStats* perf_stats,
      TensorUnit::ExeTraceData* trace_data);
};

} // namespace vortex
