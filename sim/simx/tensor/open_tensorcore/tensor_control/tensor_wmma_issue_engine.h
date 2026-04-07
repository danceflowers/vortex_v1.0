// tensor_wmma_issue_engine.h
//
// WMMA 原语发射引擎（WMMA Primitive Issue Engine）
//
// 本模块负责将 PendingWmmaJob（由 TensorAsyncFrontend 创建的宏 WMMA 操作）
// 分解为 4 个原语操作（primitive），逐周期下发到 TensorCore 计算管线。
//
// 原语分解方式（Primitive Decomposition, m16n16k8）：
//   一条 WMMA 被分解为 kWmmaPrimitiveCount = 4 个 8x8 原语：
//     - 4 个子块（subtile）: storage_m=[0,1], storage_n=[0,1]  (2x2 = 4 子块)
//     - K=8 无需拆分，K 累加由软件发射多条 WMMA 指令控制
//     - 子块内映射: c_subtile_id = next_uop, storage_m = c_subtile_id/2, storage_n = c_subtile_id%2
//
// 发射条件（每周期最多发射 1 个原语）：
//   - TensorCore 管线就绪（ready(true)）
//   - AMem、BMem、CMem 读端口可用（通过 TensorLocalMemArbiter 仲裁）
//
// 阻塞分类（Stall Classification）：
//   当无法发射时，本模块会细分阻塞原因用于性能分析：
//   - TensorCore 管线忙（stall_tc_busy）
//   - AMem/BMem/CMem 端口冲突（stall_amem/bmem/cmem_port_busy）
//   - 无就绪的 WMMA 作业（stall_no_wmma_job_ready），进一步细分为：
//     - 等待 mma_load 完成
//     - 等待 handle 分配
//     - 等待槽位释放（WMMA inflight drain / store pending / dirty flush 等）
//     - 作业构建器为空

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>

#include "open_tensorcore/tensor_top/tensor_unit.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/local_memory/tensor_local_mem_arbiter.h"
#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"

namespace vortex {

namespace tud = tensor_unit_detail;

class TensorWmmaIssueEngine {
public:
  // 返回当前待处理的 WMMA 作业深度（队列中的 + 正在发射中的 active job）
  static uint32_t current_pending_wmma_depth(
      const std::deque<tud::PendingWmmaJob>& pending_wmma_jobs,
      bool active_wmma_job_valid);

  // 采样待处理 WMMA 深度到性能统计（按深度 0/1/2/3+ 分桶计数）
  static void sample_pending_wmma_depth(
      const std::deque<tud::PendingWmmaJob>& pending_wmma_jobs,
      bool active_wmma_job_valid,
      TensorUnit::PerfStats* perf_stats);

  // 检查是否有正在执行的 mma_load 填充操作（mem_ops 队列或槽位 pending 状态）
  static bool has_inflight_mma_load_build(
      const std::deque<tud::MemUop>& mem_ops,
      const std::array<tud::ASlotState, tud::kNumOperandSlots>& a_slots,
      const std::array<tud::BSlotState, tud::kNumOperandSlots>& b_slots,
      const std::array<tud::CSlotState, tud::kNumOperandSlots>& c_slots);

  // 分类槽位无法释放的具体原因（用于性能分析）
  static tud::SlotReleaseReason classify_slot_release_reason(
      const std::array<tud::ASlotState, tud::kNumOperandSlots>& a_slots,
      const std::array<tud::BSlotState, tud::kNumOperandSlots>& b_slots,
      const std::array<tud::CSlotState, tud::kNumOperandSlots>& c_slots);

  // 分类"无就绪 WMMA 作业"的根本原因
  static tud::NoWmmaReadyReason classify_no_wmma_ready_reason(
      Core* core,
      const std::deque<tud::MemUop>& mem_ops,
      const std::array<tud::ASlotState, tud::kNumOperandSlots>& a_slots,
      const std::array<tud::BSlotState, tud::kNumOperandSlots>& b_slots,
      const std::array<tud::CSlotState, tud::kNumOperandSlots>& c_slots);

  // 记录"无就绪 WMMA"阻塞事件到性能计数器
  static void record_no_wmma_ready_stall(
      Core* core,
      const std::deque<tud::MemUop>& mem_ops,
      const std::array<tud::ASlotState, tud::kNumOperandSlots>& a_slots,
      const std::array<tud::BSlotState, tud::kNumOperandSlots>& b_slots,
      const std::array<tud::CSlotState, tud::kNumOperandSlots>& c_slots,
      TensorUnit::PerfStats* perf_stats);

  // --------------------------------------------------------------------------
  // issue_wmma_primitives（核心方法）
  // --------------------------------------------------------------------------
  // 每周期调用一次。尝试从 active_wmma_job 中发射下一个原语到 TensorCore：
  //   1. 若无 active job，从 pending_wmma_jobs 队列取出一个
  //   2. 计算当前原语的 storage_m/n/k 和所需的 SRAM bank mask
  //   3. 通过 mem_arbiter 仲裁 AMem/BMem/CMem 读端口
  //   4. 从 SRAM 读取 A/B/C 数据块
  //   5. 构建 TensorCoreMeta 并调用 tensorcore->push_uop() 下发
  //   6. 当所有 8 个原语发射完毕后，释放 A/B 的 wmma_pending 状态
  static void issue_wmma_primitives(
      Core* core,
      const Arch& arch,
      TensorLocalMemArbiter* mem_arbiter,
      AMem* amem,
      BMem* bmem,
      CMem* cmem,
      TensorCoreTop* tensorcore,
      std::array<tud::ASlotState, tud::kNumOperandSlots>* a_slots,
      std::array<tud::BSlotState, tud::kNumOperandSlots>* b_slots,
      std::array<tud::CSlotState, tud::kNumOperandSlots>* c_slots,
      const std::deque<tud::MemUop>& mem_ops,
      std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
      std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
      tud::PendingWmmaJob* active_wmma_job,
      bool* active_wmma_job_valid,
      TensorUnit::PerfStats* perf_stats);
};

} // namespace vortex
