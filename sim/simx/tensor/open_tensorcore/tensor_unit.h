// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// ============================================================================
// TensorUnit —— TensorCore 近核前端控制器
// ============================================================================
//
// TensorUnit 是 TensorCore 的近核前端控制器，位于 execute stage 与底层
// TensorCore 计算阵列之间。其主要职责包括：
//
//   1. 接收来自 execute stage 的描述符级宏操作（mma_load / mma_store / wmma），
//      这些宏操作由编译器或运行时通过描述符打包产生。
//
//   2. 将 TMEM（Tensor Memory）数据包逐行搬运到本地 SRAM 操作数存储器
//      （AMem / BMem / CMem），完成数据预备。
//
//   3. 将一条宏 WMMA（Warp-level Matrix Multiply-Accumulate）展开为
//      8 条原始（primitive）TensorCore 发射，每条对应一个子块运算。
//
//   4. 将原始计算结果写回 CMem / DMem，并在全部子块完成后触发异步操作
//      完成信号（TensorAsyncOpCompletion）。
//
// 本类采用 Pimpl（Pointer to Implementation）模式：对外只暴露接口，
// 所有内部状态（槽位、本地存储器、队列等）封装在 Impl 类中，降低头文件
// 耦合度并加速编译。
// ============================================================================

#include <algorithm>
#include <iosfwd>
#include <simobject.h>
#include "instr_trace.h"
#include "tensor_mem_port_types.h"
#include "open_tensorcore/tensor_helper/tensor_debug_utils.h"

namespace vortex {

class Core;

class TensorUnit : public SimObject<TensorUnit> {
public:

  // 执行追踪数据：附在每条指令 trace 上，用于指示是否需要写回目标寄存器
  // 以及是否需要重试（retry）。
  struct ExeTraceData : public ITraceData {
    using Ptr = std::shared_ptr<ExeTraceData>;
    bool rd_write = false;   // 是否写回目的寄存器
    bool retry = false;      // 是否需要重新发射（调度器层面的反馈）
  };

  // --------------------------------------------------------------------------
  // PerfStats —— 性能统计计数器
  // --------------------------------------------------------------------------
  // 用于收集 TensorUnit 运行期间的各类延迟与阻塞统计信息，最终汇总到
  // 全局性能报告中。各字段含义如下：
  //
  //   latency                       : 总延迟周期数（从第一条发射到最后一条退休）
  //   tc_active_cycles              : TensorCore 计算阵列处于活跃状态的周期数
  //   tc_idle_cycles                : TensorCore 计算阵列处于空闲状态的周期数
  //   issued_primitive_tiles        : 已发射的原始子块总数
  //   retired_primitive_tiles       : 已退休的原始子块总数
  //   first_tc_issue_cycle          : 第一条原始子块发射的周期号
  //   last_tc_issue_cycle           : 最后一条原始子块发射的周期号
  //   first_tc_retire_cycle         : 第一条原始子块退休的周期号
  //   last_tc_retire_cycle          : 最后一条原始子块退休的周期号
  //   stall_a_not_ready             : 因 A 操作数未就绪而阻塞的周期数
  //   stall_b_not_ready             : 因 B 操作数未就绪而阻塞的周期数
  //   stall_c_not_ready             : 因 C 累加器未就绪而阻塞的周期数
  //   stall_tc_busy                 : 因 TensorCore 忙碌而阻塞的周期数
  //   stall_no_wmma_job_ready       : 无可发射的 WMMA 作业而阻塞的周期数
  //   stall_no_tensor_instr_candidate: 调度器中无张量指令候选而阻塞的周期数
  //   stall_mma_load_handle_not_ready: MMA_LOAD 句柄尚未就绪而阻塞的周期数
  //   stall_handle_reuse            : 句柄被复用导致冲突而阻塞的周期数
  //   stall_handle_busy_due_to_tma_load : 句柄因 TMA_LOAD 占用而阻塞
  //   stall_handle_busy_due_to_tma_store_or_shift : 句柄因 TMA_STORE/SHIFT 占用而阻塞
  //   stall_slot_busy               : 操作数槽位忙碌而阻塞的周期数
  //   stall_tmem_read_port_busy     : TMEM 读端口忙碌而阻塞
  //   stall_tmem_write_port_busy    : TMEM 写端口忙碌而阻塞
  //   stall_amem_port_busy          : AMem 端口忙碌而阻塞
  //   stall_bmem_port_busy          : BMem 端口忙碌而阻塞
  //   stall_cmem_port_busy          : CMem 端口忙碌而阻塞
  //   stall_meta_port_busy          : MetaMem 端口忙碌而阻塞
  //   issued_macro_wmma             : 已发射的宏 WMMA 指令数
  //   retired_macro_wmma            : 已退休的宏 WMMA 指令数
  //   pending_wmma_jobs_max         : 待处理 WMMA 作业队列峰值深度
  //   pending_wmma_depth_cycles_0/1/2/3plus : 各深度档位的持续周期数（用于排队分析）
  //   mem_queue_max                 : 内存操作队列峰值深度
  //   stall_a_meta_not_ready        : A 操作数元数据未就绪而阻塞
  //   stall_no_wmma_job_builder_empty         : WMMA 作业构造器为空而阻塞
  //   stall_no_wmma_waiting_for_mma_load      : WMMA 等待 MMA_LOAD 完成而阻塞
  //   stall_no_wmma_waiting_for_handle_alloc  : WMMA 等待句柄分配而阻塞
  //   stall_no_wmma_waiting_for_slot_release  : WMMA 等待槽位释放而阻塞
  //   stall_no_wmma_waiting_for_c_wmma_inflight_drain : WMMA 等待 C 槽位在飞计算排空
  //   stall_no_wmma_waiting_for_accum_live_only       : WMMA 等待累加器独占而阻塞
  //   stall_no_wmma_waiting_for_dirty_flush_only      : WMMA 等待脏数据刷出而阻塞
  //   stall_no_wmma_waiting_for_store_pending         : WMMA 等待 store 操作完成而阻塞
  //   stall_no_wmma_waiting_for_ab_wmma_pending_clear : WMMA 等待 AB 槽位的 wmma_pending 清除
  // --------------------------------------------------------------------------
	struct PerfStats {
		uint64_t latency;
    uint64_t tc_active_cycles;
    uint64_t tc_idle_cycles;
    uint64_t issued_primitive_tiles;
    uint64_t retired_primitive_tiles;
    uint64_t first_tc_issue_cycle;
    uint64_t last_tc_issue_cycle;
    uint64_t first_tc_retire_cycle;
    uint64_t last_tc_retire_cycle;
    uint64_t stall_a_not_ready;
    uint64_t stall_b_not_ready;
    uint64_t stall_c_not_ready;
    uint64_t stall_tc_busy;
    uint64_t stall_no_wmma_job_ready;
    uint64_t stall_no_tensor_instr_candidate;
    uint64_t stall_mma_load_handle_not_ready;
    uint64_t stall_handle_reuse;
    uint64_t stall_handle_busy_due_to_tma_load;
    uint64_t stall_handle_busy_due_to_tma_store_or_shift;
    uint64_t stall_slot_busy;
    uint64_t stall_tmem_read_port_busy;
    uint64_t stall_tmem_write_port_busy;
    uint64_t stall_amem_port_busy;
    uint64_t stall_bmem_port_busy;
    uint64_t stall_cmem_port_busy;
    uint64_t stall_meta_port_busy;
    uint64_t issued_macro_wmma;
    uint64_t retired_macro_wmma;
    uint64_t pending_wmma_jobs_max;
    uint64_t pending_wmma_depth_cycles_0;
    uint64_t pending_wmma_depth_cycles_1;
    uint64_t pending_wmma_depth_cycles_2;
    uint64_t pending_wmma_depth_cycles_3plus;
    uint64_t mem_queue_max;
    uint64_t stall_a_meta_not_ready;
    uint64_t stall_no_wmma_job_builder_empty;
    uint64_t stall_no_wmma_waiting_for_mma_load;
    uint64_t stall_no_wmma_waiting_for_handle_alloc;
    uint64_t stall_no_wmma_waiting_for_slot_release;
    uint64_t stall_no_wmma_waiting_for_c_wmma_inflight_drain;
    uint64_t stall_no_wmma_waiting_for_accum_live_only;
    uint64_t stall_no_wmma_waiting_for_dirty_flush_only;
    uint64_t stall_no_wmma_waiting_for_store_pending;
    uint64_t stall_no_wmma_waiting_for_ab_wmma_pending_clear;

		PerfStats()
			: latency(0)
      , tc_active_cycles(0)
      , tc_idle_cycles(0)
      , issued_primitive_tiles(0)
      , retired_primitive_tiles(0)
      , first_tc_issue_cycle(0)
      , last_tc_issue_cycle(0)
      , first_tc_retire_cycle(0)
      , last_tc_retire_cycle(0)
      , stall_a_not_ready(0)
      , stall_b_not_ready(0)
      , stall_c_not_ready(0)
      , stall_tc_busy(0)
      , stall_no_wmma_job_ready(0)
      , stall_no_tensor_instr_candidate(0)
      , stall_mma_load_handle_not_ready(0)
      , stall_handle_reuse(0)
      , stall_handle_busy_due_to_tma_load(0)
      , stall_handle_busy_due_to_tma_store_or_shift(0)
      , stall_slot_busy(0)
      , stall_tmem_read_port_busy(0)
      , stall_tmem_write_port_busy(0)
      , stall_amem_port_busy(0)
      , stall_bmem_port_busy(0)
      , stall_cmem_port_busy(0)
      , stall_meta_port_busy(0)
      , issued_macro_wmma(0)
      , retired_macro_wmma(0)
      , pending_wmma_jobs_max(0)
      , pending_wmma_depth_cycles_0(0)
      , pending_wmma_depth_cycles_1(0)
      , pending_wmma_depth_cycles_2(0)
      , pending_wmma_depth_cycles_3plus(0)
      , mem_queue_max(0)
      , stall_a_meta_not_ready(0)
      , stall_no_wmma_job_builder_empty(0)
      , stall_no_wmma_waiting_for_mma_load(0)
      , stall_no_wmma_waiting_for_handle_alloc(0)
      , stall_no_wmma_waiting_for_slot_release(0)
      , stall_no_wmma_waiting_for_c_wmma_inflight_drain(0)
      , stall_no_wmma_waiting_for_accum_live_only(0)
      , stall_no_wmma_waiting_for_dirty_flush_only(0)
      , stall_no_wmma_waiting_for_store_pending(0)
      , stall_no_wmma_waiting_for_ab_wmma_pending_clear(0)
		{}

		PerfStats& operator+=(const PerfStats& rhs) {
			this->latency += rhs.latency;
      this->tc_active_cycles += rhs.tc_active_cycles;
      this->tc_idle_cycles += rhs.tc_idle_cycles;
      this->issued_primitive_tiles += rhs.issued_primitive_tiles;
      this->retired_primitive_tiles += rhs.retired_primitive_tiles;
      if (0 == this->first_tc_issue_cycle || (rhs.first_tc_issue_cycle != 0 && rhs.first_tc_issue_cycle < this->first_tc_issue_cycle))
        this->first_tc_issue_cycle = rhs.first_tc_issue_cycle;
      this->last_tc_issue_cycle = std::max(this->last_tc_issue_cycle, rhs.last_tc_issue_cycle);
      if (0 == this->first_tc_retire_cycle || (rhs.first_tc_retire_cycle != 0 && rhs.first_tc_retire_cycle < this->first_tc_retire_cycle))
        this->first_tc_retire_cycle = rhs.first_tc_retire_cycle;
      this->last_tc_retire_cycle = std::max(this->last_tc_retire_cycle, rhs.last_tc_retire_cycle);
      this->stall_a_not_ready += rhs.stall_a_not_ready;
      this->stall_b_not_ready += rhs.stall_b_not_ready;
      this->stall_c_not_ready += rhs.stall_c_not_ready;
      this->stall_tc_busy += rhs.stall_tc_busy;
      this->stall_no_wmma_job_ready += rhs.stall_no_wmma_job_ready;
      this->stall_no_tensor_instr_candidate += rhs.stall_no_tensor_instr_candidate;
      this->stall_mma_load_handle_not_ready += rhs.stall_mma_load_handle_not_ready;
      this->stall_handle_reuse += rhs.stall_handle_reuse;
      this->stall_handle_busy_due_to_tma_load += rhs.stall_handle_busy_due_to_tma_load;
      this->stall_handle_busy_due_to_tma_store_or_shift += rhs.stall_handle_busy_due_to_tma_store_or_shift;
      this->stall_slot_busy += rhs.stall_slot_busy;
      this->stall_tmem_read_port_busy += rhs.stall_tmem_read_port_busy;
      this->stall_tmem_write_port_busy += rhs.stall_tmem_write_port_busy;
      this->stall_amem_port_busy += rhs.stall_amem_port_busy;
      this->stall_bmem_port_busy += rhs.stall_bmem_port_busy;
      this->stall_cmem_port_busy += rhs.stall_cmem_port_busy;
      this->stall_meta_port_busy += rhs.stall_meta_port_busy;
      this->issued_macro_wmma += rhs.issued_macro_wmma;
      this->retired_macro_wmma += rhs.retired_macro_wmma;
      this->pending_wmma_jobs_max = std::max(this->pending_wmma_jobs_max, rhs.pending_wmma_jobs_max);
      this->pending_wmma_depth_cycles_0 += rhs.pending_wmma_depth_cycles_0;
      this->pending_wmma_depth_cycles_1 += rhs.pending_wmma_depth_cycles_1;
      this->pending_wmma_depth_cycles_2 += rhs.pending_wmma_depth_cycles_2;
      this->pending_wmma_depth_cycles_3plus += rhs.pending_wmma_depth_cycles_3plus;
      this->mem_queue_max = std::max(this->mem_queue_max, rhs.mem_queue_max);
      this->stall_a_meta_not_ready += rhs.stall_a_meta_not_ready;
      this->stall_no_wmma_job_builder_empty += rhs.stall_no_wmma_job_builder_empty;
      this->stall_no_wmma_waiting_for_mma_load += rhs.stall_no_wmma_waiting_for_mma_load;
      this->stall_no_wmma_waiting_for_handle_alloc += rhs.stall_no_wmma_waiting_for_handle_alloc;
      this->stall_no_wmma_waiting_for_slot_release += rhs.stall_no_wmma_waiting_for_slot_release;
      this->stall_no_wmma_waiting_for_c_wmma_inflight_drain += rhs.stall_no_wmma_waiting_for_c_wmma_inflight_drain;
      this->stall_no_wmma_waiting_for_accum_live_only += rhs.stall_no_wmma_waiting_for_accum_live_only;
      this->stall_no_wmma_waiting_for_dirty_flush_only += rhs.stall_no_wmma_waiting_for_dirty_flush_only;
      this->stall_no_wmma_waiting_for_store_pending += rhs.stall_no_wmma_waiting_for_store_pending;
      this->stall_no_wmma_waiting_for_ab_wmma_pending_clear += rhs.stall_no_wmma_waiting_for_ab_wmma_pending_clear;
			return *this;
		}
	};

  // --------------------------------------------------------------------------
  // IssueBlockReason —— 发射阻塞原因枚举
  // --------------------------------------------------------------------------
  // 当调度器尝试向 TensorUnit 发射张量指令但被拒绝时，classify_issue_block()
  // 返回以下原因之一，用于驱动性能计数器和调试日志。
  //
  //   None                        : 无阻塞，可正常发射
  //   ANotReady                   : A 操作数槽位数据尚未就绪
  //   BNotReady                   : B 操作数槽位数据尚未就绪
  //   CNotReady                   : C 累加器槽位数据尚未就绪
  //   TcBusy                      : TensorCore 计算阵列正忙，无法接受新的原始发射
  //   NoTensorInstrCandidate      : 调度器中当前没有可选的张量指令
  //   MmaLoadHandleNotReady       : MMA_LOAD 的目标句柄尚未分配或未就绪
  //   HandleBusyDueToTmaLoad      : 句柄正被 TMA_LOAD 操作占用
  //   HandleBusyDueToTmaStoreOrShift : 句柄正被 TMA_STORE 或 TMEM_SHIFT 占用
  //   HandleReuse                 : 句柄复用冲突（同一句柄上有未完成的操作）
  //   SlotBusy                    : 目标操作数槽位正忙（正在被其他操作填充或使用）
  //   AMetaNotReady               : A 操作数的元数据（meta）尚未就绪
  // --------------------------------------------------------------------------
  enum class IssueBlockReason : uint8_t {
    None = 0,
    ANotReady,
    BNotReady,
    CNotReady,
    TcBusy,
    NoTensorInstrCandidate,
    MmaLoadHandleNotReady,
    HandleBusyDueToTmaLoad,
    HandleBusyDueToTmaStoreOrShift,
    HandleReuse,
    SlotBusy,
    AMetaNotReady,
  };

  // 指令输入端口（来自 execute stage），每个 issue-width 通道一个
  std::vector<SimPort<instr_trace_t*>> Inputs;
  // 指令输出端口（经过延迟管线后送出），每个 issue-width 通道一个
	std::vector<SimPort<instr_trace_t*>> Outputs;
#ifdef EXT_TCU_ENABLE
  // 以下三个端口是 TensorExecuteSystem 与 TensorMemSystem 之间的数据包边界：
  SimPort<TensorMemPortReq> TensorMemReqOut;           // TMEM 读写请求输出
  SimPort<TensorMemPortRsp> TensorMemRspIn;            // TMEM 读写响应输入
  SimPort<TensorAsyncOpCompletion> TensorAsyncOpCompletionOut; // 异步操作完成通知输出
#endif

  TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core);
  virtual ~TensorUnit();

  virtual void reset();

  virtual void tick();

  // ---- 协处理器统一指令接口 ----
  // Vortex Core 只转发原始字段，TensorUnit 内部由 TcDecode 解释语义。
  // PTX path: each tcgen05.mma carries its idesc directly.

  /// Phase-2 PTX-aligned tcgen05.mma dispatch (single-instruction fan-out).
  /// 接收 decode.cpp 抓的原始字段：rs1_value=idesc(32-bit),
  /// rs2_value=operand_block_t LMEM 指针, qualifier=funct7。d_taddr 从
  /// operand_block_t 读取，rd 固定 x0。
  /// TensorUnit 内部走 TcDecode::decode_tcu_mma 译码 + 调
  /// TensorAsyncFrontend::enqueue_async_tcu_mma 展开 fill->compute->drain。
  void dispatch_tcu_mma(uint32_t wid,
                        uint32_t rs1_value,
                        uint32_t rs2_value,
                        uint32_t qualifier,
                        ExeTraceData* trace_data);

  // ---- 调度器查询接口 ----
  // Core 侧 warp scheduler 用于判断 tensor 指令是否可发射。

  uint32_t scheduler_score(uint32_t wid, TcuType tcu_type) const;
  IssueBlockReason classify_issue_block(uint32_t wid, TcuType tcu_type) const;
  void record_issue_stall(IssueBlockReason reason);
  void record_no_tensor_instr_candidate_stall();

	const PerfStats& perf_stats() const;

  void dump_debug_state(std::ostream& os) const;

private:
  // Pimpl 模式：所有内部状态（槽位数组、本地存储器、队列、TensorCore 实例等）
  // 封装在 Impl 类中，避免头文件暴露实现细节，减少编译依赖。
	class Impl;
	Impl* impl_;
};

} // namespace vortex
