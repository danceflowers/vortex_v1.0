// tensor_mem_manager.h
//
// A/B/C/DMem 单实例状态管理器。
//
// 状态语义:
//   - *_ready    : 可接受新的 fill lifecycle（数据空闲 / 已使用完毕）
//   - *_pending  : fill 搬运进行中
//   - *_valid    : 数据完整有效，可供 WMMA 或 MMA_STORE 消费
//   - *_wmma_pending (A/B) : 正被 WMMA 原语消费
//   - b_ws_locked : weight-stationary 锁定（多个 WMMA 共享 BMem）
//
// 生命周期:
//   [ready] --(MMA_LOAD 发起)--> [pending] --(4 行 fill 完成)--> [valid]
//   [valid] --(WMMA 发射)--> [wmma_pending] --(8 uop push 完)--> [ready]
//
// 对 BMem ws=1: 8 uop push 完后保持 [valid]+[ws_locked]，不回 [ready]。

#pragma once

#include "types.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"
#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/dmem.h"
#include "open_tensorcore/local_memory/meta_mem.h"

namespace vortex {

namespace tud = tensor_unit_detail;

class TensorMemManager {
public:
  // ---- A ----
  static bool is_a_ready(const tud::AMemState& s);
  static void clear_a_storage(AMem* amem, MetaMem* metamem);
  static void set_amemstate(tud::AMemState* s,
                                const IntrTcuArgs& args,
                                AMem* amem,
                                MetaMem* metamem);
  static void mark_a_pending(tud::AMemState* s);
  static void mark_a_valid(tud::AMemState* s);
 // static void clear_a_after_wmma(tud::AMemState* s);

  // ---- B ----
  static bool is_b_ready(const tud::BMemState& s);
  static void clear_b_storage(BMem* bmem);
  static void set_bmemstate(tud::BMemState* s,
                                const IntrTcuArgs& args,
                                BMem* bmem);
  static void mark_b_pending(tud::BMemState* s);
  static void mark_b_valid(tud::BMemState* s);
  //static void clear_b_after_wmma(tud::BMemState* s, bool ws);

  // ---- C ----
  static bool is_c_ready(const tud::CMemState& s);
  static void clear_c_storage(CMem* cmem);
  static void set_cmemstate(tud::CMemState* s,
                                const IntrTcuArgs& args,
                                CMem* cmem);
  static void mark_c_pending(tud::CMemState* s);
  static void mark_c_valid(tud::CMemState* s);

  // ---- D ----
  static bool is_d_ready(const tud::DMemState& s);
  static void clear_d_storage(DMem* dmem);
  static void set_dmemstate(tud::DMemState* s,
                                const IntrTcuArgs& args,
                                DMem* dmem);

  // ---- 发布快照供调度器使用 ----
  static void snapshot_for_scheduler(const tud::AMemState& a_state,
                                     const tud::BMemState& b_state,
                                     const tud::CMemState& c_state,
                                     const tud::DMemState& d_state,
                                     tud::AMemState* pub_a,
                                     tud::BMemState* pub_b,
                                     tud::CMemState* pub_c,
                                     tud::DMemState* pub_d);
};

} // namespace vortex
