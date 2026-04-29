// ============================================================================
// tensor_tick_utils.cpp —— TensorUnit 每周期指令锁存辅助函数实现
// ============================================================================
// 详细说明参见 tensor_tick_utils.h。
// ============================================================================

#include "open_tensorcore/tensor_top/tensor_tick_utils.h"

#include "open_tensorcore/tensor_top/tensor_unit.h"

namespace vortex {

// 将张量指令从 Inputs 端口锁存到 Outputs 延迟管线。
// 遍历所有 issue-width 通道，对每条有效指令根据 TcuType 类型计算
// 类型相关延迟，然后以 (2 + delay) 的总延迟推入输出管线。
//
// 延迟分解：
//   基础延迟 = 2 周期（指令获取 + 初步译码）
//   MMA_LOAD / MMA_STORE 额外延迟 = 2 周期（描述符解析）  -> 总计 4 周期
//   WMMA              额外延迟 = 4 周期（操作数地址计算）  -> 总计 6 周期
//   其他指令          额外延迟 = 1 周期（简单派发）         -> 总计 3 周期
void latch_tensor_instructions_into_output_pipe(TensorUnit* simobject) {
  if (nullptr == simobject) {
    return;
  }

  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    auto& input = simobject->Inputs.at(iw);
    if (input.empty())
      continue;
    auto trace = input.front();
    auto tcu_type = std::get<TcuType>(trace->op_type);
    // Phase-2: with TCU_MMA fan-out, the macro op subsumes MMA_LOAD/STORE/WMMA.
    // Use a single TCU_MMA latency bucket; legacy types collapsed.
    int delay = 1;  // 默认：简单指令 1 周期
    switch (tcu_type) {
    case TcuType::TCU_MMA:
      delay = 4;    // 宏 TCU_MMA 需要额外 4 周期进行操作数地址计算和展开准备
      break;
    default:
      delay = 1;
      break;
    }
    // 总延迟 = 基础 2 周期 + 类型相关延迟
    simobject->Outputs.at(iw).push(trace, 2 + delay);
    DT(3, simobject->name() << ": op=" << tcu_type << ", " << *trace);
    input.pop();
  }
}

} // namespace vortex
