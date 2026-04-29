#pragma once

// ============================================================================
// tensor_tick_utils.h —— TensorUnit 每周期指令锁存辅助函数
// ============================================================================
//
// 声明 latch_tensor_instructions_into_output_pipe()，该函数在 TensorUnit
// 的 tick() 中被调用，负责将来自 execute stage 的张量指令从输入端口（Inputs）
// 锁存到输出延迟管线（Outputs），并根据指令类型施加不同的管线延迟。
//
// 延迟建模说明：
//   所有指令都经过一个基础延迟 2 个周期（模拟指令获取和初步译码），
//   再加上类型相关的派发延迟：
//     - MMA_LOAD / MMA_STORE : 2 周期  -> 总延迟 2 + 2 = 4 周期
//     - WMMA                 : 4 周期  -> 总延迟 2 + 4 = 6 周期
//     - 其他（如 TMEM_ALLOC 等）: 1 周期  -> 总延迟 2 + 1 = 3 周期
//
//   这些延迟值模拟了硬件中从指令进入 TensorUnit 到实际开始执行的
//   译码与派发（decode & dispatch）延迟。
// ============================================================================

namespace vortex {

class TensorUnit;

// 将张量指令从 execute stage 输入端口锁存到输出延迟管线，
// 施加类型相关的管线延迟以模拟译码和派发时间。
void latch_tensor_instructions_into_output_pipe(TensorUnit* simobject);

} // namespace vortex
