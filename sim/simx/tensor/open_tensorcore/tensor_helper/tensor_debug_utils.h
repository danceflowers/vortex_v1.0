#pragma once

// ============================================================================
// tensor_debug_utils.h —— TensorUnit 调试与日志（单实例简化版）
// ============================================================================

#include <deque>
#include <iosfwd>
#include <unordered_map>

#include "tensor_mem_port_types.h"
#include "types.h"
#include "tmem_window_planner.h"
#include "open_tensorcore/tensor_control/tensor_unit_types.h"

namespace vortex {

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

void log_window_plan_summary(const char* prefix,
                             uint32_t handle,
                             uint32_t window_id,
                             uint32_t tile_id,
                             const TmemWindowPlan* window);

void dump_tensor_unit_state(
    std::ostream& os,
    const std::deque<tensor_unit_detail::MemUop>& mem_ops,
    const std::unordered_map<uint32_t, uint32_t>& pending_mem_ops,
    const std::deque<tensor_unit_detail::PendingWmmaJob>& pending_wmma_jobs,
    bool active_wmma_job_valid,
    const tensor_unit_detail::PendingWmmaJob& active_wmma_job,
    const std::unordered_map<uint32_t, uint32_t>& pending_wmma_uops,
    const std::unordered_map<uint64_t, TensorMemPortRsp>& completed_tensor_mem_responses,
    const tensor_unit_detail::AMemState& amem_state,
    const tensor_unit_detail::BMemState& bmem_state,
    const tensor_unit_detail::CMemState& cmem_state,
    const tensor_unit_detail::DMemState& dmem_state,
    const tensor_unit_detail::AMemState& pub_amem_state,
    const tensor_unit_detail::BMemState& pub_bmem_state,
    const tensor_unit_detail::CMemState& pub_cmem_state,
    const tensor_unit_detail::DMemState& pub_dmem_state);

} // namespace vortex
