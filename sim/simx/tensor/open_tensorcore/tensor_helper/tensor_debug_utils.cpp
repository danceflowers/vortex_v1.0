// ============================================================================
// tensor_debug_utils.cpp —— TensorUnit 调试与日志实现（单实例简化版）
// ============================================================================

#include "tensor_debug_utils.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "tensor_cfg.h"

namespace vortex {

namespace vt = vortex::tensor;

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args) {
  auto fmt_a = args.fmt_a;
  auto fmt_b = args.fmt_b;
  (void)fmt_a; (void)fmt_b;
  switch (tcu_type) {
  case TcuType::TMEM_ALLOC:
    return {"TMEM_ALLOC", ""};
  case TcuType::TMEM_REL_PERMIT:
    return {"TMEM_REL_PERMIT", ""};
  case TcuType::TMEM_SHIFT:
    return {"TMEM_SHIFT", ""};
  case TcuType::MBAR_INIT:
    return {"MBAR_INIT", ""};
  case TcuType::MBAR_ARRIVE:
    return {"MBAR_ARRIVE", ""};
  case TcuType::MBAR_WAIT:
    return {"MBAR_WAIT", ""};
  // ----- New tcgen05-aligned ISA -----
  case TcuType::TMEM_DEALLOC:
    return {"TMEM_DEALLOC", ""};
  case TcuType::TMEM_CP: {
    auto shape = static_cast<uint32_t>(args.cp_shape);
    auto decomp = static_cast<uint32_t>(args.cp_decompress);
    return {"TMEM_CP.shape" + std::to_string(shape) + ".decomp" + std::to_string(decomp), ""};
  }
  case TcuType::CPABULK_TENSOR_LD:
    return {"CPABULK_TENSOR_LD." + std::to_string(args.dim_count) + "d", ""};
  case TcuType::CPABULK_TENSOR_ST:
    return {"CPABULK_TENSOR_ST." + std::to_string(args.dim_count) + "d", ""};
  case TcuType::MBAR_FENCE:
    return {std::string("MBAR_FENCE.") + (args.fence_mode == TcuFenceMode::After ? "AFTER" : "BEFORE"), ""};
  case TcuType::MBAR_COMMIT:
    return {"MBAR_COMMIT", ""};
  case TcuType::MBAR_EXPECT_TX:
    return {"MBAR_EXPECT_TX", ""};
  case TcuType::MBAR_COMPLETE_TX:
    return {"MBAR_COMPLETE_TX", ""};
  case TcuType::MBAR_TEST_TRY_WAIT:
    return {std::string("MBAR_") + (args.test_or_try == TcuTestTryWait::Try ? "TRY_WAIT" : "TEST_WAIT"), ""};
  case TcuType::TCU_MMA: {
    std::string suffix;
    if (args.ws) suffix += ".ws";
    if (args.sp) suffix += ".sp";
    if (args.enable_input_d) suffix += ".accum";
    return {"TCU_MMA" + suffix, ""};
  }
  case TcuType::TCU_LD: {
    auto shape = static_cast<uint32_t>(args.ld_shape);
    return {"TCU_LD.shape" + std::to_string(shape) + ".num" + std::to_string(args.ld_num), ""};
  }
  case TcuType::TCU_ST: {
    auto shape = static_cast<uint32_t>(args.ld_shape);
    return {"TCU_ST.shape" + std::to_string(shape) + ".num" + std::to_string(args.ld_num), ""};
  }
  case TcuType::TCU_WAIT_LD:
    return {"TCU_WAIT_LD", ""};
  case TcuType::TCU_WAIT_ST:
    return {"TCU_WAIT_ST", ""};
  default:
    std::abort();
  }
}

void log_window_plan_summary(const char* prefix,
                             uint32_t handle,
                             uint32_t window_id,
                             uint32_t tile_id,
                             const TmemWindowPlan* window) {
  std::cerr << prefix
            << " handle=" << handle
            << " window=" << window_id
            << " tile=" << tile_id;
  if (nullptr == window) {
    std::cerr << " plan=null";
  } else {
    std::cerr << " target=" << static_cast<uint32_t>(window->target)
              << " layout=" << static_cast<uint32_t>(window->layout_kind)
              << " fmt=" << window->fmt
              << " packets_per_tile=" << window->packets_per_tile
              << " tile_count=" << window->tile_count
              << " shape=(" << window->elem_shape.rows << "x" << window->elem_shape.cols << ")";
  }
  std::cerr << std::endl;
}

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
    const tensor_unit_detail::DMemState& pub_dmem_state) {
  os << "[TensorUnit] mem_ops=" << mem_ops.size()
     << " pending_mem_ops=" << pending_mem_ops.size()
     << " pending_wmma_jobs=" << pending_wmma_jobs.size()
     << " active_wmma_job_valid=" << active_wmma_job_valid
     << " pending_wmma_uops=" << pending_wmma_uops.size()
     << " completed_tensor_mem_responses=" << completed_tensor_mem_responses.size()
     << "\n";
  os << "  AMem{ready=" << amem_state.a_ready
     << ",pending=" << amem_state.a_pending
     << ",valid=" << amem_state.a_valid
     << ",wmma_pending=" << amem_state.a_wmma_pending << "}"
     << " BMem{ready=" << bmem_state.b_ready
     << ",pending=" << bmem_state.b_pending
     << ",valid=" << bmem_state.b_valid
     << ",wmma_pending=" << bmem_state.b_wmma_pending
     << ",ws_locked=" << bmem_state.b_ws_locked << "}"
     << " CMem{ready=" << cmem_state.c_ready
     << ",pending=" << cmem_state.c_pending
     << ",valid=" << cmem_state.c_valid
     << ",c_wmma_inflight=" << cmem_state.c_wmma_inflight << "}"
     << " DMem{ready=" << dmem_state.d_ready
     << ",valid=" << dmem_state.d_valid
     << ",store_pending=" << dmem_state.d_store_pending
     << ",d_wmma_inflight=" << dmem_state.d_wmma_inflight << "}"
     << "\n";
  os << "  A_pub{ready=" << pub_amem_state.a_ready << ",valid=" << pub_amem_state.a_valid << "}"
     << " B_pub{ready=" << pub_bmem_state.b_ready << ",valid=" << pub_bmem_state.b_valid << ",ws=" << pub_bmem_state.b_ws_locked << "}"
     << " C_pub{ready=" << pub_cmem_state.c_ready << ",valid=" << pub_cmem_state.c_valid << "}"
     << " D_pub{ready=" << pub_dmem_state.d_ready << ",valid=" << pub_dmem_state.d_valid << "}"
     << "\n";
  for (const auto& uop : mem_ops) {
    os << "  mem_uop kind=" << static_cast<uint32_t>(uop.kind)
       << " line_idx=" << uop.line_idx
       << " handle=" << uop.handle
       << " window=" << uop.window_id
       << " tile=" << uop.tile_idx
       << " async=" << uop.async_id
       << " rem_read_pkts=" << uop.remaining_tmem_read_packets
       << " rem_write_pkts=" << uop.remaining_tmem_write_packets
       << " rem_a_lines=" << uop.remaining_amem_fill_lines
       << " rem_b_lines=" << uop.remaining_bmem_fill_lines
       << " rem_c_fill=" << uop.remaining_cmem_fill_subtiles
       << " rem_c_dump=" << uop.remaining_cmem_dump_subtiles
       << " rem_meta_pkts=" << uop.remaining_metamem_fill_packets
       << " fill_ingress_valid=" << uop.fill_ingress_valid
       << " fill_input_valid=" << uop.fill_input_valid
       << " fill_convert_valid=" << uop.fill_convert_valid
       << " store_from_dmem=" << uop.store_from_dmem
       << " store_raw_valid=" << uop.store_raw_subtile_valid
       << " staged_store=" << uop.staged_store_count
       << "\n";
  }
  if (active_wmma_job_valid) {
    os << "  active_wmma async=" << active_wmma_job.async_id
       << " next_uop=" << active_wmma_job.next_uop
       << " or=" << active_wmma_job.output_resident
       << " ws=" << active_wmma_job.ws
       << " first_in_seq=" << active_wmma_job.is_first_in_accum_seq
       << "\n";
  }
}

} // namespace vortex
