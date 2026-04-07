// tensor_async_frontend.cpp
//
// 张量异步操作分发前端的实现。

#include "open_tensorcore/tensor_control/tensor_async_frontend.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "core.h"
#include "tensor_cfg.h"
#include "open_tensorcore/tensor_control/tensor_slot_manager.h"
#include "open_tensorcore/tensor_helper/tensor_debug_utils.h"

namespace vortex {

namespace vt = vortex::tensor;

namespace {

// --------------------------------------------------------------------------
// use_open_tensorcore
// --------------------------------------------------------------------------
// 格式兼容性检查：判断当前 A/B/C 的数据格式是否可由 open_tensorcore 路径处理。
// 要求：
//   - NUM_THREADS 必须为 32（一个 warp 32 线程）
//   - C（累加器）格式仅支持 fp16 或 fp32
//   - A 和 B 输入格式仅支持 fp8 或 fp16
// 不满足以上条件时返回 false，调用方会触发 retry。
bool use_open_tensorcore(uint32_t fmt_a, uint32_t fmt_b, uint32_t fmt_c) {
  if constexpr (NUM_THREADS != 32) {
    return false;
  }
  if (fmt_c != vt::fp16::id && fmt_c != vt::fp32::id) {
    return false;
  }
  bool supported_a = (fmt_a == vt::fp8::id || fmt_a == vt::fp16::id);
  bool supported_b = (fmt_b == vt::fp8::id || fmt_b == vt::fp16::id);
  return supported_a && supported_b;
}

} // namespace

// ============================================================================
// enqueue_async_mma_load 实现
// ============================================================================
// 整体流程：
//   1. 空指针防护
//   2. 检查 TMEM handle 是否就绪（数据是否已从全局内存搬到 TMEM）
//   3. 校验 slot_id 范围
//   4. 窗口查找 / 延迟绑定（window lookup / lazy bind）
//   5. 目标槽位状态检查：首次使用则初始化，已绑定则检查是否可接受新填充
//   6. 创建 MemUop 并推入 mem_ops 队列
//
// 重试机制（retry mechanism）:
//   任何步骤失败时设 trace_data->retry = true 并返回。核心的执行阶段会在
//   下一个周期重新发射这条 mma_load 指令，直到条件满足为止。
//   这避免了在 CModel 中引入复杂的状态机，代价是不精确地假设重试只需 1 周期。
void TensorAsyncFrontend::enqueue_async_mma_load(
    Core* core,
    const Arch& arch,
    uint32_t wid,
    uint32_t handle,
    const IntrTcuArgs& args,
    std::array<tud::ASlotState, tud::kNumOperandSlots>* a_slots,
    std::array<tud::BSlotState, tud::kNumOperandSlots>* b_slots,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* c_slots,
    std::array<tud::ASlotState, tud::kNumOperandSlots>* published_a_slots,
    std::array<tud::BSlotState, tud::kNumOperandSlots>* published_b_slots,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* published_c_slots,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    DMem* dmem,
    MetaMem* metamem,
    std::deque<tud::MemUop>* mem_ops,
    std::unordered_map<uint32_t, uint32_t>* pending_mem_ops,
    TensorUnit::PerfStats* perf_stats,
    TensorUnit::ExeTraceData* trace_data) {
  // --- 空指针防护 ---
  if (nullptr == core || nullptr == a_slots || nullptr == b_slots || nullptr == c_slots
   || nullptr == published_a_slots || nullptr == published_b_slots || nullptr == published_c_slots
   || nullptr == amem || nullptr == bmem || nullptr == cmem || nullptr == dmem || nullptr == metamem
   || nullptr == mem_ops || nullptr == pending_mem_ops || nullptr == perf_stats) {
    return;
  }
  // published_*_slots 在本函数中不使用，仅在 IssuePolicy 的调度查询中使用
  (void)published_a_slots;
  (void)published_b_slots;
  (void)published_c_slots;

  auto wgid = arch.warpgroup_id(wid);

  // --- 步骤 2: 检查 TMEM handle 是否已就绪 ---
  // handle 对应 TMA（Tensor Memory Accelerator）的异步搬运状态，
  // 只有当全局内存数据已经到达 TMEM 缓冲区时才可继续。
  if (!core->tmem_handle_ready_for_mma_load(handle, args.target, args.a_sparse_mode)) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // --- 步骤 3: 校验 slot_id 是否在合法范围内 ---
  if ((args.target == TcuTarget::A && args.slot_id >= a_slots->size())
   || (args.target == TcuTarget::B && args.slot_id >= b_slots->size())
   || (args.target == TcuTarget::C && args.slot_id >= c_slots->size())
   || args.target == TcuTarget::None) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  auto slot_id = static_cast<uint32_t>(args.slot_id);

  // --- 步骤 4: 窗口查找与延迟绑定 ---
  // TmemWindowPlan 描述了 TMEM 中数据的布局方式（每个 tile 需要多少 packets 等）。
  // 先尝试直接查找已有窗口；若未找到，则调用 ensure_tmem_window_bound() 进行
  // 延迟绑定（lazy binding），然后再次查找。
  // 如果窗口仍不可用或目标类型不匹配，则触发 retry。
  const TmemWindowPlan* source_window = nullptr;
  bool use_window = false;
  uint32_t source_payload_fmt = 0;
  bool lookup_ok = core->lookup_tmem_window(handle, args.window_id, &source_window);
  use_window = lookup_ok && source_window->packets_per_tile != 0;
  if (!use_window) {
    // 延迟绑定：首次使用该 window_id 时，核心会根据 descriptor 和 target 创建窗口计划
    if (!core->ensure_tmem_window_bound(handle, args.descriptor, args.target, args.window_id)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    lookup_ok = core->lookup_tmem_window(handle, args.window_id, &source_window);
    use_window = lookup_ok && source_window->packets_per_tile != 0;
  }
  // 验证窗口的 target 类型与 load 请求的 target 一致（A 对 A，B 对 B，C 对 C/D）
  if (!use_window || !tud::window_matches_load_target(args.target, source_window->target)) {
    log_window_plan_summary("TensorUnit retry: mma_load window not ready/mismatched",
                            handle,
                            args.window_id,
                            args.tile_id,
                            source_window);
    std::cerr << "TensorUnit retry detail:"
              << " descriptor=" << args.descriptor
              << " target=" << static_cast<uint32_t>(args.target)
              << " slot=" << args.slot_id
              << " fmt_a=" << args.fmt_a
              << " fmt_b=" << args.fmt_b
              << " fmt_c=" << args.fmt_c
              << " fmt_d=" << args.fmt_d
              << " sparse=" << args.a_sparse_mode
              << " ws=" << static_cast<uint32_t>(args.ws)
              << std::endl;
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  source_payload_fmt = source_window->fmt;

  // --- 步骤 5: 目标槽位状态检查与绑定 ---
  // 双缓冲槽位（kNumOperandSlots = 2）的绑定逻辑：
  //   - 若槽位未绑定、或 owner/descriptor 不匹配 → 尝试重新绑定（rebind）
  //     - rebind 前提：槽位未在执行中的操作（can_rebind 检查）
  //     - rebind 后调用 init_*_slot_for_descriptor 初始化槽位并清空旧数据
  //   - 若槽位已绑定且 owner/descriptor 匹配 → 检查是否可接受新的填充
  //     - 如果 slot 正在被填充(pending)或正在被 WMMA 使用，则需要 retry
  switch (args.target) {
  case TcuTarget::A: {
    auto& slot = a_slots->at(slot_id);
    if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
      if (!TensorSlotManager::a_slot_can_rebind(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
      TensorSlotManager::init_a_slot_for_descriptor(slot_id, &slot, wgid, args, amem, metamem);
    } else if (!TensorSlotManager::is_a_slot_ready_for_new_fill(slot)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
  } break;
  case TcuTarget::B: {
    auto& slot = b_slots->at(slot_id);
    if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
      if (!TensorSlotManager::b_slot_can_rebind(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
      TensorSlotManager::init_b_slot_for_descriptor(slot_id, &slot, wgid, args, bmem);
    } else if (!TensorSlotManager::is_b_slot_ready_for_new_fill(slot)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
  } break;
  case TcuTarget::C: {
    auto& slot = c_slots->at(slot_id);
    if (!slot.valid || slot.owner_wgid != wgid || slot.descriptor != args.descriptor) {
      if (!TensorSlotManager::c_slot_can_rebind(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
      TensorSlotManager::init_c_slot_for_descriptor(slot_id, &slot, wgid, args, cmem, dmem);
    } else if (!TensorSlotManager::is_c_slot_ready_for_new_fill(slot)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
  } break;
  default:
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // --- 步骤 6: 创建 MemUop 并入队 ---
  // 通知核心此 mma_load 已成功发射，获取 async_id 用于后续完成追踪
  auto async_id = core->mma_load_async_issue(wid, handle, args.descriptor);
  uint32_t num_uops = 0;

  // push_fill: 创建一个 Fill 类型的 MemUop 的辅助 lambda
  // 根据 target 类型设置不同的剩余 packet 数和 fill line 数：
  //   - FillA: 计算数据 packets + 稀疏元数据 packets，填充 AMem 行 + MetaMem 行
  //   - FillB: 计算数据 packets，填充 BMem 行
  //   - FillC: 计算数据 packets，填充 CMem 子块
  // 创建完成后将槽位标记为 pending（加载中）
  auto push_fill = [&](tud::MemUop::Kind kind,
                       TcuTarget target,
                       bool separate_handle,
                       uint32_t payload_fmt) {
    tud::MemUop op{};
    op.kind = kind;
    op.wgid = wgid;
    op.slot_id = slot_id;
    op.handle = handle;
    op.window_id = args.window_id;
    op.payload_fmt = payload_fmt;
    op.tile_idx = args.tile_id;
    op.async_id = async_id;
    op.separate_handle = separate_handle;
    switch (target) {
    case TcuTarget::A: {
      auto& slot = a_slots->at(slot_id);
      // 计算需要从 TMEM 读取的总 packet 数（数据 + 稀疏元数据）
      op.remaining_tmem_read_packets = tud::a_packet_count((payload_fmt != tud::kUnsetPayloadFmt) ? payload_fmt : slot.fmt_a)
                              + ((slot.a_sparse_mode != vt::sparse_none) ? tud::meta_packet_count(slot.a_sparse_mode) : 0);
      op.remaining_amem_fill_lines = AMem::fill_lines();
      op.remaining_metamem_fill_packets = (slot.a_sparse_mode != vt::sparse_none) ? MetaMem::fill_packets() : 0;
      TensorSlotManager::mark_a_pending(&slot, true);
    } break;
    case TcuTarget::B: {
      auto& slot = b_slots->at(slot_id);
      op.remaining_tmem_read_packets = tud::b_packet_count((payload_fmt != tud::kUnsetPayloadFmt) ? payload_fmt : slot.fmt_b);
      op.remaining_bmem_fill_lines = BMem::fill_lines();
      TensorSlotManager::mark_b_pending(&slot, true);
    } break;
    case TcuTarget::C: {
      auto& slot = c_slots->at(slot_id);
      op.remaining_tmem_read_packets = tud::c_load_packet_count((payload_fmt != tud::kUnsetPayloadFmt) ? payload_fmt : slot.fmt_c);
      op.remaining_cmem_fill_subtiles = CMem::fill_subtiles((payload_fmt != tud::kUnsetPayloadFmt) ? payload_fmt : slot.fmt_c);
      TensorSlotManager::mark_c_pending(&slot, true);
    } break;
    default:
      std::abort();
    }
    mem_ops->push_back(op);
    // 更新性能统计：记录 mem_ops 队列的历史最大深度
    perf_stats->mem_queue_max = std::max<uint64_t>(perf_stats->mem_queue_max, mem_ops->size());
    ++num_uops;
  };

  // 根据 target 类型创建对应的 Fill MemUop
  switch (args.target) {
  case TcuTarget::A:
    push_fill(tud::MemUop::Kind::FillA, TcuTarget::A, true, source_payload_fmt);
    break;
  case TcuTarget::B:
    push_fill(tud::MemUop::Kind::FillB, TcuTarget::B, true, source_payload_fmt);
    break;
  case TcuTarget::C:
    push_fill(tud::MemUop::Kind::FillC, TcuTarget::C, true, source_payload_fmt);
    break;
  default:
    std::abort();
  }

  // 记录该 async_id 对应的未完成 MemUop 数量，用于后续完成追踪
  (*pending_mem_ops)[async_id] = num_uops;
}

// ============================================================================
// enqueue_async_mma_store 实现
// ============================================================================
// 整体流程：
//   1. 检查 TMEM handle 是否就绪（store 通道空闲）
//   2. 查找/绑定 store 窗口（目标类型必须为 D）
//   3. C 槽位验证：
//      - 槽位必须 valid，且 owner/descriptor 匹配
//      - 不能有正在进行的 store（store_pending）或 WMMA（c_wmma_inflight）
//      - CMem 数据必须有效（cmem_final_valid）
//      - 若输出不驻留在 CMem（output_resident=false），则 DMem 必须有效
//   4. 创建 StoreC 类型的 MemUop 推入 mem_ops 队列
void TensorAsyncFrontend::enqueue_async_mma_store(
    Core* core,
    const Arch& arch,
    uint32_t wid,
    uint32_t handle,
    const IntrTcuArgs& args,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* c_slots,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* published_c_slots,
    CMem* cmem,
    DMem* dmem,
    std::deque<tud::MemUop>* mem_ops,
    TensorUnit::PerfStats* perf_stats,
    TensorUnit::ExeTraceData* trace_data) {
  (void)cmem;
  (void)dmem;
  (void)published_c_slots;
  if (nullptr == core || nullptr == c_slots || nullptr == mem_ops || nullptr == perf_stats) {
    return;
  }
  auto wgid = arch.warpgroup_id(wid);

  // --- 步骤 1: 检查 TMEM handle 的 store 通道是否就绪 ---
  if (!core->tmem_handle_ready_for_mma_store(handle)) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // 校验 slot_id 范围
  if (args.slot_id >= c_slots->size()) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  auto slot_id = static_cast<uint32_t>(args.slot_id);
  const auto& live_slot = c_slots->at(slot_id);

  // --- 步骤 2: 查找 / 延迟绑定 store 窗口 ---
  // store 窗口的 target 必须为 TmemWindowTarget::D，
  // 且 packets_per_tile 必须与输出格式 fmt_d 所需的 packet 数匹配。
  const TmemWindowPlan* store_window = nullptr;
  bool use_window = core->lookup_tmem_window(handle, args.window_id, &store_window)
                 && store_window->target == TmemWindowTarget::D
                 && store_window->packets_per_tile == tud::d_store_packet_count(live_slot.fmt_d);
  if (!use_window
   && args.window_id != 0
   && !core->ensure_tmem_window_bound(handle, args.descriptor, TcuTarget::C, args.window_id, true)) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  // 延迟绑定后再次查找
  if (!use_window && args.window_id != 0) {
    use_window = core->lookup_tmem_window(handle, args.window_id, &store_window)
              && store_window->target == TmemWindowTarget::D
              && store_window->packets_per_tile == tud::d_store_packet_count(live_slot.fmt_d);
  }
  if (args.window_id != 0 && !use_window) {
    log_window_plan_summary("TensorUnit retry: mma_store window not ready/mismatched",
                            handle,
                            args.window_id,
                            args.tile_id,
                            store_window);
    std::cerr << "TensorUnit retry detail:"
              << " descriptor=" << args.descriptor
              << " slot=" << args.slot_id
              << " fmt_d=" << live_slot.fmt_d
              << " ws=" << static_cast<uint32_t>(args.ws)
              << std::endl;
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // --- 步骤 3: C 槽位状态验证 ---
  // 必须满足以下所有条件才能发起 store：
  //   (a) 槽位有效且归当前 warp group 所有，descriptor 匹配
  //   (b) 没有正在进行的 store 操作（store_pending == false）
  //   (c) 没有正在执行的 WMMA 原语（c_wmma_inflight == 0）
  //   (d) CMem 中的最终结果有效（cmem_final_valid == true）
  //   (e) 若输出不驻留在 CMem，则 DMem 中的结果必须有效
  if (!live_slot.valid || live_slot.owner_wgid != wgid || live_slot.descriptor != args.descriptor
   || live_slot.store_pending || live_slot.c_wmma_inflight != 0) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  if (!live_slot.cmem_final_valid) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  // output_resident 为 false 时，WMMA 结果写在 DMem 中而非 CMem，
  // 因此需要从 DMem 读取数据进行 store
  if (!live_slot.output_resident && !live_slot.dmem_valid) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // --- 步骤 4: 标记槽位为 store_pending 并创建 StoreC MemUop ---
  auto& store_slot = c_slots->at(slot_id);
  store_slot.store_pending = true;
  store_slot.busy = true;
  store_slot.store_async_id = core->mma_store_async_issue(wid, handle, args.descriptor);
  tud::MemUop op{};
  op.kind = tud::MemUop::Kind::StoreC;
  op.wgid = wgid;
  op.slot_id = slot_id;
  op.handle = handle;
  op.window_id = args.window_id;
  op.tile_idx = args.tile_id;
  op.async_id = store_slot.store_async_id;
  op.separate_handle = use_window;
  // store_from_dmem: 当 output 不驻留在 CMem 时，需从 DMem 读取结果
  op.store_from_dmem = !store_slot.output_resident;
  op.remaining_cmem_dump_subtiles = CMem::dump_subtiles(store_slot.fmt_c);
  op.remaining_tmem_write_packets = tud::d_store_packet_count(store_slot.fmt_d);
  mem_ops->push_back(op);
  perf_stats->mem_queue_max = std::max<uint64_t>(perf_stats->mem_queue_max, mem_ops->size());
}

// ============================================================================
// enqueue_async_wmma 实现
// ============================================================================
// 整体流程：
//   1. 格式兼容性检查（use_open_tensorcore）
//   2. 校验三个 slot_id 的范围
//   3. WMMA 就绪检查（readiness check）：
//      - 三个槽位均 valid、owner/descriptor 匹配
//      - A/B/C 数据均已 ready（已完成加载）
//      - A/B 槽位未被其他 WMMA 占用（wmma_pending == false）
//      - C 槽位未在执行 store（store_pending == false）
//   4. 标记槽位状态、创建 PendingWmmaJob 入队
//
// 一个宏 WMMA 指令会被 TensorWmmaIssueEngine 分解为 kWmmaPrimitiveCount(=4)
// 个原语操作下发到 TensorCore 计算管线。
void TensorAsyncFrontend::enqueue_async_wmma(
    Core* core,
    const Arch& arch,
    uint32_t wid,
    const IntrTcuArgs& args,
    uint32_t fmt_a,
    uint32_t fmt_b,
    uint32_t fmt_c,
    std::array<tud::ASlotState, tud::kNumOperandSlots>* a_slots,
    std::array<tud::BSlotState, tud::kNumOperandSlots>* b_slots,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* c_slots,
    std::array<tud::ASlotState, tud::kNumOperandSlots>* published_a_slots,
    std::array<tud::BSlotState, tud::kNumOperandSlots>* published_b_slots,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* published_c_slots,
    std::unordered_map<uint32_t, uint32_t>* pending_wmma_uops,
    std::deque<tud::PendingWmmaJob>* pending_wmma_jobs,
    bool active_wmma_job_valid,
    TensorUnit::PerfStats* perf_stats,
    TensorUnit::ExeTraceData* trace_data) {
  (void)published_a_slots;
  (void)published_b_slots;
  (void)published_c_slots;
  if (nullptr == core || nullptr == a_slots || nullptr == b_slots || nullptr == c_slots
   || nullptr == pending_wmma_uops || nullptr == pending_wmma_jobs || nullptr == perf_stats) {
    return;
  }
  auto wgid = arch.warpgroup_id(wid);

  // --- 步骤 1: 格式兼容性检查 ---
  // 当前 open_tensorcore 路径仅支持 A/B 为 fp8/fp16、C 为 fp16/fp32 的组合
  if (!use_open_tensorcore(fmt_a, fmt_b, fmt_c)) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // --- 步骤 2: 校验 slot_id 范围 ---
  if (args.a_slot_id >= a_slots->size()
   || args.b_slot_id >= b_slots->size()
   || args.c_slot_id >= c_slots->size()) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }
  auto a_slot_id = static_cast<uint32_t>(args.a_slot_id);
  auto b_slot_id = static_cast<uint32_t>(args.b_slot_id);
  auto c_slot_id = static_cast<uint32_t>(args.c_slot_id);
  auto& a_slot = a_slots->at(a_slot_id);
  auto& b_slot = b_slots->at(b_slot_id);
  auto& c_slot = c_slots->at(c_slot_id);

  // --- 步骤 3: WMMA 就绪检查 ---
  // 所有三个槽位必须同时满足以下条件：
  //   - valid == true（槽位已初始化）
  //   - owner_wgid 和 descriptor 匹配当前请求
  //   - 数据已 ready（a_ready/b_ready/c_ready，即 mma_load 已完成）
  //   - A/B 未被其他 WMMA 占用、C 未在执行 store
  if (!a_slot.valid || a_slot.owner_wgid != wgid || a_slot.descriptor != args.descriptor
   || !b_slot.valid || b_slot.owner_wgid != wgid || b_slot.descriptor != args.descriptor
   || !c_slot.valid || c_slot.owner_wgid != wgid || c_slot.descriptor != args.descriptor
   || !(a_slot.a_ready && b_slot.b_ready && c_slot.c_ready)
   || a_slot.wmma_pending || b_slot.wmma_pending || c_slot.store_pending) {
    if (trace_data) {
      trace_data->retry = true;
    }
    return;
  }

  // --- 步骤 4: 标记槽位状态并创建 PendingWmmaJob ---
  // 将 A/B 标记为 wmma_pending（不可被新的 mma_load 或 wmma 使用）
  a_slot.wmma_pending = true;
  a_slot.busy = TensorSlotManager::is_a_slot_busy(a_slot);
  b_slot.wmma_pending = true;
  b_slot.busy = TensorSlotManager::is_b_slot_busy(b_slot);

  // C 槽位：标记为 busy、清除 cmem_final_valid（即将被新结果覆写）、
  // 设 c_dirty（表示 CMem 内容已被修改但尚未写回 TMEM）、
  // 递增 c_wmma_inflight 计数（一个 C 槽位可以有多个 WMMA 连续累加）
  c_slot.busy = true;
  c_slot.cmem_final_valid = false;
  c_slot.c_dirty = true;
  ++c_slot.c_wmma_inflight;

  // 获取异步操作 ID，用于后续完成追踪
  auto async_id = core->wmma_async_issue(wid);
  a_slot.wmma_async_id = async_id;
  b_slot.wmma_async_id = async_id;

  // 记录该 WMMA 宏操作对应的未完成原语数 = kWmmaPrimitiveCount(4)
  (*pending_wmma_uops)[async_id] = tud::kWmmaPrimitiveCount;

  // 创建 PendingWmmaJob 并推入队列，等待 TensorWmmaIssueEngine 逐周期下发原语
  // next_uop 初始化为 0，将从第 0 个原语开始发射
  pending_wmma_jobs->push_back(tud::PendingWmmaJob{
    wgid,
    a_slot_id,
    b_slot_id,
    c_slot_id,
    fmt_a,
    fmt_b,
    fmt_c,
    a_slot.a_sparse_mode,
    async_id,
    c_slot.output_resident,
    0,  // next_uop = 0
  });

  // 更新性能统计
  ++perf_stats->issued_macro_wmma;
  perf_stats->pending_wmma_jobs_max = std::max<uint64_t>(perf_stats->pending_wmma_jobs_max,
                                                         pending_wmma_jobs->size() + (active_wmma_job_valid ? 1u : 0u));
}

} // namespace vortex
