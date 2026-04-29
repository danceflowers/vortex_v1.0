// ============================================================================
// TensorLocalMemPipeline 实现 -- TMEM ↔ SRAM 传输引擎
// ============================================================================
//
// 本文件实现 fill(加载) 和 store(回存) 两条流水线的所有阶段函数,
// 以及主调度函数 advance_tensor_memory_pipeline()。
//
// 关键数据结构:
//   - tud::MemUop: 描述一个内存操作的全部状态 (kind, slot, 各 remaining 计数器,
//     fill/store 各阶段的缓冲和标志位)
//   - TensorMemPortReq/Rsp: TMEM 端口请求/响应, 携带 64B 数据包
//   - TmemWindowPlan: 描述 TMEM 窗口映射 (tile_count, packets_per_tile)
// ============================================================================

#include "open_tensorcore/local_memory/tensor_local_mem_pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "core.h"
#include "open_tensorcore/local_memory/tensor_local_mem_arbiter.h"
#include "open_tensorcore/tensor_control/tensor_slot_manager.h"
#include "open_tensorcore/tensor_helper/tensor_debug_utils.h"

namespace vortex {

namespace {

// 将 Core::TmemPacket 转换为 MetaMem 的 64B 数据包格式
static MetaMem::packet_t to_meta_packet(const Core::TmemPacket& packet) {
  MetaMem::packet_t out{};
  std::copy_n(packet.bytes.begin(), out.size(), out.begin());
  return out;
}

// 将任意 SRAM 数据包转换为 Core::TmemPacket (用于 store 写回)
template <typename PacketT>
static Core::TmemPacket to_tmem_packet(const PacketT& packet) {
  Core::TmemPacket out;
  std::copy_n(packet.begin(), packet.size(), out.bytes.begin());
  return out;
}

// 计算 MemUop 请求的仲裁优先级,给TMEM使用，而不是给amem、bmem的
// 生成请求的优先级 age: 高 32 位为 async_id, 低 32 位为包序号
// 数值越小表示越早发出, 仲裁时优先服务
static uint64_t mem_uop_request_age(const tud::MemUop& uop, uint32_t ordinal) {
  return (static_cast<uint64_t>(uop.async_id) << 32) | ordinal;
}

// 查找数据窗口映射: 将 (handle, window_id, tile_idx, local_packet_idx) 映射为全局 packet_idx
static bool lookup_window_packet_idx(const tud::MemUop& uop,
                                     uint32_t local_packet_idx,
                                     Core* core,
                                     uint32_t* packet_idx) {
  if (!uop.separate_handle || nullptr == core || nullptr == packet_idx) {
    return false;
  }
  const TmemWindowPlan* window = nullptr;
  if (!core->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
    return false;
  }
  if (window->packets_per_tile == 0 || local_packet_idx >= window->packets_per_tile || uop.tile_idx >= window->tile_count) {
    return false;
  }
  *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
  return true;
}

// 查找元数据窗口映射: 稀疏元数据使用独立的窗口 ID
static bool lookup_meta_window_packet_idx(const tud::MemUop& uop,
                                          uint32_t local_packet_idx,
                                          Core* core,
                                          uint32_t* packet_idx) {
  if (!uop.separate_handle || nullptr == core || nullptr == packet_idx) {
    return false;
  }
  const TmemWindowPlan* window = nullptr;
  auto shadow_window_id = tud::meta_shadow_window_id(uop.window_id);
  if (!core->lookup_tmem_window(uop.handle, shadow_window_id, &window)) {
    return false;
  }
  if (window->packets_per_tile == 0 || local_packet_idx >= window->packets_per_tile || uop.tile_idx >= window->tile_count) {
    return false;
  }
  *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
  return true;
}

// 根据 MemUop 当前缓冲状态重建 fill 大阶段。
// 该 helper 只负责根据寄存器/缓冲占用关系选择下一拍应执行的唯一阶段，
// 不会直接推进任何数据传输。
static void refresh_fill_stage(tud::MemUop* uop) {
  if (nullptr == uop) {
    return;
  }
  if (uop->fill_convert_valid
   || (uop->remaining_metamem_fill_packets != 0 && !uop->staged_meta_packets.empty())) {
    uop->fill_stage = tud::FillPipelineStage::LocalWrite;
    return;
  }
  if (uop->fill_input_valid) {
    uop->fill_stage = tud::FillPipelineStage::Conversion;
    return;
  }
  if (uop->fill_ingress_valid) {
    uop->fill_stage = tud::FillPipelineStage::Ingress;
    return;
  }
  if (uop->pending_tmem_request_tag != 0
   || uop->remaining_tmem_read_packets != 0
   || uop->fill_input_packet_count != 0) {
    uop->fill_stage = tud::FillPipelineStage::WaitTmemRsp;
    return;
  }
  uop->fill_stage = tud::FillPipelineStage::Done;
}

// 根据 MemUop 当前缓冲状态重建 store 大阶段。
// 注意优先级必须保证 conversion 寄存器先进入 staged buffer，
// 然后后续周期才能进入 TMEM emit 阶段。
static void refresh_store_stage(tud::MemUop* uop) {
  if (nullptr == uop) {
    return;
  }
  if (uop->store_conv_valid) {
    if (uop->staged_store_count < tud::kOutputPacketBufferDepth) {
      uop->store_stage = tud::StorePipelineStage::StageBuffer;
    } else if (uop->pending_tmem_request_tag != 0 || uop->staged_store_count != 0) {
      uop->store_stage = tud::StorePipelineStage::Emit;
    } else {
      uop->store_stage = tud::StorePipelineStage::StageBuffer;
      // 转换结果已准备好但缓冲未满且无未完成请求时直接进入 StageBuffer, 否则先进入 Emit 等待请求完成
    }
    return;
  }
  if (uop->store_raw_subtile_valid) {
    if (uop->staged_store_count >= tud::kOutputPacketBufferDepth) {
      uop->store_stage = tud::StorePipelineStage::Emit;
    } else {
      uop->store_stage = tud::StorePipelineStage::Conversion;
      
    }
    return;
  }
  
  // 只要有未完成的 TMEM 写请求或未发出的缓冲数据就继续发出请求, 否则进入 Read 等待读取 subtile
  if (uop->pending_tmem_request_tag != 0 || uop->staged_store_count != 0) {
    uop->store_stage = tud::StorePipelineStage::Emit;
    return;
  }

  // 只要还有未读的 CMem subtile 就继续读, 否则进入 Done
  if (uop->remaining_cmem_dump_subtiles != 0) {
    uop->store_stage = tud::StorePipelineStage::Read;
    return;
  }
  uop->store_stage = tud::StorePipelineStage::Done;
}

} // namespace

// 检查 MemUop 是否完全完成: 所有 remaining 计数器为零, 且所有中间缓冲均已排空
bool TensorLocalMemPipeline::mem_op_complete(const tud::MemUop& op) {
  return op.remaining_tmem_read_packets == 0       // 无待读 TMEM 包
      && op.remaining_tmem_write_packets == 0      // 无待写 TMEM 包
      && op.remaining_amem_fill_lines == 0         // 无待写 AMem 行
      && op.remaining_bmem_fill_lines == 0         // 无待写 BMem 行
      && op.remaining_cmem_fill_subtiles == 0      // 无待写 CMem subtile
      && op.remaining_cmem_dump_subtiles == 0      // 无待读 CMem subtile (store)
      && op.remaining_metamem_fill_packets == 0    // 无待写 MetaMem 包
      && !op.fill_ingress_valid                    // ingress 缓冲为空
      && !op.fill_input_valid                      // conversion 输入缓冲为空
      && op.fill_input_packet_count == 0           // 已收集的输入包数为零
      && !op.fill_convert_valid                    // conversion 输出缓冲为空
      && !op.store_raw_subtile_valid               // store 原始 subtile 缓冲为空
      && !op.store_conv_valid                      // store 转换缓冲为空
      && op.staged_store_count == 0;               // store 缓冲为空
}

// 从 TMEM 响应端口取出一个响应并按 request_id 存入完成映射表
// 每次只取一个 (非阻塞), 由上层 tick() 循环多次调用以排空
void TensorLocalMemPipeline::receive_one_tmem_response(
    SimPort<TensorMemPortRsp>* rsp_in,
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses) {
  if (nullptr == rsp_in || nullptr == completed_tensor_mem_responses) {
    return;
  }
  if (rsp_in->empty()) {
    return;
  }
  auto response = rsp_in->front();
  (*completed_tensor_mem_responses)[response.request_id] = response;
  rsp_in->pop();
}

// 按 request_id 查找已完成的 TMEM 响应, 找到则取出并从映射表中删除
bool TensorLocalMemPipeline::take_tensor_mem_response(
    uint64_t request_id,
    std::unordered_map<uint64_t, TensorMemPortRsp>* completed_tensor_mem_responses,
    TensorMemPortRsp* out) {
  if (nullptr == completed_tensor_mem_responses) {
    return false;
  }
  auto it = completed_tensor_mem_responses->find(request_id);
  if (it == completed_tensor_mem_responses->end()) {
    return false;
  }
  if (out) {
    *out = it->second;
  }
  completed_tensor_mem_responses->erase(it);
  return true;
}

// 根据 uop 当前状态选择下一步应写入哪个 SRAM
// 优先处理已完成精度转换的数据 (fill_convert_valid), 再处理暂存的 meta 包
TensorLocalMemPipeline::LocalFillAction TensorLocalMemPipeline::select_fill_local_write(const tud::MemUop& uop) {
  if (uop.fill_convert_valid) {
    switch (uop.fill_convert_kind) {
    case tud::FillConvertKind::ALine:
      return LocalFillAction::AData;
    case tud::FillConvertKind::BLine:
      return LocalFillAction::BData;
    case tud::FillConvertKind::CSubtile:
      return LocalFillAction::CData;
    case tud::FillConvertKind::None:
    default:
      break;
    }
  }

  //“普通 A/B/C 数据写入”已经在前半段处理完了
  // 这段只负责 sparse FillA 的 metadata 尾巴，FillB/FillC 在这段没有动作
  switch (uop.kind) {
  case tud::MemUop::Kind::FillA:
    if (uop.remaining_metamem_fill_packets != 0
     && !uop.staged_meta_packets.empty()) {
      return LocalFillAction::Meta;
    }
    return LocalFillAction::None;
  case tud::MemUop::Kind::FillB:
    return LocalFillAction::None;
  case tud::MemUop::Kind::FillC:
    return LocalFillAction::None;
  default:
    return LocalFillAction::None;
  }
}

// Fill 阶段 3: 收集 ingress 数据包到 fill_input_packets[]
// fp16 模式需要 2 个包凑齐一行/subtile, fp8 模式 1 个包即可
// 凑齐所需包数后设置 fill_input_valid = true, 等待下一步转换
bool TensorLocalMemPipeline::stage_fill_conversion_input(tud::MemUop* uop, uint32_t payload_fmt) {
  if (nullptr == uop) {
    return false;
  }
  if (!uop->fill_ingress_valid || uop->fill_input_valid || uop->fill_convert_valid) {
    return false;
  }

  if (uop->fill_input_packet_count == 0) {
    switch (uop->kind) {
    case tud::MemUop::Kind::FillA:
      if (uop->remaining_amem_fill_lines == 0) {
        return false;
      }
      uop->fill_input_kind = tud::FillConvertKind::ALine;
      uop->fill_input_index = AMem::fill_lines() - uop->remaining_amem_fill_lines;
      uop->fill_input_packets_needed = AMem::packets_per_fill_line(payload_fmt);
      break;
    case tud::MemUop::Kind::FillB:
      if (uop->remaining_bmem_fill_lines == 0) {
        return false;
      }
      uop->fill_input_kind = tud::FillConvertKind::BLine;
      uop->fill_input_index = BMem::fill_lines() - uop->remaining_bmem_fill_lines;
      uop->fill_input_packets_needed = BMem::packets_per_fill_line(payload_fmt);
      break;
    case tud::MemUop::Kind::FillC:
      if (uop->remaining_cmem_fill_subtiles == 0) {
        return false;
      }
      uop->fill_input_kind = tud::FillConvertKind::CSubtile;
      uop->fill_input_index = CMem::fill_subtiles(payload_fmt) - uop->remaining_cmem_fill_subtiles;
      uop->fill_input_packets_needed = CMem::packets_per_subtile(payload_fmt);
      break;
    default:
      return false;
    }
    if (uop->fill_input_packets_needed == 0 || uop->fill_input_packets_needed > uop->fill_input_packets.size()) {
      std::abort();
    }
  }

  
  if (uop->fill_input_packet_count >= uop->fill_input_packets_needed) {
    std::abort();
  }

  uop->fill_input_packets.at(uop->fill_input_packet_count++) = uop->fill_ingress_packet;
  uop->fill_ingress_valid = false;
  if (uop->fill_input_packet_count == uop->fill_input_packets_needed) {
    uop->fill_input_valid = true;
  }
  refresh_fill_stage(uop);
  return true;
}

// Fill 阶段 4: 执行精度转换
// - ALine / BLine: 调用 AMem/BMem::convert_fill_packets() 将 fp8/fp16 → fp9
// - CSubtile: 调用 CMem::convert_fill_packets() 将 fp8/fp16/fp32 → fp22
// 转换结果存入 uop->fill_convert_fp9[] 或 fill_convert_fp22[], 并清空输入缓冲
bool TensorLocalMemPipeline::advance_fill_conversion(tud::MemUop* uop, uint32_t payload_fmt) {
  if (nullptr == uop) {
    return false;
  }
  if (uop->fill_convert_valid || !uop->fill_input_valid) {
    return false;
  }

  std::vector<Core::TmemPacket> packet_slice;
  packet_slice.reserve(uop->fill_input_packet_count);
  for (uint32_t packet = 0; packet < uop->fill_input_packet_count; ++packet) {
    packet_slice.push_back(uop->fill_input_packets.at(packet));
  }

  switch (uop->fill_input_kind) {
  case tud::FillConvertKind::ALine: {
    if (!AMem::convert_fill_packets(payload_fmt, tud::copy_packets<AMem::packet_t>(packet_slice), uop->fill_convert_fp9)) {
      std::abort();
    }
    uop->fill_convert_kind = tud::FillConvertKind::ALine;
    uop->fill_convert_valid = true;//表示“转换输出缓冲有效了”，下一阶段可以消费转换结果了
    uop->fill_convert_index = uop->fill_input_index;
    uop->fill_input_valid = false;//表示输入缓冲已经被消费完了
    uop->fill_input_kind = tud::FillConvertKind::None;//清掉输入缓冲的类型标记
    uop->fill_input_packet_count = 0;//输入包计数清零，准备收下一组包
    uop->fill_input_packets_needed = 0;//这一次组包需求结束，等待下次重新计算
    refresh_fill_stage(uop);//重新计算下一拍应该进入哪个阶段。因为 fill_convert_valid = true，这里通常会把阶段切到 LocalWrite
    return true;
  }
  case tud::FillConvertKind::BLine: {
    if (!BMem::convert_fill_packets(payload_fmt, tud::copy_packets<BMem::packet_t>(packet_slice), uop->fill_convert_fp9)) {
      std::abort();
    }
    uop->fill_convert_kind = tud::FillConvertKind::BLine;
    uop->fill_convert_valid = true;
    uop->fill_convert_index = uop->fill_input_index;
    uop->fill_input_valid = false;
    uop->fill_input_kind = tud::FillConvertKind::None;
    uop->fill_input_packet_count = 0;
    uop->fill_input_packets_needed = 0;
    refresh_fill_stage(uop);
    return true;
  }
  case tud::FillConvertKind::CSubtile: {
    if (!CMem::convert_fill_packets(payload_fmt, tud::copy_packets<CMem::packet_t>(packet_slice), uop->fill_convert_fp22)) {
      std::abort();
    }
    uop->fill_convert_kind = tud::FillConvertKind::CSubtile;
    uop->fill_convert_valid = true;
    uop->fill_convert_index = uop->fill_input_index;
    uop->fill_input_valid = false;
    uop->fill_input_kind = tud::FillConvertKind::None;
    uop->fill_input_packet_count = 0;
    uop->fill_input_packets_needed = 0;
    refresh_fill_stage(uop);
    return true;
  }
  case tud::FillConvertKind::None:
  default:
    return false;
  }
}

// Fill 阶段 5: 将转换后的数据写入目标 SRAM
// 根据 action 类型分发到对应的 write_converted_line / write_converted_subtile / write_fill_packet
// 写入成功后递减对应的 remaining 计数器, 并清空 fill_convert 缓冲
bool TensorLocalMemPipeline::perform_fill_local_write(
    tud::MemUop* uop,
    LocalFillAction action,
    AMem* amem,
    BMem* bmem,
    CMem* cmem,
    MetaMem* metamem) {
  if (nullptr == uop || nullptr == amem || nullptr == bmem || nullptr == cmem || nullptr == metamem) {
    return false;
  }
  switch (action) {
  case LocalFillAction::AData: {
    auto line_idx = uop->fill_convert_index;
    if (!amem->write_converted_line(uop->slot_id, line_idx, uop->fill_convert_fp9)) {
      std::cerr << "TensorUnit error: AMem fill write failed"
                << " kind=" << static_cast<uint32_t>(uop->kind)
                << " slot=" << uop->slot_id
                << " line=" << line_idx
                << std::endl;
      std::abort();
    }
    uop->fill_convert_valid = false;
    uop->fill_convert_kind = tud::FillConvertKind::None;
    --uop->remaining_amem_fill_lines;
    refresh_fill_stage(uop);
    return true;
  }
  case LocalFillAction::BData: {
    auto line_idx = uop->fill_convert_index;
    if (!bmem->write_converted_line(uop->slot_id, line_idx, uop->fill_convert_fp9)) {
      std::cerr << "TensorUnit error: BMem fill write failed"
                << " kind=" << static_cast<uint32_t>(uop->kind)
                << " slot=" << uop->slot_id
                << " line=" << line_idx
                << std::endl;
      std::abort();
    }
    uop->fill_convert_valid = false;
    uop->fill_convert_kind = tud::FillConvertKind::None;
    --uop->remaining_bmem_fill_lines;
    refresh_fill_stage(uop);
    return true;
  }
  case LocalFillAction::CData: {
    auto subtile_idx = uop->fill_convert_index;
    if (!cmem->write_converted_subtile(uop->slot_id, subtile_idx, uop->fill_convert_fp22)) {
      std::cerr << "TensorUnit error: CMem fill write failed"
                << " kind=" << static_cast<uint32_t>(uop->kind)
                << " slot=" << uop->slot_id
                << " subtile=" << subtile_idx
                << std::endl;
      std::abort();
    }
    uop->fill_convert_valid = false;
    uop->fill_convert_kind = tud::FillConvertKind::None;
    --uop->remaining_cmem_fill_subtiles;
    refresh_fill_stage(uop);
    return true;
  }
  case LocalFillAction::Meta: {
    if (uop->staged_meta_packets.empty()) {
      std::abort();
    }
    auto packet = to_meta_packet(uop->staged_meta_packets.front());
    if (!metamem->write_fill_packet(uop->slot_id, packet)) {
      std::abort();
    }
    uop->staged_meta_packets.erase(uop->staged_meta_packets.begin());
    --uop->remaining_metamem_fill_packets;
    refresh_fill_stage(uop);
    return true;
  }
  case LocalFillAction::None:
  default:
    return false;
  }
}

// Fill 阶段 2: 接收 TMEM 读响应
// - 若当前包序号 < payload_packets: 为数据包, 存入 fill_ingress (单缓冲)
// - 否则为稀疏元数据包, 追加到 staged_meta_packets 列表
// 每接收一个包, remaining_tmem_read_packets 递减
bool TensorLocalMemPipeline::accept_fill_read_response(
    tud::MemUop* uop,
    const TensorMemPortRsp& response,
    uint32_t payload_packets) {
  if (nullptr == uop) {
    return false;
  }
  if (uop->remaining_tmem_read_packets == 0) {
    return false;
  }

  if (uop->next_payload_packet_idx < payload_packets) {
    if (uop->fill_ingress_valid) {
      std::cerr << "TensorUnit error: payload ingress buffer overflow"
                << " kind=" << static_cast<uint32_t>(uop->kind)
                << " slot=" << uop->slot_id
                << " next_payload=" << uop->next_payload_packet_idx
                << std::endl;
      std::abort();
    }
    uop->fill_ingress_packet = response.read_packet;
    uop->fill_ingress_valid = true;
    ++uop->next_payload_packet_idx;
  } else {
    uop->staged_meta_packets.push_back(response.read_packet);
    ++uop->next_meta_packet_idx;
  }

  --uop->remaining_tmem_read_packets;
  refresh_fill_stage(uop);
  return true;
}

// Fill 阶段 1: 向 TMEM 端口发出读请求
// 根据当前进度决定读数据包还是元数据包:
// - next_payload_packet_idx < payload_packets: 读数据窗口
// - 否则: 读元数据影子窗口 (shadow window)
// 通过 lookup_window_packet_idx / lookup_meta_window_packet_idx 计算全局包索引
uint64_t TensorLocalMemPipeline::issue_fill_read_request(
    const tud::MemUop& uop,
    Core* core,
    SimPort<TensorMemPortReq>* req_out,
    uint64_t* next_tensor_mem_request_id,
    const std::array<tud::ASlotState, tud::kNumOperandSlots>& a_slots,
    const std::array<tud::BSlotState, tud::kNumOperandSlots>& b_slots,
    const std::array<tud::CSlotState, tud::kNumOperandSlots>& c_slots) {
  if (nullptr == core || nullptr == req_out || nullptr == next_tensor_mem_request_id) {
    return 0;
  }

  TensorMemPortReq request{};
  request.request_id = (*next_tensor_mem_request_id)++;
  request.arbitration_age = mem_uop_request_age(uop, uop.next_payload_packet_idx + uop.next_meta_packet_idx);
  request.access_type = TensorMemPortReq::AccessType::Read;
  request.port_request.age = request.arbitration_age;
  auto payload_packets = tud::fill_payload_packet_count(uop, a_slots, b_slots, c_slots);

  if (uop.next_payload_packet_idx < payload_packets) {
    uint32_t packet_idx = 0;
    if (!lookup_window_packet_idx(uop, uop.next_payload_packet_idx, core, &packet_idx)) {
      std::cerr << "TensorUnit error: missing TMEM window mapping for tensor payload read"
                << " handle=" << uop.handle
                << " window=" << uop.window_id
                << " tile=" << uop.tile_idx
                << " local_packet=" << uop.next_payload_packet_idx
                << " kind=" << static_cast<uint32_t>(uop.kind)
                << " payload_fmt=" << tud::mem_uop_payload_fmt(uop, a_slots, b_slots, c_slots)
                << std::endl;
      const TmemWindowPlan* window = nullptr;
      if (core->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
        log_window_plan_summary("TensorUnit detail: payload window", uop.handle, uop.window_id, uop.tile_idx, window);
      }
      std::abort();
    }
    request.port_request.kind = Core::TmemRequestKind::WindowRead;
    request.port_request.handle = uop.handle;
    request.port_request.window_id = uop.window_id;
    request.port_request.packet_idx = packet_idx;
  } else {
    uint32_t meta_packet_idx = 0;
    auto shadow_window_id = tud::meta_shadow_window_id(uop.window_id);
    if (!lookup_meta_window_packet_idx(uop, uop.next_meta_packet_idx, core, &meta_packet_idx)) {
      std::cerr << "TensorUnit error: missing TMEM shadow window mapping for tensor meta read"
                << " handle=" << uop.handle
                << " window=" << shadow_window_id
                << " tile=" << uop.tile_idx
                << " local_packet=" << uop.next_meta_packet_idx
                << " kind=" << static_cast<uint32_t>(uop.kind)
                << std::endl;
      const TmemWindowPlan* window = nullptr;
      if (core->lookup_tmem_window(uop.handle, shadow_window_id, &window)) {
        log_window_plan_summary("TensorUnit detail: meta shadow window", uop.handle, shadow_window_id, uop.tile_idx, window);
      }
      std::abort();
    }
    request.port_request.kind = Core::TmemRequestKind::WindowRead;
    request.port_request.handle = uop.handle;
    request.port_request.window_id = shadow_window_id;
    request.port_request.packet_idx = meta_packet_idx;
  }

  req_out->push(request);
  return request.request_id;
}

// Store 阶段 1: 从 CMem 或 DMem 读出一个 fp22 subtile
// store_from_dmem 标志决定读取来源 (非驻留模式从 DMem 读, 否则从 CMem 读)
// 读出的 fp22 数据存入 uop->store_raw_subtile[], 等待后续精度转换
bool TensorLocalMemPipeline::stage_store_read(
    tud::MemUop* uop,
    const tud::CSlotState& slot,
    CMem* cmem,
    DMem* dmem) {
  if (nullptr == uop || nullptr == cmem || nullptr == dmem) {
    return false;
  }
  if (uop->remaining_cmem_dump_subtiles == 0 || uop->store_raw_subtile_valid) {
    return false;
  }

  auto subtile_idx = CMem::dump_subtiles(slot.fmt_c) - uop->remaining_cmem_dump_subtiles;
  if (uop->store_from_dmem) {
    if (!dmem->valid(uop->slot_id)) {
      std::cerr << "TensorUnit error: DMem dump failed"
                << " slot=" << uop->slot_id
                << " subtile=" << subtile_idx
                << " fmt_d=" << slot.fmt_d
                << std::endl;
      std::abort();
    }
    dmem->read_subtile_fp22(uop->slot_id, subtile_idx, uop->store_raw_subtile);
  } else {
    if (!cmem->valid(uop->slot_id)) {
      std::cerr << "TensorUnit error: CMem dump failed"
                << " slot=" << uop->slot_id
                << " subtile=" << subtile_idx
                << " fmt_d=" << slot.fmt_d
                << std::endl;
      std::abort();
    }
    cmem->read_subtile_fp22(uop->slot_id, subtile_idx, uop->store_raw_subtile);
  }

  uop->store_raw_subtile_valid = true;
  uop->store_raw_next_segment = 0;
  --uop->remaining_cmem_dump_subtiles;
  refresh_store_stage(uop);
  return true;
}

// Store 阶段 2: fp22 → 目标精度 (fp8/fp16/fp32) 转换
// 将当前 raw subtile 的一个 segment 转换到单包 conversion 寄存器中。
// 每个 subtile 根据目标精度可拆分为 1/2/4 个包 (segment)。
bool TensorLocalMemPipeline::advance_store_conversion(tud::MemUop* uop, const tud::CSlotState& slot) {
  if (nullptr == uop) {
    return false;
  }
  auto packets_per_subtile = uop->store_from_dmem
                           ? DMem::packets_per_subtile(slot.fmt_d)
                           : CMem::packets_per_subtile(slot.fmt_d);
  if (packets_per_subtile == 0) {
    std::abort();
  }

  if (uop->store_conv_valid || !uop->store_raw_subtile_valid) {
    return false;
  }

  if (uop->store_from_dmem) {
    DMem::packet_t packet{};
    if (!DMem::build_dump_packet(slot.fmt_d, uop->store_raw_subtile, uop->store_raw_next_segment, &packet)) {
      std::abort();
    }
    uop->store_conv_packet = to_tmem_packet(packet);
  } else {
    CMem::packet_t packet{};
    if (!CMem::build_dump_packet(slot.fmt_d, uop->store_raw_subtile, uop->store_raw_next_segment, &packet)) {
      std::abort();
    }
    uop->store_conv_packet = to_tmem_packet(packet);
  }
  uop->store_conv_valid = true;
  ++uop->store_raw_next_segment;
  if (uop->store_raw_next_segment >= packets_per_subtile) {
    uop->store_raw_subtile_valid = false;
    uop->store_raw_next_segment = 0;
  }
  refresh_store_stage(uop);
  return true;
}

// Store 阶段 3: 将 conversion 寄存器中的单包结果推进到 staged_store 环形缓冲
bool TensorLocalMemPipeline::stage_store_converted_packet(tud::MemUop* uop) {
  if (nullptr == uop) {
    return false;
  }
  if (!uop->store_conv_valid || uop->staged_store_count >= tud::kOutputPacketBufferDepth) {
    return false;
  }
  uop->staged_store_packets.at(uop->staged_store_tail) = uop->store_conv_packet;
  uop->staged_store_tail = (uop->staged_store_tail + 1) % tud::kOutputPacketBufferDepth;
  ++uop->staged_store_count;
  uop->store_conv_valid = false;
  refresh_store_stage(uop);
  return true;
}

// Store 阶段 4: 从环形缓冲弹出一个已转换的数据包
// 更新 head 指针和计数器, 同时递减 remaining_tmem_write_packets
bool TensorLocalMemPipeline::emit_store_packet(tud::MemUop* uop) {
  if (nullptr == uop) {
    return false;
  }
  if (uop->staged_store_count == 0 || uop->remaining_tmem_write_packets == 0) {
    return false;
  }
  ++uop->next_store_packet_idx;
  // 从环形缓冲头部取出一个包准备发出, 但实际发出由 issue_store_write_request() 负责
  uop->staged_store_head = (uop->staged_store_head + 1) % tud::kOutputPacketBufferDepth;
  --uop->staged_store_count;
  --uop->remaining_tmem_write_packets;
  refresh_store_stage(uop);
  return true;
}

// Store 阶段 4: 向 TMEM 端口发出写请求
// 从环形缓冲头部取出待发送的数据包, 通过窗口映射计算目标地址
uint64_t TensorLocalMemPipeline::issue_store_write_request(
    const tud::MemUop& uop,
    Core* core,
    SimPort<TensorMemPortReq>* req_out,
    uint64_t* next_tensor_mem_request_id) {
  if (nullptr == core || nullptr == req_out || nullptr == next_tensor_mem_request_id) {
    return 0;
  }
  TensorMemPortReq request{};
  request.request_id = (*next_tensor_mem_request_id)++;
  request.arbitration_age = mem_uop_request_age(uop, uop.next_store_packet_idx);
  request.access_type = TensorMemPortReq::AccessType::Write;
  request.port_request.age = request.arbitration_age;
  if (uop.staged_store_count == 0) {
    std::abort();
  }
  request.write_packet = uop.staged_store_packets.at(uop.staged_store_head);

  uint32_t packet_idx = 0;
  if (!lookup_window_packet_idx(uop, uop.next_store_packet_idx, core, &packet_idx)) {
    std::cerr << "TensorUnit error: missing TMEM window mapping for tensor store write"
              << " handle=" << uop.handle
              << " window=" << uop.window_id
              << " tile=" << uop.tile_idx
              << " local_packet=" << uop.next_store_packet_idx
              << " slot=" << uop.slot_id
              << std::endl;
    const TmemWindowPlan* window = nullptr;
    if (core->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
      log_window_plan_summary("TensorUnit detail: store window", uop.handle, uop.window_id, uop.tile_idx, window);
    }
    std::abort();
  }

  request.port_request.kind = Core::TmemRequestKind::WindowWrite;
  request.port_request.handle = uop.handle;
  request.port_request.window_id = uop.window_id;
  request.port_request.packet_idx = packet_idx;
  req_out->push(request);
  return request.request_id;
}

// ============================================================================
// advance_tensor_memory_pipeline -- 主调度入口
// ============================================================================
//
// 每个 tick() 调用一次, 遍历 mem_ops 队列中所有活跃的 MemUop 并尝试推进。
//
// 处理逻辑按 uop.kind 分三大分支, 并由显式 stage 驱动:
//
// 【分支 1】 FillA / FillB / FillC:
//   每拍只允许执行一个显式阶段:
//   a) WaitTmemRsp: 检查 TMEM 响应是否到达, 或发出新的 TMEM 读请求
//   b) Ingress: 将 ingress 寄存器中的 64B 包送入 input 组装缓冲
//   c) Conversion: 执行一次精度转换
//   d) LocalWrite: 将转换结果或 meta 包写入本地 SRAM
//
// 【分支 2】 StoreC:
//   每拍只允许执行一个显式阶段:
//   a) Read: 从 CMem/DMem 读出一个 fp22 subtile 到 raw 寄存器
//   b) Conversion: 将 raw subtile 的一个 segment 转到 conversion 寄存器
//   c) StageBuffer: 将 conversion 寄存器推进到 staged_store 环形缓冲
//   d) Emit: 发射 TMEM 写请求或等待其完成确认
//
// 【分支 3】 其他 (fallback 路径):
//   简单的预算消耗模式, 用于不需要精度转换的操作
//
// 完成处理:
//   当一个 uop 的 mem_op_complete() 返回 true 时:
//   - Fill 完成: 调用 TensorSlotManager::mark_*_ready() 标记 slot 就绪
//   - Store 完成: 清除 slot 的 store_pending 标志, 发出 completion 通知
//   - 递减 pending_mem_ops 中的计数, 全部子操作完成后通知上游
//   - 从 mem_ops 队列中移除该 uop 并返回 (每次最多完成一个 uop)
// ============================================================================
void TensorLocalMemPipeline::advance_tensor_memory_pipeline(
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
    std::array<tud::ASlotState, tud::kNumOperandSlots>* a_slots,
    std::array<tud::BSlotState, tud::kNumOperandSlots>* b_slots,
    std::array<tud::CSlotState, tud::kNumOperandSlots>* c_slots,
    TensorUnit::PerfStats* perf_stats,
    uint32_t* rr_index) {
  if (nullptr == core || nullptr == req_out || nullptr == completion_out || nullptr == mem_arbiter
   || nullptr == amem || nullptr == bmem || nullptr == cmem || nullptr == dmem || nullptr == metamem
   || nullptr == mem_ops || nullptr == pending_mem_ops || nullptr == completed_tensor_mem_responses
   || nullptr == next_tensor_mem_request_id || nullptr == a_slots || nullptr == b_slots
   || nullptr == c_slots || nullptr == perf_stats || nullptr == rr_index) {
    return;
  }

  if (mem_ops->empty()) {
    return;
  }

  // ---- Round-Robin 轮询遍历所有活跃的 MemUop ----
  // 每拍从 rr_index 指向的 uop 开始遍历, 公平分配 TMEM bus 带宽。
  // 当某个 uop 成功使用了 TMEM bus (发出请求或收到响应),
  // rr_index 前进到下一个 uop, 保证下一拍优先服务其他 uop。
  auto n = mem_ops->size();
  if (*rr_index >= n) {
    *rr_index = 0;
  }
  bool tmem_bus_used = false;  // 本拍 TMEM bus 是否已被使用

  for (size_t round = 0; round < n; ++round) {
    size_t idx = (*rr_index + round) % n;
    auto& uop = mem_ops->at(idx);
    bool progressed = false;  // 本轮迭代是否有任何阶段成功推进

    // ============================================================
    // 【分支 1】 Fill 操作 (FillA / FillB / FillC)
    // ============================================================
    if (uop.kind == tud::MemUop::Kind::FillA
     || uop.kind == tud::MemUop::Kind::FillB
     || uop.kind == tud::MemUop::Kind::FillC) {
      refresh_fill_stage(&uop);
      auto payload_fmt = tud::mem_uop_payload_fmt(uop, *a_slots, *b_slots, *c_slots);

      switch (uop.fill_stage) {
      case tud::FillPipelineStage::LocalWrite: {
        auto fill_action = select_fill_local_write(uop);
        bool local_blocked = false;
        switch (fill_action) {
        case LocalFillAction::AData:
          // AMem 仲裁已注释：双端口 SRAM + 双缓冲确定性数据流保证无冲突
          // if (mem_arbiter->try_claim_amem_write(AMem::line_bank_mask(uop.fill_convert_index))) {
            progressed = perform_fill_local_write(&uop, fill_action, amem, bmem, cmem, metamem);
          // } else {
          //   local_blocked = true;
          // }
          break;
        case LocalFillAction::BData:
          // BMem 仲裁已注释：双端口 SRAM + 双缓冲确定性数据流保证无冲突
          // if (mem_arbiter->try_claim_bmem_write(BMem::line_bank_mask(uop.fill_convert_index))) {
            progressed = perform_fill_local_write(&uop, fill_action, amem, bmem, cmem, metamem);
          // } else {
          //   local_blocked = true;
          // }
          break;
        case LocalFillAction::CData:
#ifdef ENABLE_CMEM_ARBITRATION
          if (mem_arbiter->try_claim_cmem_write(CMem::subtile_bank_mask(uop.fill_convert_index))) {
            progressed = perform_fill_local_write(&uop, fill_action, amem, bmem, cmem, metamem);
          } else {
            local_blocked = true;
          }
#else
          progressed = perform_fill_local_write(&uop, fill_action, amem, bmem, cmem, metamem);
#endif
          break;
        case LocalFillAction::Meta:
          // Meta 仲裁已注释：确定性数据流保证无冲突
          // if (mem_arbiter->try_claim_meta_write()) {
            progressed = perform_fill_local_write(&uop, fill_action, amem, bmem, cmem, metamem);
          // } else {
          //   local_blocked = true;
          // }
          break;
        case LocalFillAction::None:
          break;
        }
        if (!progressed && local_blocked) {
          switch (fill_action) {
          // case LocalFillAction::AData:
          //   ++perf_stats->stall_amem_port_busy;
          //   break;
          // case LocalFillAction::BData:
          //   ++perf_stats->stall_bmem_port_busy;
          //   break;
          case LocalFillAction::CData:
            ++perf_stats->stall_cmem_port_busy;
            break;
          // case LocalFillAction::Meta:
          //   ++perf_stats->stall_meta_port_busy;
          //   break;
          case LocalFillAction::None:
          default:
            break;
          }
          continue;
        }
        break;
      }
      case tud::FillPipelineStage::Conversion:
        progressed = advance_fill_conversion(&uop, payload_fmt);
        break;
      case tud::FillPipelineStage::Ingress:
        progressed = stage_fill_conversion_input(&uop, payload_fmt);
        break;
      case tud::FillPipelineStage::WaitTmemRsp: {
        // 背压检测: 若 ingress/input/convert 缓冲满, 不能接收新包
        bool fill_read_backpressured = uop.fill_ingress_valid
                                    || uop.fill_input_valid
                                    || uop.fill_convert_valid
                                    || (uop.fill_input_packets_needed != 0
                                     && uop.fill_input_packet_count >= uop.fill_input_packets_needed);
        if (uop.remaining_tmem_read_packets != 0 && !fill_read_backpressured) {
          if (uop.pending_tmem_request_tag != 0) {
            // 有待处理的请求: 尝试收取响应
            TensorMemPortRsp response{};
            if (take_tensor_mem_response(uop.pending_tmem_request_tag, completed_tensor_mem_responses, &response)) {
              uop.pending_tmem_request_tag = 0;
              progressed = accept_fill_read_response(
                  &uop, response, tud::fill_payload_packet_count(uop, *a_slots, *b_slots, *c_slots));
            } else {
              ++perf_stats->stall_tmem_read_port_busy;
              continue;
            }
          } else if (!tmem_bus_used) {
            // Round-Robin: 本拍 TMEM bus 未被占用, 本 uop 可以发出请求
            uop.pending_tmem_request_tag = issue_fill_read_request(
                uop, core, req_out, next_tensor_mem_request_id, *a_slots, *b_slots, *c_slots);
            tmem_bus_used = true;
            *rr_index = (idx + 1) % n;  // 下一拍从下一个 uop 开始轮询
            continue;
          } else {
            // Round-Robin: 本拍 TMEM bus 已被其他 uop 占用, 等下一拍
            continue;
          }
        }
        break;
      }
      case tud::FillPipelineStage::Done:
        break;
      }

    // ============================================================
    // 【分支 2】 Store 操作 (StoreC)
    // ============================================================
    } else if (uop.kind == tud::MemUop::Kind::StoreC) {
      refresh_store_stage(&uop);
      switch (uop.store_stage) {
      case tud::StorePipelineStage::Emit:
        if (uop.pending_tmem_request_tag != 0) {
          TensorMemPortRsp response{};
          if (take_tensor_mem_response(uop.pending_tmem_request_tag, completed_tensor_mem_responses, &response)) {
            uop.pending_tmem_request_tag = 0;
            refresh_store_stage(&uop);
            progressed = emit_store_packet(&uop);
          } else {
            ++perf_stats->stall_tmem_write_port_busy;
            continue;
          }
        } else if (uop.staged_store_count != 0 && !tmem_bus_used) {
          // Round-Robin: 本拍 TMEM bus 未被占用, 发出 store 写请求
          uop.pending_tmem_request_tag = issue_store_write_request(uop, core, req_out, next_tensor_mem_request_id);
          tmem_bus_used = true;
          *rr_index = (idx + 1) % n;
          continue;
        } else if (uop.staged_store_count != 0) {
          // Round-Robin: 本拍 TMEM bus 已被占用, 等下一拍
          continue;
        }
        break;
      case tud::StorePipelineStage::StageBuffer:
        progressed = stage_store_converted_packet(&uop);
        break;
      case tud::StorePipelineStage::Conversion:
        progressed = advance_store_conversion(&uop, c_slots->at(uop.slot_id));
        break;
      case tud::StorePipelineStage::Read: {
        auto subtile_idx = CMem::dump_subtiles(c_slots->at(uop.slot_id).fmt_c) - uop.remaining_cmem_dump_subtiles;
        if (uop.store_from_dmem) {
          // DMem 读仲裁已注释：双端口 SRAM + 确定性数据流保证无冲突
          // if (!mem_arbiter->try_claim_dmem_read(DMem::subtile_bank_mask(subtile_idx))) {
          //   ++perf_stats->stall_cmem_port_busy;
          //   continue;
          // }
        } else {
#ifdef ENABLE_CMEM_ARBITRATION
          if (!mem_arbiter->try_claim_cmem_read(CMem::subtile_bank_mask(subtile_idx))) {
            ++perf_stats->stall_cmem_port_busy;
            continue;
          }
#endif
        }
        (void)subtile_idx;
        progressed = stage_store_read(&uop, c_slots->at(uop.slot_id), cmem, dmem);
        break;
      }
      case tud::StorePipelineStage::Done:
        break;
      }

    // ============================================================
    // 【分支 3】 Fallback 路径: 简单预算消耗模式
    // 用于非精确 fill/store 的遗留操作, 仅消耗端口预算计数
    // ============================================================
    } else if (uop.remaining_amem_fill_lines != 0 || uop.remaining_bmem_fill_lines != 0
            || uop.remaining_cmem_fill_subtiles != 0 || uop.remaining_cmem_dump_subtiles != 0
            || uop.remaining_metamem_fill_packets != 0) {
      if (uop.remaining_amem_fill_lines != 0) {
        // AMem 仲裁已注释：双端口 SRAM + 确定性数据流保证无冲突
        // if (!mem_arbiter->try_consume_amem_write()) {
        //   ++perf_stats->stall_amem_port_busy;
        //   continue;
        // }
        --uop.remaining_amem_fill_lines;
        progressed = true;
      }
      if (uop.remaining_bmem_fill_lines != 0) {
        // BMem 仲裁已注释：双端口 SRAM + 确定性数据流保证无冲突
        // if (!mem_arbiter->try_consume_bmem_write()) {
        //   ++perf_stats->stall_bmem_port_busy;
        //   continue;
        // }
        --uop.remaining_bmem_fill_lines;
        progressed = true;
      }
      if (uop.remaining_cmem_fill_subtiles != 0) {
#ifdef ENABLE_CMEM_ARBITRATION
        if (!mem_arbiter->try_consume_cmem_write()) {
          ++perf_stats->stall_cmem_port_busy;
          continue;
        }
#endif
        --uop.remaining_cmem_fill_subtiles;
        progressed = true;
      }
      if (uop.remaining_cmem_dump_subtiles != 0) {
#ifdef ENABLE_CMEM_ARBITRATION
        if (!mem_arbiter->try_consume_cmem_read()) {
          ++perf_stats->stall_cmem_port_busy;
          continue;
        }
#endif
        --uop.remaining_cmem_dump_subtiles;
        progressed = true;
      }
      if (uop.remaining_metamem_fill_packets != 0) {
        // Meta 仲裁已注释：确定性数据流保证无冲突
        // if (!mem_arbiter->try_consume_meta_write()) {
        //   ++perf_stats->stall_meta_port_busy;
        //   continue;
        // }
        --uop.remaining_metamem_fill_packets;
        progressed = true;
      }

    // ============================================================
    // 【分支 4】 遗留 TMEM 写回路径 (不应到达, 到达即 abort)
    // ============================================================
    } else if (uop.remaining_tmem_write_packets != 0) {
      if (uop.pending_tmem_request_tag != 0) {
        TensorMemPortRsp response{};
        if (!take_tensor_mem_response(uop.pending_tmem_request_tag, completed_tensor_mem_responses, &response)) {
          ++perf_stats->stall_tmem_write_port_busy;
          continue;
        }
        uop.pending_tmem_request_tag = 0;
        --uop.remaining_tmem_write_packets;
        progressed = true;
      } else {
        std::cerr << "TensorUnit error: reached legacy tensor writeback path"
                  << " kind=" << static_cast<uint32_t>(uop.kind)
                  << " handle=" << uop.handle
                  << " window=" << uop.window_id
                  << " tile=" << uop.tile_idx
                  << std::endl;
        std::abort();
      }
    }

    // ============================================================
    // 完成检查: 若本轮有进展且所有阶段都已完成, 则处理 uop 完成逻辑
    // ============================================================
    if (!progressed || !mem_op_complete(uop)) {
      continue;  // 未完成, 继续处理下一个 uop
    }

    // 根据操作类型更新对应 slot 的状态
    switch (uop.kind) {
    case tud::MemUop::Kind::FillA: {
      // A 数据加载完成, 标记 A slot 就绪, WMMA 可以开始消费
      auto& slot = a_slots->at(uop.slot_id);
      TensorSlotManager::mark_a_ready(&slot);
      break;
    }
    case tud::MemUop::Kind::FillB: {
      // B 数据加载完成, 标记 B slot 就绪
      auto& slot = b_slots->at(uop.slot_id);
      TensorSlotManager::mark_b_ready(&slot);
      break;
    }
    case tud::MemUop::Kind::FillC: {
      // C 数据加载完成, 标记 C slot 就绪
      auto& slot = c_slots->at(uop.slot_id);
      TensorSlotManager::mark_c_ready(&slot);
      break;
    }
    case tud::MemUop::Kind::StoreC: {
      // Store 完成: 清除 store_pending, 更新 slot 有效性标志
      auto& slot = c_slots->at(uop.slot_id);
      slot.store_pending = false;
      slot.c_dirty = false;
      slot.cmem_final_valid = slot.output_resident;   // 驻留模式: CMem 有效
      slot.dmem_valid = !slot.output_resident;         // 非驻留模式: DMem 有效
      slot.busy = TensorSlotManager::is_c_slot_busy(slot);
      // StoreC 直接发出完成通知并立即返回 (不经过 pending_mem_ops 计数)
      completion_out->push({uop.async_id}, 1);
      mem_ops->erase(mem_ops->begin() + idx);
      return;
    }
    default:
      std::abort();
    }

    // Fill 完成: 递减 pending_mem_ops 中的子操作计数
    // 当同一个 async_id 的所有子操作 (如 FillA + FillB + FillC) 都完成时,
    // 发出一次完成通知
    auto pending_it = pending_mem_ops->find(uop.async_id);
    if (pending_it != pending_mem_ops->end()) {
      if (pending_it->second == 0) {
        std::abort();  // 不应出现: 计数已为零但仍有对应的 uop
      }
      --pending_it->second;
      if (pending_it->second == 0) {
        // 所有子操作完成, 清理并通知上游
        pending_mem_ops->erase(pending_it);
        completion_out->push({uop.async_id}, 1);
      }
    }
    // 从队列中移除已完成的 uop, 每次最多完成一个 (避免迭代器失效)
    mem_ops->erase(mem_ops->begin() + idx);
    return;
  }
}

} // namespace vortex
