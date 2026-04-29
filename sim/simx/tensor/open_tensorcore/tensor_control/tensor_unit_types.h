#pragma once

// ============================================================================
// tensor_unit_types.h -- TensorUnit 前端核心数据类型定义（单实例简化版）
// ============================================================================
//
// 简化后架构要点：
//   - AMem/BMem/CMem/DMem 各自只有一份物理实例（不再 ping-pong 双缓冲）
//   - AMem/BMem 深度 = 4 行（line 0/1 = K-phase 0，line 2/3 = K-phase 1）
//   - CMem/DMem 深度 = 4 subtile（一个 16x16 tile 的 4 个 8x8 subtile）
//   - WMMA = 8 primitive (m16n16k16 = 2 K-phase × 4 subtile)
//   - WMMA 发射单门控：a_valid && b_valid && (first ? c_valid : 1)
//   - 不再拆 K-phase 独立 ready；必须 4 行 fill 全部到齐才 a_valid=true
//   - 输出累加用 DMem 原位累加；不再使用 FIFO 循环累加
// ============================================================================

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "tensor_cfg.h"
#include "types.h"
#include "tensor_mem_port_types.h"
#include "tmem_window_planner.h"
#include "open_tensorcore/local_memory/meta_mem.h"
#include "open_tensorcore/tensor_compute/fp_types.h"

namespace vortex::tensor_unit_detail {

namespace vt = vortex::tensor;

// ============================================================================
// 全局常量
// ============================================================================

/// payload_fmt 的未设置值
inline constexpr uint32_t kUnsetPayloadFmt = 0xffffffffu;

/// 一条 m16n16k16 WMMA 指令展开后的 8×8 原语数量。
/// 分解: 2 K-phase × 4 subtile = 8 原语。
inline constexpr uint32_t kWmmaPrimitiveCount = 8;

/// 内部 K-phase 数量。m16n16k16 拆成 2 个 m16n16k8 phase。
inline constexpr uint32_t kKPhases = 2;

/// 一个 16×16 tile 内的 8×8 subtile 数量 = 4。
inline constexpr uint32_t kSubtilesPerTile = 4;

/// 每个 K-phase 的原语数量 = 4（2M × 2N）。
inline constexpr uint32_t kPrimitivesPerKPhase = kSubtilesPerTile;

/// 原语矩阵维度。
inline constexpr uint32_t kPrimitiveDim = 8;

/// fill 转换阶段输入缓冲深度。
inline constexpr uint32_t kMaxConversionInputPackets = 4;

/// store 输出缓冲深度。
inline constexpr uint32_t kOutputPacketBufferDepth = 2;

// ============================================================================
// 精度映射辅助函数
// ============================================================================

inline PrecisionType map_out_precision(uint32_t fmt_out) {
  switch (fmt_out) {
  case vt::fp8::id:  return PREC_FP8_E4M3;
  case vt::fp16::id: return PREC_FP16;
  case vt::fp32::id: return PREC_FP32;
  default:           return PREC_FP16;
  }
}

inline PrecisionType map_c_precision(uint32_t fmt_c) {
  switch (fmt_c) {
  case vt::fp8::id:  return PREC_FP8_E4M3;
  case vt::fp16::id: return PREC_FP16;
  case vt::fp32::id: return PREC_FP32;
  default:           return PREC_FP16;
  }
}

// ============================================================================
// 数据包数量计算
// ============================================================================

inline uint32_t a_packet_count(uint32_t fmt_a) {
  return (fmt_a == vt::fp16::id) ? 4 : 2;
}

inline uint32_t b_packet_count(uint32_t fmt_b) {
  return (fmt_b == vt::fp16::id) ? 4 : 2;
}

inline uint32_t meta_packet_count(uint32_t a_sparse_mode) {
  return (a_sparse_mode == vt::sparse_none) ? 0 : MetaMem::packet_count();
}

inline uint32_t meta_shadow_window_id(uint32_t window_id) {
  return window_id | 0x80000000u;
}

inline uint32_t c_load_packet_count(uint32_t fmt_c) {
  switch (fmt_c) {
  case vt::fp8::id:  return 4;
  case vt::fp16::id: return 8;
  case vt::fp32::id: return 16;
  default:           return 0;
  }
}

inline uint32_t d_store_packet_count(uint32_t fmt_d) {
  switch (fmt_d) {
  case vt::fp8::id:  return 4;
  case vt::fp16::id: return 8;
  case vt::fp32::id: return 16;
  default:           return 0;
  }
}

// ============================================================================
// Window 匹配与数据包拷贝
// ============================================================================

inline bool window_matches_load_target(TcuTarget load_target, TmemWindowTarget window_target) {
  switch (load_target) {
  case TcuTarget::A:
    return window_target == TmemWindowTarget::A;
  case TcuTarget::B:
    return window_target == TmemWindowTarget::B;
  case TcuTarget::C:
    return window_target == TmemWindowTarget::C || window_target == TmemWindowTarget::D;
  default:
    return false;
  }
}

template <typename PacketT>
inline std::vector<PacketT> copy_packets(const std::vector<TmemPacket>& packets) {
  std::vector<PacketT> out(packets.size());
  for (size_t i = 0; i < packets.size(); ++i) {
    std::copy_n(packets.at(i).bytes.begin(), packets.at(i).bytes.size(), out.at(i).begin());
  }
  return out;
}

// ============================================================================
// AMem 状态 (AMemState)
// ============================================================================
//
// 单实例。生命周期：
//   ready (未绑定 / 可接受新 fill)
//     → 绑定描述符 (descriptor/fmt_a/... 写入，a_ready 仍为 true)
//     → a_pending = true (fill MemUop 入队)
//     → 所有 4 行 fill 完成 → a_valid = true，a_pending = false
//     → WMMA 引用 (a_wmma_pending = true)
//     → 8 uop 全部 push 入计算阵列 → a_wmma_pending = false，a_valid = false
//     → a_ready = true，可接受新 fill
// ============================================================================
struct AMemState {
  //uint32_t descriptor = 0xffffffffu;
  uint32_t fmt_a = 0;
  //uint32_t a_sparse_mode = 0;
  uint32_t wmma_async_id = 0;
  bool transpose_a = false;

  bool a_ready = true;          ///< 可接受新的 fill 生命周期
  //bool a_pending = false;       ///< fill 搬运进行中
  bool a_valid = false;         ///< AMem 数据完整有效
  //bool a_wmma_pending = false;  ///< 正被 WMMA 原语消费

  //bool a_ws_locked = false;

  bool ws = false;
  bool use = false;
  bool lastuse = false;
  bool discard = false;
  bool fill = false;

  void reset() {
    //descriptor = 0xffffffffu;
    fmt_a = 0;
    //a_sparse_mode = 0;
    wmma_async_id = 0;
    transpose_a = false;
    a_ready = true;
    //a_pending = false;
    a_valid = false;
    //a_wmma_pending = false;
    bool a_ws_locked = false;
  }
};

// ============================================================================
// BMem 状态 (BMemState)
// ============================================================================
struct BMemState {
  //uint32_t descriptor = 0xffffffffu;
  uint32_t fmt_b = 0;
  //uint32_t wmma_async_id = 0;
  bool transpose_b = false;


  bool b_ready = true;
  //bool b_pending = false;
  bool b_valid = false;
  //bool b_wmma_pending = false;

  /// weight-stationary 锁定：ws=1 时 fill 完成后置 true，
  /// WMMA 消费后不清 b_valid，直到显式释放。
  //bool b_ws_locked = false;
  //uint32_t ws_descriptor = 0xffffffffu;
  
  bool ws = false;
  bool use = false;
  bool lastuse = false;
  bool discard = false;
  bool fill = false;



  void reset() {
    //descriptor = 0xffffffffu;
    fmt_b = 0;
    //wmma_async_id = 0;
    transpose_b = false;
    b_ready = true;
    //b_pending = false;
    b_valid = false;
    //b_wmma_pending = false;
    b_ws_locked = false;
    //ws_descriptor = 0xffffffffu;
  }
};

// ============================================================================
// CMem 状态 (CMemState)
// ============================================================================
//
// C 作为输入偏置被 fill 到 CMem；WMMA 发射时 first/single 模式从 CMem 读 c_bypass。
// ============================================================================
struct CMemState {
  //uint32_t descriptor = 0xffffffffu;
  uint32_t fmt_c = 0;
  //uint32_t fmt_d = 0;

  bool c_ready = true;          ///< 可接受新的 C fill
  //bool c_pending = false;       ///< C fill 搬运进行中
  bool c_valid = false;         ///< CMem 中 C 偏置已可读
  //uint32_t c_wmma_inflight = 0; ///< 正引用此 CMem 的未完成 WMMA 宏操作数

  bool output_resident = false; ///< 累加序列模式标志（OR bit）

  void reset() {
    //descriptor = 0xffffffffu;
    fmt_c = 0;
    //fmt_d = 0;
    c_ready = true;
    //c_pending = false;
    c_valid = false;
    //c_wmma_inflight = 0;
    //output_resident = false;
  }
};

// ============================================================================
// DMem 状态 (DMemState)
// ============================================================================
//
// DMem 是 WMMA 结果的最终去向。累加序列中，中间/尾 WMMA 读 DMem 做原位累加。
// ============================================================================
struct DMemState {
  //uint32_t descriptor = 0xffffffffu;
  uint32_t fmt_d = 0;
  //uint32_t store_async_id = 0;

  bool d_ready = true;             ///< 可接受新的结果写入生命周期
  //bool d_pending = false;          ///< 占位（结果尚未产生，但 DMem 已绑定描述符）
  bool d_valid = false;            ///< DMem 中已有有效最终结果供 MMA_STORE
  //bool d_store_pending = false;    ///< MMA_STORE 搬运进行中
  uint32_t d_wmma_inflight = 0;    ///< 正在原位累加 DMem 的 WMMA 宏操作数

  void reset() {
   // descriptor = 0xffffffffu;
    fmt_d = 0;
    //store_async_id = 0;
    d_ready = true;
    //d_pending = false;
    d_valid = false;
    //d_store_pending = false;
    d_wmma_inflight = 0;
  }
};

// ============================================================================
// PendingWmmaJob
// ============================================================================
//
// 单实例 SRAM + K-phase 不独立 ready。发射门控统一读 AMem/BMem/CMem 的 valid。
// 8 原语展开顺序：
//   next_uop ∈ [0, 7]
//   k_phase      = next_uop / kPrimitivesPerKPhase   (0 或 1)
//   c_subtile_id = next_uop % kPrimitivesPerKPhase   (0..3)
//   storage_m    = c_subtile_id / 2
//   storage_n    = c_subtile_id % 2
//   AMem line_idx = k_phase * 2 + storage_m
//   BMem line_idx = k_phase * 2 + storage_n
//
// is_first_in_accum_seq 由发射引擎根据 prev_or 上升沿设置：
//   - OR=0                 → first (单次模式)
//   - OR=1 && prev_or=0    → first (累加序列的第一条)
//   - OR=1 && prev_or=1    → middle/tail（原位累加 DMem）
// ============================================================================
struct PendingWmmaJob {
  uint32_t wgid = 0;
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t a_sparse_mode = 0;
  uint32_t async_id = 0;
  bool enable_input_d = false;
  //bool ws = false;
 // bool is_first_in_accum_seq = false;  ///< 第一条（或单次模式）：C 从 CMem 读
  uint32_t next_uop = 0;
};

// ============================================================================
// Fill 转换类型枚举
// ============================================================================
enum class FillConvertKind : uint8_t {
  None = 0,
  ALine,
  BLine,
  CSubtile,
};

// ============================================================================
// Fill / Store 显式流水阶段
// ============================================================================
enum class FillPipelineStage : uint8_t {
  WaitTmemRsp = 0,
  Ingress,
  Conversion,
  LocalWrite,
  Done,
};

enum class StorePipelineStage : uint8_t {
  Read = 0,
  Conversion,
  StageBuffer,
  Emit,
  Done,
};

// ============================================================================
// 内存微操作 (MemUop)
// ============================================================================
//
// 单实例 SRAM：
//   - FillA/FillB: line_idx ∈ [0, 4)，覆盖 K-phase 0/1 的全部 M/N-block
//   - FillC: line_idx 等同 subtile_id ∈ [0, 4)
//   - StoreC: 从 DMem 读 subtile 并回写 TMEM
// ============================================================================
struct MemUop {
  enum class Kind : uint8_t {
    FillA = 0,
    FillB,
    FillC,
    StoreC,
  };

  Kind kind = Kind::FillA;
  uint32_t wgid = 0;
  uint32_t line_idx = 0;    ///< FillA/B: AMem/BMem 行号；FillC: CMem subtile 号
  uint32_t taddr = 0;
  uint32_t payload_fmt = kUnsetPayloadFmt;
  uint32_t async_id = 0;

  uint32_t remaining_tmem_read_packets = 0;
  uint32_t remaining_tmem_write_packets = 0;
  uint32_t remaining_amem_fill_lines = 0;
  uint32_t remaining_bmem_fill_lines = 0;
  uint32_t remaining_cmem_fill_subtiles = 0;
  uint32_t remaining_dmem_dump_subtiles = 0;
  uint32_t remaining_metamem_fill_packets = 0;

  uint64_t pending_tmem_request_tag = 0;
  

  uint32_t next_payload_packet_idx = 0;
  uint32_t next_meta_packet_idx = 0;
 
  TmemPacket fill_ingress_packet{};
  std::vector<TmemPacket> staged_meta_packets;

  FillConvertKind fill_input_kind = FillConvertKind::None;
  FillConvertKind fill_convert_kind = FillConvertKind::None;
  uint32_t fill_input_index = 0;
  uint32_t fill_input_packet_count = 0;
  uint32_t fill_input_packets_needed = 0;
  std::array<TmemPacket, kMaxConversionInputPackets> fill_input_packets{};


  uint32_t fill_convert_index = 0;
  uint16_t fill_convert_fp9[kPrimitiveDim][kPrimitiveDim] = {};
  uint32_t fill_convert_fp22[kPrimitiveDim][kPrimitiveDim] = {};

  uint32_t next_store_packet_idx = 0;
  bool store_from_dmem = true;  ///< 当前 MMA_STORE 统一从 DMem 读
  uint32_t store_raw_subtile[kPrimitiveDim][kPrimitiveDim] = {};
  uint32_t store_raw_next_segment = 0;

  
  TmemPacket store_conv_packet{};

  std::array<TmemPacket, kOutputPacketBufferDepth> staged_store_packets{};
  uint32_t staged_store_head = 0;
  uint32_t staged_store_tail = 0;
  uint32_t staged_store_count = 0;

  FillPipelineStage fill_stage = FillPipelineStage::WaitTmemRsp;
  StorePipelineStage store_stage = StorePipelineStage::Read;

  bool fill_ingress_valid = false;
  bool fill_input_valid = false;
  bool fill_convert_valid = false;
  bool fill_memwrite_valid = false;

  bool store_conv_valid = false;
  bool store_raw_subtile_valid = false;
};

// ============================================================================
// WMMA 未就绪原因 / 回收原因枚举
// ============================================================================
enum class NoWmmaReadyReason : uint8_t {
  JobBuilderEmpty = 0,
  WaitingForMmaLoad,
  WaitingForHandleAlloc,
  WaitingForMemRelease,
};

enum class MemReleaseReason : uint8_t {
  None = 0,
  CWmmaInflightDrain,
  CStorePending,
  AbWmmaPendingClear,
  DAccumBusy,
};

// ============================================================================
// MemUop 辅助查询
// ============================================================================

inline uint32_t mem_uop_payload_fmt(
    const MemUop& uop,
    const AMemState& amem_state,
    const BMemState& bmem_state,
    const CMemState& cmem_state) {
  if (uop.payload_fmt != kUnsetPayloadFmt) {
    return uop.payload_fmt;
  }
  switch (uop.kind) {
  case MemUop::Kind::FillA:
    return amem_state.fmt_a;
  case MemUop::Kind::FillB:
    return bmem_state.fmt_b;
  case MemUop::Kind::FillC:
    return cmem_state.fmt_c;
  default:
    return 0;
  }
}

inline uint32_t fill_payload_packet_count(
    const MemUop& uop,
    const AMemState& amem_state,
    const BMemState& bmem_state,
    const CMemState& cmem_state) {
  auto payload_fmt = mem_uop_payload_fmt(uop, amem_state, bmem_state, cmem_state);
  switch (uop.kind) {
  case MemUop::Kind::FillA:
    return a_packet_count(payload_fmt);
  case MemUop::Kind::FillB:
    return b_packet_count(payload_fmt);
  case MemUop::Kind::FillC:
    return c_load_packet_count(payload_fmt);
  default:
    std::abort();
  }
}

} // namespace vortex::tensor_unit_detail
