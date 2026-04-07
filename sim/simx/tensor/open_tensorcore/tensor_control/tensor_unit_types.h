#pragma once

// ============================================================================
// tensor_unit_types.h -- TensorUnit 前端核心数据类型定义
// ============================================================================
//
// 架构定位:
//   TensorUnit 位于 GPU 核心执行阶段 (execute stage) 和 TensorCore 计算阵列
//   之间, 承担以下职责:
//     1. 管理操作数从 TMEM (Tensor Memory) 加载到本地 SRAM
//        (AMem / BMem / CMem / DMem)
//     2. 将宏 WMMA 指令展开为 8 个 TensorCore 原语操作
//     3. 协调 fill (加载) 和 store (回写) 的异步流水线
//
// 双缓冲机制 (kNumOperandSlots = 2):
//   A/B/C 操作数各有 2 个 slot, 实现经典的双缓冲:
//   - slot[0] 正在被 TensorCore 读取执行 WMMA, slot[1] 可同时从 TMEM 填充
//   - 下一轮 WMMA 时两个 slot 角色互换, 消除加载与计算之间的流水线气泡
//
// WMMA 分解 (kWmmaPrimitiveCount = 4):
//   基本计算粒度为 m16n16k8, 分解为 4 个独立的 8×8 原语操作:
//   - A 矩阵 (16×8) 沿 M 维分为 2 块 → 2 个子块
//   - B 矩阵 (8×16) 沿 N 维分为 2 块 → 2 个子块
//   - 输出矩阵 (16×16) 为 2×2 = 4 个 8×8 subtile
//   - K 维累加由外部 WMMA 指令序列控制, 不在单条 WMMA 内拆分
//
// 命名空间:
//   - 完整命名空间: vortex::tensor_unit_detail
//   - 常用别名: tud (在外部文件中通过 namespace tud = tensor_unit_detail 引入)
//   - vt 别名指向 vortex::tensor, 用于访问数据格式 ID 和稀疏常量
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

/// payload_fmt 的未设置值; 当 MemUop 尚未绑定具体格式时使用此值,
/// 运行时会回退到对应 slot 中记录的格式
inline constexpr uint32_t kUnsetPayloadFmt = 0xffffffffu;

/// 每种操作数 (A/B/C) 的 slot 数量 = 2, 即双缓冲深度。
/// 一个 slot 供当前 WMMA 读取, 另一个 slot 可同时被 fill 流水线写入。
inline constexpr uint32_t kNumOperandSlots = 2;

/// 一条 m16n16k8 WMMA 指令展开后的 8×8 原语操作数量。
/// 分解方式: M 维 2 步 × N 维 2 步 = 4 次原语调用 (K=8 无需拆分)。
/// K 累加由软件控制: 输出驻留模式下发射多条 WMMA 共享同一 CMem slot。
inline constexpr uint32_t kWmmaPrimitiveCount = 4;

/// 一个 16×16 输出 tile 内包含的 8×8 子块 (subtile) 数量 = 4。
/// 布局为 2×2: 沿 M 维 2 块, 沿 N 维 2 块。
inline constexpr uint32_t kSubtilesPerTile = 4;

/// 单个原语操作的矩阵维度 = 8, 即 TensorCore 硬件一次处理 8×8 的乘加运算。
inline constexpr uint32_t kPrimitiveDim = 8;

/// fill 路径转换阶段的输入缓冲区深度 (最多可暂存的 64B 数据包数量)。
/// fp16 需要 2 个包拼成一个 8×8 block; fp32 累加器加载需要 4 个包。
inline constexpr uint32_t kMaxConversionInputPackets = 4;

/// store 路径的输出数据包缓冲区深度 = 2, 实现简单的双缓冲:
/// 转换阶段生产一个包的同时, TMEM 写端口可以发射上一个包。
inline constexpr uint32_t kOutputPacketBufferDepth = 2;

// ============================================================================
// 精度映射辅助函数
// ============================================================================

/// 将输出格式 ID (vt::fp8/fp16/fp32 的 id) 映射为 PrecisionType 枚举。
/// 用于 store 路径中 fp22 累加器值向目标精度转换时选择转换器。
inline PrecisionType map_out_precision(uint32_t fmt_out) {
  switch (fmt_out) {
  case vt::fp8::id:
    return PREC_FP8_E4M3;
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp32::id:
    return PREC_FP32;
  default:
    return PREC_FP16;
  }
}

/// 将 C 累加器格式 ID 映射为 PrecisionType 枚举。
/// 用于 fill 路径中外部 C 偏置数据向 fp22 内部格式转换时选择转换器。
inline PrecisionType map_c_precision(uint32_t fmt_c) {
  switch (fmt_c) {
  case vt::fp8::id:
    return PREC_FP8_E4M3;
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp32::id:
    return PREC_FP32;
  default:
    return PREC_FP16;
  }
}

// ============================================================================
// 数据包数量计算函数
// ============================================================================
// 这些函数根据操作数精度计算从 TMEM 传输一个 m16n16k8 tile 所需的包数。
//
// 计算依据 (以 A 矩阵为例):
//   A 矩阵为 16×8 = 128 个元素。
//   - fp16 (2B/elem): 128 × 2 = 256B → 256 / 64 = 4 个包
//   - fp8  (1B/elem): 128 × 1 = 128B → 128 / 64 = 2 个包
// ============================================================================

/// A 操作数从 TMEM 加载一个 m16n16k8 tile 所需的 64B 数据包数量。
/// fp16 格式 4 包, fp8 格式 2 包。
inline uint32_t a_packet_count(uint32_t fmt_a) {
  return (fmt_a == vt::fp16::id) ? 4 : 2;
}

/// B 操作数从 TMEM 加载一个 m16n16k8 tile 所需的 64B 数据包数量。
/// fp16 格式 4 包, fp8 格式 2 包。
inline uint32_t b_packet_count(uint32_t fmt_b) {
  return (fmt_b == vt::fp16::id) ? 4 : 2;
}

/// 稀疏元数据 (metadata) 从 TMEM 加载所需的数据包数量。
/// 非稀疏模式 (sparse_none) 返回 0, 即不需要传输元数据;
/// 2:4 或 1:4 稀疏模式返回 MetaMem::packet_count() = 1 个 64B 包。
inline uint32_t meta_packet_count(uint32_t a_sparse_mode) {
  return (a_sparse_mode == vt::sparse_none) ? 0 : MetaMem::packet_count();
}

/// 为稀疏元数据生成一个 "影子 window ID"。
/// 稀疏 A 矩阵的数据和元数据共享同一个 TMEM window, 但元数据的 TMEM 读请求
/// 需要一个独立的 window_id 以区分调度。此函数通过设置最高位 (bit31)
/// 生成影子 ID, 确保与数据 window_id 不冲突。
inline uint32_t meta_shadow_window_id(uint32_t window_id) {
  return window_id | 0x80000000u;
}

/// C 累加器 (偏置) 从 TMEM 加载一个完整 16×16 tile 所需的 64B 数据包数量。
/// - fp8  (1B/elem): 256B → 4 包
/// - fp16 (2B/elem): 512B → 8 包
/// - fp32 (4B/elem): 1024B → 16 包
/// C 的包数量与 A/B 不同, 因为 C 支持更宽的 fp32 精度。
inline uint32_t c_load_packet_count(uint32_t fmt_c) {
  switch (fmt_c) {
  case vt::fp8::id:
    return 4;
  case vt::fp16::id:
    return 8;
  case vt::fp32::id:
    return 16;
  default:
    return 0;
  }
}

/// D (输出矩阵) 回写到 TMEM 所需的 64B 数据包数量。
/// 包数量与 C 加载完全对称, 由输出精度决定:
/// - fp8 → 4 包, fp16 → 8 包, fp32 → 16 包
inline uint32_t d_store_packet_count(uint32_t fmt_d) {
  switch (fmt_d) {
  case vt::fp8::id:
    return 4;
  case vt::fp16::id:
    return 8;
  case vt::fp32::id:
    return 16;
  default:
    return 0;
  }
}

// ============================================================================
// Window 匹配与数据包拷贝辅助函数
// ============================================================================

/// 判断一个 TMEM 加载目标 (TcuTarget: A/B/C) 是否与给定的 TMEM window 目标
/// (TmemWindowTarget: A/B/C/D/Meta) 兼容。
/// 注意: TcuTarget::C 同时匹配 TmemWindowTarget::C 和 TmemWindowTarget::D,
/// 因为 C (累加器输入) 和 D (输出结果) 共享相同的 C-slot 管理逻辑。
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

/// 将 TmemPacket 向量深拷贝为指定类型 PacketT 的向量。
/// PacketT 通常是 std::array<uint8_t, 64> (与 AMem::packet_t / BMem::packet_t 等兼容)。
/// 用于在 fill 路径中将 TMEM 响应包转换为本地存储模块所需的包类型。
template <typename PacketT>
inline std::vector<PacketT> copy_packets(const std::vector<TmemPacket>& packets) {
  std::vector<PacketT> out(packets.size());
  for (size_t i = 0; i < packets.size(); ++i) {
    std::copy_n(packets.at(i).bytes.begin(), packets.at(i).bytes.size(), out.at(i).begin());
  }
  return out;
}

// ============================================================================
// A 操作数 Slot 状态 (ASlotState)
// ============================================================================
//
// 每个 ASlotState 实例管理一个 A 操作数 SRAM slot 的完整生命周期。
// 双缓冲阵列: std::array<ASlotState, kNumOperandSlots> a_slots;
//
// Slot 生命周期:
//   空闲 (valid=false)
//     → 分配 (valid=true, busy=true): 收到 WMMA_LOAD 指令, 绑定 descriptor/格式
//     → 填充中 (a_pending=true): fill 微操作已入队, 等待 TMEM 数据传输完成
//     → 就绪 (a_ready=true): AMem 中数据已完整, 可供 WMMA 原语读取
//     → WMMA 执行中 (wmma_pending=true): 已被 PendingWmmaJob 引用, 正在逐步发射原语
//     → 回收 (reset): 所有原语完成后释放 slot, 回到空闲状态
// ============================================================================
struct ASlotState {
  /// 拥有此 slot 的 workgroup ID; 用于多 workgroup 共享 TensorUnit 时的隔离
  uint32_t owner_wgid = 0;

  /// MMA 描述符 ID; 标识此次加载所对应的矩阵乘法配置
  /// (包含 M/N/K 维度、精度、步长等信息)。
  /// 0xffffffff 表示未设置。
  uint32_t descriptor = 0xffffffffu;

  /// A 操作数的数据精度格式 ID (vt::fp8::id=13, vt::fp16::id=1 等)
  uint32_t fmt_a = 0;

  /// A 操作数的稀疏模式 (vt::sparse_none=0, vt::sparse_2_4=1, vt::sparse_1_4=2)。
  /// 稀疏模式下, A 矩阵只存储非零元素, 额外需要加载稀疏元数据到 MetaMem。
  uint32_t a_sparse_mode = 0;

  /// 此 slot 关联的异步操作 ID; 当 fill 完成后, 通过此 ID 向 CPU 核心
  /// 发送 TensorAsyncOpCompletion 通知, 使得后续的 fence/barrier 能够同步。
  uint32_t wmma_async_id = 0;

  /// 是否需要对 A 矩阵进行转置。转置在 AMem 读取时实现 (交换行列索引),
  /// 不影响 fill 路径的数据布局。
  bool transpose_a = false;

  /// slot 是否已被分配给某个加载操作 (生命周期开始标志)
  bool valid = false;

  /// slot 是否处于忙碌状态 (正在 fill 或被 WMMA 引用, 不可被新的加载指令重用)
  bool busy = false;

  /// AMem 中的数据是否已完整可读; fill 完成时置 true, slot 释放时清除
  bool a_ready = false;

  /// fill 微操作是否已入队等待执行; 从入队到 fill 完成期间为 true
  bool a_pending = false;

  /// 此 slot 是否已被某个 PendingWmmaJob 引用, 正在等待或执行 WMMA 原语;
  /// 所有 8 个原语完成后清除。此标志防止 slot 在 WMMA 执行过程中被回收。
  bool wmma_pending = false;

  /// 将所有字段重置为初始值, 使 slot 回到空闲状态
  void reset() {
    owner_wgid = 0;
    descriptor = 0xffffffffu;
    fmt_a = 0;
    a_sparse_mode = 0;
    wmma_async_id = 0;
    transpose_a = false;
    valid = false;
    busy = false;
    a_ready = false;
    a_pending = false;
    wmma_pending = false;
  }
};

// ============================================================================
// B 操作数 Slot 状态 (BSlotState)
// ============================================================================
//
// 与 ASlotState 结构高度对称, 管理 B 操作数 SRAM slot 的生命周期。
// B 矩阵不支持稀疏模式, 因此没有 sparse_mode 字段。
//
// 生命周期与 ASlotState 完全相同:
//   空闲 → 分配 → 填充中 → 就绪 → WMMA 执行中 → 回收
// ============================================================================
struct BSlotState {
  /// 拥有此 slot 的 workgroup ID
  uint32_t owner_wgid = 0;

  /// MMA 描述符 ID (0xffffffff 表示未设置)
  uint32_t descriptor = 0xffffffffu;

  /// B 操作数的数据精度格式 ID
  uint32_t fmt_b = 0;

  /// 此 slot 关联的异步操作 ID
  uint32_t wmma_async_id = 0;

  /// 是否需要对 B 矩阵进行转置
  bool transpose_b = false;

  /// slot 是否已被分配
  bool valid = false;

  /// slot 是否处于忙碌状态
  bool busy = false;

  /// BMem 中的数据是否已完整可读
  bool b_ready = false;

  /// fill 微操作是否已入队等待执行
  bool b_pending = false;

  /// 此 slot 是否已被某个 PendingWmmaJob 引用, 正在执行 WMMA
  bool wmma_pending = false;

  /// 将所有字段重置为初始值
  void reset() {
    owner_wgid = 0;
    descriptor = 0xffffffffu;
    fmt_b = 0;
    wmma_async_id = 0;
    transpose_b = false;
    valid = false;
    busy = false;
    b_ready = false;
    b_pending = false;
    wmma_pending = false;
  }
};

// ============================================================================
// C 累加器 Slot 状态 (CSlotState)
// ============================================================================
//
// C slot 的管理比 A/B 更复杂, 因为它同时承担:
//   1. 输入角色: 加载 C 偏置 (bias) 到 CMem, 作为 WMMA 的初始累加值
//   2. 输出角色: WMMA 计算结果写回 CMem (或 DMem), 然后回写到 TMEM
//
// 生命周期:
//   空闲
//     → 分配 (valid=true, busy=true)
//     → C 填充中 (c_pending=true): 从 TMEM 加载 C 偏置到 CMem
//     → C 就绪 (c_ready=true): CMem 数据完整, 可作为 WMMA 输入
//     → WMMA 执行 (c_wmma_inflight > 0): 多个原语正在向 CMem 写入累加结果
//     → 脏 (c_dirty=true): 至少一个 WMMA 原语已完成, CMem 内容已被修改
//     → 最终有效 (cmem_final_valid=true): 所有 WMMA 原语完成, 累加器结果完整
//     → 回写 (store_pending=true): 正在将 CMem/DMem 数据写回 TMEM
//     → 回收 (reset)
//
// output_resident 标志:
//   - true: WMMA 结果直接累加到 CMem (就地模式, 适用于多次 WMMA 累加同一 C)
//   - false: WMMA 结果写入 DMem (解耦模式, 下一次 WMMA 可以立即开始填充新的 C)
// ============================================================================
struct CSlotState {
  /// 拥有此 slot 的 workgroup ID
  uint32_t owner_wgid = 0;

  /// MMA 描述符 ID (0xffffffff 表示未设置)
  uint32_t descriptor = 0xffffffffu;

  /// C 累加器 (输入偏置) 的数据精度格式 ID
  uint32_t fmt_c = 0;

  /// D 输出矩阵的数据精度格式 ID; 可以与 fmt_c 不同
  /// (例如以 fp16 加载 C 偏置, 以 fp32 写回 D 结果)
  uint32_t fmt_d = 0;

  /// store 操作的异步 ID; store 完成后通过此 ID 发送完成通知
  uint32_t store_async_id = 0;

  /// 当前正在飞行中 (已发射但未完成) 的 WMMA 原语数量。
  /// 每发射一个原语 +1, 每完成一个 -1。
  /// 当此计数降为 0 且 c_dirty=true 时, 表示所有 WMMA 操作完成。
  uint32_t c_wmma_inflight = 0;

  /// 输出驻留模式标志:
  /// - true: WMMA 结果直接写入 CMem (就地累加, 占用 CMem 直到 store 完成)
  /// - false: WMMA 结果写入 DMem (解耦模式, CMem 可提前释放给下一次加载)
  bool output_resident = false;

  /// DMem 中是否持有有效的输出数据 (仅在 output_resident=false 时使用)
  bool dmem_valid = false;

  /// slot 是否已被分配
  bool valid = false;

  /// slot 是否处于忙碌状态 (正在 fill/WMMA/store 中的任何一个阶段)
  bool busy = false;

  /// CMem 中的 C 偏置数据是否已完整可读, 可供 WMMA 使用
  bool c_ready = false;

  /// C fill 微操作是否已入队等待执行
  bool c_pending = false;

  /// CMem 中的最终累加结果是否已完整有效 (所有 WMMA 原语均已完成写回)。
  /// 这是 store 路径的前置条件: 只有 cmem_final_valid=true 时才能开始回写。
  bool cmem_final_valid = false;

  /// CMem 内容是否已被 WMMA 修改 (至少一个原语完成了累加写回)。
  /// 用于区分 "C slot 仅加载了偏置但尚未做任何计算" 和 "已有计算结果"。
  bool c_dirty = false;

  /// store (回写 TMEM) 微操作是否正在执行
  bool store_pending = false;

  /// 将所有字段重置为初始值
  void reset() {
    owner_wgid = 0;
    descriptor = 0xffffffffu;
    fmt_c = 0;
    fmt_d = 0;
    store_async_id = 0;
    c_wmma_inflight = 0;
    output_resident = false;
    dmem_valid = false;
    valid = false;
    busy = false;
    c_ready = false;
    c_pending = false;
    cmem_final_valid = false;
    c_dirty = false;
    store_pending = false;
  }
};

// ============================================================================
// 待执行的 WMMA job (PendingWmmaJob)
// ============================================================================
//
// 当 A/B/C 三个 slot 都就绪后, issue 策略模块会创建一个 PendingWmmaJob,
// 表示一条宏 WMMA 指令已准备好被展开执行。
//
// 展开过程:
//   next_uop 从 0 递增到 kWmmaPrimitiveCount-1 (即 0~3), 每步发射一个
//   8×8 原语操作给 TensorCore 计算阵列。原语的 (step_m, step_n)
//   坐标由 next_uop 的编码位决定 (K=8 无需拆分)。
//
// 引用关系:
//   a_slot_id / b_slot_id / c_slot_id 分别指向 a_slots / b_slots / c_slots
//   中的下标。在所有 4 个原语完成之前, 对应 slot 的 wmma_pending 保持为 true,
//   防止被提前回收。
// ============================================================================
struct PendingWmmaJob {
  /// 发起此 WMMA 的 workgroup ID
  uint32_t wgid = 0;

  /// 引用的 A 操作数 slot 下标 (0 或 1)
  uint32_t a_slot_id = 0;

  /// 引用的 B 操作数 slot 下标 (0 或 1)
  uint32_t b_slot_id = 0;

  /// 引用的 C 累加器 slot 下标 (0 或 1)
  uint32_t c_slot_id = 0;

  /// A 操作数精度格式 ID (冗余缓存, 避免每次查 slot)
  uint32_t fmt_a = 0;

  /// B 操作数精度格式 ID
  uint32_t fmt_b = 0;

  /// C 累加器精度格式 ID
  uint32_t fmt_c = 0;

  /// A 操作数的稀疏模式 (影响原语计算时的元数据读取路径)
  uint32_t a_sparse_mode = 0;

  /// 此 WMMA 宏操作的异步 ID; 最后一个原语完成时用于发送完成通知
  uint32_t async_id = 0;

  /// 输出是否就地写入 CMem (true) 还是写入 DMem (false)
  bool output_resident = false;

  /// 下一个待发射的原语编号 (0~7)。
  /// 编码方式: bit0 = step_k 的低位, bit1 = step_m 或 step_n 的选择,
  /// bit2 = 更高维步进。具体映射由 issue 逻辑决定。
  uint32_t next_uop = 0;
};

// ============================================================================
// Fill 转换类型枚举 (FillConvertKind)
// ============================================================================
//
// 标识 fill 路径中精度转换阶段正在处理的数据类型。
// 不同操作数的转换粒度不同:
//   - A/B 以 "行" (line) 为粒度: 一行 = 一个 8×8 block = 一次转换操作
//   - C 以 "子块" (subtile) 为粒度: 一个子块 = 一个 8×8 block = 一次转换操作
//   (两者语义相同, 但在代码中使用不同名称以区分来源)
// ============================================================================
enum class FillConvertKind : uint8_t {
  None = 0,      ///< 无转换任务
  ALine,         ///< 正在转换 A 操作数的一行 (8×8 block → fp9)
  BLine,         ///< 正在转换 B 操作数的一行 (8×8 block → fp9)
  CSubtile,      ///< 正在转换 C 累加器的一个子块 (8×8 block → fp22)
};

// ============================================================================
// Fill / Store 显式流水阶段枚举
// ============================================================================
//
// 为了避免多个 TMEM↔SRAM 传输阶段在同一个 tick() 中连续下落，
// 每个 MemUop 都显式记录当前所处的大阶段。TensorLocalMemPipeline
// 在每个周期最多只推进一个阶段，并在阶段边界之间插入寄存器语义。
// ============================================================================

/// Fill (TMEM → 本地 SRAM) 流水线的显式阶段。
enum class FillPipelineStage : uint8_t {
  WaitTmemRsp = 0,  ///< 等待 / 发射 TMEM 读请求，或等待响应返回
  Ingress,          ///< ingress 寄存器中的 64B 包进入 input 组装缓冲
  Conversion,       ///< input 组装缓冲进入精度转换寄存器
  LocalWrite,       ///< 转换结果或 meta 包写入本地 SRAM
  Done,             ///< 当前 fill 微操作的所有阶段均已完成
};

/// Store (本地 SRAM → TMEM) 流水线的显式阶段。
enum class StorePipelineStage : uint8_t {
  Read = 0,         ///< 从 CMem/DMem 读取一个 fp22 subtile 到 raw 寄存器
  Conversion,       ///< raw subtile 进入精度转换寄存器
  StageBuffer,      ///< 转换结果写入 staged_store 环形缓冲
  Emit,             ///< 从 staged_store 发射 / 等待 TMEM 写请求完成
  Done,             ///< 当前 store 微操作的所有阶段均已完成
};

// ============================================================================
// 内存微操作 (MemUop) -- TMEM ↔ 本地 SRAM 数据传输的原子操作
// ============================================================================
//
// 每个 MemUop 代表一次完整的操作数 tile 加载 (fill) 或结果回写 (store) 操作。
// 它包含操作所需的全部状态, 由 TensorLocalMemPipeline 逐拍推进。
//
// ── Fill 流水线 (FillA / FillB / FillC) ──
//
//   数据流: TMEM → [ingress] → [input 组装] → [conversion 精度转换] → [local write 写入 SRAM]
//
//   阶段 1 -- Ingress (入口暂存):
//     从 TMEM 读取一个 64B 数据包, 存入 fill_ingress_packet。
//     每收到一个响应包, remaining_tmem_read_packets 减 1。
//
//   阶段 2 -- Input (输入组装):
//     将一个或多个 ingress 包组装为一次转换所需的输入。
//     - A/B fp8: 1 个包 → 一个 8×8 block (64 个 fp8 元素)
//     - A/B fp16: 2 个包 → 一个 8×8 block (64 个 fp16 元素, 每个 2B)
//     - C fp8: 1 个包; C fp16: 2 个包; C fp32: 4 个包
//     fill_input_packets[] 缓冲区暂存这些包, fill_input_packets_needed 记录还缺几个。
//
//   阶段 3 -- Conversion (精度转换):
//     - A/B: 将 fp8/fp16 转换为 fp9 (TensorCore 内部乘法精度), 结果写入 fill_convert_fp9[][]
//     - C: 将 fp8/fp16/fp32 转换为 fp22 (TensorCore 内部累加精度), 结果写入 fill_convert_fp22[][]
//
//   阶段 4 -- Local Write (写入本地 SRAM):
//     将转换后的数据写入 AMem / BMem / CMem 的对应行/子块。
//     每写入一个 block, remaining_amem/bmem/cmem_fill_* 减 1。全部归零时 fill 完成。
//
// ── Store 流水线 (StoreC) ──
//
//   数据流: CMem/DMem → [read 读出子块] → [conversion 精度转换] → [staged 暂存] → [emit 发射到 TMEM]
//
//   阶段 1 -- Read (读出原始子块):
//     从 CMem 或 DMem 读取一个 8×8 子块的 fp22 原始值, 存入 store_raw_subtile[][]。
//     store_from_dmem 决定读取来源。
//
//   阶段 2 -- Conversion (精度转换):
//     将 fp22 值按段 (segment) 转换为目标精度 (fp8/fp16/fp32), 每段生成一个 64B 包。
//     store_raw_next_segment 跟踪当前转换到第几段。
//
//   阶段 3 -- Staged (暂存缓冲):
//     转换后的包存入 staged_store_packets[] 环形缓冲区 (深度=2),
//     使转换和 TMEM 写入可以流水线并行。
//
//   阶段 4 -- Emit (发射写请求):
//     从 staged_store_packets[] 取出一个包, 发送 TMEM 写请求。
//     每发出一个写请求, remaining_tmem_write_packets 减 1。全部归零时 store 完成。
// ============================================================================
struct MemUop {
  /// 微操作类型
  enum class Kind : uint8_t {
    FillA = 0,  ///< 从 TMEM 加载 A 操作数到 AMem
    FillB,      ///< 从 TMEM 加载 B 操作数到 BMem
    FillC,      ///< 从 TMEM 加载 C 累加器偏置到 CMem
    StoreC,     ///< 将 CMem/DMem 中的计算结果回写到 TMEM
  };

  // ---- 基本信息 ----

  Kind kind = Kind::FillA;              ///< 此微操作的类型 (FillA/FillB/FillC/StoreC)
  uint32_t wgid = 0;                    ///< 发起操作的 workgroup ID
  uint32_t slot_id = 0;                 ///< 目标 slot 下标 (0 或 1)
  uint32_t handle = 0;                  ///< TMEM window planner 分配的 window 句柄
  uint32_t window_id = 0;              ///< TMEM window ID, 用于地址计算和仲裁
  uint32_t payload_fmt = kUnsetPayloadFmt; ///< 数据精度格式; kUnsetPayloadFmt 表示需从 slot 推断
  uint32_t tile_idx = 0;                ///< 在 window 中的 tile 偏移索引
  uint32_t async_id = 0;                ///< 异步操作 ID, 完成时用于通知
  bool separate_handle = false;          ///< 是否使用独立句柄 (用于稀疏元数据等特殊场景)

  // ---- 全局进度计数器 ----
  // 这些计数器从初始值递减到 0, 标记各阶段的剩余工作量

  uint32_t remaining_tmem_read_packets = 0;     ///< fill: 还需从 TMEM 读取的 64B 数据包数
  uint32_t remaining_tmem_write_packets = 0;    ///< store: 还需向 TMEM 写入的 64B 数据包数
  uint32_t remaining_amem_fill_lines = 0;       ///< FillA: 还需写入 AMem 的行数 (每行一个 8×8 block)
  uint32_t remaining_bmem_fill_lines = 0;       ///< FillB: 还需写入 BMem 的行数
  uint32_t remaining_cmem_fill_subtiles = 0;    ///< FillC: 还需写入 CMem 的子块数 (4 个 8×8 子块)
  uint32_t remaining_cmem_dump_subtiles = 0;    ///< StoreC: 还需从 CMem/DMem 读出的子块数
  uint32_t remaining_metamem_fill_packets = 0;  ///< FillA (稀疏): 还需写入 MetaMem 的元数据包数

  // ---- TMEM 请求追踪 ----

  /// 最近一次发出的 TMEM 读/写请求的标签 (request_id)。
  /// 用于在 TMEM 响应到达时匹配对应的 MemUop。
  uint64_t pending_tmem_request_tag = 0;

  /// Fill 路径当前所处的显式流水阶段。
  FillPipelineStage fill_stage = FillPipelineStage::WaitTmemRsp;

  // ---- Fill 阶段 1: Ingress (入口暂存) ----

  /// 下一个期望接收的 payload 数据包索引 (从 0 递增)
  uint32_t next_payload_packet_idx = 0;

  /// 下一个期望接收的 metadata 数据包索引 (仅稀疏模式使用)
  uint32_t next_meta_packet_idx = 0;

  /// ingress 阶段是否持有一个有效的 TMEM 响应包, 等待被送入 input 阶段
  bool fill_ingress_valid = false;

  /// ingress 阶段暂存的 64B TMEM 响应数据包
  TmemPacket fill_ingress_packet{};

  /// 已暂存的稀疏元数据包列表 (收齐后一次性写入 MetaMem)
  std::vector<TmemPacket> staged_meta_packets;

  // ---- Fill 阶段 2: Input (输入组装) ----

  /// 当前 input 组装阶段正在处理的数据类型
  FillConvertKind fill_input_kind = FillConvertKind::None;

  /// input 组装缓冲区中是否有正在组装的数据 (至少收到一个包)
  bool fill_input_valid = false;

  /// 当前正在组装的行/子块索引 (对应 AMem/BMem 的行号或 CMem 的子块号)
  uint32_t fill_input_index = 0;

  /// 当前 input 缓冲区中已收集的数据包数量
  uint32_t fill_input_packet_count = 0;

  /// 当前行/子块还缺少多少个数据包才能凑齐一次转换的输入
  uint32_t fill_input_packets_needed = 0;

  /// input 组装缓冲区, 最多暂存 kMaxConversionInputPackets(=4) 个 64B 数据包
  std::array<TmemPacket, kMaxConversionInputPackets> fill_input_packets{};

  // ---- Fill 阶段 3: Conversion (精度转换) ----

  /// 当前转换阶段正在处理的数据类型
  FillConvertKind fill_convert_kind = FillConvertKind::None;

  /// 转换阶段是否持有有效的转换结果, 等待写入本地 SRAM
  bool fill_convert_valid = false;

  /// 转换结果对应的行/子块索引 (与 fill_input_index 对应)
  uint32_t fill_convert_index = 0;

  /// A/B 操作数的转换输出: 8×8 的 fp9 值矩阵。
  /// fp9 是 TensorCore 内部乘法器的原生精度 (E5M3, 9 bit)。
  uint16_t fill_convert_fp9[kPrimitiveDim][kPrimitiveDim] = {};

  /// C 累加器的转换输出: 8×8 的 fp22 值矩阵。
  /// fp22 是 TensorCore 内部累加器的原生精度 (E8M13, 22 bit)。
  uint32_t fill_convert_fp22[kPrimitiveDim][kPrimitiveDim] = {};

  // ---- Store 阶段 1 & 2: Read + Conversion ----

  /// store 路径: 下一个待发射的 TMEM 写数据包索引 (从 0 递增)
  uint32_t next_store_packet_idx = 0;

  /// Store 路径当前所处的显式流水阶段。
  StorePipelineStage store_stage = StorePipelineStage::Read;

  /// store 数据来源: true = 从 DMem 读取 (解耦模式), false = 从 CMem 读取 (就地模式)
  bool store_from_dmem = false;

  /// 是否持有一个有效的原始 (fp22) 子块, 等待精度转换
  bool store_raw_subtile_valid = false;

  /// 从 CMem/DMem 读出的 8×8 fp22 原始值矩阵
  uint32_t store_raw_subtile[kPrimitiveDim][kPrimitiveDim] = {};

  /// 当前子块的下一个待转换的段 (segment) 索引。
  /// 一个 8×8 子块可能需要被分成多段转换, 每段生成一个 64B 输出包。
  /// 段数由输出精度决定: fp8 → 1 段, fp16 → 2 段, fp32 → 4 段。
  uint32_t store_raw_next_segment = 0;

  // ---- Store 阶段 2 → 3: Conversion → Staged ----

  /// 精度转换阶段是否已产生一个有效的输出数据包
  bool store_conv_valid = false;

  /// 精度转换后的 64B 输出数据包 (单包暂存, 等待进入 staged 缓冲区)
  TmemPacket store_conv_packet{};

  // ---- Store 阶段 3 → 4: Staged → Emit ----

  /// 环形输出数据包缓冲区 (深度=2), 实现转换与 TMEM 写的流水线解耦
  std::array<TmemPacket, kOutputPacketBufferDepth> staged_store_packets{};

  /// 环形缓冲区的写指针 (生产者)
  uint32_t staged_store_head = 0;

  /// 环形缓冲区的读指针 (消费者)
  uint32_t staged_store_tail = 0;

  /// 环形缓冲区中当前有效的数据包数量 (0 ~ kOutputPacketBufferDepth)
  uint32_t staged_store_count = 0;
};

// ============================================================================
// WMMA 未就绪原因枚举 (NoWmmaReadyReason)
// ============================================================================
//
// 当 issue 策略模块无法发射新的 WMMA 作业时, 返回此枚举值说明原因。
// 用于调试和性能分析 (性能计数器按原因分类统计停顿拍数)。
// ============================================================================
enum class NoWmmaReadyReason : uint8_t {
  /// WMMA 作业构建器为空: 没有宏 WMMA 指令在等待调度
  JobBuilderEmpty = 0,

  /// 等待 MMA 加载完成: A/B/C 中至少有一个 slot 的 fill 尚未完成
  /// (即 a_ready / b_ready / c_ready 尚未全部为 true)
  WaitingForMmaLoad,

  /// 等待句柄分配: TMEM window planner 无法为当前操作分配 window 句柄
  /// (所有 window 资源被占用)
  WaitingForHandleAlloc,

  /// 等待 slot 释放: 所需的 slot 当前被之前的 WMMA 占用,
  /// 需要等待其完成并释放后才能复用
  WaitingForSlotRelease,
};

// ============================================================================
// Slot 释放原因枚举 (SlotReleaseReason)
// ============================================================================
//
// 标识 slot 释放逻辑被阻塞的具体原因。
// 在 slot 回收的多步检查中, 如果某个条件不满足则返回对应的原因码。
// ============================================================================
enum class SlotReleaseReason : uint8_t {
  /// 无阻塞, slot 可以正常释放
  None = 0,

  /// 等待 C slot 的 in-flight WMMA 原语全部完成 (c_wmma_inflight > 0)。
  /// 在所有原语完成写回之前, CMem 内容不完整, 不能释放。
  CWmmaInflightDrain,

  /// C slot 仅保留累加器数据 (output_resident 且尚未标记 dirty),
  /// 等待 WMMA 开始使用此 slot
  CAccumLiveOnly,

  /// C slot 已脏 (c_dirty=true), 需要先执行 store 回写到 TMEM 才能释放。
  /// 此状态下 slot 正在等待 store 微操作被调度。
  CDirtyFlushOnly,

  /// C slot 的 store 操作正在进行中 (store_pending=true),
  /// 需要等待 store 完成才能释放
  CStorePending,

  /// A 或 B slot 的 wmma_pending=true, 需要等待所有 WMMA 原语完成后才能释放
  AbWmmaPendingClear,
};

// ============================================================================
// MemUop 辅助查询函数
// ============================================================================

/// 查询 MemUop 的有效 payload 格式。
/// 如果 uop 自身携带了明确的 payload_fmt (非 kUnsetPayloadFmt), 则直接使用;
/// 否则根据 uop 的类型 (FillA/FillB/FillC) 从对应的 slot 状态中获取格式。
/// StoreC 不走 fill 路径, 此函数对其返回 0。
inline uint32_t mem_uop_payload_fmt(
    const MemUop& uop,
    const std::array<ASlotState, kNumOperandSlots>& a_slots,
    const std::array<BSlotState, kNumOperandSlots>& b_slots,
    const std::array<CSlotState, kNumOperandSlots>& c_slots) {
  if (uop.payload_fmt != kUnsetPayloadFmt) {
    return uop.payload_fmt;
  }
  switch (uop.kind) {
  case MemUop::Kind::FillA:
    return a_slots.at(uop.slot_id).fmt_a;
  case MemUop::Kind::FillB:
    return b_slots.at(uop.slot_id).fmt_b;
  case MemUop::Kind::FillC:
    return c_slots.at(uop.slot_id).fmt_c;
  default:
    return 0;
  }
}

/// 计算一个 fill 微操作需要从 TMEM 读取的 payload 数据包总数。
/// 根据操作数类型分派到 a_packet_count / b_packet_count / c_load_packet_count。
/// 注意: 此函数不计算稀疏元数据包, 元数据包由 meta_packet_count() 单独计算。
/// StoreC 不是 fill 操作, 调用此函数会触发 std::abort()。
inline uint32_t fill_payload_packet_count(
    const MemUop& uop,
    const std::array<ASlotState, kNumOperandSlots>& a_slots,
    const std::array<BSlotState, kNumOperandSlots>& b_slots,
    const std::array<CSlotState, kNumOperandSlots>& c_slots) {
  auto payload_fmt = mem_uop_payload_fmt(uop, a_slots, b_slots, c_slots);
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
