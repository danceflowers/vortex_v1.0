#pragma once
#include <vector>
#include "fp_types.h"

// =============================================================================
//  全局配置结构体 (由命令行参数填充, 控制仿真行为)
// =============================================================================
struct Config {
    std::vector<PrecisionType> precisions;      // 要测试的输入精度列表
    std::vector<PrecisionType> out_precisions;   // 要测试的输出精度列表
    int  test_id    = 0;       // 测试用例编号: 0 = 运行全部, 1-6 = 指定单个测试
    RoundingMode rm = RNE;     // 浮点舍入模式 (默认: 最近偶数舍入)
    uint32_t seed   = 0;       // 随机数种子: 0 = 使用当前时间
    bool show_help  = false;   // 是否显示帮助信息
};

// =============================================================================
//  TensorCoreMeta — 张量核操作元数据 (随数据在流水线中逐级传递)
// =============================================================================
//
//  该结构体包含一次 8x8 矩阵原语操作所需的全部控制信息。
//  它在 push_uop() 时由上层填入, 随后跟随数据流经 tc_mul_add 内部的
//  多级流水线, 最终随计算结果一起从 retired 输出端取出。
//
//  字段详解:
//
//  --- 标识与寻址字段 ---
//    wgid          — Warpgroup ID, 标识发起此操作的 warpgroup 编号。
//                    用于结果写回时定位目标 warpgroup。
//
//    async_id      — 异步操作 ID, 用于跟踪异步张量指令的完成状态。
//                    上层可通过此 ID 匹配发射(issue)与退休(retire)。
//
//    a_slot_id     — A 矩阵在张量存储器(TMEM)中的槽位编号。
//                    指示从哪个 TMEM slot 读取 A 矩阵数据。
//
//    b_slot_id     — B 矩阵在 TMEM 中的槽位编号。
//                    指示从哪个 TMEM slot 读取 B 矩阵数据。
//
//    c_slot_id     — C 矩阵(累加偏置)在 TMEM 中的槽位编号。
//                    指示从哪个 TMEM slot 读取 C 矩阵数据。
//
//    c_subtile_id  — C 矩阵的子块编号。当大矩阵被分成多个 8x8 子块时,
//                    此字段指示当前操作对应的是 C 矩阵的第几个子块。
//
//  --- 精度控制字段 ---
//    in_prec       — 输入精度类型 (A/B 矩阵的数据格式)。
//                    支持 FP4(E2M1)、FP8(E4M3/E5M2)、FP9(E5M3)、FP16。
//                    计算单元会根据此字段选择对应的格式转换逻辑。
//
//    out_prec      — 输出精度类型 (D 矩阵写回时的目标格式)。
//                    内部始终以 FP22 精度计算, 写回时按此字段转换。
//
//    c_prec        — C 偏置矩阵的精度类型。
//                    决定 C 矩阵从 TMEM 读出后如何转换为内部 FP22 格式。
//
//  --- 数据通路控制标志 ---
//    c_bypass_is_fp22   — C 旁路标志。当为 1 时, 表示 C 数据已经是 FP22 格式,
//                         无需经过格式转换, 直接通过 passthrough 通路参与
//                         最终加法。用于 C 来自上一次矩阵运算结果的场景。
//
//    use_cmem_operand1  — 备用操作数选择。当为 1 时, 最终加法阶段使用
//                         operand1_in 替代 C 矩阵作为加数。operand1 通过
//                         独立的延迟匹配流水线(operand1_pipe)传输, 以保证
//                         与点积结果在同一拍到达最终加法器。
//
//    sparse_mode    — A 操作数结构化稀疏模式。0=dense, 1=2:4, 2=1:4。
//                     tensor_compute 只消费这个压缩后的模式位，不直接依赖
//                     tc_decode 的完整宏指令结构。
//
//    sparse_meta    — 与当前 8x8 primitive 对应的 16B 稀疏元数据行。
//                     当 sparse_mode!=0 时，用它把压缩 A payload 和 dense B
//                     恢复到固定 8-lane primitive 拓扑。
//
//  --- 有效标志 ---
//    valid          — 元数据有效标志。在 push_uop() 时被置为 true,
//                     随数据流经流水线各级, 用于控制各级的有效性判断。
// =============================================================================
struct TensorCoreMeta {
    uint32_t wgid = 0;                             // Warpgroup ID
    uint32_t async_id = 0;                          // 异步操作追踪 ID
    uint32_t a_slot_id = 0;                         // A 矩阵 TMEM 槽位
    uint32_t b_slot_id = 0;                         // B 矩阵 TMEM 槽位
    uint32_t c_slot_id = 0;                         // C 矩阵 TMEM 槽位
    uint32_t c_subtile_id = 0;                      // C 矩阵子块编号
    PrecisionType in_prec = PREC_FP9;               // 输入精度 (A/B)
    PrecisionType out_prec = PREC_FP16;             // 输出精度 (D 写回格式)
    PrecisionType c_prec = PREC_FP16;               // C 偏置精度
    uint8_t c_bypass_is_fp22 = 0;                   // C 旁路: 已为 FP22 格式
    uint8_t use_cmem_operand1 = 0;                  // 使用备用操作数替代 C
    uint8_t sparse_mode = 0;                        // 0=dense, 1=2:4, 2=1:4
    uint8_t sparse_meta[16] = {};                   // 当前 primitive 的 16B 稀疏元数据
    bool valid = false;                             // 元数据有效标志
};

// =============================================================================
//  全局配置实例 (在 main 中由命令行参数初始化)
// =============================================================================
static Config g_cfg;
