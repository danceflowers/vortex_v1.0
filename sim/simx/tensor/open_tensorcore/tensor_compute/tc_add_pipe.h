#pragma once
#include <cstdint>
#include "config_register.h"
#include "fadd_s1.h"
#include "fadd_s2.h"

// =====================================================================================
// fp22_add_bits: 组合逻辑 FP22 加法（非流水线版本）
//   将 fadd_s1 和 fadd_s2 两级合并为单周期调用，用于不需要流水线的场合。
//   FP22 格式：8-bit 指数 + 14-bit 尾数（含隐含位），总共 22 有效位 + 1 符号位。
// =====================================================================================
inline uint32_t fp22_add_bits(uint32_t a_bits, uint32_t b_bits, RoundingMode rm) {
    return fadd_s2(fadd_s1(a_bits, b_bits, 8, 14, 14, rm), 8, 14);
}

// =====================================================================================
// add_pipe —— 两级流水线浮点加法器（中间精度：5-bit 指数，8-bit 尾数）
//
//   用途：在归约树（reduction tree）中完成部分积的逐级累加。归约树将多个乘法结果
//         两两相加，最终归约为一个求和值。本加法器采用中间精度（5-bit exp, 8-bit
//         mantissa）以在面积和精度之间取得平衡——乘积本身为低位宽格式，累加过程
//         无需全精度。
//
//   流水线结构：
//     Stage 1 (r1)：执行 fadd_s1 —— 操作数分类、指数对齐、远路径/近路径并行计算
//     Stage 2 (r2)：执行 fadd_s2 —— 路径选择、舍入、结果组装，输出最终加法结果
//
//   c_in 透传信号：
//     C 矩阵旁路数据（bypass data）随流水线向下传递，不参与加法运算。该信号将
//     C 操作数与对应的归约结果保持同步，以便后续 merge 阶段将归约结果与 C 累加。
//
//   反压机制（backpressure / in_ready-out_valid 握手）：
//     采用 valid-ready 握手协议。每一级只有在下游就绪（ready）或本级无有效数据
//     时才接受新数据。反压从输出端逐级向输入端传播：
//       s2_ready = out_ready || !r2.valid
//       s1_ready = s2_ready  || !r1.valid
//     当下游阻塞时，流水线自动暂停，不会丢失数据。
// =====================================================================================
struct add_pipe {

    // -------------------------------------------------------
    // 输入寄存器：外部在 tick() 之前写入 a_in、b_in、input_valid
    // -------------------------------------------------------
    struct {
        uint32_t a_in;           // 加法操作数 A（中间精度浮点位模式）
        uint32_t b_in;           // 加法操作数 B（中间精度浮点位模式）
        bool input_valid;        // 输入数据有效标志
        TensorCoreMeta meta;     // 随路元数据（warp id、slot id 等）
    } add_input;

    // ===================================
    //      流水线寄存器 (Pipeline Registers)
    // ===================================

    // ---------------------------------------------------------
    // 第一级寄存器 r1：保存 fadd_s1 的输出
    //   fadd_s1 完成操作数分类、指数差计算、远路径加/减与移位、
    //   近路径减法与前导零计数，输出中间结构体 fadd_s1_out。
    // ---------------------------------------------------------
    struct {
        fadd_s1_out s1_result;   // fadd_s1 输出：远/近路径中间结果
        uint32_t c_in;           // 透传的 C 旁路数据
        bool valid;              // 本级数据有效标志
        TensorCoreMeta meta;     // 随路元数据
    } r1;

    // ---------------------------------------------------------
    // 第二级寄存器 r2：保存 fadd_s2 的输出（最终加法结果）
    //   fadd_s2 根据路径选择信号选取远路径或近路径结果，
    //   执行舍入（rounding）并组装为最终浮点位模式。
    // ---------------------------------------------------------
    struct {
        uint32_t result;         // 最终加法结果（中间精度浮点位模式）
        uint32_t c_in;           // 透传的 C 旁路数据
        bool valid;              // 本级数据有效标志
        TensorCoreMeta meta;     // 随路元数据
    } r2;

    // -------------------------------------------------------
    // 复位：清除所有流水线级的有效位
    // -------------------------------------------------------
    void reset() {
        r1.valid = false;
        r2.valid = false;
        add_input.input_valid = false;
        add_input.meta = {};
    }

    // ===================================
    //  外部接口 / External Interface
    // ===================================

    // 输出有效：第二级寄存器持有有效结果时为 true
    bool out_valid() const {
        return r2.valid;
    }

    // 流水线活跃：任何一级持有有效数据时返回 true，用于外部判断是否可安全关断时钟
    bool active() const {
        return add_input.input_valid || r1.valid || r2.valid;
    }

    // 输入就绪：根据下游 out_ready 信号逐级反压，返回本加法器能否接受新输入
    //   反压链：s2_ready = out_ready || !r2.valid
    //           s1_ready = s2_ready  || !r1.valid
    //   返回 s1_ready，即第一级是否可以接受新数据
    bool in_ready(bool out_ready) const {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;
        return s1_ready;
    }

    // 输出数据：返回第二级寄存器中的加法结果
    const uint32_t& out_data() const {
        return r2.result;
    }

    // 输出元数据：返回与结果对应的 TensorCoreMeta
    const TensorCoreMeta& out_meta() const {
        return r2.meta;
    }

    // -------------------------------------------------------
    // tick()：每时钟周期调用一次，驱动两级流水线向前推进
    //   out_ready —— 下游是否准备好接收本级输出
    //   g_cfg     —— 全局配置（舍入模式等）
    //   c_in      —— 当拍输入的 C 旁路数据（透传，不参与加法）
    //
    //   更新顺序：先更新后级（r2），再更新前级（r1），
    //   确保同一拍内不会覆盖尚未被后级读取的数据。
    // -------------------------------------------------------
    void tick(bool out_ready, const Config& g_cfg, uint32_t c_in) {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;

        // --- 第二级：从 r1 读取 fadd_s1 中间结果，执行 fadd_s2 完成舍入 ---
        if (s2_ready) {
            if (r1.valid) {
                r2.result = fadd_s2(r1.s1_result, 5, 8);
                r2.c_in = r1.c_in;
                r2.valid = true;
                r2.meta = r1.meta;
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }

        // --- 第一级：从输入读取操作数，执行 fadd_s1 完成对齐与远/近路径计算 ---
        if (s1_ready) {
            if (add_input.input_valid) {
                r1.s1_result = fadd_s1(add_input.a_in, add_input.b_in, 5, 8, 8, g_cfg.rm);
                r1.c_in = c_in;
                r1.valid = true;
                r1.meta = add_input.meta;
            } else {
                r1.valid = false;
                r1.meta = {};
            }
        }
    }
};

// =====================================================================================
// add_pipe_fp22 —— 两级流水线 FP22 加法器（高精度：8-bit 指数，14-bit 尾数）
//
//   用途：在 merge 阶段和最终累加（final accumulation）中使用。归约树输出的中间
//         精度求和结果需要与 C 矩阵进行累加，此时精度提升为 FP22（8-bit exp,
//         14-bit mantissa）以保证累加精度，避免精度损失在多次累加中放大。
//
//   流水线结构（与 add_pipe 相同的两级划分）：
//     Stage 1 (r1)：执行 fadd_s1 —— 操作数分类、指数对齐、远/近路径并行计算
//     Stage 2 (r2)：执行 fadd_s2 —— 路径选择、舍入、结果组装
//
//   passthrough 透传信号：
//     与 add_pipe 中的 c_in 类似，passthrough 信号随流水线向下传递，不参与运算。
//     在 merge 阶段，该信号用于携带 C 旁路数据或其他需要与加法结果保持时序对齐
//     的辅助数据，使下游模块可以在同一拍获得加法结果及其对应的旁路数据。
//
//   反压机制（backpressure / in_ready-out_valid 握手）：
//     与 add_pipe 完全相同的 valid-ready 协议，反压从输出端逐级传播至输入端：
//       s2_ready = out_ready || !r2.valid
//       s1_ready = s2_ready  || !r1.valid
// =====================================================================================
struct add_pipe_fp22 {

    // -------------------------------------------------------
    // 输入寄存器：外部在 tick() 之前写入 a_in、b_in、input_valid
    // -------------------------------------------------------
    struct {
        uint32_t a_in;           // 加法操作数 A（FP22 浮点位模式）
        uint32_t b_in;           // 加法操作数 B（FP22 浮点位模式）
        bool input_valid;        // 输入数据有效标志
        TensorCoreMeta meta;     // 随路元数据（warp id、slot id 等）
    } add_input;

    // ===================================
    //      流水线寄存器 (Pipeline Registers)
    // ===================================

    // ---------------------------------------------------------
    // 第一级寄存器 r1：保存 fadd_s1 的输出
    //   fadd_s1 以 FP22 参数（exp=8, prec=14）运行，完成操作数分类、
    //   指数对齐、远路径加/减与移位、近路径减法与前导零计数。
    // ---------------------------------------------------------
    struct {
        fadd_s1_out s1_result;   // fadd_s1 输出：远/近路径中间结果
        uint32_t passthrough;    // 透传信号：随流水线传递的旁路数据
        bool valid;              // 本级数据有效标志
        TensorCoreMeta meta;     // 随路元数据
    } r1;

    // ---------------------------------------------------------
    // 第二级寄存器 r2：保存 fadd_s2 的输出（最终 FP22 加法结果）
    //   fadd_s2 以 FP22 参数（exp=8, prec=14）运行，选取远/近路径
    //   结果，执行舍入并组装为最终 FP22 浮点位模式。
    // ---------------------------------------------------------
    struct {
        uint32_t result;         // 最终加法结果（FP22 浮点位模式）
        uint32_t passthrough;    // 透传信号：与结果对齐的旁路数据
        bool valid;              // 本级数据有效标志
        TensorCoreMeta meta;     // 随路元数据
    } r2;

    // -------------------------------------------------------
    // 复位：清除所有流水线级的有效位和透传数据
    // -------------------------------------------------------
    void reset() {
        r1.valid = false;
        r1.passthrough = 0;
        r2.valid = false;
        r2.passthrough = 0;
        add_input.input_valid = false;
        add_input.meta = {};
    }

    // ===================================
    //  外部接口 / External Interface
    // ===================================

    // 输出有效：第二级寄存器持有有效结果时为 true
    bool out_valid() const {
        return r2.valid;
    }

    // 流水线活跃：任何一级持有有效数据时返回 true
    bool active() const {
        return add_input.input_valid || r1.valid || r2.valid;
    }

    // 输入就绪：根据下游 out_ready 逐级反压，返回是否可接受新输入
    bool in_ready(bool out_ready) const {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;
        return s1_ready;
    }

    // 输出数据：返回第二级寄存器中的 FP22 加法结果
    const uint32_t& out_data() const {
        return r2.result;
    }

    // 输出元数据：返回与结果对应的 TensorCoreMeta
    const TensorCoreMeta& out_meta() const {
        return r2.meta;
    }

    // 输出透传数据：返回与结果对齐的 passthrough 旁路值
    uint32_t out_passthrough() const {
        return r2.passthrough;
    }

    // -------------------------------------------------------
    // tick()：每时钟周期调用一次，驱动两级流水线向前推进
    //   out_ready   —— 下游是否准备好接收本级输出
    //   g_cfg       —— 全局配置（舍入模式等）
    //   passthrough —— 当拍输入的透传旁路数据（不参与加法，沿流水线传递）
    //
    //   更新顺序：先更新后级（r2），再更新前级（r1），
    //   确保同一拍内不会覆盖尚未被后级读取的数据。
    // -------------------------------------------------------
    void tick(bool out_ready, const Config& g_cfg, uint32_t passthrough = 0) {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;

        // --- 第二级：从 r1 读取 fadd_s1 中间结果，执行 fadd_s2 完成 FP22 舍入 ---
        if (s2_ready) {
            if (r1.valid) {
                r2.result = fadd_s2(r1.s1_result, 8, 14);
                r2.passthrough = r1.passthrough;
                r2.valid = true;
                r2.meta = r1.meta;
            } else {
                r2.passthrough = 0;
                r2.valid = false;
                r2.meta = {};
            }
        }

        // --- 第一级：从输入读取操作数，执行 fadd_s1 完成 FP22 对齐与远/近路径计算 ---
        if (s1_ready) {
            if (add_input.input_valid) {
                r1.s1_result = fadd_s1(add_input.a_in, add_input.b_in, 8, 14, 14, g_cfg.rm);
                r1.passthrough = passthrough;
                r1.valid = true;
                r1.meta = add_input.meta;
            } else {
                r1.passthrough = 0;
                r1.valid = false;
                r1.meta = {};
            }
        }
    }
};
