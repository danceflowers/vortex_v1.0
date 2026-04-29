#pragma once
#include <cstdint>
#include "config_register.h"
#include "fmul_s1.h"
#include "fmul_s2.h"
#include "fmul_s3.h"

// =============================================================================
//  mul_pipe — 三级流水线 FP9 乘法器 (3-Stage Pipelined FP9 Multiplier)
// =============================================================================
//
//  整体架构说明 (Overall Architecture)
//  ---------------------------------------------------------------------------
//  本模块是 8x8 TensorCore 中用于点积运算的单个乘法器单元的周期精确
//  (cycle-accurate) C++ 模拟模型。在硬件 TensorCore 中，一次矩阵乘法需要
//  对每个输出元素执行多次 FP9 乘法并累加。每个 mul_pipe 实例负责完成
//  其中一次 a * b 的乘法运算。
//
//  乘法被拆分为三级流水线，与 RTL 的寄存器级完全对应：
//
//    Stage 1 (fmul_s1) — 指数计算与特殊值检测
//        输入原始 FP9(E5M3) 位域，将其展宽到内部 FP22 算术域，
//        计算指数之和、检测 NaN/Inf/零等特殊情况，并计算次正规数
//        (subnormal) 的移位量。
//
//    Stage 2 (fmul_s2) — 尾数乘法
//        对 Stage 1 展宽后的有效数字 (significand) 进行整数乘法，
//        得到 2*PRECISION 位的原始乘积 (raw product)，同时透传
//        Stage 1 的所有中间结果。
//
//    Stage 3 (fmul_s3) — 规格化、舍入与特殊值封装
//        对乘积进行移位规格化 (normalization)，执行 IEEE 754 兼容的
//        舍入 (rounding)，处理上溢 (overflow) / 下溢 (underflow) /
//        特殊值，最终输出一个打包好的浮点结果。
//
//  反压 / 握手机制 (Backpressure / Handshaking)
//  ---------------------------------------------------------------------------
//  本流水线使用 valid/ready 握手协议实现反压 (backpressure)：
//
//    - out_valid()  : 当 r3 寄存器持有有效结果时为 true，表示输出端
//                     有数据可以被下游消费。
//    - in_ready()   : 当流水线能够接受新的输入数据时为 true。
//                     其计算方式是从输出端向输入端反向传播就绪信号：
//                       s3_ready = out_ready || !r3.valid
//                       s2_ready = s3_ready  || !r2.valid
//                       s1_ready = s2_ready  || !r1.valid
//                     只要下游能消费数据、或者本级寄存器为空(气泡)，
//                     该级就处于就绪状态。这种链式逻辑保证了在任意一级
//                     被阻塞时，数据不会被覆盖丢失。
//
//  周期精确时序模型 (Cycle-Accurate Timing Model)
//  ---------------------------------------------------------------------------
//  每次调用 tick() 模拟一个时钟上升沿。在 tick() 内部，流水线寄存器
//  的更新顺序是从后往前 (r3 -> r2 -> r1)，这与硬件中寄存器在同一
//  时钟沿同时采样的行为等效——因为先更新后级寄存器，再更新前级时，
//  前级写入的新值不会在同一拍内传播到后级，从而保证了流水线行为的
//  正确性。
// =============================================================================

struct mul_pipe {

    // =========================================================================
    //  输入锁存 (Input Latch)
    // =========================================================================
    //  外部在调用 tick() 之前将待乘的两个 FP9 操作数和控制信息写入
    //  mul_input。input_valid 指示本拍是否有有效输入数据。
    //  meta 携带该乘法对应的 TensorCore 控制信息（warp id、slot id 等），
    //  随数据一起沿流水线向下传递。
    struct {
        uint32_t a_fp9;           // FP9 操作数 A（E5M3 格式，低 9 位有效）
        uint32_t b_fp9;           // FP9 操作数 B（E5M3 格式，低 9 位有效）
        bool input_valid;         // 输入有效标志
        TensorCoreMeta meta;      // 附带的 TensorCore 元信息
    } mul_input;

    // =========================================================================
    //  流水线寄存器 (Pipeline Registers)
    // =========================================================================

    // ---- 第一级寄存器 r1 (Stage 1 Register) ----
    //  保存 fmul_s1 的计算结果：指数之和、特殊值标志、移位量等。
    //  同时保留原始 FP9 操作数位域 (a_fp9, b_fp9)，因为 Stage 2 的
    //  尾数乘法仍需使用由 s1 解包后的有效数字。
    //  c_in 是来自累加寄存器的旧值，沿流水线透传到输出端以供后续加法。
    struct {
        fmul_s1_out s1_result;    // Stage 1 输出：指数、特殊值标志等
        uint16_t a_fp9;           // 截断为 16 位的原始 FP9 操作数 A
        uint16_t b_fp9;           // 截断为 16 位的原始 FP9 操作数 B
        uint32_t c_in;            // 透传的累加输入 C
        bool valid;               // 本级数据有效标志
        TensorCoreMeta meta;      // 透传的元信息
    } r1;

    // ---- 第二级寄存器 r2 (Stage 2 Register) ----
    //  保存 fmul_s2 的计算结果：包含 2*PRECISION 位原始乘积 (prod)
    //  以及从 Stage 1 透传的所有中间量。
    struct {
        fmul_s2_out s2_result;    // Stage 2 输出：乘积 + 透传的 s1 结果
        uint32_t c_in;            // 透传的累加输入 C
        bool valid;               // 本级数据有效标志
        TensorCoreMeta meta;      // 透传的元信息
    } r2;

    // ---- 第三级寄存器 r3 (Stage 3 Register) ----
    //  保存 fmul_s3 的最终结果：经过规格化和舍入后的打包浮点值。
    //  这就是流水线的输出端，当 r3.valid 为 true 时，result 中
    //  存放着可供下游消费的乘法结果。
    struct {
        uint32_t result;          // 最终打包浮点乘法结果
        uint32_t c_in;            // 透传的累加输入 C
        bool valid;               // 输出数据有效标志（即 out_valid）
        TensorCoreMeta meta;      // 透传的元信息
    } r3;

    // =========================================================================
    //  复位 (Reset)
    // =========================================================================
    //  将所有流水线寄存器的 valid 标志清零，使整条流水线进入空闲状态。
    //  等价于硬件上电复位或软件清空流水线。
    void reset() {
        r1.valid = false;
        r2.valid = false;
        r3.valid = false;
        mul_input.input_valid = false;
        mul_input.meta = {};
    }

    // =========================================================================
    //  外部查询接口 (External Query Interface)
    // =========================================================================

    // 输出有效 — 当第三级寄存器持有已完成的乘法结果时返回 true。
    // 下游模块（如累加器）通过此信号判断是否可以读取 out_data()。
    bool out_valid() const {
        return r3.valid;
    }

    // 流水线活跃 — 只要任意一级寄存器或输入端持有有效数据，就返回 true。
    // 可用于判断整条流水线是否已完全排空 (drain)，在仿真结束时
    // 确保所有 in-flight 的运算都已完成。
    bool active() const {
        return mul_input.input_valid || r1.valid || r2.valid || r3.valid;
    }

    // 输入就绪 — 反压链：从输出端向输入端逐级计算 ready 信号。
    //  s3_ready: 如果下游可以接收 (out_ready) 或 r3 本身为空，则第三级就绪
    //  s2_ready: 如果 s3 就绪或 r2 本身为空，则第二级就绪
    //  s1_ready: 如果 s2 就绪或 r1 本身为空，则第一级就绪
    // 返回 s1_ready，即流水线入口是否可以接受新数据。
    bool in_ready(bool out_ready) const {
        bool s3_ready = out_ready || !r3.valid;
        bool s2_ready = s3_ready  || !r2.valid;
        bool s1_ready = s2_ready  || !r1.valid;
        return s1_ready;
    }

    // 输出数据 — 返回第三级寄存器中打包好的浮点乘法结果。
    // 调用前应先检查 out_valid() 为 true。
    const uint32_t& out_data() const {
        return r3.result;
    }

    // 输出元信息 — 返回与当前输出结果对应的 TensorCore 控制元信息。
    const TensorCoreMeta& out_meta() const {
        return r3.meta;
    }

    // =========================================================================
    //  时钟节拍 (Clock Tick)
    // =========================================================================
    //  每次调用模拟一个时钟上升沿。参数说明：
    //    out_ready — 下游模块是否可以接收本流水线的输出
    //    g_cfg     — 全局配置（包含舍入模式 rm 等）
    //    c_in      — 来自累加寄存器的当前值，随乘法结果一起透传
    //
    //  更新顺序：r3 -> r2 -> r1（从后往前），以保证同一拍内前级写入
    //  的新值不会立即传播到后级，与硬件并行采样语义一致。
    void tick(bool out_ready, const Config& g_cfg, uint32_t& c_in) {

        // ---- 计算各级就绪信号 (Compute per-stage ready signals) ----
        bool s3_ready = out_ready || !r3.valid;
        bool s2_ready = s3_ready  || !r2.valid;
        bool s1_ready = s2_ready  || !r1.valid;

        // ---- 第三级：规格化 + 舍入 (Stage 3: Normalization + Rounding) ----
        //  当 s3_ready 为 true 时，本级可以更新：
        //    - 若 r2 有效，调用 fmul_s3 完成规格化与舍入，产生最终结果
        //    - 若 r2 无效，将 r3 标记为气泡 (bubble)
        if (s3_ready) {
            if (r2.valid) {
                r3.result = fmul_s3(r2.s2_result, 8, 14);
                r3.c_in = r2.c_in;
                r3.valid = true;
                r3.meta = r2.meta;
            } else {
                r3.valid = false;
                r3.meta = {};
            }
        }

        // ---- 第二级：尾数乘法 (Stage 2: Mantissa Multiplication) ----
        //  当 s2_ready 为 true 时，本级可以更新：
        //    - 若 r1 有效，调用 fmul_s2 执行有效数字的整数乘法
        //    - 若 r1 无效，将 r2 标记为气泡
        if (s2_ready) {
            if (r1.valid) {
                r2.s2_result = fmul_s2(r1.a_fp9, r1.b_fp9, 8, 14, r1.s1_result);
                r2.c_in = r1.c_in;
                r2.valid = true;
                r2.meta = r1.meta;
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }

        // ---- 第一级：指数计算 + 特殊值检测 (Stage 1: Exponent + Special Cases) ----
        //  当 s1_ready 为 true 时，本级可以接受新输入：
        //    - 若 mul_input 有效，截断 FP9 操作数为 16 位后调用 fmul_s1
        //      进行指数运算和特殊值检测
        //    - 若 mul_input 无效，将 r1 标记为气泡
        if (s1_ready) {
            if (mul_input.input_valid) {
                r1.a_fp9 = static_cast<uint16_t>(mul_input.a_fp9);
                r1.b_fp9 = static_cast<uint16_t>(mul_input.b_fp9);
                r1.c_in = c_in;
                r1.s1_result = fmul_s1(r1.a_fp9, r1.b_fp9, 8, 14, g_cfg.rm);
                r1.valid = true;
                r1.meta = mul_input.meta;
            } else {
                r1.valid = false;
                r1.meta = {};
            }
        }
    }
};
