#pragma once
#include <array>
#include <cstdlib>
#include <cstdint>
#include "fp_types.h"
#include "tc_mul_pipe.h"
#include "tc_add_pipe.h"
#include "tc_add4_pipe_fp22.h"

// ============================================================================
//  tc_mul_add -- 单元素点积累加单元（TensorCore 8x8 矩阵中的一个 (i,j) 位置）
// ============================================================================
//
//  功能描述：
//    本模块实现 TensorCore 中单个输出元素 D[i][j] 的全部计算：
//
//        D[i][j] = A[i][0]*B[0][j] + ... + A[i][7]*B[7][j] + C[i][j]
//
//  流水线结构（共 ~12 拍）：
//
//    阶段            实例                             拍数    说明
//    ───────────────────────────────────────────────────────────────────
//    mul   阶段      8 x mul_pipe（k=0..7 并行）       3      FP9 乘法，8 个 FP22 乘积
//    add4  阶段      2 x add4_pipe_fp22                4      4 输入规约为 1 个部分和
//    merge 阶段      1 x add_pipe_fp22 (merge_add)     2      两个部分和合并
//    final 阶段      1 x add_pipe_fp22 (final_add)     2      加上 C（FIFO 或透传）
//    输出寄存器      r2                                 1      最终 FP22 结果锁存
//    ───────────────────────────────────────────────────────────────────
//
//  final_add 的 C 操作数双路选择：
//
//    ┌── 输出驻留模式 (circulating=true)：
//    │     C 来自 accum_fifo（CMem 预取 → FIFO → 循环反馈累加）
//    │     final_add 结果反馈回 FIFO tail
//    │
//    └── 非输出驻留模式 (circulating=false)：
//          C 来自管线透传 (c_in → mul → add4 → merge → passthrough)
//          final_add 结果输出到外部（DMem）
//
//  c_in / passthrough 透传链（非输出驻留模式专用）：
//    发射时注入 c_in → mul_pipe.c_in → add4.passthrough → merge.passthrough
//    → final_add 的 b_in 操作数
//    该链路在输出驻留模式下仍然存在但不被 final_add 使用。
//
//  accum_fifo（输出驻留模式专用）：
//    2-entry elastic buffer。
//    - 初始阶段：由 CMem 预取 C 值填入
//    - 循环累加阶段：final_add 结果反馈到 FIFO tail
//    - 不变量：FIFO.count + final_add 管线中的 entry 数 = 4（subtile 总数）
// ============================================================================

struct tc_mul_add {

    // ===================== 2-entry elastic accumulator FIFO =====================
    //
    // 输出驻留模式下，替代 c_in 透传链向 final_add 供给 C 操作数。
    // 弹性缓冲规则：
    //   - curr 被消费且 next.valid → promote (next → curr)
    //   - 同拍若有新数据到达 → 可同时 refill next
    //   - next 已满 → 不接受新的 push
    struct accum_fifo_t {
        struct entry_t {
            uint32_t data = 0;
            bool valid = false;
        };

        entry_t curr;   // 队头：当前可消费
        entry_t next;   // 队尾：预取/反馈缓冲

        void reset() {
            curr = {};
            next = {};
        }

        bool can_pop() const { return curr.valid; }

        bool can_push(bool consuming) const {
            return !next.valid || (consuming && next.valid);
        }

        uint32_t peek() const { return curr.data; }

        // 弹性缓冲每拍更新
        void advance(bool consumed, bool push_valid, uint32_t push_data) {
            auto old_curr = curr;
            auto old_next = next;

            if (consumed) {
                if (old_next.valid) {
                    curr = old_next;
                    next = push_valid ? entry_t{push_data, true} : entry_t{};
                } else {
                    curr = {};
                    next = push_valid ? entry_t{push_data, true} : entry_t{};
                }
                return;
            }

            if (push_valid) {
                if (!old_curr.valid) {
                    curr = {push_data, true};
                } else if (!old_next.valid) {
                    next = {push_data, true};
                }
            }
        }

        bool empty() const { return !curr.valid && !next.valid; }
    } accum_fifo;

    // ===================== 子流水线实例 =====================

    std::array<mul_pipe, 8> mul_array;
    std::array<add4_pipe_fp22, 2> add4_level;
    add_pipe_fp22 merge_add;
    add_pipe_fp22 final_add;

    // ===================== 模块输入端口 =====================
    struct {
        uint32_t a_in[8];       // A[i][k]，k = 0..7（FP9 格式）
        uint32_t b_in[8];       // B[k][j]，k = 0..7（FP9 格式）
        uint32_t c_in;          // C[i][j] 旁路数据（非输出驻留模式用，沿管线透传）
        PrecisionType prec;
        bool input_valid;
        TensorCoreMeta meta;
    } mul_add_input;

    // ===================== 输出寄存器 r2 =====================
    struct {
        uint32_t fp22_result;
        bool valid;
        TensorCoreMeta meta;
    } r2;

    // resident 输出驻留模式下，本拍是否发生了一次真实的 FIFO -> final_add
    // 操作数轮转。该脉冲不是“拍数计数器”，而是用于 handoff 控制器判断：
    // resident 回环是否已经实际释放出新的 FIFO 尾部窗口。
    bool resident_turnover_pulse = false;

    // ===================== 复位 =====================
    void reset() {
        for (int i = 0; i < 8; i++) mul_array[i].reset();
        for (int i = 0; i < 2; i++) add4_level[i].reset();
        merge_add.reset();
        final_add.reset();
        accum_fifo.reset();
        r2.fp22_result = 0;
        r2.valid = false;
        r2.meta = {};
        mul_add_input.meta = {};
        mul_add_input.input_valid = false;
        resident_turnover_pulse = false;
    }

    // ===================== 对外接口 =====================

    bool out_valid() const { return r2.valid; }

    bool active() const {
        if (mul_add_input.input_valid || r2.valid) return true;
        for (const auto& mul : mul_array) { if (mul.active()) return true; }
        for (const auto& add : add4_level) { if (add.active()) return true; }
        if (merge_add.active() || final_add.active()) return true;
        if (!accum_fifo.empty()) return true;
        return false;
    }

    // 仅表示算术管线本身是否仍有在途计算，不把 output-resident FIFO 中
    // 已经形成但尚未 drain 到 CMem 的累加值算作“仍在计算”。
    bool pipeline_active() const {
        if (mul_add_input.input_valid || r2.valid) return true;
        for (const auto& mul : mul_array) { if (mul.active()) return true; }
        for (const auto& add : add4_level) { if (add.active()) return true; }
        if (merge_add.active() || final_add.active()) return true;
        return false;
    }

    bool in_ready(bool out_ready) const {
        bool s8_ready = out_ready || !r2.valid;
        bool s7_ready = final_add.in_ready(s8_ready);
        bool s6_ready = merge_add.in_ready(s7_ready);
        bool s5_ready = true;
        for (int i = 0; i < 2; ++i) s5_ready &= add4_level[i].in_ready(s6_ready);
        bool s4_ready = true;
        for (int i = 0; i < 8; ++i) s4_ready &= mul_array[i].in_ready(s5_ready);
        return s4_ready;
    }

    const uint32_t& out_fp22() const { return r2.fp22_result; }
    const TensorCoreMeta& out_meta() const { return r2.meta; }

    // ===================== FIFO 外部加载（CMem 预取用） =====================
    bool load_fifo(uint32_t fp22_value) {
        if (!accum_fifo.curr.valid) {
            accum_fifo.curr = {fp22_value, true};
            return true;
        }
        if (!accum_fifo.next.valid) {
            accum_fifo.next = {fp22_value, true};
            return true;
        }
        return false;
    }

    // ===================== 主时钟推进 =====================
    //
    //  参数:
    //   out_ready   — 下游是否准备好接收 r2
    //   g_cfg       — 全局配置
    //   circulating — 输出驻留循环累加模式
    //                 true:  final_add operand2 = accum_fifo, 结果反馈 FIFO
    //                 false: final_add operand2 = merge_add passthrough (c_bypass)
    void tick(bool out_ready,
              const Config& g_cfg,
              bool circulating = false,
              bool resident_tail_to_dmem_valid = false,
              uint32_t resident_tail_to_dmem_async_id = 0) {
        resident_turnover_pulse = false;

        // ---------- ready 信号 ----------
        bool s8_ready = out_ready || !r2.valid;
        bool s7_ready = final_add.in_ready(s8_ready);
        bool s6_ready = merge_add.in_ready(s7_ready);
        bool s5_ready = true;
        for (int i = 0; i < 2; ++i) s5_ready &= add4_level[i].in_ready(s6_ready);
        bool s4_ready = true;
        for (int i = 0; i < 8; ++i) s4_ready &= mul_array[i].in_ready(s5_ready);

        // ======== s8: final_add → r2 ========
        bool r2_was_valid = false;
        uint32_t r2_feedback_data = 0;

        if (s8_ready) {
            if (final_add.out_valid()) {
                r2.fp22_result = final_add.out_data();
                r2.valid = true;
                r2.meta = final_add.out_meta();
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }
        r2_was_valid = r2.valid;
        r2_feedback_data = r2.fp22_result;

        // ======== s7: merge_add + C操作数 → final_add ========
        //
        // 双路选择：
        //   circulating=true  → operand2 = accum_fifo.peek()
        //   circulating=false → operand2 = merge_add.out_passthrough() (c_bypass 透传)
        bool merge_valid = merge_add.out_valid();
        bool fifo_consumed = false;

        if (s7_ready) {
            if (circulating) {
                // ---- 输出驻留模式：从 FIFO 取 C ----
                bool fifo_ready = accum_fifo.can_pop();
                if (merge_valid && fifo_ready) {
                    final_add.add_input.a_in = merge_add.out_data();
                    final_add.add_input.b_in = accum_fifo.peek();
                    final_add.add_input.input_valid = true;
                    final_add.add_input.meta = merge_add.out_meta();
                    fifo_consumed = true;
                } else {
                    final_add.add_input.input_valid = false;
                    final_add.add_input.meta = {};
                }
            } else {
                // ---- 非输出驻留模式：从管线透传取 C bypass ----
                if (merge_valid) {
                    final_add.add_input.a_in = merge_add.out_data();
                    final_add.add_input.b_in = merge_add.out_passthrough();
                    final_add.add_input.input_valid = true;
                    final_add.add_input.meta = merge_add.out_meta();
                } else {
                    final_add.add_input.input_valid = false;
                    final_add.add_input.meta = {};
                }
            }
            final_add.tick(s8_ready, g_cfg);
        }

        // ======== FIFO advance（仅输出驻留模式） ========
        if (circulating) {
            bool tail_to_dmem = resident_tail_to_dmem_valid
                             && r2_was_valid
                             && (r2.meta.async_id == resident_tail_to_dmem_async_id);
            bool push_valid = r2_was_valid && !tail_to_dmem;
            accum_fifo.advance(fifo_consumed, push_valid, r2_feedback_data);
            resident_turnover_pulse = fifo_consumed;
        }

        // ======== s6: add4 → merge_add ========
        bool r5_valid = true;
        for (int i = 0; i < 2; ++i) r5_valid &= add4_level[i].out_valid();

        if (s6_ready) {
            if (r5_valid) {
                merge_add.add_input.a_in = add4_level[0].out_data();
                merge_add.add_input.b_in = add4_level[1].out_data();
                merge_add.add_input.input_valid = true;
                merge_add.add_input.meta = add4_level[0].out_meta();
            } else {
                merge_add.add_input.input_valid = false;
                merge_add.add_input.meta = {};
            }
            merge_add.tick(s7_ready, g_cfg, add4_level[0].out_passthrough());
        }

        // ======== s5: mul → add4 ========
        bool r4_valid = true;
        for (int i = 0; i < 8; ++i) r4_valid &= mul_array[i].out_valid();

        if (s5_ready) {
            if (r4_valid) {
                for (int i = 0; i < 2; ++i) {
                    const int base = i * 4;
                    add4_level[i].add_input.in[0] = mul_array[base + 0].out_data();
                    add4_level[i].add_input.in[1] = mul_array[base + 1].out_data();
                    add4_level[i].add_input.in[2] = mul_array[base + 2].out_data();
                    add4_level[i].add_input.in[3] = mul_array[base + 3].out_data();
                    add4_level[i].add_input.input_valid = true;
                    add4_level[i].add_input.meta = mul_array[base].out_meta();
                    add4_level[i].tick(s6_ready, g_cfg, mul_array[base].r3.c_in);
                }
            } else {
                for (int i = 0; i < 2; ++i) {
                    const int base = i * 4;
                    add4_level[i].add_input.input_valid = false;
                    add4_level[i].add_input.meta = {};
                    add4_level[i].tick(s6_ready, g_cfg, mul_array[base].r3.c_in);
                }
            }
        }

        // ======== s4: 外部输入 → mul ========
        if (s4_ready) {
            if (mul_add_input.input_valid) {
                for (int i = 0; i < 8; ++i) {
                    mul_array[i].mul_input.a_fp9 = mul_add_input.a_in[i];
                    mul_array[i].mul_input.b_fp9 = mul_add_input.b_in[i];
                    mul_array[i].mul_input.input_valid = true;
                    mul_array[i].mul_input.meta = mul_add_input.meta;
                    mul_array[i].tick(s5_ready, g_cfg, mul_add_input.c_in);
                }
            } else {
                for (int i = 0; i < 8; ++i) {
                    mul_array[i].mul_input.input_valid = false;
                    mul_array[i].mul_input.meta = {};
                    mul_array[i].tick(s5_ready, g_cfg, mul_add_input.c_in);
                }
            }
        }
    }
};
