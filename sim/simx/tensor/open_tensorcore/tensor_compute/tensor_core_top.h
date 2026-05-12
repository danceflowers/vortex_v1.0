#pragma once

#include <cstdint>
#include "config_register.h"
#include "tc_mul_add.h"
#include "fp_types.h"

// =============================================================================
//  TensorCoreRetire — 张量核退休(输出)数据结构
// =============================================================================
struct TensorCoreRetire {
    static constexpr int M = 8;
    static constexpr int N = 8;

    TensorCoreMeta meta;
    uint32_t fp22_out[M][N];
    bool valid = false;

    void reset() {
        meta = {};
        valid = false;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                fp22_out[i][j] = 0;
    }
};

// =============================================================================
//  TensorCoreTop — 8x8 顶层模块
// =============================================================================
//
//  m16n16k16 基本计算粒度，4 个 8x8 subtile per WMMA。
//
//  C 操作数双路供给：
//    - 非输出驻留模式：c_in[M][N] 从发射时注入，沿管线透传到 final_add
//    - 输出驻留模式：C 通过 accum_fifo 预取，循环累加
//
//  push_uop 保留 c_in 参数用于非输出驻留模式的透传。
//  输出驻留模式下 c_in 可传零（不使用），C 由 FIFO 提供。
// =============================================================================
struct TensorCoreTop {
    static constexpr int M = 8, K = 8, N = 8;

    // ---- 输入缓冲区 ----
    uint16_t a_in[M][K];
    uint16_t b_in[K][N];
    uint32_t c_in[M][N];           // 非输出驻留模式的 C bypass 数据
    uint32_t fp22_out[M][N];

    // ---- 控制/状态 ----
    int jobs_completed = 0;
    int set_jobs = 0;
    bool input_loaded = false;
    int cycle_count = 0;
    bool circulating = false;       // 输出驻留循环累加模式
    bool resident_tail_to_dmem_valid = false;
    uint32_t resident_tail_to_dmem_async_id = 0;

    // ---- 8x8 计算阵列 ----
    tc_mul_add tc_dot_product[M][N];

    // ---- 暂存区 ----
    TensorCoreMeta staged_meta;
    bool staged_valid = false;

    // ---- 退休缓冲区 ----
    TensorCoreRetire retired;

    // =========================================================================
    void reset() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].reset();
                fp22_out[i][j] = 0;
                c_in[i][j] = 0;
            }
        }
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                a_in[i][k] = 0;
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                b_in[k][j] = 0;
        staged_meta = {};
        staged_valid = false;
        retired.reset();
        input_loaded = false;
        cycle_count = 0;
        set_jobs = 1;
        jobs_completed = 0;
        circulating = false;
        resident_tail_to_dmem_valid = false;
        resident_tail_to_dmem_async_id = 0;
    }

    bool ready(bool out_ready = true) const {
        if (staged_valid) return false;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (!tc_dot_product[i][j].in_ready(out_ready))
                    return false;
        return true;
    }

    bool active() const {
        if (staged_valid || retired.valid) return true;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (tc_dot_product[i][j].active())
                    return true;
        return false;
    }

    bool pipeline_active() const {
        if (staged_valid || retired.valid) return true;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (tc_dot_product[i][j].pipeline_active())
                    return true;
        return false;
    }

    bool fifo_empty() const {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (!tc_dot_product[i][j].accum_fifo.empty())
                    return false;
        return true;
    }

    // resident 回环在本拍是否真实发生了一次 FIFO -> final_add 的消费/轮转。
    // 所有 PE 的 resident 时序保持一致，因此采样 [0][0] 即可。
    bool resident_turnover_pulse() const {
        return tc_dot_product[0][0].resident_turnover_pulse;
    }

    // ---- FIFO 加载（输出驻留模式，CMem 预取用） ----
    bool load_fifo_element(int i, int j, uint32_t fp22_val) {
        return tc_dot_product[i][j].load_fifo(fp22_val);
    }

    bool load_fifo_subtile(const uint32_t c[M][N]) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (!tc_dot_product[i][j].load_fifo(c[i][j]))
                    return false;
        return true;
    }

    // ---- FIFO 弹出（排空阶段，写回 CMem 用） ----
    bool pop_fifo_element(int i, int j, uint32_t* out) {
        auto& fifo = tc_dot_product[i][j].accum_fifo;
        if (!fifo.can_pop()) return false;
        if (out) *out = fifo.peek();
        fifo.advance(true, false, 0);
        return true;
    }

    void set_circulating(bool enable) {
        circulating = enable;
    }

    void set_resident_tail_to_dmem(bool valid, uint32_t async_id) {
        resident_tail_to_dmem_valid = valid;
        resident_tail_to_dmem_async_id = async_id;
    }

    // ---- push_uop：保留 c_in 用于非输出驻留模式透传 ----
    void push_uop(const uint16_t a[M][K],
                  const uint16_t b[K][N],
                  const uint32_t c[M][N],
                  const TensorCoreMeta& meta) {
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                a_in[i][k] = a[i][k];
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                b_in[k][j] = b[k][j];
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                c_in[i][j] = c[i][j];
        staged_meta = meta;
        staged_meta.valid = true;
        staged_valid = true;
        input_loaded = true;
    }

    // // 输出驻留模式简化版（不传 C，C 由 FIFO 提供）
    // void push_uop(const uint16_t a[M][K],
    //               const uint16_t b[K][N],
    //               const TensorCoreMeta& meta) {
    //     uint32_t zero_c[M][N] = {};
    //     push_uop(a, b, zero_c, meta);
    // }

    bool pop_retired(TensorCoreRetire* out) {
        if (!retired.valid) return false;
        if (out != nullptr) *out = retired;
        retired.valid = false;
        return true;
    }

    // ---- tick ----
    void tick(bool out_ready) {
        cycle_count++;
        retired.valid = false;

        // 阶段 1: 广播输入到 64 个计算单元
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    tc_dot_product[i][j].mul_add_input.a_in[k] = a_in[i][k];
                    tc_dot_product[i][j].mul_add_input.b_in[k] = b_in[k][j];
                }
                tc_dot_product[i][j].mul_add_input.c_in = c_in[i][j];
                tc_dot_product[i][j].mul_add_input.prec =
                    staged_valid ? staged_meta.in_prec : PREC_FP9;
                tc_dot_product[i][j].mul_add_input.input_valid = staged_valid;
                tc_dot_product[i][j].mul_add_input.meta =
                    staged_valid ? staged_meta : TensorCoreMeta{};
            }
        }

        // 阶段 2: 推进 64 个单元（circulating 控制 final_add 路径选择）
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                tc_dot_product[i][j].tick(out_ready,
                                          g_cfg,
                                          circulating,
                                          resident_tail_to_dmem_valid,
                                          resident_tail_to_dmem_async_id);

        // 阶段 3: 清除暂存区
        staged_valid = false;
        input_loaded = false;
        staged_meta = {};

        // 阶段 4: 完成检测
        //
        // 非输出驻留模式:
        //   - retire 表示结果已经 materialize 为一个可写回 CMem/DMem 的 subtile
        //
        // 输出驻留模式:
        //   - retire 表示一个 primitive 的结果已经成功进入 circulating accum state
        //   - 这时可将该 primitive 记为 completed, 但不能把 C slot 标成
        //     cmem_final_valid; 最终结果仍要等待 FIFO drain 回写到 CMem
        bool all_done = true;
        for (int i = 0; i < M && all_done; i++)
            for (int j = 0; j < N && all_done; j++)
                if (!tc_dot_product[i][j].out_valid())
                    all_done = false;

        if (all_done) {
            jobs_completed++;
            retired.valid = true;
            retired.meta = tc_dot_product[0][0].out_meta();
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    fp22_out[i][j] = tc_dot_product[i][j].out_fp22();
                    retired.fp22_out[i][j] = fp22_out[i][j];
                }
        }
    }
};
