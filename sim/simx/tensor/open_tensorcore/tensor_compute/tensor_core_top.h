#pragma once

#include <cstdint>
#include "config_register.h"
#include "tc_mul_add.h"
#include "fp_types.h"

// Retired output for one 8x8 primitive.
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

// Top-level 8x8 TensorCore primitive array.
//
// One m16n16k16 macro MMA is decomposed into four 8x8 subtiles and two K
// phases. TensorCoreTop accepts one 8x8 A/B/C primitive, broadcasts it to 64
// scalar tc_mul_add lanes, and retires one 8x8 FP22 subtile result.
struct TensorCoreTop {
    static constexpr int M = 8, K = 8, N = 8;

    // Input staging buffers.
    uint16_t a_in[M][K];
    uint16_t b_in[K][N];
    uint32_t c_in[M][N];           // C bypass data for non-resident mode.
    uint32_t fp22_out[M][N];

    // Control and progress state.
    int jobs_completed = 0;
    int set_jobs = 0;
    bool input_loaded = false;
    int cycle_count = 0;
    bool circulating = false;       // Output-resident circulating accumulation mode.
    bool resident_tail_to_dmem_valid = false;
    uint32_t resident_tail_to_dmem_async_id = 0;

    // 8x8 compute array.
    tc_mul_add tc_dot_product[M][N];

    // One-cycle input metadata staging.
    TensorCoreMeta staged_meta;
    bool staged_valid = false;

    // Retire buffer.
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

    // True when the resident feedback path consumed one FIFO value this tick.
    bool resident_turnover_pulse() const {
        return tc_dot_product[0][0].resident_turnover_pulse;
    }

    // Load the output-resident FIFO with prefetched C values.
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

    // Drain one output-resident FIFO element for writeback.
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

    // Stage one primitive input. c_in is passed through for non-resident mode.
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

    // // Simplified resident-mode overload: C is supplied by the FIFO.
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

    // Advance all 64 lanes by one cycle and collect a retired primitive.
    void tick(bool out_ready) {
        cycle_count++;
        retired.valid = false;

        // Stage 1: broadcast the staged primitive to all 64 lanes.
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

        // Stage 2: advance all lanes; circulating selects the final-add input.
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                tc_dot_product[i][j].tick(out_ready,
                                          g_cfg,
                                          circulating,
                                          resident_tail_to_dmem_valid,
                                          resident_tail_to_dmem_async_id);

        // Stage 3: clear the one-cycle staging registers.
        staged_valid = false;
        input_loaded = false;
        staged_meta = {};

        // Stage 4: retire once every lane produced a valid result.
        //
        // Non-resident mode retires a materialized subtile. Resident mode only
        // means the primitive has entered the circulating accumulator state;
        // final C visibility still waits for FIFO drain.
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
